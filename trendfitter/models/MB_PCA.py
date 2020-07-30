from numpy import min, sum, mean, std, var, insert, array, multiply, where, zeros, \
                  append, isnan, nan_to_num, nansum, nanvar, nanmean, unique, ix_, \
                  nonzero, nan, diagonal, sqrt
from numpy.linalg import norm
from scipy.stats import f, chi2
from pandas import DataFrame
from sklearn.model_selection import KFold
from trendfitter.auxiliary.tf_aux import scores_with_missing_values
from . import PCA

class MB_PCA:
    """
    A sklearn-like class for the Multi-Block Principal Component Analysis.

        Parameters
        ----------
        cv_splits_number : int, optional
            number of splits used for cross validation in case latent_variables is None
        tol : float, optional
            value used to decide if model has converged
        loop_limit : int, optional
            maximum number of loops for the extraction of one latent variable
        missing_values_method : str, optional
            string to define the method that the model will deal with missing values

        Attributes
        ----------
        principal_components : int, optional
            Number of principal components extracted or to extract. If not given, a cross-validation
                internal routine will decide this value.
        loadings : array_like
            Loading parameters that define the PCA model.
        q2 : [float]
            list of r² coefficients extracted during cross validation        
        training_scores : array_like
            Scores extracted during training of the model.
        superlever_training_scores : array_like
            Scores extracted during training of the model for the superlevel matrix.
        omega : array_like
            If missing_values_method requires a scores covariance matrix ('TSR', 'CMR', 'PMP'), 
                it will be stored here. 
        omega_b : [array_like]
            If missing_values_method requires a scores covariance matrix ('TSR', 'CMR', 'PMP'), 
                the matrixes for each block will be stored here. 
        block_divs : [int]
            list with the index of every block final position. ex. [2,4] means two blocks with 
                columns 0 and 1 on the first block and columns 2 and 3 on the second block. 
                Assigned when fit method runs.
        block_loadings : [array_like]
            list of p loadings arrays of every block with all the extracted latent variables 
        superlevel_loadings : array_like
            array of all latent variables extracted p loadings for the super level
        feature_importances_ : [float]
            list of values that represent how important is each variable in the same order 
                of the X columns on the first matrix

        Methods
        -------
        fit(X, blocks_sep)
            Applies the NIPALS like method to find the best parameters that adjust the model 
                to the data
        transform(X)
            Transforms the original data to the latent variable space in the super level 
        transform_b(X, block)
            Transforms the original data to the latent variable space in the block level for
                all blocks
        predict(X)
            Predicts Y values 
        score(X)
            Returns a r² representing how much variability from X is captured in the model.
        Hotellings_T2(X)
            Calculates Hotellings_T2 values for the X data in the super level
        T2_limit(alpha)
           Returns the Hotelling's T² limit estimated with alpha confidence level
        SPEs(X)
            Calculates squared prediction errors on the X side for the super level
        SPE_limit(alpha)
            Returns the squared prediction error limit with alpha confidence level
        contributions_scores_ind(X, principal_components = None)
            Returns an array with the contributions to the scores of each X row.
        contributions_spe(X)
            calculates the contributions of each variable to the SPE on the X side for
                the super level

    """
    def __init__(self, cv_splits_number = 7, tol = 1e-16, loop_limit = 1000, missing_values_method = 'TSM'):
        
        # Parameters
        
        self.cv_splits_number = cv_splits_number # number of splits for latent variable cross-validation
        self.tol = tol # criteria for convergence
        self.loop_limit = loop_limit # maximum number of loops before convergence is decided to be not attainable        
        self.missing_values_method = missing_values_method
        
        # Attributes

        self.principal_components = None # number of principal components to be extracted
        self.loadings = None
        self.q2 = [] # list of cross validation scores
        self.block_divs = None
        self.block_loadings = None
        self.superlevel_loadings = None
        self.omega = None # score covariance matrix for missing value estimation
        self.omega_b = []
        self.training_scores = None
        self.superlevel_training_scores = None
        self.feature_importances_ = None
        self._chi2_params = None
        self._pca = None
        
    def fit(self, X, block_divs, principal_components = None, deflation = 'both', int_call = False): #require test

        """
        Adjusts the model parameters to best fit the Y using the algorithm defined in 
            Westerhuis et al's [1]

        Parameters
        ----------
        X : array_like
            Matrix with all the data to be used as predictors in one only object
        block_divs : [int]
            list with the index of every block final position. ex. [2,4] means two blocks with 
                columns 0 and 1 on the first block and columns 2 and 3 on the second block.
        Y : array_like
            Matrix with all the data to be predicted in one only object
        latent_variables : int, optional
            Number of latent variables deemed relevant from each block. If left unspecified
                a cross validation routine will define the number during fitting
        deflation : str, optional
            string defining method of deflation, only Y or both X and Y 
        int_call : Boolean, optional
            Flag to define if it is an internal call on the cross validation routine and decide
                if it is necessary to calculate the VIP values

        References 
        ----------
        [1] J. A. Westerhuis, T. Kourti, and J. F. MacGregor, “Analysis of multiblock and 
        hierarchical PCA and PLS models,” Journal of Chemometrics, vol. 12, no. 5, pp. 301–321, 
        1998, doi: 10.1002/(SICI)1099-128X(199809/10)12:5<301::AID-CEM515>3.0.CO;2-S.

        """
        if isinstance(X, DataFrame):# If X data is a pandas Dataframe
            indexes = X.index.copy()
            X_columns = X.columns.copy()
            X = array(X.to_numpy(), ndmin = 2)

        self.block_divs = block_divs
        block_coord_pairs = (*zip([0] + block_divs[:-1], block_divs),)
        missing_values_list = [isnan(sum(X[:, start:end])) for (start, end) in block_coord_pairs] 
          
            
        Orig_X = X

        
        #Using a full PLS as basis to calculate the MB-PLS
        int_PCA = PCA(tol = self.tol, loop_limit = self.loop_limit, missing_values_method = self.missing_values_method)
        int_PCA.fit(X, principal_components = principal_components,  cv_splits_number = self.cv_splits_number)

        

        block_coord_pairs = (*zip([0] + block_divs[:-1], block_divs),)

        superlevel_T = zeros((X.shape[0], len(block_coord_pairs) * int_PCA.principal_components))
        block_loadings = zeros((int_PCA.principal_components, X.shape[1]))

        for block, (start, end) in enumerate(block_coord_pairs):
            test_missing_data = isnan(sum(X[:, start:end]))
            if test_missing_data:
                b_loadings = zeros((int_PCA.principal_components, end - start))
                for i in range(int_PCA.principal_components):
                    b_loadings[i, :] = nansum(X[:, start:end] * array(int_PCA.training_scores[:, i], ndmin = 2).T, axis = 0) / nansum(((~isnan(X[:, start:end]).T * int_PCA.training_scores[:, i]) ** 2), axis = 1)
            else:
                b_loadings = array(X[:, start:end].T @ int_PCA.training_scores / diagonal(int_PCA.training_scores.T @ int_PCA.training_scores), ndmin = 2).T
            
            block_loadings[:int_PCA.principal_components, start:end] = b_loadings
            block_scores = zeros((X.shape[0], int_PCA.principal_components))
            if test_missing_data:
                for i in range(int_PCA.principal_components):
                    block_scores[:, i] = nansum(X[:, start:end] * b_loadings[i, :], axis = 1) / nansum(((~isnan(X[:, start:end]) * b_loadings[i, :]) ** 2), axis = 1)
            else:
                block_scores = (X[:, start:end] @ b_loadings.T)

            superlevel_T[:, [block + num * len(block_coord_pairs) for num, _ in enumerate(block_coord_pairs)]] = block_scores
            self.omega_b.append(block_scores.T @ block_scores)

 
        superlevel_loadings = array(superlevel_T.T @ int_PCA.training_scores / diagonal(int_PCA.training_scores.T @ int_PCA.training_scores), ndmin = 2).T

        #----------------Attribute Assignment---------------

        self.principal_components = int_PCA.principal_components
        self.block_divs = block_divs 
        self.block_loadings = block_loadings #
        self.superlevel_loadings = superlevel_loadings #
        self.superlevel_training_scores = superlevel_T #
        self.training_scores = int_PCA.training_scores
        self.loadings = int_PCA.loadings 
        self.q2 = int_PCA.q2


        self.feature_importances_ = int_PCA.feature_importances_
        self.omega = int_PCA.omega
        self._chi2_params = int_PCA._chi2_params
        self._pca = int_PCA

    def predict(self, X, principal_components = None):

        """
        Transforms the X sample to the principal component space and back to evaluate what is
        the model "prediction" of the original sample values.

        Parameters
        ----------
        X : array_like
            Samples to reproduce
        principal_components : array_like, Optional
            Number of desired principal components to be used
        
        Returns
        -------
        preds : array_like 
            returns "predicted" X values.        
        """

        return self._pca.predict(X, principal_components = principal_components)
    
    def transform(self, X, principal_components = None): 
        
        """

        Transforms the X sample to the principal component space .

        Parameters
        ----------
        X : array_like
            Samples to transform
        principal_components : array_like, Optional
            Number of desired principal components to be used
        
        Returns
        -------
        result : array_like 
            returns X samples' scores.  

        """

        return self._pca.transform(X, principal_components)
    
    def score(self, X, principal_components = None): #return r²
        
        """
        
        Returns the coefficient of determination R^2 of the model.

        R² is defined as 1 - Variance(Error) / Variance(X) with Error = X - predictions(X)

        Parameters
        ----------
        X : array_like
            Samples used for score calculation
        principal_components : array_like, Optional
            Number of desired principal components to be used
        
        Returns
        -------
        result : array_like 
            returns calculated r².        
        
        """
        return self._pca.score(X, principal_components = principal_components)

    def Hotellings_T2(self, X, principal_components = None):
        
        """
        Calculates the Hotelling's T² for the X samples.

        Parameters
        ----------
        X : array_like
            Samples Matrix
        principal_components : int, optional
            number of latent variables to be used. 

        Returns
        -------
        T2s : array_like 
            returns all calculated T²s for the X samples
        """

        return self._pca.Hotellings_T2(X, principal_components = principal_components)
    
    def T2_limit(self, alpha, principal_components = None):
        
        """
        Calculates the Hotelling's T² limit based on the training dataset.

        Parameters
        ----------
        alpha : array_like
            value ranging from 0-1 to represent the % confidence limit
        principal_components : int, optional
            number of latent variables to be used. 

        Returns
        -------
        t2_limit : array_like 
            returns the limit T² for the alpha based on the training dataset
        """

        return self._pca.T2_limit(alpha, principal_components = principal_components)
    
    def contributions_scores_ind(self, X, principal_components = None) : #contribution of each individual point in X1
        
        """
        calculates the sample individual contributions to the scores.

        Parameters
        ----------
        X : array_like
            Sample matrix 
        principal_components : int, optional
            number of latent variables to be used. 

        Returns
        -------
        contributions : array_like 
            matrix of scores contributions for every X sample
        """

        return self._pca.contributions_scores_ind(X, principal_components = principal_components)
       
    def contributions_spe(self, X, principal_components = None) :
        
        """
        calculates the individual sample individual contributions to the squared prediction error 
        of the X variables matrix reconstruction.

        Parameters
        ----------
        X : array_like
            Sample matrix 
        principal_components : int, optional
            number of latent variables to be used. 

        Returns
        -------
        SPE_contributions : array_like 
            matrix of SPE contributions for every X sample
        """
        
        return self._pca.contributions_spe(X, principal_components = principal_components)   

    def SPEs(self, X, principal_components = None) :
        
        """
        Calculates the Squared prediction error for for the X matrix rebuild.

        Parameters
        ----------
        X : array_like
            Samples Matrix
        principal_components : int, optional
            number of latent variables to be used. 

        Returns
        -------
        SPE : array_like 
            returns all calculated SPEs for the X samples
        """
        
        return self._pca.SPEs(X, principal_components = principal_components)

    def SPE_limit(self, alpha, principal_components = None):

        """
        Calculates the SPE limit based on the training dataset.

        Parameters
        ----------
        alpha : array_like
            value ranging from 0-1 to represent the % confidence limit
        principal_components : int, optional
            number of latent variables to be used. 

        Returns
        -------
        SPE_limit : array_like 
            returns the limit SPE for the alpha based on the training dataset
        """
        return self._pca.SPE_limit(alpha, principal_components = principal_components)

    def transform_b(self, X, block_index, principal_components = None): #require test

        """

        Transforms the X sample to the latent space on the chosen block level.

        Parameters
        ----------
        X : array_like 
            Samples to transform to the block scores. dim 1 of the array should be equal to the block's.
        block_index : int
            Desired block starting from zero
        principal_components : array_like, Optional
            Number of desired principal components to be used
        
        Returns
        -------
        result : array_like 
            returns X samples' scores for the chosen block.  

        """

        if isinstance( X, DataFrame ): X = X.to_numpy()            
        if principal_components == None: principal_components = self.principal_components
        
        if block_index == 0: start, end = 0, self.block_divs[block_index]
        else: start, end = self.block_divs[block_index], self.block_divs[block_index + 1]

        if isnan(sum(X)):
            result = zeros((X.shape[0], principal_components))
            X_nan = isnan(X)
            variables_missing_mask = unique(X_nan, axis = 0)
            
            for row_mask in variables_missing_mask:
                
                rows_indexes = where((X_nan == row_mask).all(axis = 1))                
                
                if sum(row_mask) == 0: 

                    result[rows_indexes, :] = X[rows_indexes, :] @ self.loadings[:principal_components, start:end].T 
                
                else:
                    
                    result[rows_indexes, :] = scores_with_missing_values(self.omega_b[block_index], self.block_loadings[:, start:end][:, ~row_mask], X[rows_indexes[0][:, None], ~row_mask], 
                                                                            LVs = principal_components, method = self.missing_values_method)
                 
                    
        else: result = X @ self.block_loadings[:principal_components, start:end].T

        return result

    def predict_b(self, X, block_index, principal_components = None): #require test

        """

        Transforms the X sample to the latent space on the chosen block level.

        Parameters
        ----------
        X : array_like 
            Samples to rebuild using the block scores. dim 1 of the array should be equal to the block's.
        block_index : int
            Desired block starting from zero
        principal_components : array_like, Optional
            Number of desired principal components to be used
        
        Returns
        -------
        result : array_like 
            returns X samples rebuilt using the model's block loadings.  

        """
         
        if principal_components == None: principal_components = self.principal_components
        
        if block_index == 0: start, end = 0, self.block_divs[block_index]
        else: start, end = self.block_divs[block_index], self.block_divs[block_index + 1]

        b_scores = self.transform_b(X, block_index, principal_components = principal_components)

        result = b_scores @ self.block_loadings[:principal_components, start:end]

        return result

    def score_b(self, X, block_index, principal_components = None): #require test

        """
        
        Returns the coefficient of determination R^2 referring to a specific block.
        dim 1 of the array should be equal to the block's.

        R² is defined as 1 - Variance(Error) / Variance(X) with Error = X - predictions(X)

        Parameters
        ----------
        X : array_like
            Samples used for score calculation
        block_index : int
            Desired block starting from zero
        principal_components : array_like, Optional
            Number of desired principal components to be used
        
        Returns
        -------
        result : array_like 
            returns calculated r².        
        
        """
        if isinstance(X, DataFrame) : X = X.to_numpy()
        
        error = X - self.predict_b(X, block_index, principal_components = principal_components)
        result = 1 - nanvar(error) / nanvar(X)
        return result
