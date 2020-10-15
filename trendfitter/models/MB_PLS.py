from numpy import array, isnan, nansum, nan_to_num, multiply, sum, sqrt, \
                  append, zeros, place, nan, concatenate, mean, nanvar,  \
                  std, unique, where, nanmean, diagonal
from numpy.linalg import norm, pinv
from sklearn.model_selection import KFold
from pandas import DataFrame, Series
from trendfitter.auxiliary.tf_aux import scores_with_missing_values
from . import PLS

class MB_PLS:
    """
    A sklearn-like class for the Multi-Block Projection to Latent Structures.

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
        latent_variables : int
            list of number of latent variables deemed relevant from each block. 
        block_divs : [int]
            list with the index of every block final position. ex. [2,4] means two blocks with 
                columns 0 and 1 on the first block and columns 2 and 3 on the second block. 
                Assigned when fit method runs.
        p_loadings_block : [array_like]
            list of p loadings arrays of every block with all the extracted latent variables 
        superlevel_p_loadings : array_like
            array of all latent variables extracted p loadings for the super level
        weights_block : [array_like]
            list of weights arrays of every block with all the extracted latent variables 
        weights_super : array_like
            array of all latent variables extracted weights for the super level
        c_loadings : array_like
            array of c_loadings for all extracted latent variables
        q2 : [float]
            list of r² coefficients extracted during cross validation
        feature_importances : [float]
            list of values that represent how important is each variable in the same order 
                of the X columns on the first matrix

        Methods
        -------
        fit(X, blocks_sep, Y, latent_variables, deflation, int_call)
            Applies the NIPALS like method to find the best parameters that adjust the model 
                to the data
        transform(X, latent_variables)
            Transforms the original data to the latent variable space in the super level 
        transform_inv(super_scores, latent_variables)
        transform_b(X, block, latent_variables)
            Transforms the original data to the latent variable space in the block level for
                all blocks
        predict(X, latent_variables)
            Predicts Y values 
        score(X, Y, latent_variables)
            calculates the r² value for Y
        Hotellings_T2(X)
            Calculates Hotellings_T2 values for the X data in the super level
        T2_limit(alpha)
            Returns the Hotelling's T² limit estimated with alpha confidence level
        SPEs_X(X)
            Calculates squared prediction errors on the X side for the super level
        SPEs_X_limit(X)
            Calculates squared prediction errors for the block level for all blocks
        SPEs_Y(X, Y)
            Calculates squared prediction errors for the predictions
        SPEs_Y_limit(X, Y)
            Calculates squared prediction errors for the predictions
        contributions_scores_ind(X)
            calculates the contributions of each variable to the scores on the super level
        contributions_SPE(X)
            calculates the contributions of each variable to the SPE on the X side for
                the super level

    """
    def __init__(self, cv_splits_number = 7, tol = 1e-8, loop_limit = 1000, missing_values_method = 'TSM'):
        
        # Parameters
        
        self.cv_splits_number = cv_splits_number # number of splits for latent variable cross-validation
        self.tol = tol # criteria for convergence
        self.loop_limit = loop_limit # maximum number of loops before convergence is decided to be not attainable
        self.q2y = [] # list of cross validation scores
        self.missing_values_method = missing_values_method
        
        # Attributes

        self.latent_variables = None # number of principal components to be extracted
        self.block_divs = None
        self.block_p_loadings = None
        self.superlevel_p_loadings = None
        self.block_weights = None
        self.superlevel_weights = None
        self.x_weights_star = None
        self.x_weights = None
        self.c_loadings = None 
        self.feature_importances_ = None
        self.coefficients = None
        self.omega = None # score covariance matrix for missing value estimation
        self._int_PLS = None
        self.training_scores = None
        self.training_sl_scores = None

    def fit(self, X, block_divs, Y, latent_variables = None, deflation = 'both', int_call = False):
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
        [1] J. A. Westerhuis, T. Kourti, and J. F. MacGregor, “Analysis of multiblock and hierarchical 
        PCA and PLS models,” Journal of Chemometrics, vol. 12, no. 5, pp. 301–321, 1998, 
        doi: 10.1002/(SICI)1099-128X(199809/10)12:5<301::AID-CEM515>3.0.CO;2-S.


        """
        if isinstance(X, DataFrame):# If X data is a pandas Dataframe
            X = array(X.to_numpy(), ndmin = 2)

        self.block_divs = block_divs
        block_coord_pairs = (*zip([0] + block_divs[:-1], block_divs),)
          
        if isinstance( Y, DataFrame ) or isinstance( Y, Series ): # If Y data is a pandas Dataframe or Series 
            Y = array( Y.to_numpy(), ndmin = 2 ).T
        
        #Using a full PLS as basis to calculate the MB-PLS
        int_PLS = PLS(cv_splits_number = self.cv_splits_number, tol = self.tol, loop_limit = self.loop_limit, missing_values_method = self.missing_values_method)
        int_PLS.fit(X, Y, latent_variables = latent_variables, deflation = deflation)

        

        block_coord_pairs = (*zip([0] + block_divs[:-1], block_divs),)

        superlevel_T = zeros((X.shape[0], len(block_coord_pairs) * int_PLS.latent_variables))
        superlevel_weights = zeros((int_PLS.latent_variables , len(block_coord_pairs) * int_PLS.latent_variables))
        block_weights = zeros((int_PLS.latent_variables * len(block_coord_pairs), X.shape[1]))
        block_p_loadings = zeros(block_weights.shape)

        for block, (start, end) in enumerate(block_coord_pairs):
            test_missing_data = isnan(sum(X[:, start:end]))
            if test_missing_data:
                b_weights = zeros((int_PLS.latent_variables, end - start))
                for i in range(int_PLS.latent_variables):
                    b_weights[i, start:end] = nansum(X[:, start:end] * array(int_PLS.training_scores[:, i], ndmin = 2).T, axis = 0) / nansum(((~isnan(X[:, start:end]).T * int_PLS.training_scores[:, i]) ** 2), axis = 1)
            else:
                b_weights = array(X[:, start:end].T @ int_PLS.training_scores / diagonal(int_PLS.training_scores.T @ int_PLS.training_scores), ndmin = 2).T

            block_weights[(block * int_PLS.latent_variables):((block + 1) * int_PLS.latent_variables), start:end] = b_weights
            
            block_scores = zeros((X.shape[0], int_PLS.latent_variables))
            
            
            if test_missing_data:
                for i in range(int_PLS.latent_variables):
                    block_scores[:, i] = nansum(X[:, start:end] * b_weights[i, :], axis = 1) / nansum(((~isnan(X[:, start:end]) * b_weights[i, :]) ** 2), axis = 1)
                    b_p_loadings = array(sum(nan_to_num(X) * block_scores[:, i], axis = 0) / sum((~isnan(X) * block_scores[:, i]) ** 2, axis = 0), ndmin = 2)
                    
            else:
                block_scores = (X[:, start:end] @ b_weights.T)
                b_p_loadings = zeros((int_PLS.latent_variables, end-start))
                for i in range(int_PLS.latent_variables):                  
                    b_p_loadings[i:i+1, :] = array(X[:, start:end].T @ block_scores[:, i] / (block_scores[:, i].T @ block_scores[:, i]), ndmin = 2)
            block_p_loadings[(block * int_PLS.latent_variables):((block + 1) * int_PLS.latent_variables), start:end] = b_p_loadings
            superlevel_T[:, [block + num * len(block_coord_pairs) for num, _ in enumerate(block_coord_pairs)]] = block_scores
        
        for i in range(int_PLS.latent_variables):
            numerator = (superlevel_T[:, i * len(block_coord_pairs) : (i + 1) * len(block_coord_pairs)].T @ int_PLS.training_scores[:, i])
            denominator = (int_PLS.training_scores[:, i].T @ int_PLS.training_scores[:, i])
            superlevel_weights[i , i * len(block_coord_pairs) : (i + 1) * len(block_coord_pairs)] = numerator / denominator #Error here
        
        
        """-------------------------------- p loadings calculation section ----------------------------"""                        
        

        #----------------Attribute Assignment---------------

        

        self.latent_variables = int_PLS.latent_variables
        self.block_divs = block_divs
        

        self.block_p_loadings = block_p_loadings
        self.block_weights = block_weights
        self.superlevel_weights = superlevel_weights
        self.training_sl_scores = superlevel_T

        self.x_weights_star = int_PLS.weights_star
        self.x_weights = int_PLS.weights
        self.c_loadings = int_PLS.c_loadings
        self.feature_importances_ = int_PLS.feature_importances_
        self.omega = int_PLS.omega
        self._int_PLS = int_PLS
        self.training_scores = self._int_PLS.training_scores
        
    def transform(self, X, latent_variables = None):

        """

        Transforms the X sample to the principal component space.

        Parameters
        ----------
        X : array_like
            Samples to transform
        latent_variables : int, Optional
            Number of desired principal components to be used
        
        Returns
        -------
        result : array_like 
            returns X samples' superlevel scores.  

        """

        return self._int_PLS.transform(X, latent_variables)

    def transform_inv(self, super_scores, latent_variables = None):

        """
        Transforms the super_scores matrix to the original X.

        Parameters
        ----------
        super_scores : array_like
            Matrix with all the scores to be used to rebuild X
        latent_variables : int, optional
            number of latent variables to be used. 

        Returns
        -------
        result : array_like 
            matrix of rebuilt X from scores
        """

        return self._int_PLS.transform_inv(super_scores, latent_variables = None)

    def predict(self, X, latent_variables = None):

        """
        
        Predicts Y for X samples.

        Parameters
        ----------

        X : array_like 
            Samples to use for prediction
        latent_variables : int, Optional
            Number of desired latent variables to be used. If 
        none are supplied, all variables will be used.

        Returns
        -------
        result : array_like
            Returns Y predictions

        """

        return self._int_PLS.predict(X, latent_variables = latent_variables)

    def score(self, X, Y, latent_variables = None):

        """
        Return the coefficient of determination R^2 of the prediction.

        R² is defined as 1 - Variance(Error) / Variance(Y) with Error = Y - predictions(X)

        Parameters
        ----------
        X : array_like
            Matrix with all the X to be used
        Y : array_like
            Matrix with all the Y ground truth values
        latent_variables : int, optional
            number of latent variables to be used. 

        Returns
        -------
        score : float 
            returns calculated r².
        """

        return self._int_PLS.score(X, Y, latent_variables = latent_variables)

    def Hotellings_T2(self, X, latent_variables = None):
        
        """
        Calculates the Hotelling's T² for the X samples.

        Parameters
        ----------
        X : array_like
            Samples Matrix
        latent_variables : int, optional
            number of latent variables to be used. 

        Returns
        -------
        T2s : array_like 
            returns all calculated T²s for the X samples
        """

        return self._int_PLS.Hotellings_T2(X, latent_variables = latent_variables)

    def T2_limit(self, alpha, latent_variables = None):
        
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
        return self._int_PLS.T2_limit(alpha, latent_variables = latent_variables)
    
    def SPEs_X(self, X, latent_variables = None):
        
        """
        Calculates the Squared prediction error for for the X matrix rebuild.

        Parameters
        ----------
        X : array_like
            Samples Matrix
        latent_variables : int, optional
            number of latent variables to be used. 

        Returns
        -------
        SPE : array_like 
            returns all calculated SPEs for the X samples
        """
        return self._int_PLS.SPEs_X(X, latent_variables = latent_variables)

    def SPE_X_limit(self, alpha, latent_variables = None):
        """
        Calculates the SPE limit for the X rebuild based on the training dataset.

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
        return self._int_PLS.SPE_X_limit(alpha, latent_variables = latent_variables)

    def SPEs_Y(self, X, latent_variables = None):
        """
        Calculates the Squared prediction error for the Y values.

        Parameters
        ----------
        X : array_like
            Samples X Matrix
        Y : array_like
            ground truth Y Matrix
        latent_variables : int, optional
            number of latent variables to be used. 

        Returns
        -------
        SPE : array_like 
            returns all calculated SPEs for the X samples
        """
        return self._int_PLS.SPEs_Y(X, latent_variables = latent_variables)
    
    def SPE_Y_limit(self, alpha, latent_variables = None):
        """
        Calculates the SPE limit for the X rebuild based on the training dataset.

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
        
        return self._int_PLS.SPEs_Y_limit(alpha, latent_variables = latent_variables)

    def RMSEE (self, X, Y, latent_variables = None):

        """
        Returns the root mean squared prediction error.

        Parameters
        ----------
        X : array_like
            Matrix with all the X to be used
        Y : array_like
            Matrix with all the Y ground truth values
        latent_variables : int, optional
            number of latent variables to be used. 

        Returns
        -------
        RMSEE : float 
            returns calculated RMSEE.
        """
        return self._int_PLS.RMSEE(X, Y, latent_variables = latent_variables)

    def contributions_scores_ind(self, X, latent_variables = None):
        """
        calculates the sample individual contributions to the scores.

        Parameters
        ----------
        X : array_like
            Sample matrix 
        latent_variables : int, optional
            number of latent variables to be used. 

        Returns
        -------
        contributions : array_like 
            matrix of scores contributions for every X sample
        """
        
        return self.int_PLS.contributions_scores_ind(X, latent_variables = latent_variables)

    def contributions_SPE_X(self, X, latent_variables = None):
        """
        calculates the individual sample individual contributions to the squared prediction error 
            of the X variables matrix reconstruction.

        Parameters
        ----------
        X : array_like
            Sample matrix 
        latent_variables : int, optional
            number of latent variables to be used. 

        Returns
        -------
        SPE_contributions : array_like 
            matrix of SPE contributions for every X sample
        """
        return self.int_PLS.contributions_SPE_X(X, latent_variables = latent_variables)

    def transform_b(self, X, block, latent_variables = None): # requires testing & development

        """
        
        Transform the X sample of a specific block to the block space.
        Does not handle missing data.

        Parameters 
        ----------
        X : array_like
            Samples to transform
        latent_variables : int, Optional
            Number of desired latent variables to be used. If
        none are supplied, all variables will be used.

        Returns
        -------
        result : array_like
            returns X samples' block level scores

        """

        if isinstance(X, DataFrame): X = X.to_numpy()      
        if latent_variables is None: latent_variables = self.latent_variables

        if block == 0:
            divs = [0 , self.block_divs[block]]
        else:
            divs = [self.block_divs[block - 1] , self.block_divs[block]]
        result = X @ self.block_weights[(block * latent_variables):(block * latent_variables) + latent_variables, divs[0]:divs[1]].T
        
        return result

    def transform_b_inv(self, scores, block, latent_variables = None): # requires testing & development

        """
        
        Transform the X sample of a specific block to the block space.
        Does not handle missing data.

        Parameters 
        ----------
        scores : array_like
            Sample scores to transform back to block domain
        latent_variables : int, Optional
            Number of desired latent variables to be used. If
        none are supplied, all variables will be used.

        Returns
        -------
        result : array_like
            returns X samples
        """

        if isinstance(scores, DataFrame): scores = scores.to_numpy()      
        if latent_variables is None: latent_variables = self.latent_variables

        if block == 0:
            divs = [0 , self.block_divs[block]]
        else:
            divs = [self.block_divs[block - 1] , self.block_divs[block]]
        result = scores @ self.block_weights[(block * latent_variables):(block * latent_variables) + latent_variables, divs[0]:divs[1]]
        
        return result

    def score_b(self, X, block, latent_variables = None): #requires testing & development

        """
        
        Transform the X sample of a specific block to the block space.
        Does not handle missing data.

        Parameters 
        ----------
        X : array_like
            Samples to transform
        latent_variables : int, Optional
            Number of desired latent variables to be used. If
        none are supplied, all variables will be used.

        Returns
        -------
        result : array_like
            returns X samples' block level scores

        """

        error = X - self.transform_b_inv(self.transform_b(X, block, latent_variables = latent_variables), block, latent_variables = latent_variables)
        
        result = 1 - nanvar(error) / nanvar(X)

        return result
