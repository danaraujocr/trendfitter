from numpy import min, sum, mean, std, var, insert, array, multiply, where, zeros, append, isnan, nan_to_num, nansum, nanvar, nanmean, unique, ix_, nonzero, nan
from numpy.linalg import norm
from scipy.stats import f, chi2
from pandas import DataFrame
from sklearn.model_selection import KFold
from trendfitter.auxiliary.tf_aux import scores_with_missing_values


class PCA:
    """
    A sklearn-like class for the NIPALs algorithm for PCA together with a toolset for 
    investigation.
    
    Parameters
    ---------- 
    tol : float, Optional
        Value used to decide if model has converged.
    loop_limit : int, Optional
        Maximum number of loops before forced stop. Resets every new component.
    missing_values_method : str, Optional
        Defines which method will be used to evaluate missing values in future transformations.
    keep_scores : boolean, Optional
        Decision to save scores extracted during model fitting. If not given, assumed False.

    Attributes
    ----------
    principal_components : int, optional
        Number of principal components extracted or to extract. If not given, a cross-validation
            internal routine will decide this value.
    cv_splits_number : int, optional
        Number of splits used for cross-validation. If not given, it will be 7.
    loadings : array_like
        Loading parameters that define the PCA model.
    q2 : [float]
        Average score on the test sets during the cross-validation procedure
    feature_importances_ : array_like
        An array that describes the importance of each feature used to build the model
            using the VIP value of each.
    training_scores : array_like
        Scores extracted during training of the model.
    omega : array_like
        If missing_values_method requires a scores covariance matrix ('TSR', 'CMR', 'PMP'), 
            it will be stored here.

    Methods
    -------
    fit(X, principal_components = None, cv_splits_number = 7, int_call = False)
        Runs the NIPALS algorithm to extract the principal components from X. 
    predict(X, principal_components = None)
        Uses the model to reconstruct X using the principal components.
    transform(X, principal_components = None)
        Transforms the X from its original space to the latent variable space.
    score(X, principal_components = None)
        Returns a r² representing how much variability from X is captured in the model. 
    Hotellings_T2(X, principal_components = None)
        Returns an array with the Hotelling's T² calculated for each row in X.
    T2_limit(alpha)
        Returns the Hotelling's T² limit estimated with alpha confidence level
    contributions_scores_ind(X, principal_components = None)
        Returns an array with the contributions to the scores of each X row.
    contributions_spe(X, principal_components = None)
        Returns an array with the contributions to the SPE of each X row.
    SPEs(X, principal_components = None)
        Returns the squared prediction errors of each row's X reconstruction.
    SPE_limit(alpha)
        Returns the squared prediction error limit with alpha confidence level
    """

    def __init__(self, tol = 1e-12, loop_limit = 100, missing_values_method = 'TSM'):
           
        self.principal_components = None # number of principal components to be extracted
        self.cv_splits_number = None # number of splits for latent variable cross-validation
        self.tol = tol # criteria for convergence
        self.loop_limit = loop_limit # maximum number of loops before convergence is decided to be not attainable
        self.missing_values_method = missing_values_method 

        self.loadings = None #loadings 
        self.q2 = [] # list of cross validation scores
        self.feature_importances_ = None #for scikit learn use with feature selection methods
        self.omega = None # scores covariance matrix for missing values score estimation
        self.training_scores = None
        self._training_scores_stds = None
        self._chi2_params = []

    def fit(self, X, principal_components = None, cv_splits_number = 7, int_call = False):
        """
        Extracts the model parameters using the NIPALs algorithm [1].

        Parameters
        ----------
        X : array_like
            Data used to extract the parameters and fit the model
        principal_components : array_like, Optional
            Number of desired principal components to be extracted
        cv_splits_number : int, Optional
            Number of desired splits to be used during the cross-validation routine        
        int_call : Boolean, optional
            Flag to determine if the method should calculate certain values. If not specified
            
        References 
        [1] S. Wold, K. Esbensen, and P. Geladi, “Principal component analysis,” 
            Chemometrics and Intelligent Laboratory Systems, vol. 2, no. 1–3, 
            pp. 37–52, Aug. 1987, doi: 10.1016/0169-7439(87)80084-9.

        """

        self.principal_components = principal_components # number of principal components to be extracted
        self.cv_splits_number = cv_splits_number # number of splits for latent variable cross-validation


        if isinstance(X, DataFrame): X = X.to_numpy() #ensuring data in the numpy format
        dataset_incomplete = False 
        if isnan(sum(X)): dataset_incomplete = True #checking if there is missing data on the input X

        #CV = False
        if not(self.principal_components is None) : 
            numberOfLVs = self.principal_components 
            #CV = True
        else :
            numberOfLVs = X.shape[1] #maximum amount of extractable latent variables
            kf = KFold(n_splits = self.cv_splits_number, shuffle = True, random_state = 1)

        MatrixXModel = array(zeros(X.shape), ndmin = 2) #initializing matrix space for manipulation 
        
        #----------------------------------------------------------------------------------
        #------------------------------NIPALS implementation-------------------------------
        #----------------------------------------------------------------------------------
        q2_final = []
        for latent_variable in range(1, numberOfLVs + 1) :
            scores_vec = nan_to_num(array(X[ :, 1], ndmin = 2).T) #initializing the guess by using a specific vector
            MatrixX2 = X - MatrixXModel #deflation of the X matrix
            counter = 0
            conv = 1
            while conv > self.tol and counter < self.loop_limit:
                counter += 1
                
                if dataset_incomplete:

                    loadings_vec = array(nansum(MatrixX2 * scores_vec, axis = 0) / nansum(((~isnan(X) * scores_vec) ** 2).T, axis = 1), ndmin = 2) # equivalent to loadings = scores'*data/scores'*scores
                    loadings_vec = loadings_vec / norm(loadings_vec) #normalization of loading vector
                    new_scores_vec = array(nansum(MatrixX2 * loadings_vec, axis = 1) / nansum(((~isnan(X) * loadings_vec) ** 2), axis = 1), ndmin = 2).T #scores calculation w/ missing data
                
                else:
                    
                    loadings_vec = array(scores_vec.T @ MatrixX2 / (scores_vec.T @ scores_vec), ndmin = 2) #loadings = scores'*data/scores'*scores        
                    loadings_vec = loadings_vec / norm(loadings_vec) #normalization of loading vector
                    new_scores_vec = MatrixX2 @ loadings_vec.T #scores calculation
                
                conv = sum((scores_vec - new_scores_vec) ** 2)  #scores comparation in between loops to assess convergency
                scores_vec = new_scores_vec # old t becomes new t

            #After convergency, if the principal components desired quantity is undefined
            #then we check if the Component is in fact significant and keep on until all 
            #components are there            
            if self.principal_components is None :
                
                testq2 = []
                
                for train_index, test_index in kf.split(X):

                    q2_model = PCA(missing_values_method = self.missing_values_method)
                    q2_model.fit(X[train_index], principal_components = latent_variable, int_call = True)
                    testq2.append(q2_model.score(X[test_index]))

                q2_final.append(mean(testq2))
                
                if latent_variable > 1:
                    
                    if (q2_final[-1] < q2_final[-2] or \
                        q2_final[-1] - q2_final[-2] < 0.01 or \
                        latent_variable > min(X.shape) / 2):
                        self.q2 = q2_final[:-1]
                        self.principal_components = latent_variable - 1
                        if self.missing_values_method == 'TSM': break 
                        
            #if significant, then we add them to the loadings and score matrixes that will be returned as method result

            #prediction of this model
            

            if latent_variable < 2:

                self.loadings = loadings_vec
                self.training_scores = scores_vec
                
            else:

                self.loadings = insert(self.loadings, self.loadings.shape[0], loadings_vec, axis = 0)
                self.training_scores = insert(self.training_scores, self.training_scores.shape[1], scores_vec.T, axis = 1)

            if not (self.principal_components) is None and latent_variable > 2 * self.principal_components: break # Ensuring to extract at least double the useful components for missing values estimation:

            MatrixXModel = self.training_scores @ self.loadings 
            error = nan_to_num(X) - MatrixXModel
            SPE = sum(error ** 2 , axis = 1)
            self._chi2_params.append([mean(SPE), var(SPE)]) #for future SPE analysis

        if not int_call: 
            self.feature_importances_ = self._VIPs_calc(X,  principal_components = self.principal_components)
            self.omega = self.training_scores.T @ self.training_scores # calculation of the covariance matrix
            self._training_scores_stds = std(self.training_scores, axis = 0)
        pass
        
    def predict(self, X, principal_components = None):

        """
        Transforms the X sample to the principal component space and back to evaluate what is
        the model "prediction" of the original sample values.

        Parameters
        ----------
        X : array_like
            Data used to extract the parameters and fit the model
        principal_components : array_like, Optional
            Number of desired principal components to be used
        
        Returns
        -------
        preds : array_like 
            returns "predicted" X values.        
        """

        if isinstance(X, DataFrame): X = X.to_numpy()
        if principal_components == None : principal_components = self.principal_components
        
        preds = (self.transform(X, principal_components = principal_components)) @ self.loadings[:principal_components, :] 
        
        return preds
    
    def transform(self, X, principal_components = None): 
        
        """

        Transforms the X sample to the principal component space .

        Parameters
        ----------
        X : array_like
            Data used to extract the parameters and fit the model
        principal_components : array_like, Optional
            Number of desired principal components to be used
        
        Returns
        -------
        result : array_like 
            returns X samples' scores.  

        """

        if isinstance(X, DataFrame) : X = X.to_numpy()            
        if principal_components == None : principal_components = self.principal_components
        
        if isnan(sum(X)) :
            result = zeros((X.shape[0], principal_components))
            X_nan = isnan(X)
            variables_missing_mask = unique(X_nan, axis = 0)
            
            for row_mask in variables_missing_mask:
                
                rows_indexes = where((X_nan == row_mask).all(axis = 1))                
                
                if sum(row_mask) == 0: 

                    result[rows_indexes, :] = X[rows_indexes, :] @ self.loadings[:principal_components, :].T 
                
                else:
                    
                    result[rows_indexes, :] = scores_with_missing_values(self.omega, self.loadings[:, ~row_mask], X[rows_indexes[0][:, None], ~row_mask], 
                                                                            LVs = principal_components, method = self.missing_values_method)
                 
                    
        else: result = X @ self.loadings[:principal_components, :].T 
        
        return result
    
    def score(self, X, principal_components = None): #return r²
        
        """
        
        Returns the coefficient of determination R^2 of the model.

        R² is defined as 1 - Variance(Error) / Variance(X) with Error = X - predictions(X)

        Parameters
        ----------
        X : array_like
            Data used to extract the parameters and fit the model
        principal_components : array_like, Optional
            Number of desired principal components to be used
        
        Returns
        -------
        result : array_like 
            returns calculated r².        
        
        """

        if isinstance(X, DataFrame) : X = X.to_numpy()
        
        ErrorQ2 = X - self.predict(X, principal_components = principal_components)
        result = 1 - nanvar(ErrorQ2) / nanvar(X)
        return result

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

        if isinstance(X, DataFrame) : X = X.to_numpy()
            
        if principal_components == None : principal_components = self.principal_components # Unless specified, the number of PCs is the one in the trained model 
        
        scores_matrix = self.transform(X, principal_components = principal_components)
        
        T2s = sum((scores_matrix / self._training_scores_stds[:principal_components]) ** 2 , axis = 1)
        return T2s
    
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

        if principal_components == None: principal_components = self.principal_components # Unless specified, the number of PCs is the one in the trained model 

        F_value = f.isf(1 - alpha , principal_components, self.training_scores.shape[0])
        t2_limit = ((principal_components * (self.training_scores.shape[0] ** 2 - 1)) / 
                    (self.training_scores.shape[0] * (self.training_scores.shape[0] - principal_components))) * \
                    F_value

        return t2_limit
    
    def _VIPs_calc(self, X, principal_components = None):
        
        if principal_components is None: principal_components = self.principal_components
              
        if isinstance(X, DataFrame) : X = X.to_numpy()

        SSX = array(sum((nan_to_num(X) - nanmean(X, axis = 0)) ** 2))   
        
        for i in range(1, principal_components + 1):

            pred = self.predict(X, principal_components = i)
            res = nan_to_num(X) - pred
            SSX = append(SSX, (res - res.mean(axis = 0) ** 2).sum())
            
        SSXdiff = SSX[:-1] - array(SSX)[1:]

        VIPs = array(((self.loadings[:principal_components, :].T ** 2) @ SSXdiff * 
                          self.loadings.shape[1] / (SSX[0] - SSX[len(SSX) - 1])) ** 1 / 2 , ndmin = 2)                
        return VIPs
    
    def contributions_scores_ind(self, X, principal_components = None): #contribution of each individual point in X1
        
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

        if principal_components == None : principal_components = self.principal_components

        if isinstance(X, DataFrame) : X = X.to_numpy()
        
        scores = self.transform(X, principal_components = principal_components)
        scores = (scores / scores.std(axis = 0) ) ** 2 
        contributions = multiply(X, (scores @ self.loadings[:principal_components, :] ** 2) ** 1 / 2) 

        return contributions
       
    def contributions_spe(self, X, principal_components = None):
        
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
        
        if principal_components == None : principal_components = self.principal_components
        if isinstance(X, DataFrame): X = X.to_numpy()

        error = nan_to_num(X) - self.predict(X)
        SPE_contributions = multiply(error ** 2 , where(error > 0, 1, -1))
        
        return SPE_contributions   

    def SPEs(self, X, principal_components = None):
        
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
        
        if principal_components == None : principal_components = self.principal_components
        if isinstance(X, DataFrame): X = X.to_numpy()

        error = nan_to_num(X) - self.predict(X)
        SPE = sum(error ** 2 , axis = 1)
        
        return SPE

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

        if principal_components == None: principal_components = self.principal_components # Unless specified, the number of PCs is the one in the trained model 
        chi2_df = 2 * self._chi2_params[principal_components - 1][0] ** 2 / self._chi2_params[principal_components - 1][1]
        chi2_val = chi2.isf(1 - alpha, chi2_df)
        SPE_limit = self._chi2_params[principal_components - 1][1] / (2 * self._chi2_params[principal_components - 1][0])  * chi2_val
        
        return SPE_limit