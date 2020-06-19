from numpy import min, sum, mean, std, nanvar, insert, array, where, isnan, nan_to_num, nansum, nanmean, identity, zeros, unique
from numpy.linalg import norm
from pandas import DataFrame, Series
from sklearn.model_selection import KFold
from trendfitter.auxiliary.tf_aux import scores_with_missing_values



class PLS:

    """
    A sklearn-like class for the Projection to Latent Structures.

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
            number of latent variables one wants to extract.
        p_loadings_block : [array_like]
            list of p loadings arrays of every block with all the extracted latent variables 
        weights : array_like
            array of all latent variables extracted weights 
        weights_star : array_like
            array of all latent variables extracted weights for the non-deflated X matrix
        c_loadings : array_like
            array of c_loadings for all extracted latent variables
        q2 : [float]
            list of r² coefficients extracted during cross validation
        feature_importances_ : [float]
            list of values that represent how important is each variable in the same order 
                of the X columns on the first matrix

        Methods
        -------
        fit(X, blocks_sep, Y)
            Applies the NIPALS like method to find the best parameters that adjust the model 
                to the data
        transform(X)
            Transforms the original data to the latent variable space in the super level 
        predict(X)
            Predicts Y values 
        score(X, Y)
            calculates the r² value for Y
        Hotellings_T2(X)
            Calculates Hotellings_T2 values for the X data in the super level
        SPEs_X(X)
            Calculates squared prediction errors on the X side 
        SPEs_Y(X, Y)
            Calculates squared prediction errors for the predictions
        contributions_scores_ind(X)
            calculates the contributions of each variable to the scores on the super level
        contributions_SPE_X(X)
            calculates the contributions of each variable to the SPE on the X side for 

    """
    def __init__(self, cv_splits_number = 7, tol = 1e-8, loop_limit = 1000, missing_values_method = 'TSM'):
        
        self.p_loadings = None
        self.weights = None
        self.weights_star = None
        self.c_loadings = None
        self.latent_variables = None # number of principal components to be extracted
        self.cv_splits_number = cv_splits_number # number of splits for latent variable cross-validation
        self.tol = tol # criteria for convergence
        self.loop_limit = loop_limit # maximum number of loops before convergence is decided to be not attainable
        self.q2y = [] # list of cross validation scores
        self.deflation = None
        self.VIPs = None
        self.coefficients = None
        self.scores = None
        self.omega = None
        self.missing_values_method = missing_values_method
        self.feature_importances_ = None #for scikit learn use with feature selection methods
        
    def fit(self, X, Y, latent_variables = None, deflation = 'both', random_state = None):

        """
        Adjusts the model parameters to best fit the Y using the algorithm defined in 
            Wold's [1]

        Parameters
        ----------
        X : array_like
            Matrix with all the data to be used as predictors in one only object
        Y : array_like
            Matrix with all the data to be predicted in one only object
        latent_variables : int, optional
            number of latent variables deemed relevant. If left unspecified
                a cross validation routine will define the number during fitting.
        deflation : str, optional
            string defining method of deflation, only Y or both X and Y 
        random_state : int, optional
            value used as a seed in the random number generator for cross validation

        References 
        ----------
        [1] S. Wold, K. Esbensen, and P. Geladi, “Principal component analysis,” 
            Chemometrics and Intelligent Laboratory Systems, vol. 2, no. 1–3, 
            pp. 37–52, Aug. 1987, doi: 10.1016/0169-7439(87)80084-9.

        """


        """----------------------- Dealing with data possibly coming in the form of a pandas dataframe ------------------------"""
        
        self.latent_variables = latent_variables
        self.deflation = deflation
        
        if isinstance(X, DataFrame): # in case the function receives a dataframe as X data
            X = array(X.to_numpy(), ndmin = 2)
        Orig_X = X
            
        if isinstance(Y, DataFrame) : Y = array( Y.to_numpy(), ndmin = 2)
        elif isinstance(Y, Series): Y = array( Y.to_numpy(), ndmin = 2).T # in case the function receives a dataframe as Y data
        else : Y = array(Y, ndmin = 2)
        Orig_Y = Y
    
        """------------------- Checking if the data is in fact problematic -------------------------"""

        if isnan(sum(X)) or isnan(sum(Y)):
            dataset_complete = False
        else: dataset_complete = True
        
        """------------------- Handling the case where the amount of latent variables is not defined -------------------"""
        if self.latent_variables != None :
            latent_variables = self.latent_variables
        else :
            latent_variables = min(X.shape) #maximum amount of extractable latent variables
            kf = KFold(n_splits = self.cv_splits_number, shuffle = True , random_state = random_state)

        """------------------- model Calculation -------------"""
        G = identity(X.shape[1])
        q2_final = []
        for latent_variable in range(1, latent_variables + 1):
            
            loop_count = 0
            conv = 1
            
            y_scores1 = array(nan_to_num( Y[:,0] ), ndmin = 2).T #handling possible NaNs that come from faulty datasets - Step 1

            while conv > self.tol and loop_count < self.loop_limit:#NIPALS internal Loop 
                
                if dataset_complete: # if the dataset is without problematic data, matrix implementation is possible and faster
                    
                    weights = array(X.T @ y_scores1 / (y_scores1.T @ y_scores1), ndmin = 2).T #calculating w vector by regressing all u values into X transpose
                    weights = weights / norm(weights) #normalizing w vector
                    x_scores = X @ weights.T / (weights @ weights.T) #regressing all w values into X matrix and producing a t matrix
                    c_loadings = array(Y.T @ x_scores / (x_scores.T @ x_scores), ndmin = 2).T #regressing all t values into Y and producing the c matrix
                    y_scores = Y @ c_loadings.T / (c_loadings @ c_loadings.T) #calculating new u values by regressing cs into Ys
                    
                    conv = norm(y_scores1 - y_scores)
                    y_scores1 = y_scores
                    
                else: # if it isn't without problematic data, then one needs to deal with it by looping for every regression
                                        
                    """ This segment of the code is a lot slower. 
                    - An opportunity to make it faster is to separate the lines and columns
                    with missing data from the ones complete and iterate only on the ones with missing data. The ones with complete lines can 
                    be used for one single matrix operation. 
                    - A second opportunity is to group the missing data columns and lines based on which variables are missing to then execute a matrix 
                    operation for each group."""
                       
                    weights = array(nansum((X * y_scores1), axis = 0) / nansum(((~isnan(X) * y_scores1) ** 2).T, axis = 1), ndmin = 2) #calculating w vector by regressing all u values into X transpose
                    weights = weights / norm(weights) #normalizing w vector
                    x_scores = array(nansum(X * weights, axis = 1) / nansum(((~isnan(X) * weights) ** 2), axis = 1), ndmin = 2).T #regressing all w values into X matrix and producing a t matrix
                    c_loadings = array(nansum(Y * x_scores, axis = 0) / nansum(((~isnan(Y) * x_scores) ** 2).T, axis = 1), ndmin = 2) #regressing all t values into Y and producing the c matrix
                    y_scores =  array(nansum(Y * c_loadings, axis = 1) / nansum(((~isnan(Y) * c_loadings) ** 2), axis = 1), ndmin = 2).T #calculating new u values by regressing cs into Ys

                    conv = norm(y_scores1 - y_scores)
                    y_scores1 = y_scores
                    
               
                loop_count += 1
                if loop_count == self.loop_limit: 
                    print(f'Model could not converge with {self.loop_limit} loops')
                    return
                
            
            
            """-------------------------------- Cross validation section -----------------------------------"""
            if self.latent_variables == None: # if the quantity of principal components is undefined, a cross-validation routine decides when to stop
                
                testq2 = []
                

                q2_model = PLS(latent_variables = latent_variable, tol = self.tol, deflation = self.deflation, loop_limit = self.loop_limit
                                , missing_values_method = 'TSM') # an internal model is initialized
                for train_index, test_index in kf.split(X):
                    q2_model.fit(Orig_X[train_index, :], Orig_Y[train_index, :]) # this model is trained with only a partition of the total dataset
                    testq2.append(q2_model.score(Orig_X[test_index, :], Orig_Y[test_index, :])) # its performance is registered in a list
                q2_final.append(mean(testq2))
                """ ------------------ coefficients and VIP calculations ----------------- """
             
                if latent_variable > 1 : 
                    
                    if (q2_final[-1] < q2_final[-2] or  # mean cross-validation performance of the model has decreased with the addition of another latent variable
                        q2_final[-1] - q2_final[-2] < 0.01 or  # mean cross-validation performance of the model has increased less than 1% with the addition of another latent variable
                        latent_variable > min(X.shape) / 2): # the amount of latent variables added is more than half the variables on X
                        self.q2y = q2_final[:-1]
                        self.latent_variables = latent_variable - 1 
                        if self.missing_values_method != 'TSM' : break  #In case of TSM use there is no need of more components for missing value estimation
                        
            """-------------------------------- p loadings calculation section ----------------------------"""                        
            if dataset_complete: # if the dataset is without problematic data, matrix implementation is possible and faster
                p_loadings = array(X.T @ x_scores / (x_scores.T @ x_scores), ndmin = 2).T
                
            else: # if it isn't without problematic data, then one needs to deal with it
                p_loadings = array(sum(nan_to_num(X) * x_scores, axis = 0) / sum((~isnan(X) * x_scores) ** 2, axis = 0), ndmin = 2)
                

            
            """-------------------------------- deflation section -----------------------------------------"""
            if self.deflation == 'both':                

                X = X - x_scores @ p_loadings
                Y = Y - x_scores @ c_loadings 
                
            elif self.deflation == 'Y':

                Y = Y - x_scores.T @ c_loadings.T
                
            else :
                raise ValueError("impossible deflation parameter")
            
            """-------------------------------- class property assignment section -------------------------"""
            if latent_variable < 2:
                self.p_loadings = p_loadings
                self.weights = weights
                self.c_loadings = c_loadings
                self.weights_star = weights
                G = G - weights.T @ p_loadings 
                self.scores = array(x_scores, ndmin = 2)
            else:
                weights_star = array(G @ weights.T , ndmin = 2).T
                G = G - weights_star.T @ p_loadings 
                self.weights_star = insert(self.weights_star, self.weights_star.shape[0], weights_star, axis = 0)          
                self.p_loadings = insert(self.p_loadings, self.p_loadings.shape[0], p_loadings, axis = 0)
                self.weights = insert(self.weights, self.weights.shape[0], weights, axis = 0)
                self.c_loadings = insert(self.c_loadings, self.c_loadings.shape[0], c_loadings, axis = 0)
                self.scores = insert(self.scores, self.scores.shape[1], array( x_scores, ndmin = 2).T, axis = 1)
            self.omega = self.scores.T @ self.scores 

            """-------------------------------- Coefficients property calculations ----------------"""
           
            #coefficients_calc = self.weights_star.dot( self.c_loadings.T )
            
            """-------------------------------- VIPs property calculations ------------------------"""
            
            if self.latent_variables != None and latent_variable > 2 * self.latent_variables: break
        #VIP calculation
        self.feature_importances_ = self._VIPs_calc(Orig_X,Orig_Y)

        return
        
    def transform( self, X, latent_variables = None) :

        """
        Transforms the X matrix to the model-fitted space.

        Parameters
        ----------
        X : array_like
            Matrix with all the data to be used as predictors in one only object
        latent_variables : int, optional
            number of latent variables to be used. 

        Returns
        -------
        result : array_like of shape (X.shape[0], latent_variables)
            Scores for the X values transformed

        """
        
        if isinstance( X, DataFrame ) : X = X.to_numpy()      
        if latent_variables == None : latent_variables = self.latent_variables
        
        if isnan( sum( X ) ) :
            
            result = zeros( ( X.shape[ 0 ], latent_variables ) )
            X_nan = isnan( X )
            variables_missing_mask = unique( X_nan, axis = 0 )

            for row_mask in variables_missing_mask :
                
                rows_indexes = where( ( X_nan == row_mask ).all( axis = 1 ) )                
                
                if sum( row_mask ) == 0 : 

                    result[ rows_indexes, : ] = X[ rows_indexes, :] @ self.weights_star[ :latent_variables, : ].T 
                
                else :
                    
                    result[ rows_indexes, : ] = scores_with_missing_values( self.omega, self.weights_star[ : , ~row_mask ], X[ rows_indexes[ 0 ][ :, None ], ~row_mask], 
                                                                            LVs = latent_variables, method = self.missing_values_method )
                    
        else : result = X @ self.weights_star[ :latent_variables, : ].T 

        return result
    
    def transform_inv( self, scores, latent_variables = None) :

        """
        Transforms the scores matrix to the original X.

        Parameters
        ----------
        scores : array_like
            Matrix with all the scores to be used to rebuild X
        latent_variables : int, optional
            number of latent variables to be used. 

        Returns
        -------
        result : array_like 
            matrix of rebuilt X from scores
        """
                  
        if latent_variables == None : latent_variables = self.latent_variables
        result = scores @ self.weights_star[ :latent_variables, : ] 
        
        return result
    
    def predict( self, X, latent_variables = None ) :
     
        """
        Predicts Y values using X array.

        Parameters
        ----------
        X : array_like
            Samples Matrix
        latent_variables : int, optional
            number of latent variables to be used. 

        Returns
        -------
        preds : array_like 
            returns predictions
        """

        if isinstance( X, DataFrame ) : X = X.to_numpy()        
        if latent_variables == None : latent_variables = self.latent_variables        
        preds = self.transform(X, latent_variables = latent_variables ) @ self.c_loadings[ :latent_variables, : ] 
        
        return preds
    
    def score( self, X, Y, latent_variables = None ):
        
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

        if isinstance(Y, DataFrame) or isinstance(Y, Series) : Y = array( Y.to_numpy() , ndmin = 2 ).T       
        if latent_variables == None : latent_variables = self.latent_variables 

        Y_hat = self.predict( X, latent_variables = latent_variables )
        F = Y - Y_hat
        score = 1 - nanvar( F ) / nanvar( Y )
        
        return score
    
    def Hotellings_T2( self, X, latent_variables = None ):

        """
        Calculates the Hotelling's T² for the X saples.

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
        
        if isinstance( X, DataFrame ) : X = X.to_numpy()     #dataframe, return it
   
        if latent_variables == None : latent_variables = self.latent_variables # Unless specified, the number of PCs is the one in the trained model 
        
        scores_matrix = self.transform( X, latent_variables = latent_variables )
        
        T2s = sum( ( ( scores_matrix / std( scores_matrix) ) ** 2), axis = 1 )
        
        return T2s
    
    def SPEs_X( self, X, latent_variables = None ):

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
        
        if latent_variables == None : latent_variables = self.latent_variables

        if isinstance(X, DataFrame) : X = X.to_numpy()
        
        
        error = X - nan_to_num( X ) @ self.weights_star.T @ self.weights_star 
        SPE = nansum( error ** 2, axis = 1 )
        
        return SPE

    def SPEs_Y(self, X, Y, latent_variables = None) :

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
        
        if latent_variables == None : latent_variables = self.latent_variables

        if isinstance(X, DataFrame) : X = X.to_numpy()
        
        
        error = Y - self.predict(X, latent_variables = latent_variables)
        SPE = nansum( error ** 2, axis = 1 )
        
        return SPE
    
    def RMSEE ( self, X, Y, latent_variables = None ):

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
        
        if latent_variables == None : latent_variables = self.latent_variables
        if isinstance(Y, DataFrame) : Y = array( Y.to_numpy(), ndmin = 2)
        elif isinstance(Y, Series): Y = array( Y.to_numpy(), ndmin = 2).T # in case the function receives a dataframe as Y data
        else : Y = array( Y, ndmin = 2 )

        Y_hat = self.predict( X )
        error = sum( ( Y - Y_hat ) ** 2, axis = 0)
        RMSEE = ( error / ( Y.shape[ 0 ] - latent_variables - 1 ) ) ** (1 / 2)  # il faut ajouter une manière de calculer multiples Y simultaneement
        
        return RMSEE
      
    def contributions_scores_ind( self, X, latent_variables = None ): #contribution of each individual point in X1

        """
        calculates the individual sample individual contributions to the scores.

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

        if latent_variables == None : latent_variables = self.latent_variables
        if isinstance(X, DataFrame) : X = X.to_numpy()

        scores = self.transform( X, latent_variables = latent_variables )
        scores = ( scores / std( scores, axis = 0 ) ** 2 )
        contributions = X * ( scores @ (self.weights_star[ :latent_variables, : ] ** 2 ) ** 1 / 2 )

        return contributions
    
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

        if latent_variables == None : latent_variables = self.latent_variables
        if isinstance(X, DataFrame) : X = X.to_numpy()
        
        error = X - self.transform_inv( self.transform( X ) )
        
        SPE_contributions = ( error ** 2 ) * where( error > 0, 1, -1 )
               
        return SPE_contributions

    def _VIPs_calc( self, X, Y ): # code for calculation of VIPs

        """
        Calculates the VIP scores for all the variables for the prediction
        """
        
        SSY = sum( ( Y - nanmean( Y, axis = 0 ) ) ** 2)
        for i in range( 1, self.weights.shape[ 1 ] + 1 ):
            pred = self.predict( X, latent_variables = i )
            res = Y - pred
            SSY.loc[ i ] = sum(( ( res - res.mean( axis = 0 ) ) ** 2))
            
        SSYdiff = SSY.iloc[ :-1 ]-SSY.iloc[ 1: ]
        VIPs = ( ( ( self.weights ** 2) @ SSYdiff.values ) * self.weights.shape[1] / ( SSY[0] - SSY[-1] ) ** 1 / 2 )
       
        return VIPs