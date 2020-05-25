from numpy import min, sum, mean, std, nanvar, insert, array, where, isnan, nan_to_num, nansum, nanmean, identity, zeros, unique
from numpy.linalg import norm
from pandas import DataFrame, Series
from sklearn.model_selection import KFold
from trendfitter.auxiliary.tf_aux import scores_with_missing_values



class PLS:
    def __init__(self, latent_variables = None, cv_splits_number = 7, tol = 1e-8, loop_limit = 1000, deflation = 'both', missing_values_method = 'TSM' ):
        
        self.p_loadings = None
        self.weights = None
        self.weights_star = None
        self.c_loadings = None
        self.latent_variables = latent_variables # number of principal components to be extracted
        self.cv_splits_number = cv_splits_number # number of splits for latent variable cross-validation
        self.tol = tol # criteria for convergence
        self.loop_limit = loop_limit # maximum number of loops before convergence is decided to be not attainable
        self.q2y = [] # list of cross validation scores
        self.deflation = deflation
        self.VIPs = None
        self.coefficients = None
        self.scores = None
        self.omega = None
        self.missing_values_method = missing_values_method
        
    def fit(self, X, Y, random_state = None ):
        """----------------------- Dealing with data possibly coming in the form of a pandas dataframe ------------------------"""
        
        if isinstance(X, DataFrame): # in case the function receives a dataframe as X data
            X = array( X.to_numpy(), ndmin = 2)
        Orig_X = X
            
        if isinstance(Y, DataFrame) : Y = array( Y.to_numpy(), ndmin = 2)
        elif isinstance(Y, Series): Y = array( Y.to_numpy(), ndmin = 2).T # in case the function receives a dataframe as Y data
        else : Y = array( Y, ndmin = 2 )
        Orig_Y = Y
    
        """------------------- Checking if the data is in fact problematic -------------------------"""

        if isnan(sum(X)) or isnan(sum(Y)):
            dataset_complete = False
        else: dataset_complete = True
        
        """------------------- Handling the case where the amount of latent variables is not defined -------------------"""
        if self.latent_variables != None :
            latent_variables = self.latent_variables
        else :
            latent_variables = X.shape[1] #maximum amount of extractable latent variables
            kf = KFold(n_splits = self.cv_splits_number, shuffle = True , random_state = random_state )
        """------------------- model Calculation -------------"""
        G = identity(X.shape[1])
        q2_final=[]
        for latent_variable in range(1,latent_variables+1):
            
            loop_count = 0
            conv = 1
            
            y_scores1 = array( nan_to_num( Y[:,0] ), ndmin = 2 ).T #handling possible NaNs that come from faulty datasets - Step 1

            while conv>self.tol and loop_count<self.loop_limit:#NIPALS internal Loop 
                
                if dataset_complete: # if the dataset is without problematic data, matrix implementation is possible and faster
                    
                    weights = array( X.T @ y_scores1  / ( y_scores1.T @ y_scores1 ), ndmin = 2 ).T #calculating w vector by regressing all u values into X transpose
                    weights = weights / norm( weights ) #normalizing w vector
                    x_scores = X @ weights.T / ( weights @ weights.T ) #regressing all w values into X matrix and producing a t matrix
                    c_loadings = array( Y.T @ x_scores / ( x_scores.T @ x_scores ), ndmin = 2 ).T #regressing all t values into Y and producing the c matrix
                    y_scores = Y @ c_loadings.T / ( c_loadings @ c_loadings.T ) #calculating new u values by regressing cs into Ys
                    
                    conv = norm( y_scores1 - y_scores )
                    y_scores1 = y_scores
                    
                else: # if it isn't without problematic data, then one needs to deal with it by looping for every regression
                                        
                    """ This segment of the code is a lot slower. 
                    - An opportunity to make it faster is to separate the lines and columns
                    with missing data from the ones complete and iterate only on the ones with missing data. The ones with complete lines can 
                    be used for one single matrix operation. 
                    - A second opportunity is to group the missing data columns and lines based on which variables are missing to then execute a matrix 
                    operation for each group."""
                       
                    weights = array( nansum( ( X * y_scores1 ), axis = 0 ) / nansum( ( ( ~isnan( X ) * y_scores1 ) ** 2 ).T, axis = 1), ndmin = 2 )
                    weights = weights / norm( weights )
                    x_scores = array( nansum( X * weights, axis = 1 ) / nansum( ( ( ~isnan( X ) * weights ) ** 2 ), axis = 1), ndmin = 2 ).T
                    c_loadings = array( nansum( Y * x_scores, axis = 0 ) / nansum( ( ( ~isnan( Y ) * x_scores ) ** 2 ).T, axis = 1 ), ndmin = 2)
                    y_scores =  array( nansum( Y * c_loadings, axis = 1 ) / nansum( ( ( ~isnan( Y ) * c_loadings ) ** 2 ), axis = 1), ndmin = 2 ).T

                    conv = norm( y_scores1 - y_scores )
                    y_scores1 = y_scores
                    
               
                loop_count+=1
                if loop_count == self.loop_limit : 
                    print('Model could not converge with %i loops' %(self.loop_limit))
                    return
                
            
            
            """-------------------------------- Cross validation section -----------------------------------"""
            if self.latent_variables == None: # if the quantity of principal components is undefined, a cross-validation routine decides when to stop
                
                testq2 = []
                

                q2_model = PLS(latent_variables = latent_variable, tol = self.tol, deflation = self.deflation, loop_limit = self.loop_limit
                                , missing_values_method = 'TSM') # an internal model is initialized
                for train_index, test_index in kf.split(X):
                    q2_model.fit(Orig_X[train_index,:], Orig_Y[train_index,:]) # this model is trained with only a partition of the total dataset
                    testq2.append(q2_model.score(Orig_X[test_index,:], Orig_Y[test_index,:])) # its performance is registered in a list
                q2_final.append(mean(testq2))
                """ ------------------ coefficients and VIP calculations ----------------- """
             
                if latent_variable > 1 : 
                    
                    if (q2_final[-1] < q2_final[-2] or  # mean cross-validation performance of the model has decreased with the addition of another latent variable
                        q2_final[-1] - q2_final[-2] < 0.01 or  # mean cross-validation performance of the model has increased less than 1% with the addition of another latent variable
                        latent_variable > min(X.shape)/2): # the amount of latent variables added is more than half the variables on X
                        self.q2y=q2_final[:-1]
                        self.latent_variables = latent_variable -1 
                        if self.missing_values_method != 'TSM' : break  #In case of TSM use there is no need of more components for missing value estimation
                        
            """-------------------------------- p loadings calculation section ----------------------------"""                        
            if dataset_complete: # if the dataset is without problematic data, matrix implementation is possible and faster
                p_loadings = array( X.T @ x_scores / ( x_scores.T @ x_scores ), ndmin = 2 ).T
                
            else: # if it isn't without problematic data, then one needs to deal with it by looping for every regression
                p_loadings = array( sum( nan_to_num( X ) * x_scores, axis = 0 ) / sum( ( ~isnan( X ) * x_scores ) ** 2, axis = 0 ), ndmin = 2 )
                

            
            """-------------------------------- deflation section -----------------------------------------"""
            if self.deflation == 'both':                

                X = X - x_scores @ p_loadings
                Y = Y - x_scores @ c_loadings 
                
            elif self.deflation =='Y':

                Y = Y - x_scores.T @ c_loadings.T
                
            else :
                raise ValueError("impossible deflation parameter")
            
            """-------------------------------- class property assignment section -------------------------"""
            if latent_variable<2:
                self.p_loadings = p_loadings
                self.weights = weights
                self.c_loadings = c_loadings
                self.weights_star = weights
                G = G - weights.T @ p_loadings 
                self.scores = array ( x_scores, ndmin = 2 )
            else:
                weights_star = array( G @ weights.T , ndmin = 2 ).T
                G = G - weights_star.T @ p_loadings 
                self.weights_star = insert( self.weights_star, self.weights_star.shape[0], weights_star, axis=0 )          
                self.p_loadings = insert( self.p_loadings, self.p_loadings.shape[0], p_loadings, axis=0 )
                self.weights = insert( self.weights, self.weights.shape[0], weights, axis=0 )
                self.c_loadings = insert( self.c_loadings, self.c_loadings.shape[0], c_loadings, axis=0 )
                self.scores = insert( self.scores, self.scores.shape[1], array( x_scores, ndmin = 2).T, axis = 1)
            self.omega = self.scores.T @ self.scores 

            """-------------------------------- Coefficients property calculations ----------------"""
           
            #coefficients_calc = self.weights_star.dot( self.c_loadings.T )
            
            """-------------------------------- VIPs property calculations ------------------------"""
            #VIPs_calc = self.VIPs_calc(Orig_X,Orig_Y)
            if self.latent_variables != None and latent_variable > 2 * self.latent_variables: break


        return
        
    def transform( self, X, latent_variables = None) :
        
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
                  
        if latent_variables == None : latent_variables = self.latent_variables
        
        return scores @ self.weights_star[ :latent_variables, : ] 
    
    def predict( self, X, latent_variables = None ) :
     
        """ to make a prediction, one takes the X matrix, multiply it by the weights_stars to get the X-scores 
         then the x scores are multiplied by the c_loadings so that we get our predictions"""

        if isinstance( X, DataFrame ) : X = X.to_numpy()        
        if latent_variables == None : latent_variables = self.latent_variables        
        preds = self.transform(X, latent_variables = latent_variables ) @ self.c_loadings[ :latent_variables, : ] 
        
        return preds

 
    def score( self, X, Y, latent_variables = None ):
        
        if isinstance(Y, DataFrame) or isinstance(Y, Series) : Y = array( Y.to_numpy() , ndmin = 2 ).T       
        if latent_variables == None : latent_variables = self.latent_variables 

        Y_hat = self.predict( X, latent_variables = latent_variables )
        F = Y - Y_hat
        score = 1 - nanvar( F ) / nanvar( Y )
        
        return score
    
    def Hotellings_T2( self, X, latent_variables = None ):
        
        if isinstance( X, DataFrame ) : X = X.to_numpy()     #dataframe, return it
   
        if latent_variables == None : latent_variables = self.latent_variables # Unless specified, the number of PCs is the one in the trained model 
        
        scores_matrix = self.transform( X, latent_variables = latent_variables )
        
        T2s = sum( ( ( scores_matrix / std( scores_matrix) ) ** 2), axis = 1 )
        
        return T2s
    
    def SPEs_X( self, X, latent_variables = None ):
        
        if latent_variables == None : latent_variables = self.latent_variables

        if isinstance(X, DataFrame) : X = X.to_numpy()
        
        
        error = X - nan_to_num( X ) @ self.weights_star.T @ self.weights_star 
        SPE = nansum( error ** 2, axis = 1 )
        
        return SPE
    
    def RMSEE ( self, X, Y, latent_variables = None ):
        
        if latent_variables == None : latent_variables = self.latent_variables
        if isinstance(Y, DataFrame) : Y = array( Y.to_numpy(), ndmin = 2)
        elif isinstance(Y, Series): Y = array( Y.to_numpy(), ndmin = 2).T # in case the function receives a dataframe as Y data
        else : Y = array( Y, ndmin = 2 )

        Y_hat = self.predict( X )
        error = sum( ( Y - Y_hat ) ** 2, axis = 0)
        RMSEE = ( error / ( Y.shape[ 0 ] - latent_variables - 1 ) ) ** (1 / 2)  # il faut ajouter une manière de calculer multiples Y simultaneement
        
        return RMSEE
    
    def VIPs_calc( self, X, Y ): # code for calculation of VIPs

        
        SSY = sum( ( Y - nanmean( Y, axis = 0 ) ) ** 2)
        for i in range( 1, self.weights.shape[ 1 ] + 1 ):
            pred = self.predict( X, latent_variables = i )
            res = Y - pred
            SSY.loc[ i ] = sum(( ( res - res.mean( axis = 0 ) ) ** 2))
            
        SSYdiff = SSY.iloc[ :-1 ]-SSY.iloc[ 1: ]
        VIPs = ( ( ( self.weights ** 2) @ SSYdiff.values ) * self.weights.shape[1] / ( SSY[0] - SSY[-1] ) ** 1 / 2 )
       
        return VIPs
    
    
    def contributions_scores_ind( self, X, latent_variables = None ): #contribution of each individual point in X1
        if latent_variables == None : latent_variables = self.latent_variables
        if isinstance(X, DataFrame) : X = X.to_numpy()

        scores = self.transform( X, latent_variables = latent_variables )
        scores = ( scores / std( scores, axis = 0 ) ** 2 )
        contributions = X * ( scores @ (self.weights_star[ :latent_variables, : ] ** 2 ) ** 1 / 2 )

        return contributions
    
    def contributions_spe(self, X, latent_variables = None):
        if latent_variables == None : latent_variables = self.latent_variables
        if isinstance(X, DataFrame) : X = X.to_numpy()
        
        error = X - self.transform_inv( self.transform( X ) )
        
        SPE_contributions = ( error ** 2 ) * where( error > 0, 1, -1 )
               
        return SPE_contributions