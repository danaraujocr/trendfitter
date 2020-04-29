from numpy import min, sum, mean, std, var, insert, array, multiply, where, zeros, append, isnan, nan_to_num, nansum, nanvar, nanmean, unique, ix_, nonzero, nan
from numpy.linalg import norm
from pandas import DataFrame
from sklearn.model_selection import KFold
from tf_aux import scores_with_missing_values

"""
    This Script is for a function that calculates a PCA model in the standards of the
    scikit library implementation so that one can use it with all the functions that
    can receive a regressor or classifier as an object that exists in that library.
    
    The script uses NIPALS implementation, for the extraction of Principal Components 
    according to the PCA methodology (Put a ref) . For its use, it only requires the data matrix which can come
    in the form of a pandas dataframe or numpy matrix, but it will also receive the number of components 
    one wants to extract on the principal_components argument.
    PreProcessing must be done by user before PCA method call, that allows for different strategies to
    be used based on user experience. 
    Tolerance refers to the convergence criterion which is the norm of the difference vector 
    between scores before and after one loop, and loopLimit is the maximum acceptable 
    number of loops to extract a component. 
    The fit method extracts the loadings of the PCA model and saves them into a property.
    
    Additionaly, the class has methods that return the:

    -SPEs 
    -T2s 
    -Contributions to scores and SPEs
    -VIPs
    
    Which can be used for actual multivariate statistical process control
    Author: Daniel de Araujo Costa Rodrigues - Ulaval PhD Student 2019-11-22
"""


class PCA:

    def __init__( self, principal_components = None, cv_splits_number = 7, tol = 1e-12, loop_limit = 100, missing_values_method = 'TRI' ):
        
        self.loadings = None #loadings       
        self.principal_components = principal_components # number of principal components to be extracted
        self.cv_splits_number = cv_splits_number # number of splits for latent variable cross-validation
        self.tol = tol # criteria for convergence
        self.loop_limit = loop_limit # maximum number of loops before convergence is decided to be not attainable
        self.q2 = [] # list of cross validation scores
        self.feature_importances_ = None #for scikit learn use with feature selection methods
        self.omega = None # scores covariance matrix for missing values score estimation
        self.all_loadings = None
        self.missing_values_method = missing_values_method

        
    def fit( self, X, Y = None, int_call = False ):

        if isinstance( X, DataFrame ) : X = X.to_numpy() #ensuring data in the numpy format
        dataset_incomplete = False 
        if isnan( sum( X ) ): dataset_incomplete = True #checking if there is missing data on the input X

        #CV = False
        if self.principal_components != None : 
            numberOfLVs = self.principal_components 
            #CV = True
        else :
            numberOfLVs = X.shape[1] #maximum amount of extractable latent variables
            kf = KFold( n_splits = self.cv_splits_number, shuffle = True, random_state = 1 )

        MatrixXModel = array( zeros( X.shape ), ndmin = 2 ) #initializing matrix space for manipulation 
        
        #----------------------------------------------------------------------------------
        #------------------------------NIPALS implementation-------------------------------
        #----------------------------------------------------------------------------------
        q2_final = []
        for latent_variable in range( 1, numberOfLVs + 1 ) :
            scores_vec = nan_to_num( array( X[ :, 1 ], ndmin = 2 ).T ) #initializing the guess by using a specific vector
            MatrixX2 = X - MatrixXModel #deflation of the X matrix
            counter = 0
            conv = 1
            while conv > self.tol and counter < self.loop_limit:
                counter += 1
                
                if dataset_incomplete:

                    loadings_vec = array( nansum( MatrixX2 * scores_vec, axis = 0 ) / nansum( ( ( ~isnan( X ) * scores_vec ) ** 2 ).T, axis = 1 ), ndmin = 2 ) # equivalent to loadings = scores'*data/scores'*scores
                    loadings_vec = loadings_vec / norm( loadings_vec ) #normalization of loading vector
                    new_scores_vec = array( nansum( MatrixX2 * loadings_vec, axis = 1 ) / nansum( ( ( ~isnan( X ) * loadings_vec ) ** 2 ), axis = 1), ndmin = 2 ).T #scores calculation w/ missing data
                
                else:
                    
                    loadings_vec = array( scores_vec.T @ MatrixX2 / ( scores_vec.T @ scores_vec  ), ndmin = 2 ) #loadings = scores'*data/scores'*scores        
                    loadings_vec = loadings_vec / norm( loadings_vec ) #normalization of loading vector
                    new_scores_vec = MatrixX2 @ loadings_vec.T #scores calculation
                
                conv = sum( ( scores_vec - new_scores_vec ) ** 2 )  #scores comparation in between loops to assess convergency
                scores_vec = new_scores_vec # old t becomes new t

            #After convergency, if the principal components desired quantity is undefined
            #then we check if the Component is in fact significant and keep on until all 
            #components are there            
            if self.principal_components == None :
                
                testq2 = []
                
                for train_index, test_index in kf.split( X ):

                    q2_model = PCA( principal_components = latent_variable, missing_values_method = 'self.missing_values_method' )
                    q2_model.fit( X[ train_index ], int_call = True )
                    testq2.append( q2_model.score( X[ test_index ], X[ test_index ] ) )

                q2_final.append( mean( testq2 ) )
                
                if latent_variable > 1 :
                    
                    if ( q2_final[ -1 ] < q2_final[ -2 ] or \
                        q2_final[ -1 ] - q2_final[ -2 ] < 0.01 or \
                        latent_variable > min( X.shape ) / 2 ):
                        self.q2 = q2_final[ :-1 ]
                        break #stop adding new Components if any of the above rules of thumbs are not respected
                        
            #if significant, then we add them to the loadings and score matrixes that will be returned as method result
            if latent_variable < 2 :

                self.loadings = loadings_vec
                self.training_scores = scores_vec

            else:

                self.loadings = insert( self.loadings, self.loadings.shape[0], loadings_vec, axis = 0 )
                self.training_scores = insert( self.training_scores, self.training_scores.shape[1], scores_vec.T, axis = 1 )

             
            #prediction of this model
            MatrixXModel = self.training_scores @ self.loadings 
        self.principal_components = self.loadings.shape[ 0 ]        

        if not int_call : 
            self.feature_importances_ = self.VIPs_calc(X)
            self.omega = self.training_scores.T @ self.training_scores # calculation of the covariance matrix

        pass
        
    def predict( self, X, principal_components = None ):

        if isinstance( X, DataFrame ) : X = X.to_numpy()
        if principal_components == None : principal_components = self.principal_components
        
        preds = ( self.transform( X, principal_components = principal_components ) ) @ self.loadings[ :principal_components, : ] 
        
        return preds
    
    def transform( self, X, principal_components = None ): 
        
        if isinstance( X, DataFrame ) : X = X.to_numpy()            
        if principal_components == None : principal_components = self.principal_components
        
        if isnan( sum( X ) ) :
            result = zeros( ( X.shape[ 0 ], principal_components ) )
            X_nan = isnan( X )
            variables_missing_mask = unique( X_nan, axis = 0 )
            
            for row_mask in variables_missing_mask :
                
                rows_indexes = where( ( X_nan == row_mask ).all( axis = 1 ) )                
                
                if sum( row_mask ) == 0 : 

                    result[ rows_indexes, : ] = X[ rows_indexes, :] @ self.loadings[ :principal_components, : ].T 
                
                else :
                    
                    result[ rows_indexes, : ] = scores_with_missing_values( self.omega, self.loadings[ : , ~row_mask ], X[ rows_indexes[ 0 ][ :, None ], ~row_mask ], 
                                                                            LVs = principal_components, method = self.missing_values_method )
                 
                    
        else : result = X @ self.loadings[ :principal_components, : ].T 
        
        return result
    
    def score( self, X, Y = None ): #return r²
        
        if isinstance( X, DataFrame ) : X = X.to_numpy()
        
        Y = X
        ErrorQ2 = Y - self.predict( X  )
        return 1 - nanvar( ErrorQ2 ) / nanvar( X )

    def Hotellings_T2( self, X, principal_components = None ):
        
        if isinstance( X, DataFrame ) : X = X.to_numpy()
            
        if principal_components == None : principal_components = self.principal_components # Unless specified, the number of PCs is the one in the trained model 
        
        scores_matrix = self.transform( X, principal_components = principal_components )
        
        T2s = sum( scores_matrix / std( scores_matrix, axis = 0 ) ** 2 , axis = 1 )
        return T2s
    
    def VIPs_calc( self, X, principal_components = None, confidence_intervals = False ):
        
        if principal_components == None : principal_components = self.principal_components
              
        if isinstance( X, DataFrame ) : X = X.to_numpy()

        SSX = array( sum( ( nan_to_num(X) - nanmean( X, axis = 0 ) ) ** 2 ) )   

        for i in range( 1, principal_components + 1 ) :

            pred = self.predict( X, principal_components = i )
            res = nan_to_num(X) - pred
            SSX = append( SSX, ( res - res.mean( axis = 0 ) ** 2 ).sum() )
            
        SSXdiff = SSX[ :-1 ] - array( SSX )[ 1: ]
        VIPs = array( ( ( self.loadings[ :principal_components, : ].T ** 2 ) @  SSXdiff * 
                          self.loadings.shape[ 1 ] / ( SSX[ 0 ] - SSX[ len( SSX ) - 1 ] ) ) ** 1 / 2 , ndmin = 2)                
        return VIPs
    
    

    def contributions_scores_ind( self, X, principal_components = None) : #contribution of each individual point in X1
        if principal_components == None : principal_components = self.principal_components
        if isinstance( X, DataFrame ) : X = X.to_numpy()
        
        scores = self.transform( X, principal_components = principal_components )
        scores = ( scores / scores.std( axis=0 ) ) ** 2 
        contributions = multiply( X, ( scores @ self.loadings[ :principal_components, : ] ** 2 ) ** 1 / 2  ) 

        return contributions
 
    
        
    def contributions_spe( self, X, principal_components = None ) :
        if principal_components == None : principal_components = self.principal_components
        if isinstance( X, DataFrame ) : X = X.to_numpy()

        error = nan_to_num(X) - self.predict( X )
        SPE_contributions = multiply(  error ** 2 , where( error > 0, 1, -1 ) )
        
        return SPE_contributions   


    def SPEs( self, X, principal_components = None ) :
        if principal_components == None : principal_components = self.principal_components
        if isinstance( X, DataFrame ) : X = X.to_numpy()

        error = nan_to_num(X) - self.predict( X )
        SPE = sum( error ** 2 , axis = 1 )
        
        return SPE