from numpy import array, isnan, nansum, nan_to_num, multiply, sum, sqrt, append, zeros, place, nan
from numpy.linalg import norm, pinv
from sklearn.model_selection import KFold
from pandas import DataFrame, Series
from tf_aux import scores_with_missing_values

""" The way this function will work is:
    As it is a multi-block PLS algorithm, the X's should be given in a list where every block is an element. These elements
    may be both pandas Dataframes or numpy arrays. The algorithm will handle the pandas Dataframes to rebuild them at the 
    return part so that the columns and row have their identities preserved.
    
    The model is a class and should be initialized as an object. It will hold the blocks' weights in a list in the 
    same order as the X's given as input in the fit function. It will also hold the superlevel's parameters. 
    If X's are dataframes, model parameters will be too.
    
    If the user wishes to define the amount of latent variables, he may by giving one number that will be used equally in all
    blocks or a list of the same size of the X list with a value for every block and in the same order as X.
    
    The implementation of its functions will be according to: 2018, Lauzon-Gauthier 
    "The Sequential Multi-lock PLS Algorithm (SMB-PLS): Comparison of performance and interpretability". """
    
class SMB_PLS:
    def __init__( self, latent_variables = None, cv_splits_number = 7, tol = 1e-16, loop_limit = 1000, deflation = 'both' ):
        
        self.block_p_loadings = None
        self.block_weights = None
        self.superlevel_weights = None
        self.x_weights = None
        self.x_weights_star = None
        self.c_loadings = None
        self.latent_variables = latent_variables # number of principal components to be extracted
        self.cv_splits_number = cv_splits_number # number of splits for latent variable cross-validation
        self.tol = tol # criteria for convergence
        self.loop_limit = loop_limit # maximum number of loops before convergence is decided to be not attainable
        self.q2y = [] # list of cross validation scores
        self.deflation = deflation
        self.VIPs = None
        self.coefficients = None
        
    def fit(self, X, Y):
        """----------------------- Dealing with data possibly not coming in the form of a pandas dataframe, and missing data points ------------------------"""

        X_columns = None
        Y_columns = None
        indexes = None
        Y_columns = None
        is_incomplete_X = []
        counter = 0

        for block in X:
            if isinstance( block, DataFrame ) : #If X data is a pandas Dataframe
                if X_columns == None : 
                    X_columns = {}
                    indexes = block.index
                X_columns[counter] = block.columns
                X[counter] = array( block.values, ndmin = 2 )
   
            if isnan( sum( X[ counter ] ) ) : is_incomplete_X.append( counter ) # This is a check for missing data it'll allow for dealing with it in the next steps
            counter += 1
        Orig_X = X.copy()
          
        if isinstance( Y, DataFrame ) or isinstance( Y, Series ): # If Y data is a pandas Dataframe or Series 
            if isinstance( Y, DataFrame ) : Y_columns = Y.columns
            Y = array( Y.values, ndmin = 2 ).T
            
        Orig_Y = Y.copy()
        is_incomplete_Y = isnan( sum( Y ) ) # This is a check for missing data it'll allow for dealing with it in the next steps
        
        """------------------- Handling the case where the amount of latent variables is not defined -------------------"""
        if self.latent_variables != None : #not implemented
            latent_variables = self.latent_variables #not implemented
        else :
            latent_variables = [] #maximum amount of extractable latent variables
            for block in X :
                latent_variables.append( block.shape[ 1 ] )
            kf = KFold( n_splits = self.cv_splits_number, shuffle = True, random_state = 2 )
            
        """------------------- model Calculation -------------"""
        q2_final = [] #not implemented
        block_weights_outer = {}
        block_p_loadings_outer = {}
        c_loadings_outer = {}
        for number1, block in enumerate( X ):            
            for latent_variable in range( 1, 1 + latent_variables[ number1 ] ):
                loop_count = 0
                conv = 1
                full_x = block

                y_scores1 = nan_to_num( array( Y[ :,0 ], ndmin = 2 ).T ) #handling possible NaNs that come from faulty datasets - Step 1
                super_scores_old = y_scores1
                block_weights_inner = {}
                
                while conv > self.tol and loop_count < self.loop_limit : # Step 2
                    loop_count += 1
                    
                    """---------------- If there's missing data -----------------"""
                    if number1 in is_incomplete_X :
                        block_weights = array( nansum( ( block * y_scores1 ), axis = 0 ), ndmin = 2 ).T / nansum( ( array(~isnan( sum( block, axis = 1 ) ), ndmin =2 ).T * y_scores1 ) ** 2, axis = 0 )
                        block_weights_inner[ number1 ] = block_weights / norm( block_weights )
                        T_scores = array( nansum( block.T * block_weights_inner[ number1 ], axis = 0 ), ndmin = 2).T
                        """---------------- If there isn't missing data -----------------"""
                    else : 
                        block_weights = block.T @ y_scores1 / ( y_scores1.T @ y_scores1 ) # calculating Xb block weights (as step 2.1 in L-G's 2018 paper)
                        block_weights_inner[ number1 ] = block_weights / norm( block_weights )
                        T_scores = block @ block_weights_inner[ number1 ]
                        """---------------- Independent of missing data -----------------"""
                     # normalizing block weight vectors (as step 2.2 in L-G's 2018 paper)
                     # calculating the block scores (as step 2.3 in L-G's 2018 paper)
                    
                    for number2, block2 in zip( range( number1 + 1, len( X ) ), X[ number1 + 1 : ] ) :
                        X_corr_coeffs = ( T_scores / ( T_scores.T @ T_scores) ) @ T_scores.T
                        full_x = append( full_x, block2, axis = 1 ) 
                        """---------------- If there's missing data -----------------"""
                        if number2 in is_incomplete_X : #To DO    
                            X_corr = X_corr_coeffs @ nan_to_num( block2 )  # Attention on this part
                            place( X_corr, isnan( block2 ), nan )
                            block_weights_inner[ number2 ] = array( nansum( ( X_corr * y_scores1 ), axis = 0 ), ndmin = 2 ).T / nansum( ( array( ~isnan( sum( block2, axis = 1 ) ), ndmin =2 ).T * y_scores1 ) ** 2, axis = 0 )
                            block_weights_inner[ number2 ] = block_weights_inner[ number2 ] / norm( block_weights_inner[ number2 ] ) #step 2.6
                            block2_x_scores =  array( nansum( X_corr * block_weights_inner[ number2 ].T, axis = 1 ), ndmin = 2 ).T # step 2.7
                            """---------------- If there isn't missing data -----------------"""
                        else :    
                            X_corr = X_corr_coeffs @ block2 # finishing step 2.4 for no missing data
                            block_weights_inner[ number2 ] = X_corr.T @ y_scores1 / ( y_scores1.T @ y_scores1 ) # step 2.5
                            block_weights_inner[ number2 ] = block_weights_inner[ number2 ] / norm( block_weights_inner[ number2 ] ) #step 2.6
                            block2_x_scores =  X_corr @ block_weights_inner[ number2 ] # step 2.7
                            
                            """---------------- Independent of missing data -----------------"""    
                        T_scores = append( T_scores, block2_x_scores, axis = 1 ) # step 2.8 done step by step during the loop
                     
                    super_weights = T_scores.T @ y_scores1 / ( y_scores1.T @ y_scores1 )  #step 2.9
                    super_weights = super_weights / norm( super_weights ) #step 2.10
                    super_scores = T_scores @ super_weights / ( super_weights.T @ super_weights ) #step 2.11
                    
                    if is_incomplete_Y :
                        c_loadings = nansum( ( Y.T * super_scores ).T, axis = 0 ) / nansum( ( ( isnan( Y ).T * super_scores ) ** 2 ).T, axis = 0 )
                    else :
                        c_loadings = Y.T @ super_scores / (super_scores.T @ super_scores ) # step 2.12
                    
                    y_scores = Y @ c_loadings / ( c_loadings.T @ c_loadings ) # step 2.13
                    
                    conv = norm( super_scores - super_scores_old ) / norm( super_scores )
                    super_scores_old = super_scores
                    y_scores1 = y_scores 
                block_p_loadings_inner = {}
                for number2, block2 in zip( range( number1, len( X ) ), X[ number1: ] ) : # Step 3
                    
                    if number2 in is_incomplete_X :
                        block_p_loadings_inner[ number2 ] = array( nansum( block2 * super_scores, axis = 0 ) / ( super_scores.T @ super_scores ), ndmin = 2 ).T #Step 3.1
                        
                    else:
                        block_p_loadings_inner[ number2 ] = block2.T @ super_scores / (super_scores.T @ super_scores) #Step 3.1
                    
                    block2 -= super_scores @ block_p_loadings_inner[ number2 ].T # Step 3.2
                Y_pred = super_scores @ c_loadings.T
                Y -= Y_pred #Step 4

                x_weights = array( nansum( ( full_x * y_scores1 ), axis = 0 ), ndmin = 2 ).T / nansum( ( array(~isnan( sum( full_x, axis = 1 ) ), ndmin =2 ).T * y_scores1 ) ** 2, axis = 0 )
                
                """--------------------------Property assignment section--------------------"""
            
                if latent_variable < 2 :
                    if number1 == 0: 
                        self.superlevel_weights = super_weights
                    else :
                        super_weights = append( zeros( ( self.superlevel_weights.shape[0] , 1 ) ), super_weights, axis = 0 )
                        self.superlevel_weights = append( self.superlevel_weights, zeros( ( len( X ) - number1, self.superlevel_weights.shape[1] ) ), axis = 0 )
                        self.superlevel_weights = append( self.superlevel_weights, super_weights, axis = 1 )
                    self.x_weights = x_weights
                    block_weights_outer[number1] = block_weights_inner
                    block_p_loadings_outer[number1] = block_p_loadings_inner
                    c_loadings_outer[number1] = c_loadings
                else : 
                    super_weights = append( zeros( ( self.superlevel_weights.shape[0] , 1 ) ), super_weights, axis = 0 )
                    self.superlevel_weights = append( self.superlevel_weights, zeros( ( len( X ) - number1, self.superlevel_weights.shape[1] ) ) )
                    if len( self.superlevel_weights.shape ) == 1 : self.superlevel_weights = array( self.superlevel_weights, ndmin = 2 ).T
                    self.superlevel_weights = append( self.superlevel_weights, super_weights, axis = 1 )
                    self.x_weights = append( self.x_weights, x_weights, axis = 1 )
                    
                    for number2, val in block_weights_inner.items():
                        block_weights_outer[number1][number2] = append( block_weights_outer[number1][number2] , val, axis = 1 )
                    
                    for number2, val in block_p_loadings_inner.items():
                        block_p_loadings_outer[number1][number2] = append( block_p_loadings_outer[number1][number2], val, axis = 1 )
                    
                    c_loadings_outer[number1] = append( c_loadings_outer[number1], c_loadings, axis = 1 )
                    
        self.block_p_loadings = block_p_loadings_outer
        self.block_weights = block_weights_outer
        self.c_loadings = c_loadings_outer
        # Need to calculate the x_loadings, which are not the block p_loadings
        #self.block_weights_star = self.x_weights @ pinv( self.block_p_loadings @ self.block_weights.T )
        
        return

    def transform( self, X, latent_variables = None, mode = 'star' ):


        pass
                    
    def predict( self, X, latent_variables = None, mode = 'star' ):
     
        """ to make a prediction, one takes the X matrix, multiply it by the weights_stars to get the Y-scores 
         then the y scores are multiplied by the c_loadings so that we get our predictions"""
        df_index = None
        df_columns = None
        if isinstance( X, DataFrame ):
            X = X.values
            df_index = X.index
            df_columns = X.columns
        
            
        if latent_variables == None : latent_variables = self.latent_variables
        
        
        
        return preds

 
    def score(self, X, Y):

        return 0
    
    def Hotellings_T2(self, X, latent_variables = None):
        return 0
    
    def SPEs_X(self, X, latent_variables = None):
             
        return 0 

    def VIPs_calc(self, X, Y): # code for calculation of VIPs
        
        return VIPs
    
    
    def contributions_scores_ind(self, X, latent_variables = None): #contribution of each individual point in X1
        
        return 0
    
    def contributions_spe(self, X, latent_variables = None):
        
        return 0
        
from pandas import read_csv
#Test Section
        
data = read_csv('toy_example csv.csv', delimiter=';', index_col=0)
#data = read_csv('SMBPLS toy_example csv missing.csv', delimiter=';', index_col=0)
#data = (data - data.mean())/data.std()

X=[data[['Var1','Var2','Var3']]/sqrt(3),data[['Var4','Var5']]/sqrt(2)]
Y=data['y']

model=SMB_PLS(tol=1e-8, latent_variables=[2,1])

model.fit(X,Y)
p_loadings=model.block_p_loadings
weights=model.block_weights
super_weights=model.superlevel_weights
c_loadings=model.c_loadings
"""
T2=model.Hotellings_T2(X)
VIPs = model.VIPs
coefficients = model.coefficients
contributions = model.contributions_scores_ind(X)
contributions_spe = model.contributions_spe(X)"""

