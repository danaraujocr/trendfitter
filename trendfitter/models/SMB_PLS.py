from numpy import array, isnan, nansum, nan_to_num, multiply, sum, sqrt, append, zeros, place, nan, concatenate, mean, nanvar, std, unique, where, nanmean
from numpy.linalg import norm, pinv
from sklearn.model_selection import KFold
from pandas import DataFrame, Series
from trendfitter.auxiliary.tf_aux import scores_with_missing_values

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
    """
    A sklearn-like class for the Sequential Multi-Block Projection to Latent Structures.

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
        latent_variables : [int]
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
        fit(X, blocks_sep, Y)
            Applies the NIPALS like method to find the best parameters that adjust the model 
                to the data
        transform(X)
            Transforms the original data to the latent variable space in the super level 
        transform_inv(scores)
            Returns the superlevel scores to the original data space
        transform_b(X, block)
            Transforms the original data to the latent variable space in the block level for
                all blocks
        predict(X)
            Predicts Y values 
        score(X, Y)
            calculates the r² value for Y
        Hotellings_T2(X)
            Calculates Hotellings_T2 values for the X data in the super level
        Hotellings_T2_blocks(X)
            Calculates Hotellings_T2 values for in the block level for all blocks
        SPEs_X(X)
            Calculates squared prediction errors on the X side for the super level
        SPEs_X_blocks(X)
            Calculates squared prediction errors for the block level for all blocks
        SPEs_Y(X, Y)
            Calculates squared prediction errors for the predictions
        contributions_scores(X)
            calculates the contributions of each variable to the scores on the super level
        contributions_scores_b(X)
            calculates the contributions of each variable to the scores on the super level
        contributions_SPE(X)
            calculates the contributions of each variable to the SPE on the X side for
                the super level
        contributions_SPE_b(X)
            calculates the contributions of each variable to the SPE on the X side for
                the block level for all blocks

    """
    def __init__( self, cv_splits_number = 7, tol = 1e-16, loop_limit = 1000, missing_values_method = 'TSM' ):
        
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
        self.c_loadings = None 
        self.VIPs = None
        self.coefficients = None
        self._omega = None # score covariance matrix for missing value estimation
            
    def fit(self, X, block_divs, Y, latent_variables = None, deflation = 'both', int_call = False):
        """
        Adjusts the model parameters to best fit the Y using the algorithm defined in 
            Lauzon-Gauthier et al's [1]

        Parameters
        ----------
        X : array_like
            Matrix with all the data to be used as predictors in one only object
        block_divs : [int]
            list with the index of every block final position. ex. [2,4] means two blocks with 
                columns 0 and 1 on the first block and columns 2 and 3 on the second block.
        Y : array_like
            Matrix with all the data to be predicted in one only object
        latent_variables : [int], optional
            list of number of latent variables deemed relevant from each block. If left unspecified
                a cross validation routine will define the number during fitting
        deflation : str, optional
            string defining method of deflation, only Y or both X and Y 
        int_call : Boolean, optional
            Flag to define if it is an internal call on the cross validation routine and decide
                if it is necessary to calculate the VIP values

        References 
        ----------
        [1] J. Lauzon-Gauthier, P. Manolescu, C. Duchesne, "The Sequential Multi-block PLS 
            algorithm (SMB-PLS): Comparison of performance and interpretability," Chemometrics 
            and Intelligent Laboratory Systems, vol 180, pp. 72-83, 2018.

        """

        
        # check if X and Y make sense
        # check if block_divs make sense 
        # check if latent_variables make sense


        self.block_divs = block_divs
        block_coord_pairs = (*zip([0] + block_divs[:-1], block_divs),)

        if isinstance(X, DataFrame):# If X data is a pandas Dataframe
            indexes = X.index.copy()
            X_columns = X.columns.copy()
            X = array(X.to_numpy(), ndmin = 2)

        missing_values_list = [isnan(sum(X[:, start:end])) for (start, end) in block_coord_pairs] 
          
        if isinstance( Y, DataFrame ) or isinstance( Y, Series ): # If Y data is a pandas Dataframe or Series 
            if isinstance( Y, DataFrame ) : Y_columns = Y.columns
            Y = array( Y.to_numpy(), ndmin = 2 ).T
            
        Orig_X = X
        Orig_Y = Y

        missing_values_list = missing_values_list + [isnan(sum(Y))] # This is a check for missing data it'll allow for dealing with it in the next steps
        
        """------------------- Handling the case where the amount of latent variables is not defined -------------------"""
        if latent_variables != None : #not implemented
            self.latent_variables = latent_variables #not implemented
            max_LVs = latent_variables
        else:
            max_LVs = array(block_divs) - array([0] + block_divs[:-1])
            
            
        """------------------- model Calculation -------------"""
        q2_final = [0]
        LV_valid = [0 for _ in block_divs]
        for block, LVs in enumerate(max_LVs):

            for LV in range(LVs):

                if self.latent_variables != None and  LV > 2 * self.latent_variables[block]: break

                result = self._SMBPLS_1LV(X, block_coord_pairs[block:], Y, missing_values_list, block)

                #--------------------------cross-validation------------------------------
                if (block != 0 or LV != 0) and self.latent_variables == None:

                    LV_valid[block] = LV + 1
                    q2_final.append(self._cross_validate(Orig_X, block_divs, Orig_Y, LV_valid))

                    if (q2_final[-1] < q2_final[-2] or  # mean cross-validation performance of the model has decreased with the addition of another latent variable
                        q2_final[-1] - q2_final[-2] < 0.01 or  # mean cross-validation performance of the model has increased less than 1% with the addition of another latent variable
                        LV > min(block_coord_pairs[block][1] - block_coord_pairs[block][1]) / 2): # the amount of latent variables added is more than half the variables on X
                        
                        self.q2y = q2_final[:-1]
                        LV_valid[block] -= 1
                        self.latent_variables = LV_valid 
                        if self.missing_values_method == 'TSM' : break  #In case of TSM use there is no need of more components for missing value estimation


                #------------------------------------deflation----------------------------

                if deflation == 'both' :
                    X -= result['tT'] @ result['p'].T
                    Y -= result['tT'] @ result['c'].T
                elif deflation == 'Y':
                    Y -= result['tT'] @ result['c'].T
                else : raise Exception(f'Deflation method "{deflation}" non-existing.')
                    
                #--------------------------Property assignment section--------------------
                if LV == 0 and block == 0 :
                    self.superlevel_weights = result['wT']
                    self.x_weights = result['w']
                    self.block_weights = result['wb']
                    self.block_p_loadings = result['pb']
                    self.superlevel_p_loadings = result['p']
                    self.c_loadings = result['c']
                
                else:                    
                    self.superlevel_weights = append(self.superlevel_weights, result['wT'], axis = 1)
                    self.x_weights = append(self.x_weights, result['w'], axis = 1)
                    self.block_weights = append(self.block_weights, result['wb'], axis = 1)
                    self.block_p_loadings = append(self.block_p_loadings, result['pb'], axis = 1)
                    self.superlevel_p_loadings = append(self.superlevel_p_loadings, result['p'], axis = 1)
                    self.c_loadings = append(self.c_loadings, result['c'], axis = 1)
                
                
                #elif self.latent_variables[block] ==
        
        self.x_weights_star = self.x_weights @ pinv(self.superlevel_p_loadings.T @ self.x_weights)

        return

    def transform(self, X, latent_variables = None): 
        
        """
        Transforms the X matrix to the model-fitted space returning scores
        of the super level.

        Parameters
        ----------
        X : array_like
            Matrix with all the data to be used as predictors in one only object
        latent_variables : [int], optional
            list with number of latent variables to be used. 

        Returns
        -------
        result : array_like of shape (X.shape[0], sum(latent_variables))
            Scores for the X values transformed on the super level

        """

        if isinstance(X, DataFrame): 
            X_values = X.to_numpy()
        else:
            X_values = X    

        if latent_variables == None : latent_variables = self.latent_variables
        
        if isnan( sum( X ) ) :
            
            result = zeros( ( X.shape[ 0 ], latent_variables ) )
            X_nan = isnan( X )
            variables_missing_mask = unique( X_nan, axis = 0 )

            for row_mask in variables_missing_mask :
                
                rows_indexes = where( ( X_nan == row_mask ).all( axis = 1 ) )                
                
                if sum( row_mask ) == 0 : 

                    result[ rows_indexes, : ] = X[ rows_indexes, :] @ self.x_weights_star[ :latent_variables, : ].T 
                
                else :
                    
                    result[ rows_indexes, : ] = scores_with_missing_values( self.omega, self.x_weights_star[ : , ~row_mask ], X[ rows_indexes[ 0 ][ :, None ], ~row_mask], 
                                                                            LVs = latent_variables, method = self.missing_values_method )
                    
        else : result = X @ self.x_weights_star[ :latent_variables, : ].T 

        # TO DO : check if X makes sense with latent variables
        return result

    def transform_inv(self, scores, latent_variables = None):

        """
        Transforms the superlevel scores matrix to the original X.

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
        result = scores @ self.x_weights_star[ :latent_variables, : ] 
        
        return result
    
    def transform_b( self, X, latent_variables = None ): # To Do

        pass
                    
    def predict( self, X, latent_variables = None): 

        """
        Predicts Y values using X array.

        Parameters
        ----------
        X : array_like
            Samples Matrix
        latent_variables : [int], optional
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

        if latent_variables == None : latent_variables = self.latent_variables
     
        Y_hat = self.transform(X, latent_variables = latent_variables) @ self.c_loadings[:, :sum(latent_variables)].T

        return Y_hat

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
        latent_variables : [int], optional
            number of latent variables to be used. 

        Returns
        -------
        score : float 
            returns calculated r².
        """

        if latent_variables == None : latent_variables = self.latent_variables
        if isinstance(Y, DataFrame) or isinstance(Y, Series): 
            Y_values = array(Y.to_numpy(), ndmin = 2).T
        else: 
            Y_values = Y

        Y_hat = self.predict(X, latent_variables = latent_variables)
        F = Y_values - Y_hat
        score = 1 - nanvar( F ) / nanvar( Y_values )

        return score
    
    def Hotellings_T2(self, X, latent_variables = None): 
        
        """
        Calculates the Hotelling's T² for the X samples on the superlevel.

        Parameters
        ----------
        X : array_like
            Samples Matrix
        latent_variables : [int], optional
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
        
        if latent_variables == None : latent_variables = self.latent_variables

        if isinstance(X, DataFrame) : X = X.to_numpy()
        
        
        error = X - self.transform_inv(self.transform(X))   
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
 
    def contributions_scores_ind(self, X, latent_variables = None): 
        
        """
        calculates the sample individual contributions to the super level scores.

        Parameters
        ----------
        X : array_like
            Sample matrix 
        latent_variables : [int], optional
            List of number of latent variables to be used. 

        Returns
        -------
        contributions : array_like 
            matrix of scores contributions for every X sample
        """

        if latent_variables == None : latent_variables = self.latent_variables
        if isinstance(X, DataFrame) : X = X.to_numpy()

        scores = self.transform(X, latent_variables = latent_variables)
        scores = (scores / std(scores, axis = 0) ** 2)
        contributions = X * ( scores @ (self.x_weights_star[ :latent_variables, : ] ** 2 ) ** 1 / 2 )

        return contributions
    
    def contributions_SPE_X(self, X, latent_variables = None): 
        
        """
        calculates the individual sample individual contributions to the squared prediction error 
            of the X variables matrix reconstruction.

        Parameters
        ----------
        X : array_like
            Sample matrix 
        latent_variables : [int], optional
            number of latent variables to be used. 

        Returns
        -------
        SPE_contributions : array_like 
            matrix of SPE contributions for every X sample
        """

        if latent_variables == None : latent_variables = self.latent_variables
        if isinstance(X, DataFrame) : X = X.to_numpy()
        
        error = X - self.transform_inv(self.transform(X))
        
        SPE_contributions = (error ** 2) * where(error > 0, 1, -1)
               
        return SPE_contributions

    def _SMBPLS_1LV(self, X, block_coord_pairs, Y, missing_values, block):
        
        conv = 1
        loops = 0

        y_scores1 = nan_to_num( array( Y[ :,0 ], ndmin = 2 ).T ) #handling possible NaNs that come from faulty datasets - Step 1
        superlevel_scores1 = y_scores1 #initializing superlevel scores vector

        while (conv > self.tol and loops < self.loop_limit) :

            first_block_done = False
            for missing, (start, end) in zip(missing_values, block_coord_pairs) :

                # calculates the block weights and scores for the uncorrelated part 
                if not first_block_done: 
                    block_weights, T_scores = self._uncorrelated_part(X[:, start:end], y_scores1, missing)
                    first_block_done = True
                    continue

                # calculating the block weights and scores for the blocks following the correlated parts of the following blocks
                corr_block_weights, corr_T_scores =  self._correlated_part(array(T_scores[:, 0], ndmin = 2).T, X[:, start:end], y_scores1, missing)
                block_weights = append(block_weights, corr_block_weights, axis = 0)
                T_scores = append(T_scores, corr_T_scores, axis = 1)
            
            # calculating the superlevel weights and scores
            superlevel_weights, superlevel_scores = self._superlevel_part(T_scores, y_scores1)

            # calculating the y side loadings and scores
            c_loadings, y_scores = self._y_part(superlevel_scores, Y, missing_values[-1])

            conv = norm( superlevel_scores1 - superlevel_scores ) / norm( superlevel_scores1 )
            superlevel_scores1 = superlevel_scores
            y_scores1 = y_scores

        # Calculating the p_loadings that connect from the raw X values to the superlevel scores
        start, end = block_coord_pairs[0][0], block_coord_pairs[-1][-1]
        superlevel_p_loadings = self._p_loadings(X[:, start:end], superlevel_scores, True in missing_values[:-1]).T
        x_weights = self._p_loadings(X[:, start:end], y_scores, True in missing_values).T
        x_weights = x_weights / norm(x_weights)

        for missing, scores, (start, end) in zip(missing_values, T_scores.T, block_coord_pairs) :
            
            if start == block_coord_pairs[0][0] : 
                block_p_loadings = self._p_loadings(X[:, start:end], scores, missing)
                if start > 0:
                    block_p_loadings = append(zeros((start, 1)), block_p_loadings, axis = 0 )
                    superlevel_p_loadings = append(zeros((start, 1)), superlevel_p_loadings, axis = 0 )
                    x_weights = append(zeros((start, 1)), x_weights, axis = 0 )
                    superlevel_weights = append(zeros((block, 1)), superlevel_weights, axis = 0 )
                    block_weights = append(zeros((start, 1)), block_weights, axis = 0 )

            else : 
                block_p_loadings = append(block_p_loadings, self._p_loadings(X[:, start:end], scores, missing), axis = 0)

        result_dict = {'wb':block_weights,
                       'w':x_weights,
                       'pb':block_p_loadings,
                       'tb':T_scores,
                       'wT':superlevel_weights,
                       'p':superlevel_p_loadings,
                       'tT':superlevel_scores,
                       'u':y_scores,
                       'c':c_loadings}

        return result_dict
    
    def _uncorrelated_part(self, X, y_scores, missing_value):

        if missing_value:
            block_weights = array(nansum((X * y_scores), axis = 0), ndmin = 2).T 
            block_weights = block_weights / nansum((array(~isnan(sum(X, axis = 1)), ndmin = 2).T * y_scores) ** 2, axis = 0)
        
        else:
            block_weights = X.T @ y_scores / ( y_scores.T @ y_scores ) # calculating Xb block weights (as step 2.1 in L-G's 2018 paper)
        
        block_weights = block_weights / norm( block_weights )
        T_scores = X @ block_weights

        return block_weights, T_scores
    
    def _correlated_part(self, scores, X, y_scores, missing_value):
        
        X_corr_coeffs = (scores / (scores.T @ scores)) @ scores.T
        
        if missing_value :     
            X_corr = X_corr_coeffs @ nan_to_num(X)  # Attention on this part
            place(X_corr, isnan(X), nan) # Keeping the NaN value as an NaN
            block_weights = array(nansum((X_corr * y_scores), axis = 0), ndmin = 2).T 
            block_weights = block_weights / nansum((array(~isnan(sum(X, axis = 1)), ndmin = 2).T * y_scores) ** 2, axis = 0)
            block_weights = block_weights / norm(block_weights) #step 2.6
            T_scores =  array(nansum(X_corr * block_weights.T, axis = 1), ndmin = 2).T # step 2.7
        else :    
            X_corr = X_corr_coeffs @ X # finishing step 2.4 for no missing data
            block_weights = X_corr.T @ y_scores / (y_scores.T @ y_scores) # step 2.5
            block_weights = block_weights / norm(block_weights) #step 2.6
            T_scores =  X_corr @ block_weights # step 2.7                
        
        return block_weights, T_scores
    
    def _superlevel_part(self, T_scores, y_scores):
        
        superlevel_weights = T_scores.T @ y_scores / ( y_scores.T @ y_scores )  #step 2.9
        superlevel_weights = superlevel_weights / norm( superlevel_weights ) #step 2.10
        superlevel_scores = T_scores @ superlevel_weights / ( superlevel_weights.T @ superlevel_weights ) #step 2.11

        return superlevel_weights, superlevel_scores

    def _y_part(self, superlevel_scores, Y, missing_value_Y):

        if missing_value_Y :
            c_loadings = nansum( ( Y.T * superlevel_scores ).T, axis = 0 ) 
            c_loadings = c_loadings / nansum( ( ( isnan( Y ).T * superlevel_scores ) ** 2 ).T, axis = 0 )
        else : c_loadings = Y.T @ superlevel_scores / (superlevel_scores.T @ superlevel_scores ) # step 2.12
        
        y_scores = Y @ c_loadings / ( c_loadings.T @ c_loadings ) # step 2.13

        return c_loadings, y_scores

    def _p_loadings(self, X, scores, missing_value):

        if missing_value:
            p_loadings = nansum(X * scores, axis = 0) 
            p_loadings = array( p_loadings / (scores.T @ scores), ndmin = 2).T #Step 3.1
            
        else: p_loadings = X.T @ scores / (scores.T @ scores) #Step 3.1

        return array(p_loadings, ndmin = 2).T

    def _cross_validate(self, X_orig, block_divs, Y_orig, LVs):

        
        cv_folds = KFold(n_splits = self.cv_splits_number)
        q2 = []

        for train_index, test_index in cv_folds.split(X_orig):
            cv_model = SMB_PLS(tol = self.tol)
            cv_model.fit(X_orig[train_index], block_divs, Y_orig[train_index], latent_variables = LVs)
            q2.append(cv_model.score(X_orig[test_index], Y_orig[test_index]))

        q2 = mean(q2)

        return q2

    def _VIPs_calc( self, X, Y ): # code for calculation of VIPs

        """
        Calculates the VIP scores for all the variables for the prediction
        """
        
        SSY = sum( ( Y - nanmean( Y, axis = 0 ) ) ** 2)
        for i in range( 1, self.x_weights_star.shape[ 1 ] + 1 ):
            pred = self.predict( X, latent_variables = i )
            res = Y - pred
            SSY.loc[ i ] = sum(( ( res - res.mean( axis = 0 ) ) ** 2))
            
        SSYdiff = SSY.iloc[ :-1 ]-SSY.iloc[ 1: ]
        VIPs = ( ( ( self.weights ** 2) @ SSYdiff.values ) * self.weights.shape[1] / ( SSY[0] - SSY[-1] ) ** 1 / 2 )
       
        return VIPs