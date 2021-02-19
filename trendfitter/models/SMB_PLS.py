from numpy import array, isnan, nansum, nan_to_num, multiply, sum, sqrt, append, zeros, place, nan, concatenate, mean, nanvar, std, unique, where, nanmean, identity
from numpy.linalg import norm, pinv
from sklearn.model_selection import KFold
from pandas import DataFrame, Series
from trendfitter.auxiliary.tf_aux import scores_with_missing_values

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
    def __init__(self, cv_splits_number = 7, tol = 1e-16, loop_limit = 1000, missing_values_method = 'TSM'):
        
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
        self.block_weights_star = None
        self.superlevel_weights = None
        self.x_weights_star = None
        self.x_weights = None
        self.c_loadings = None 
        self.VIPs = None
        self.coefficients = None
        self.omega = None # score covariance matrix for missing value estimation
        self.training_superlevel_scores = None
        self.training_y_scores = None
            
    def fit(self, X, block_divs, Y, latent_variables = None, deflation = 'both', int_call = False, random_state = None):
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
        G = [identity(pair[1] - pair[0]) for pair in block_coord_pairs]

        if isinstance(X, DataFrame):# If X data is a pandas Dataframe
            indexes = X.index.copy()
            X_columns = X.columns.copy()
            X = array(X.to_numpy(), ndmin = 2)

        missing_values_list = [isnan(sum(X[:, start:end])) for (start, end) in block_coord_pairs] 
          
        if isinstance(Y, DataFrame) or isinstance(Y, Series): # If Y data is a pandas Dataframe or Series 
            if isinstance(Y, DataFrame) : Y_columns = Y.columns
            Y = array(Y.to_numpy(), ndmin = 2).T
            
        Orig_X, X = X.copy(), X.copy()
        Orig_Y, Y = Y.copy(), Y.copy() 

        missing_values_list = missing_values_list + [isnan(sum(Y))] # This is a check for missing data it'll allow for dealing with it in the next steps
        
        """------------------- Handling the case where the amount of latent variables is not defined -------------------"""
        if latent_variables != None : 
            self.latent_variables = latent_variables 
            max_LVs = latent_variables
        else:
            max_LVs = array(block_divs) - array([0] + block_divs[:-1])
            
            
        """------------------- model Calculation -------------"""
        q2_final = []
        LV_valid = [0 for _ in block_divs]
        for block, LVs in enumerate(max_LVs):

            for LV in range(LVs):

                if not self.latent_variables is None and  LV > 2 * self.latent_variables[block]: break

                result = self._SMBPLS_1LV(X, block_coord_pairs[block:], Y, missing_values_list, block)

                #--------------------------cross-validation------------------------------
                if (block != 0 or LV != 0) and self.latent_variables is None and not int_call:

                    LV_valid[block] = LV + 1
                    q2_final.append(self._cross_validate(Orig_X, block_divs, Orig_Y, LV_valid, random_state))

                    if (q2_final[-1] < q2_final[-2] or  # mean cross-validation performance of the model has decreased with the addition of another latent variable
                        q2_final[-1] - q2_final[-2] < 0.01 or  # mean cross-validation performance of the model has increased less than 1% with the addition of another latent variable
                        LV > (block_coord_pairs[block][1] - block_coord_pairs[block][0]) // 2): # the amount of latent variables added is more than half the variables on X
                        
                        q2_final = q2_final[:-1]
                        self.q2y = q2_final
                        LV_valid[block] -= 1
                        if block == len(max_LVs) - 1 : self.latent_variables = LV_valid 
                        if self.missing_values_method == 'TSM': break  #In case of TSM use there is no need of more components for missing value estimation
                elif int_call and not (latent_variables is None):
                    LV_valid[block] = LV + 1
                    #q2_final.append(self._cross_validate(Orig_X, block_divs, Orig_Y, LV_valid, random_state))
                else:
                    LV_valid[block] = LV + 1    
                    q2_final.append(self._cross_validate(Orig_X, block_divs, Orig_Y, LV_valid, random_state))
                    
                #------------------------------------deflation----------------------------

                if deflation == 'both' :
                    X -= result['tT'] @ result['p']
                    Y -= result['tT'] @ result['c']
                elif deflation == 'Y':
                    Y -= result['tT'] @ result['c']
                else : raise Exception(f'Deflation method "{deflation}" non-existing.')
                    
                #--------------------------Property assignment section--------------------
                if LV == 0 and block == 0 :
                    self.superlevel_weights = result['wT']
                    self.x_weights = result['w']
                    self.block_weights = result['wb']
                    self.block_p_loadings = result['pb']
                    self.superlevel_p_loadings = result['p']
                    self.c_loadings = result['c']
                    self.training_superlevel_scores = result['tT']
                    self.training_block_scores = result['tb']
                    self.training_y_scores = result['u']
                    self.x_weights_star2 = result['w']
                    for bl, (start, end) in enumerate(block_coord_pairs):
                        G[bl] = G[bl] - result['w'][:, start:end].T @ result['p'][:, start:end]

                
                else:                    
                    self.superlevel_weights = append(self.superlevel_weights, result['wT'], axis = 0)
                    self.x_weights = append(self.x_weights, result['w'], axis = 0)
                    self.block_weights = append(self.block_weights, result['wb'], axis = 0)
                    self.block_p_loadings = append(self.block_p_loadings, result['pb'], axis = 0)
                    self.superlevel_p_loadings = append(self.superlevel_p_loadings, result['p'], axis = 0)
                    self.c_loadings = append(self.c_loadings, result['c'], axis = 0)
                    self.training_superlevel_scores = append(self.training_superlevel_scores, result['tT'], axis = 1)
                    self.training_block_scores = append(self.training_block_scores, result['tb'], axis = 1)
                    self.training_y_scores = append(self.training_y_scores, result['u'], axis = 1)

                    for bl, (start, end) in enumerate(block_coord_pairs[block:]):
                        if bl == 0: x_weights_star2 = array(G[block + bl] @ result['w'][:, start:end].T, ndmin = 2).T
                        else: x_weights_star2 = append(x_weights_star2, array(G[block + bl] @ result['w'][:, start:end].T, ndmin = 2 ).T, axis = 1)
                        if block > 0 and bl == 0: x_weights_star2 = append(zeros((1, start)), x_weights_star2, axis = 1)

                        G[block + bl] = G[block + bl] - result['w'][:, start:end].T @ result['p'][:, start:end]
                    
                    self.x_weights_star2 = append(self.x_weights_star2, x_weights_star2, axis = 0)
                    


        
        self.block_weights_star = (self.block_weights @ pinv(self.block_p_loadings.T @ self.block_weights))
        self.x_weights_star = (self.x_weights @ pinv(self.superlevel_p_loadings.T @ self.x_weights))

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

        if latent_variables is None : 
            latent_variables = sum(self.latent_variables)
        else:
            latent_variables = sum(latent_variables)
        
        if isnan(sum(X_values)):
            
            result = zeros((X_values.shape[0], latent_variables))
            X_nan = isnan(X_values)
            variables_missing_mask = unique(X_nan, axis = 0)

            for row_mask in variables_missing_mask:
                
                rows_indexes = where((X_nan == row_mask).all(axis = 1))                
                
                if sum(row_mask) == 0: 

                    result[rows_indexes, :] = X[rows_indexes, :] @ self.x_weights_star[:latent_variables, :].T 
                
                else:
                    
                    result[rows_indexes, :] = scores_with_missing_values(self.omega, self.x_weights_star[:, ~row_mask], X[rows_indexes[0][:, None], ~row_mask], 
                                                                            LVs = latent_variables, method = self.missing_values_method)
                    
        else : result = X_values @ self.x_weights_star[:latent_variables, :].T

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
                  
        if latent_variables is None : 
            latent_variables = sum(self.latent_variables)
        else:
            latent_variables = sum(latent_variables)

        result = scores @ self.p[:latent_variables, :] 
        
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
        if latent_variables is None : 
            latent_variables = sum(self.latent_variables)
        else:
            latent_variables = sum(latent_variables)

        preds = self.transform(X, latent_variables = latent_variables) @ self.c_loadings[ :latent_variables, :]
        
        return preds

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

        if latent_variables is None : latent_variables = self.latent_variables
        if isinstance(Y, DataFrame) or isinstance(Y, Series): 
            Y_values = array(Y.to_numpy(), ndmin = 2).T
        else: 
            Y_values = Y

        Y_hat = self.predict(X, latent_variables = latent_variables)
        F = Y_values - Y_hat
        score = 1 - nanvar(F) / nanvar(Y_values)

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
        
        if isinstance(X, DataFrame) : X = X.to_numpy()     #dataframe, return it
   
        if latent_variables is None : latent_variables = self.latent_variables # Unless specified, the number of PCs is the one in the trained model 
        
        scores_matrix = self.transform( X, latent_variables = latent_variables)
        
        T2s = array(sum(((scores_matrix / std(scores_matrix)) ** 2), axis = 1), ndmin = 2).T
        
        return T2s
    
    #def Hotellings_T2_block(self, X, block, latent_variables = None):

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


    #    return T2s_block
        
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
        
        if latent_variables is None : latent_variables = self.latent_variables

        if isinstance(X, DataFrame) : X = X.to_numpy()
        
        
        error = X - self.transform_inv(self.transform(X, latent_variables = latent_variables), latent_variables = latent_variables)   
        SPE = nansum(error ** 2, axis = 1)
        
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
        
        if latent_variables is None : latent_variables = self.latent_variables

        if isinstance(X, DataFrame) : X = X.to_numpy()
        
        
        error = Y - self.predict(X, latent_variables = latent_variables)
        SPE = nansum(error ** 2, axis = 1)
        
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

        if latent_variables is None : latent_variables = self.latent_variables
        if isinstance(X, DataFrame) : X = X.to_numpy()

        scores = self.transform(X, latent_variables = latent_variables)
        scores = (scores / std(scores, axis = 0)) ** 2

        if latent_variables is None : 
            latent_variables = sum(self.latent_variables)
        else:
            latent_variables = sum(latent_variables)

        contributions = X * (scores @ (self.x_weights_star[:latent_variables, :] ** 2) ** (1 / 2))

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

        if latent_variables is None : latent_variables = self.latent_variables
        if isinstance(X, DataFrame) : X = X.to_numpy()
        
        error = X - self.transform_inv(self.transform(X))
        
        SPE_contributions = (error ** 2) * where(error > 0, 1, -1)
               
        return SPE_contributions

    def _SMBPLS_1LV(self, X, block_coord_pairs, Y, missing_values, block):
        
        conv = 1
        loops = 0

        y_scores1 = nan_to_num(array(Y[ :,0 ], ndmin = 2).T) #handling possible NaNs that come from faulty datasets - Step 1
        superlevel_scores1 = nan_to_num(y_scores1) #initializing superlevel scores vector

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
                block_weights = append(block_weights, corr_block_weights, axis = 1)
                T_scores = append(T_scores, corr_T_scores, axis = 1)
            
            # calculating the superlevel weights and scores
            superlevel_weights, superlevel_scores = self._superlevel_part(T_scores, y_scores1)

            # calculating the y side loadings and scores
            c_loadings, y_scores = self._y_part(superlevel_scores, Y, missing_values[-1])

            conv = norm(superlevel_scores1 - superlevel_scores) / norm(superlevel_scores1)
            superlevel_scores1 = superlevel_scores
            y_scores1 = y_scores

        # Calculating the p_loadings that connect from the raw X values to the superlevel scores
        start, end = block_coord_pairs[0][0], block_coord_pairs[-1][-1]
        superlevel_p_loadings = self._p_loadings(X[:, start:end], superlevel_scores, True in missing_values[:-1]).T
        x_weights = self._p_loadings(X[:, start:end], y_scores, True in missing_values).T
        
        x_weights = x_weights / norm(x_weights)
        y_scores_test = X[:, start:end] @ x_weights

        for missing, scores, (start, end) in zip(missing_values, T_scores.T, block_coord_pairs) :
            
            if start == block_coord_pairs[0][0] : 
                block_p_loadings = self._p_loadings(X[:, start:end], array(scores, ndmin = 2).T, missing)
                if start > 0:
                    block_p_loadings = append(zeros((1, start)), block_p_loadings, axis = 1 )
                    superlevel_p_loadings = append(zeros((start, 1)), superlevel_p_loadings, axis = 0 )
                    x_weights = append(zeros((start, 1)), x_weights, axis = 0 )
                    superlevel_weights = append(zeros((block, 1)), superlevel_weights, axis = 0 )
                    block_weights = append(zeros((1, start)), block_weights, axis = 1 )

            else : 
                block_p_loadings = append(block_p_loadings, self._p_loadings(X[:, start:end], array(scores, ndmin = 2).T, missing), axis = 1)

        result_dict = {'wb':block_weights,
                       'w':x_weights.T,
                       'pb':block_p_loadings,
                       'tb':T_scores,
                       'wT':superlevel_weights.T,
                       'p':superlevel_p_loadings.T,
                       'tT':superlevel_scores,
                       'u':y_scores,
                       'c':c_loadings.T}

        return result_dict
    
    def _uncorrelated_part(self, X, y_scores, missing_value):

        if missing_value:
            block_weights = array(nansum((X * y_scores), axis = 0), ndmin = 2) 
            block_weights = block_weights / nansum((array(~isnan(sum(X, axis = 1)), ndmin = 2).T * y_scores) ** 2, axis = 0)
            block_weights = block_weights / norm(block_weights)
            T_scores = array(nansum(X * block_weights, axis = 1) / nansum(((~isnan(X) * block_weights) ** 2), axis = 1), ndmin = 2).T
        
        else:
            block_weights = X.T @ y_scores / (y_scores.T @ y_scores) # calculating Xb block weights (as step 2.1 in L-G's 2018 paper)
            block_weights = (block_weights / norm(block_weights)).T
            T_scores = X @ block_weights.T
        
        return block_weights, T_scores
    
    def _correlated_part(self, scores, X, y_scores, missing_value):
        
        X_corr_coeffs = (scores / (scores.T @ scores)) @ scores.T
        
        if missing_value :     
            X_corr = X_corr_coeffs @ nan_to_num(X)  # Attention on this part
            place(X_corr, isnan(X), nan) # Keeping the NaN value as an NaN
            block_weights = array(nansum((X_corr * y_scores), axis = 0), ndmin = 2) 
            block_weights = block_weights / nansum((array(~isnan(sum(X, axis = 1)), ndmin = 2).T * y_scores) ** 2, axis = 0)
            block_weights = block_weights / norm(block_weights) #step 2.6
            T_scores =  array(nansum(X_corr * block_weights.T, axis = 1), ndmin = 2).T # step 2.7
        else :    
            X_corr = X_corr_coeffs @ X # finishing step 2.4 for no missing data
            block_weights = X_corr.T @ y_scores / (y_scores.T @ y_scores) # step 2.5
            block_weights = (block_weights / norm(block_weights)).T #step 2.6
            T_scores =  X_corr @ block_weights.T # step 2.7                
        
        return block_weights, T_scores
    
    def _superlevel_part(self, T_scores, y_scores):
        
        superlevel_weights = T_scores.T @ y_scores / (y_scores.T @ y_scores)  #step 2.9
        superlevel_weights = superlevel_weights / norm(superlevel_weights) #step 2.10
        superlevel_scores = T_scores @ superlevel_weights / (superlevel_weights.T @ superlevel_weights) #step 2.11

        return superlevel_weights, superlevel_scores

    def _y_part(self, superlevel_scores, Y, missing_value_Y):

        if missing_value_Y :
            c_loadings = nansum((Y.T * superlevel_scores).T, axis = 0) 
            c_loadings = c_loadings / nansum(((isnan(Y).T * superlevel_scores) ** 2).T, axis = 0)
        else : c_loadings = Y.T @ superlevel_scores / (superlevel_scores.T @ superlevel_scores) # step 2.12
        
        y_scores = Y @ c_loadings / (c_loadings.T @ c_loadings) # step 2.13

        return c_loadings, y_scores

    def _p_loadings(self, X, scores, missing_value):

        if missing_value:
            p_loadings = nansum(X * scores, axis = 0) 
            p_loadings = array( p_loadings / nansum(scores ** 2), ndmin = 2) #Step 3.1
            
        else: p_loadings = array(X.T @ scores / (scores.T @ scores), ndmin = 2).T #Step 3.1

        return p_loadings

    def _cross_validate(self, X_orig, block_divs, Y_orig, LVs, random_state):

        
        cv_folds = KFold(n_splits = self.cv_splits_number, shuffle = True, random_state = random_state)
        q2 = []

        for train_index, test_index in cv_folds.split(X_orig):
            cv_model = SMB_PLS(tol = self.tol)
            cv_model.fit(X_orig[train_index], block_divs, Y_orig[train_index], latent_variables = LVs, int_call = True)
            q2.append(cv_model.score(X_orig[test_index], Y_orig[test_index]))

        q2 = mean(q2)

        return q2

    def _VIPs_calc( self, X, Y ): # code for calculation of VIPs

        """
        Calculates the VIP scores for all the variables for the prediction
        """
        
        SSY = sum((Y - nanmean(Y, axis = 0)) ** 2)
        for i in range(1, self.x_weights_star.shape[1] + 1):
            pred = self.predict(X, latent_variables = i)
            res = Y - pred
            SSY.loc[i] = sum(((res - res.mean(axis = 0)) ** 2))
            
        SSYdiff = SSY.iloc[:-1] - SSY.iloc[1:]
        VIPs = (((self.x_weights ** 2) @ SSYdiff.values) * self.weights.shape[1] / (SSY[0] - SSY[-1]) ** 1 / 2)
       
        return VIPs