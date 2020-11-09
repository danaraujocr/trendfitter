from numpy import array, isnan, nansum, nan_to_num, multiply, sum, sqrt, append, zeros, place, nan, concatenate, mean, nanvar, std, unique, where, nanmean, identity
from numpy.linalg import norm, pinv
from sklearn.model_selection import KFold
from pandas import DataFrame, Series
from trendfitter.auxiliary.tf_aux import scores_with_missing_values

class MLSMB_PLS:
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
        self.third_level_divs = None
        self.second_level_divs = None
        self.p_loadings_3 = None
        self.p_loadings_2 = None
        self.p_loadings_1 = None
        self.weights_3 = None
        self.weights_2 = None
        self.weights_1 = None
        self.x_weights_star = None
        self.x_weights = None
        self.c_loadings = None 
        self.VIPs = None
        self.coefficients = None
        self.omega = None # score covariance matrix for missing value estimation
        self.training_scores_3 = None
        self.training_scores_2 = None
        self.training_scores_1 = None
        self.training_scores_y = None
            
    def fit(self, X, third_level_divs, second_level_divs, Y, latent_variables = None, deflation = 'both', int_call = False, random_state = None):
        """
        Adjusts the model parameters to best fit the Y using the algorithm defined in 
            

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
        

        """

        
        # check if X and Y make sense
        # check if block_divs make sense 
        # check if latent_variables make sense


        self.third_level_divs = third_level_divs
        self.second_level_divs = second_level_divs
        third_level_pairs = (*zip([0] + third_level_divs[:-1], third_level_divs),)
        second_level_pairs = (*zip(second_level_divs[:-1], second_level_divs[1:]),)

        if isinstance(X, DataFrame):# If X data is a pandas Dataframe
            X = array(X.to_numpy(), ndmin = 2)
          
        if isinstance(Y, DataFrame) or isinstance(Y, Series): # If Y data is a pandas Dataframe or Series 
            Y = array(Y.to_numpy(), ndmin = 2).T
            
        Orig_X = X.copy()
        Orig_Y = Y.copy()
        
        """------------------- model Calculation -------------"""

        all_blocks = set(third_level_pairs).union(set(second_level_pairs))
        max_LVs = [min([pair[1] - pair[0] for pair in third_level_pairs])] + [pair[1] - pair[0]  for pair in second_level_pairs]

        for block, LVs in enumerate(max_LVs):
            for LV in range(LVs):
                if LV >= latent_variables[block]: break

                if block == 0: 
                    result = self._MLSMBPLS_1LV(X,third_level_pairs, second_level_pairs, Y, 0)
                else:
                    missing_values_list = [0 for _ in second_level_pairs]
                    result = self._SMBPLS_1LV(X, second_level_pairs[block-1:], Y, missing_values_list, block)

                
                #------------------------------------deflation----------------------------

                if deflation == 'both' :
                    X -= result['t1l'] @ result['p1l']
                    Y -= result['t1l'] @ result['c']
                elif deflation == 'Y':
                    Y -= result['t1l'] @ result['c']
                else : raise Exception(f'Deflation method "{deflation}" non-existing.')
                    
                #--------------------------Property assignment section--------------------
                if LV == 0 and block == 0 :
                    self.superlevel_weights = result['w1l']
                    #self.x_weights = result['w']
                    #self.block_weights = result['wb']
                    #self.block_p_loadings = result['pb']
                    self.superlevel_p_loadings = result['p1l']
                    self.c_loadings = result['c']
                    self.training_superlevel_scores = result['t1l']
                    #self.training_block_scores = result['tb']
                    self.training_y_scores = result['u']
                    #self.x_weights_star2 = result['w']
                    #for bl, (start, end) in enumerate(block_coord_pairs):
                        #G[bl] = G[bl] - result['w'][:, start:end].T @ result['p'][:, start:end]

                
                else:                    
                    self.superlevel_weights = append(self.superlevel_weights, result['w1l'], axis = 0)
                    #self.x_weights = append(self.x_weights, result['w'], axis = 0)
                    #self.block_weights = append(self.block_weights, result['wb'], axis = 0)
                    #self.block_p_loadings = append(self.block_p_loadings, result['pb'], axis = 0)
                    self.superlevel_p_loadings = append(self.superlevel_p_loadings, result['p1l'], axis = 0)
                    self.c_loadings = append(self.c_loadings, result['c'], axis = 0)
                    self.training_superlevel_scores = append(self.training_superlevel_scores, result['t1l'], axis = 1)
                    #self.training_block_scores = append(self.training_block_scores, result['tb'], axis = 1)
                    self.training_y_scores = append(self.training_y_scores, result['u'], axis = 1)

                    #for bl, (start, end) in enumerate(block_coord_pairs[block:]):
                        #if bl == 0: x_weights_star2 = array(G[block + bl] @ result['w'][:, start:end].T, ndmin = 2).T
                        #else: x_weights_star2 = append(x_weights_star2, array(G[block + bl] @ result['w'][:, start:end].T, ndmin = 2 ).T, axis = 1)
                        #if block > 0 and bl == 0: x_weights_star2 = append(zeros((1, start)), x_weights_star2, axis = 1)

                        #G[block + bl] = G[block + bl] - result['w'][:, start:end].T @ result['p'][:, start:end]
                    
                    #self.x_weights_star2 = append(self.x_weights_star2, x_weights_star2, axis = 0)
                    
        #self.block_weights_star = (self.block_weights @ pinv(self.block_p_loadings.T @ self.block_weights))
        #self.x_weights_star = (self.x_weights @ pinv(self.superlevel_p_loadings.T @ self.x_weights))

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

        if latent_variables == None : 
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
                  
        if latent_variables == None : 
            latent_variables = sum(self.latent_variables)
        else:
            latent_variables = sum(latent_variables)

        result = scores @ self.p[:latent_variables, :] 
        
        return result
                     
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
        if latent_variables == None : 
            latent_variables = sum(self.latent_variables)
        else:
            latent_variables = sum(latent_variables)

        preds = self.transform(X, latent_variables = latent_variables) @ self.c_loadings[ :latent_variables, :]
        
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
        score = 1 - nanvar(F) / nanvar(Y_values)

        return score
    
    def _MLSMBPLS_1LV(self, X, third_level_pairs, second_level_pairs, Y, missing_values):
        
        num_blocks3l = len(third_level_pairs)
        conv = 1
        loop = 0 
        p_3rd = zeros((1, third_level_pairs[-1][-1] - third_level_pairs[0][0]))
        wb_2nd = zeros((1, second_level_pairs[-1][-1] - second_level_pairs[0][0] + num_blocks3l))
        p_2nd = zeros((1, second_level_pairs[-1][-1]  - second_level_pairs[0][0] + num_blocks3l))

        u = array(Y[:, 0], ndmin = 2).T

        while conv > self.tol and loop < self.loop_limit:
            
            wb_3rd, wb_2nd[0, 0:num_blocks3l], t3_final, superT = self._3rdlevel_part(X, third_level_pairs, u, 0)

            start = num_blocks3l
            T1l = superT.copy()
            for i, pair in enumerate(second_level_pairs): #consequence of the second level stuff
                wbcorr, T_scores = self._correlated_part(superT, X[:, pair[0]:pair[1]], u, 0)
                #correlated part
                end = start + pair[1] - pair[0]
                wb_2nd[0, start:end] = wbcorr
                start = end
                T1l = concatenate([T1l, T_scores], axis = 1)
            
            #superlevel stuff 
            # calculating the superlevel weights and scores
            wt, tT = self._superlevel_part(T1l, u)

            # calculating the y side loadings and scores
            q, u_new = self._y_part(tT, Y, 0)

            conv = norm(u - u_new)
            u = u_new
            loop +=1

        
        for i, pair in enumerate(third_level_pairs): #third level
            p_3rd[0, pair[0]:pair[1]] = self._p_loadings(X[:, pair[0]:pair[1]], t3_final[:,i], 0)
        
        p_2nd[0, 0:num_blocks3l] = self._p_loadings(t3_final, T1l[:,0], 0)

        start = num_blocks3l
        for i, pair in enumerate(second_level_pairs):
            end = start + pair[1] - pair[0]
            p_2nd[0, start:end] = self._p_loadings(X[:, pair[0]:pair[1]], T1l[:,i+1], 0)
            start = end


        result_dict = {'w3l':wb_3rd, #weights 3rd level
                       'w2l':wb_2nd, #weights 2rd level
                       'w1l':wt.T, #weights 1rd level
                       'p3l':p_3rd, #p loadings 3rd level
                       'p2l':p_2nd, #p loadings 2rd level
                       'p1l':self._p_loadings(X, tT, 0), #p loadings 1rd level
                       't3l':t3_final, #scores 3rd level
                       't2l':superT, #scores 2nd level
                       't1l':tT, #scores 1st level
                       'u':u,   #y training scores
                       'c':q.T}   #c loadings

        return result_dict

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

        result_dict = {'w2l':block_weights,
                       'w1l':x_weights.T,
                       'p2l':block_p_loadings,
                       't2l':T_scores,
                       'w1l':superlevel_weights.T,
                       'p1l':superlevel_p_loadings.T,
                       't1l':superlevel_scores,
                       'u':y_scores,
                       'c':c_loadings.T}

        return result_dict
    
    def _3rdlevel_part(self, X, third_level_pairs, y_scores, missing_value):
        
        wb_3rd = zeros((1, third_level_pairs[-1][-1] - third_level_pairs[0][0]))
        for i, pair in enumerate(third_level_pairs): #third level
                
            wb = array(X[:, pair[0]:pair[1]].T @ y_scores / (y_scores.T @ y_scores), ndmin = 2).T
            wb /= norm(wb)
            wb_3rd[0, pair[0]:pair[1]] = wb
            t = X[:, pair[0]:pair[1]] @ wb.T
            if i < 1: 
                t3_final = t
            else:
                t3_final = concatenate([t3_final, t], axis = 1)

        #consequence of the third level
        wb12 = array(t3_final.T @ y_scores / (y_scores.T @ y_scores), ndmin = 2).T
        wb12 /= norm(wb12)
        superT = t3_final @ wb12.T


        return wb_3rd, wb12, t3_final, superT

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
            p_loadings = array(p_loadings / nansum(scores ** 2), ndmin = 2) #Step 3.1
            
        else: p_loadings = array(X.T @ scores / (scores.T @ scores), ndmin = 2) #Step 3.1
        if p_loadings.shape[0]> p_loadings.shape[1] : p_loadings = p_loadings.T
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