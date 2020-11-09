from numpy import min, sum, mean, std, nanvar, insert, array, where, isnan, nan_to_num, \
                    nansum, nanmean, identity, zeros, unique, var, append, nansum, any, \
                    concatenate
from numpy.core.numeric import ones, outer
from numpy.linalg import norm, pinv
from pandas import DataFrame, Series
from sklearn.model_selection import KFold
from scipy.stats import f, chi2
from trendfitter.auxiliary.tf_aux import scores_with_missing_values

class DiPLS:

    """
    A sklearn-like class for the Projection to Latent Structures.

        Parameters
        ----------
        cv_splits_number : int, optional
            number of splits used for cross validation in case latent_variables is None

        Attributes
        ----------
        latent_variables : int
            number of latent variables one wants to extract.
        
        Methods
        -------
        fit(X, Y, latent_variables = None, deflation = 'both', random_state = None, int_call = False)
            Applies the NIPALS like method to find the best parameters that adjust the model 
                to the data 
    
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
        self.coefficients = None
        self.training_scores = None
        self.omega = None
        self.missing_values_method = missing_values_method
        self.feature_importances_ = None #for scikit learn use with feature selection methods
        self._x_chi2_params = []
        self._y_chi2_params = []

    def fit(self, X, Y, latent_variables = None, s = 0, deflation = 'both', random_state = None, int_call = False):

        """
        Adjusts the model parameters to best fit the Y using the algorithm defined in 
            Dong et Qin's [1]

        Parameters
        ----------
        X : array_like
            Matrix with all the data to be used as predictors in one only object
        Y : array_like
            Matrix with all the data to be predicted in one only object
        latent_variables : int, optional
            number of latent variables deemed relevant. If left unspecified
                a cross validation routine will define the number during fitting.
        s : int

        deflation : str, optional
            string defining method of deflation, only Y or both X and Y 
        random_state : int, optional
            value used as a seed in the random number generator for cross validation

        References 
        ----------
        [1] Y. Dong and S. J. Qin, “Regression on dynamic PLS structures for supervised 
        learning of dynamic data,” Journal of Process Control, vol. 68, pp. 64–72, 
        Aug. 2018, doi: 10.1016/j.jprocont.2018.04.006.

        """

        """----------------------- Dealing with data possibly coming in the form of a pandas dataframe ------------------------"""
        
        self.latent_variables = latent_variables
        self.s = s
        self.deflation = deflation
        
        if isinstance(X, DataFrame): # in case the function receives a dataframe as X data
            X = array(X.to_numpy(), ndmin = 2)
        Orig_X = X
            
        if isinstance(Y, DataFrame): Y = array( Y.to_numpy(), ndmin = 2)
        elif isinstance(Y, Series): Y = array( Y.to_numpy(), ndmin = 2).T # in case the function receives a dataframe as Y data
        else: Y = array(Y, ndmin = 2)
        Orig_Y = Y

        G = identity(X.shape[1])

        for latent_variable in range(1, latent_variables + 1):
            
            """-------------------------------- Model Building Loop ---------------------------------------"""
            y_scores, y_scores_hat, x_scores, weights, q_loadings, outer_beta, inner_beta, outer_alpha, inner_alpha = self._1LV_loop(X, Y, s)
            p_loadings = (X.T @ x_scores / (x_scores.T @ x_scores)).T


            """-------------------------------- Deflation -------------------------------------------------"""
            X -= x_scores @ p_loadings
            Y[s:, :]-= y_scores_hat @ q_loadings.T


            """-------------------------------- class property assignment section -------------------------"""
            if latent_variable < 2:
                self.p_loadings = p_loadings
                self.weights = weights
                self.q_loadings = q_loadings
                self.weights_star = weights
                self.outer_beta = outer_beta
                self.inner_beta = inner_beta
                self.outer_alpha = outer_alpha
                self.inner_alpha = inner_alpha
                G = G - weights.T @ p_loadings 
                self.training_x_scores = array(x_scores, ndmin = 2)

            else:
                weights_star = array(G @ weights.T , ndmin = 2).T
                G = G - weights_star.T @ p_loadings 
                self.weights_star = insert(self.weights_star, self.weights_star.shape[0], weights_star, axis = 0)          
                self.p_loadings = insert(self.p_loadings, self.p_loadings.shape[0], p_loadings, axis = 0)
                self.weights = insert(self.weights, self.weights.shape[0], weights, axis = 0)
                self.q_loadings = insert(self.q_loadings, self.q_loadings.shape[0], q_loadings, axis = 0)
                self.outer_beta = insert(self.outer_beta, self.outer_beta.shape[0], outer_beta, axis = 0)
                self.inner_beta = insert(self.inner_beta, self.inner_beta.shape[0], inner_beta, axis = 0)
                self.outer_alpha = insert(self.outer_alpha, self.outer_alpha.shape[0], outer_alpha, axis = 0)
                self.inner_alpha = insert(self.inner_alpha, self.inner_alpha.shape[0], inner_alpha, axis = 0)
                self.training_x_scores = insert(self.training_x_scores, self.training_x_scores.shape[1], array( x_scores, ndmin = 2).T, axis = 1)
                
            self.omega = self.training_x_scores.T @ self.training_x_scores 

        pass

    def transform_to_y_scores(self, X, latent_variables = None):

        if isinstance(X, DataFrame): X = X.to_numpy()      
        if latent_variables is None: latent_variables = self.latent_variables

        t_scores = X @ self.weights_star[:latent_variables, :].T

        return t_scores

    def transform_di(self, X, old_Ys = None, latent_variables = None, s = None):

        if isinstance(X, DataFrame): X = X.to_numpy()      
        if latent_variables is None: latent_variables = self.latent_variables
        if s is None: s = self.s
        
        if old_Ys is None:
            t_scores = self.transform_to_y_scores(X, latent_variables = latent_variables)
            y_len = t_scores.shape[0]
            y_scores_hat = sum([t_scores[(s - i):y_len - i, :] * beta for i, beta in enumerate(self.inner_beta.T)], axis = 0)
        else:
            if isinstance(old_Ys, DataFrame): old_Ys = old_Ys.to_numpy()
            t_scores = self.transform_to_y_scores(X, latent_variables = latent_variables)
            y_len = t_scores.shape[0]
            y_scores_hat = sum([t_scores[(s - i):y_len - i, :] * beta for i, beta in enumerate(self.inner_beta.T)], axis = 0)
            y_scores = old_Ys @ self.q_loadings.T
            y_scores_hat += sum([y_scores[(s - (i + 1)):y_len - (i + 1), :] * beta for i, beta in enumerate(self.inner_alpha.T)], axis = 0)


        return y_scores_hat

    def predict(self, X, old_Ys = None, latent_variables = None, s = None):
    
        y_scores_hat = self.transform_di(X, old_Ys = old_Ys, latent_variables = latent_variables, s = s)
        
        preds = y_scores_hat @ self.q_loadings

        return preds
        
    def score(self, X, Y, latent_variables = None, s = None):

        if isinstance(Y, DataFrame) or isinstance(Y, Series): Y = array(Y.to_numpy(), ndmin = 2).T  
        if latent_variables is None: latent_variables = self.latent_variables
        if s is None: s = self.s 

        Y_hat = self.predict(X, Y, latent_variables = latent_variables, s = s)
        F = Y[s:, :] - Y_hat
        score = 1 - nanvar(F) / nanvar(Y)
        
        return score

    def _1LV_loop(self, X, Y, s):

        y_scores, x_scores, weights, q_loadings, outer_beta, outer_alpha = self._outer_loop(X, Y, s)

        y_scores_hat, inner_beta, inner_alpha = self._inner_loop(x_scores, y_scores, s)

        return y_scores, y_scores_hat, x_scores, weights, q_loadings, outer_beta.T, inner_beta.T, outer_alpha.T, inner_alpha.T
    
    def _outer_loop(self, X, Y, s):
        
        outer_alpha = array([0 for i in range(s)], ndmin = 2).T
        if len(outer_alpha>0): outer_alpha[0] = 1
        outer_beta = array([1] + [0 for i in range(s)], ndmin = 2).T
        y_scores = array(Y[:, 0], ndmin = 2).T
        q_loadings = ones((1, Y.shape[1]))
        conv = 1
        loops = 0
        X_len = X.shape[0]

        while (conv > self.tol and loops < self.loop_limit):
            
            
            x_beta = sum([X[(s - i):X_len - i, :] * beta for i, beta in enumerate(outer_beta)], axis = 0).T
            weights = array(x_beta @ y_scores[s:], ndmin = 2).T
            weights /= norm(weights)
            x_scores = X @ weights.T
            t_beta = sum([x_scores[s - i:X_len - i] * beta for i, beta in enumerate(outer_beta)], axis = 0)
            y_alpha = sum([Y[(s - (i + 1)): X_len - (i + 1), :] * alpha for i, alpha in enumerate(outer_alpha)], axis = 0)
            u_alpha = y_alpha @ q_loadings.T

            q_loadings_calc = Y[s:, :].T @ t_beta + y_alpha.T @ y_scores[s:, :] + Y[s:, :].T @ u_alpha
            #q_loadings = Y[s:, :].T @ t_beta versão funcional
            q_loadings_calc /= norm(q_loadings_calc)


            y_scores_calc = Y @ q_loadings.T
            
            T_s = concatenate([x_scores[(s - i):X_len - i, :] for i in range(s + 1)], axis = 1)
            outer_beta_calc = T_s.T @ y_scores_calc[s:, :]
            outer_beta_calc /= norm(outer_beta_calc)

            U_s = concatenate([y_scores[(s - i):X_len - i, :] for i in range(s)], axis = 1)
            outer_alpha_calc = U_s.T @ y_scores_calc[s:, :]
            outer_alpha_calc /= norm(outer_alpha_calc)
             

            loops += 1
            conv = norm(y_scores - y_scores_calc) + norm(outer_beta - outer_beta_calc) + \
                    norm(outer_alpha - outer_alpha_calc) + norm(q_loadings - q_loadings_calc)
            y_scores = y_scores_calc
            outer_beta = outer_beta_calc
            outer_alpha = outer_alpha_calc
            q_loadings = q_loadings_calc

        return y_scores_calc, x_scores, weights, q_loadings, outer_beta, outer_alpha

    def _inner_loop(self, x_scores, y_scores, s):

        X_len = x_scores.shape[0]
        big_Ts = concatenate([x_scores[(s - i):X_len - i, :] for i in range(s+1)] + \
                             [y_scores[(s - (i + 1)):- (i + 1)] for i in range(s)], axis = 1)
        inner_beta_alpha = (pinv(big_Ts.T @ big_Ts) @ big_Ts.T) @ y_scores[s:]
        y_scores_hat = big_Ts @ inner_beta_alpha

        return y_scores_hat, inner_beta_alpha[:s+1, :], inner_beta_alpha[s+1:2 * s + 1 , :]


