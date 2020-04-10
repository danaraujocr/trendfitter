import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

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
    
class PLS_SMB:
    def __init__(self, latent_variables = None, cv_splits_number = 7, tol = 1e-16, loop_limit = 1000, deflation = 'both'):
        
        self.block_p_loadings = None
        self.block_weights = None
        self.superlevel_weights = None
        self.superlevel_weights_star = None
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

        is_incomplete_X=[]
        counter=0
        for block in X:
            if not isinstance(block, pd.DataFrame): 
                X[counter] = pd.DataFrame(block)                
            if np.isnan(np.sum(X[counter].values)) : is_incomplete_X.append(counter) # This is a check for missing data it'll allow for dealing with it in the next steps
            counter+=1
        Orig_X=X.copy()
          
        if not isinstance(Y, pd.DataFrame) or isinstance(Y, pd.Series): # in case the function does not receive a dataframe as Y data
            Y=pd.DataFrame(Y)

        elif isinstance(Y, pd.Series): #In case the dataframe has only one column/is a series
            Y=pd.DataFrame(Y)
            
        Orig_Y=Y.copy()
        is_incomplete_Y = np.isnan(np.sum(Y.values)) # This is a check for missing data it'll allow for dealing with it in the next steps
        
        """------------------- Handling the case where the amount of latent variables is not defined -------------------"""
        if self.latent_variables != None :
            latent_variables = self.latent_variables
        else :
            latent_variables = [] #maximum amount of extractable latent variables
            for block in X:
                latent_variables.append(block.shape[1])
            kf = KFold(n_splits = self.cv_splits_number, 
                       shuffle = True, 
                       random_state = 2)
            
        """------------------- model Calculation -------------"""
        q2_final=[]
        block_weights_outer={}
        block_p_loadings_outer={}
        c_loadings_outer={}
        for number1, block in enumerate(X):            
            for latent_variable in range(1,1+latent_variables[number1]):
                block=X[number1]
                loop_count = 0
                conv = 1
                
                y_scores1 = pd.DataFrame(Y.iloc[:,0].where(Y.notna(),0)) #handling possible NaNs that come from faulty datasets - Step 1
                super_scores_old = y_scores1.copy()
                block_weights_inner={}
                
                while conv > self.tol and loop_count < self.loop_limit: # Step 2
                    loop_count+=1
                    
                    """---------------- If there's missing data -----------------"""
                    if number1 in is_incomplete_X:#(number1 in is_incomplete_X) : #To DO
                        block_weights = np.nansum((block.T * y_scores1).T, axis = 0)/np.nansum(((~np.isnan(X).T*y_scores1) ** 2).T, axis = 0)
                        """---------------- If there isn't missing data -----------------"""
                    else : 
                        block_weights = block.T.dot(y_scores1)/(y_scores1.T.dot(y_scores1)).values # calculating Xb block weights (as step 2.1 in L-G's 2018 paper)
                        block_weights = block_weights.rename(columns={'y' : 'LV'+str(latent_variable)+' B'+str(number1)})
                        """---------------- Independent of missing data -----------------"""
                    block_weights_inner[number1] = block_weights/np.linalg.norm(block_weights) # normalizing block weight vectors (as step 2.2 in L-G's 2018 paper)
                    T_scores = block.dot(block_weights_inner[number1]) # calculating the block scores (as step 2.3 in L-G's 2018 paper)
                    
                    for number2, block2 in zip(range(number1+1, len(X)), X[number1+1:]):
                        X_corr_coeffs = (T_scores/(T_scores.T.dot(T_scores)).values).dot(T_scores.T)
                        """---------------- If there's missing data -----------------"""
                        if number2 in is_incomplete_X: #To DO    
                            X_corr = np.multiply(X_corr_coeffs.dot(np.nan_to_num(block2)), ~np.isnan(block2))
                            block_weights_inner[number2] = np.nansum((X_corr.T * y_scores1).T, axis = 0)/np.nansum(((~np.isnan(X).T * y_scores1) ** 2).T, axis = 0)
                            
                            """---------------- If there isn't missing data -----------------"""
                        else:    
                            X_corr = X_corr_coeffs.dot(block2) # finishing step 2.4 for no missing data
                            block_weights_inner[number2] = X_corr.T.dot(y_scores1)/(y_scores1.T.dot(y_scores1)).values # step 2.5
                            block_weights_inner[number2] = block_weights_inner[number2].rename(columns={'y':'LV'+str(latent_variable)+' B'+str(number2)+ 'Corr'+str(number1)})
                            """---------------- Independent of missing data -----------------"""    
                        block_weights_inner[number2] = block_weights_inner[number2]/np.linalg.norm(block_weights_inner[number2]) #step 2.6
                        block2_x_scores = X_corr.dot(block_weights_inner[number2]) # step 2.7
                        T_scores = pd.concat([T_scores, block2_x_scores], axis=1) # step 2.8 done step by step during the loop
                     
                    super_weights = T_scores.T.dot(y_scores1)/(y_scores1.T.dot(y_scores1).values) #step 2.9
                    super_weights = super_weights/np.linalg.norm(super_weights) #step 2.10
                    super_weights = super_weights.rename(columns = {'y' : 'LV' +str(latent_variable) + ' B' + str(number1) })
                    super_scores = T_scores.dot(super_weights)/(super_weights.T.dot(super_weights)).values #step 2.11
                    
                    if is_incomplete_Y:
                        c_loadings = np.nansum((Y.T * super_scores).T, axis = 0)/np.nansum(((np.isnan(Y).T * super_scores) ** 2).T, axis = 0)
                    else:
                        c_loadings = Y.T.dot(super_scores)/(super_scores.T.dot(super_scores)).values # step 2.12
                    
                    y_scores = Y.dot(c_loadings)/(c_loadings.T.dot(c_loadings)).values # step 2.13
                    
                    #conv = np.linalg.norm(y_scores1.values-y_scores.values)/np.linalg.norm(y_scores.values)
                    conv = np.linalg.norm(super_scores.values-super_scores_old.values)/np.linalg.norm(super_scores.values)
                    super_scores_old = super_scores
                    y_scores1 = y_scores 
                block_p_loadings_inner={}
                for number2, block2 in zip(range(number1, len(X)), X[number1:]): # Step 3 
                    block_p_loadings_inner[number2] = block2.T.dot(super_scores)/(super_scores.T.dot(super_scores)).values #Step 3.1
                    if number1!=number2 :
                        block_p_loadings_inner[number2] = block_p_loadings_inner[number2].rename(columns={'LV'+str(latent_variable)+' B'+str(number1):'LV'+str(latent_variable)+' B'+str(number2)+ 'Corr'+str(number1)})
                    X[number2]=block2-super_scores.values.dot(block_p_loadings_inner[number2].T) # Step 3.2
                #X[number1]=block-super_scores.vales.dot(block_p_loadings_inner[number2].T)
                Y_pred = super_scores.dot(c_loadings.T)
                Y=Y- Y_pred#Step 4
                
                """--------------------------Property assignment section--------------------"""
                
                if latent_variable<2 :
                    if number1 == 0: 
                        self.superlevel_weights = super_weights
                    else:
                        self.superlevel_weights = pd.concat([self.superlevel_weights, super_weights], sort = True)
                    
                    block_weights_outer[number1] = block_weights_inner
                    block_p_loadings_outer[number1] = block_p_loadings_inner
                    c_loadings_outer[number1] = c_loadings
                else:
                    self.superlevel_weights = pd.concat([self.superlevel_weights, super_weights], sort = True)
                    for number2, val in block_weights_inner.items():
                        block_weights_outer[number1][number2] = pd.concat([block_weights_outer[number1][number2], val], axis=1)
                    
                    for number2, val in block_p_loadings_inner.items():
                        block_p_loadings_outer[number1][number2] = pd.concat([block_p_loadings_outer[number1][number2], val], axis=1)
                    
                    c_loadings_outer[number1] = pd.concat([c_loadings_outer[number1],c_loadings],axis=1)
                    
        self.block_p_loadings = block_p_loadings_outer
        self.block_weights = block_weights_outer
        self.c_loadings = c_loadings_outer
        
        return
                    
                            
                            
        
            
    def predict1(self, X):
     
        """ to make a prediction, one takes the X matrix, multiply it by the weights_stars to get the Y-scores 
         then the y scores are multiplied by the c_loadings so that we get our predictions"""
        df_index=None
        df_columns=None
        if isinstance(X, pd.DataFrame):
            X = X.values
            df_index=X.index
            df_columns=X.columns
        
            
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
        





#Test Section
        
data = pd.read_csv('toy example csv.csv', delimiter=';', index_col=0)
#data = (data - data.mean())/data.std()

X=[data[['Var1','Var2','Var3']]/np.sqrt(3),data[['Var4','Var5']]/np.sqrt(2)]
Y=data['y']

model=PLS_SMB(tol=1e-8, latent_variables=[2,1])
import time
times=[]
times.append(time.time())
model.fit(X,Y)
times.append(time.time())
print('Time to train was : {0}'.format(times[-1]-times[0]))

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

