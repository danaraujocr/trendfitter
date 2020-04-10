import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
"""
    This Script is for a function that calculates a PCA model in the standards of the
    scikit library implementation so that one can use it with all the functions that
    can receive a regressor or classifier as an object that exist in that library.
    
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
    def __init__(self, 
                 principal_components = None, 
                 cv_splits_number = 7, 
                 tol = 1e-16, 
                 loop_limit = 100):
        
        self.loadings = np.zeros((2,2)) #loadings       
        self.principal_components = principal_components # number of principal components to be extracted
        self.cv_splits_number = cv_splits_number # number of splits for latent variable cross-validation
        self.tol = tol # criteria for convergence
        self.loop_limit = loop_limit # maximum number of loops before convergence is decided to be not attainable
        self.q2=[] # list of cross validation scores
        
    def fit(self, 
            X,
            Y=None):
        if isinstance(X, pd.DataFrame):X=X.to_numpy()
        if self.principal_components != None :
            numberOfLVs = self.principal_components
        else :
            numberOfLVs = X.shape[1] #maximum amount of extractable latent variables
            kf = KFold(n_splits = self.cv_splits_number, 
                       shuffle = True, 
                       random_state = 1)

        MatrixXModel = np.matrix(np.zeros(X.shape)) #initializing matrix space for manipulation 
        
        #----------------------------------------------------------------------------------
        #------------------------------NIPALS implementation-------------------------------
        #----------------------------------------------------------------------------------
        q2_final=[]
        for latent_variable in range(1,numberOfLVs+1):
            scores_vec = np.matrix(X[:,1]).T #initializing the guess by using a specific vector
            MatrixX2 = X-MatrixXModel #deflation of the X matrix
            counter = 0
            conv = 1
            while conv > self.tol and counter < self.loop_limit:
                counter += 1
                loadings_vec = scores_vec.T.dot(np.matrix(MatrixX2)) / (scores_vec.T.dot(scores_vec)) #loadings = scores'*data/scores'*scores        
                loadings_vec = loadings_vec / np.linalg.norm(loadings_vec) #normalization of loading vector
                new_scores_vec = MatrixX2.dot(loadings_vec.T) #scores calculation
                conv = np.sum(np.power((scores_vec - new_scores_vec), 2)) #scores comparation in between loops to assess convergency
                scores_vec = new_scores_vec # old t becomes new t
            #After convergency, if the principal components desired quantity is undefined
            #then we check if the Component is in fact significant and keep on until all 
            #components are there            
            if self.principal_components == None:
                
                testq2 = []
                
                for train_index, test_index in kf.split(X):
                    q2_model = PCA(principal_components = latent_variable)
                    q2_model.fit(X[train_index])
                    testq2.append(q2_model.score(X[test_index],
                                             X[test_index]))
                q2_final.append(np.mean(testq2))
                
                if latent_variable > 1 :
                    
                    if (q2_final[-1] < q2_final[-2] or \
                        q2_final[-1] - q2_final[-2] < 0.01 or \
                        latent_variable > np.min(X.shape)/2):
                        self.q2=q2_final[:-1]
                        break #stop adding new Components if any of the above rules of thumbs are not respected
                        
            #if significant, then we add them to the loadings and score matrixes that will be returned as method result
            if np.array_equal(self.loadings,
                              np.zeros((2,2))):
                self.loadings = loadings_vec
            else:
                self.loadings = np.insert(self.loadings,
                                          self.loadings.shape[0], 
                                          loadings_vec, axis=0)
            #prediction of this model
            MatrixXModel = (X.dot(self.loadings.T)).dot(self.loadings)
        self.principal_components = self.loadings.shape[0]
        pass
        
    def predict(self, X, principal_components = None): 
        if isinstance(X, pd.DataFrame): X = X.to_numpy()
        if principal_components == None : principal_components = self.principal_components
        
        return (self.transform(X,principal_components=principal_components)).dot(self.loadings[:principal_components,:])
    
    def transform(self, X, principal_components = None): 
        
        df=False
        if isinstance(X, pd.DataFrame) :       #In case the data comes as a                   
            df = True                          #dataframe, return it  
            indexes=X.index                    #with the labels correctly
            X = X.to_numpy()                   #positioned
            
        if principal_components == None : principal_components = self.principal_components
        
        result=X.dot(self.loadings[:principal_components,:].T)
        
        if df : result=pd.DataFrame(result,index=indexes)
        return result
    
    def score(self, X, Y = None): #return r²
        
        if isinstance(X, pd.DataFrame) : X = X.to_numpy()
        
        Y = X
        ErrorQ2 = Y - self.predict(X)
        return 1 - np.var(ErrorQ2) / np.var(X)

    def Hotellings_T2(self, X, principal_components=None):
        
        df=False                               #In case the data comes as a 
        if isinstance(X, pd.DataFrame) :       #dataframe, return it
            df=True                            #with the labels correctly
            indexes=X.index                    #positioned
            X = X.to_numpy()
            
        if principal_components == None : principal_components=self.principal_components # Unless specified, the number of PCs is the one in the trained model 
        
        scores_matrix = self.transform(X, principal_components = principal_components)
        
        T2s = np.sum(np.power(scores_matrix / np.std(scores_matrix, axis = 0), 2), axis=1)
        if df : T2s=pd.DataFrame(T2s, index=indexes)
        return T2s
    
    def VIPs(self, X, principal_components = None, confidence_intervals = False, groups_CI = 10):
        
        if principal_components == None : principal_components = self.principal_components
        
        df=False        
        if isinstance(X, pd.DataFrame) : 
            df=True
            columns=X.columns            
            X = X.to_numpy()

        SSX = np.array(np.power(X - X.mean(axis = 0), 2).sum())       
        for i in range(1, principal_components+1):
            pred = self.predict(X, principal_components = i)
            res = X - pred
            SSX=np.append(SSX,np.power(res - res.mean(axis=0), 2).sum())
            
        SSXdiff = SSX[:-1] - np.array(SSX)[1:]
        VIPs = np.power((np.power(self.loadings[:principal_components,:].T, 2).dot(SSXdiff)) * 
                        self.loadings.shape[1] / (SSX[0] - SSX[len(SSX)-1]),
                        1/2) 
        if df : VIPs=pd.DataFrame(VIPs, columns=columns)
        if confidence_intervals == True:
            VIPs_IC = np.zeros((groups_CI,self.trainingDataSet.to_numpy().shape[1]))
            counter = 0
            kf = KFold(n_splits = np.min([groups_CI,self.trainingDataSet.to_numpy().shape[0]]), 
                     shuffle = True, random_state = 1)
            for train_index, test_index in kf.split(self.trainingDataSet.to_numpy()):
                PCA_VIP = PCA(self.trainingDataSet.iloc[train_index])
                VIPs_IC[counter] = PCA_VIP.VIPs()
                counter += 1
            CIs = [np.max(VIPs_IC,axis=0), np.min(VIPs_IC,axis=0)]
            if df : CIs=pd.DataFrame(CIs, columns=columns)
            return VIPs, CIs
                
        return VIPs
    
    

    def contributions_scores_ind(self, X, principal_components = None): #contribution of each individual point in X1
        if principal_components == None : principal_components = self.principal_components
        df=False
        if isinstance(X, pd.DataFrame) : 
            df=True
            indexes=X.index
            columns=X.columns
            X = X.to_numpy()
        
        scores=self.transform(X, principal_components=principal_components)
        scores=np.power(scores/scores.std(axis=0), 2)
        contributions = np.multiply(X,np.power(scores.dot(np.power(self.loadings[:principal_components,:], 2)), 1/2))
        if df: contributions=pd.DataFrame(contributions, columns = columns, index = indexes )

        return contributions
 
    
        
    def contributions_spe(self, X, principal_components = None):
        if principal_components == None : principal_components = self.principal_components
        df=False
        if isinstance(X, pd.DataFrame) : 
            df=True
            indexes=X.index
            columns=X.columns
            X = X.to_numpy()

        error = X - self.predict(X)
        SPE_contributions=np.multiply(np.power(error,2),np.where(error>0,1,-1))
        
        if df: SPE_contributions=pd.DataFrame(SPE_contributions, columns = columns, index = indexes )
        
        return SPE_contributions   


    def SPEs(self, X, principal_components = None):
        if principal_components == None : principal_components = self.principal_components
        df=False
        if isinstance(X, pd.DataFrame) : 
            df=True
            indexes=X.index
            columns=X.columns
            X = X.to_numpy()

        error = X - self.predict(X)
        SPE = np.sum(np.power(error,2), axis = 1)
        
        if df: SPE=pd.DataFrame(SPE, index = indexes, columns=['SPE'] )
        return SPE
    


    