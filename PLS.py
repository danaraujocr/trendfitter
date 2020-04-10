import numpy as np

def PLS(Xmatrix,Ymatrix,LVSel=True,LVN=1,tol=1e-16,lpLimit=100):
    """
    20/06/2019
    This is an implementation of a PLS algorithm.
    It should receive at least two arguments which are the X matrix and the Y matrix.
    a third variable called LVSel will be a boolean andis going to say if the 
    number of latent variables should be automatically defined by a criterion or 
    if it will be a manual input. if it is, then LVN needs to be supplied or it will
    assume it is one LV. the tol variable defines the value of the convergence criterion
    and the loopLimit defines the maximum number of loops before the algorithm stops running
    The resulting model is returned as a PLSModelResults object, which has all the 
    characterizing as properties and methods to predict new values and calculate 
    Hotelling T2 stats and SPEs. The plotting is left for the user to work out as he
    wishes.

    In this first implementation the values are all done in np.matrixes, but in the 
    future i wish to implement it in a way that it receives a pandas dataframe and keeps 
    all the t's, u's, T2's and SPE's organized using the dataframe index so that one
    can easily reorganize stuff.
    """
    res=PLSModelResults()
    lpCount=0
    tMatrix=np.matrix(np.zeros((2,2)))
    pMatrix=0
    uMatrix=0
    wMatrix=0
    cMatrix=0
    #X=Xmatrix
    #Y=Ymatrix
    
    X=np.matrix(Xmatrix.values)
    Y=np.matrix(Ymatrix.values)
    
    u=np.matrix(Y[:,0]) #Getting the starting vector
    #print(u)
    for i in range(0,LVN):
        u2=u-1
        lpCount=0
        
        while np.linalg.norm(u2-u)>tol and lpCount<lpLimit:#NIPALS internal Loop        
            #print(np.linalg.norm(u2-u))
            u=u2
            w=(X.T.dot(u)/(u.T.dot(u))) #calculating w vector by regressing all u values into X transpose
            w=w/np.linalg.norm(w) #normalizing w vector
            t=(X.dot(w)/(w.T.dot(w)))#regressing all w values into X matrix and producing a t matrix
            c=(Y.T.dot(t)/(t.T.dot(t)))#regressing all t values into Y and producing the c matrix
            u2=(Y.dot(c)/(c.T.dot(c)))#calculating new u values by regressing cs into Ys
            lpCount=lpCount+1
        #print(np.linalg.norm(u2-u))
        p=X.T.dot(t)/(t.T.dot(t))
        X=X-t.dot(p.T)
        Y=Y-t.dot(c.T)
        if np.array_equal(tMatrix,np.zeros((2,2))):
            tMatrix=np.matrix(t)
            pMatrix=np.matrix(p.T)
            uMatrix=np.matrix(u)
            wMatrix=np.matrix(w.T)
            cMatrix=np.matrix(c)
        else:
            tMatrix=np.insert(tMatrix,tMatrix.shape[1],t.T,axis=1)
            pMatrix=np.insert(pMatrix,pMatrix.shape[0],p.T,axis=0)
            uMatrix=np.insert(uMatrix,uMatrix.shape[1],u.T,axis=1)
            wMatrix=np.insert(wMatrix,wMatrix.shape[0],w.T,axis=0)
            cMatrix=np.insert(cMatrix,cMatrix.shape[0],c.T,axis=1)
    res.t=tMatrix
    res.p=pMatrix
    res.u=uMatrix
    res.w=wMatrix
    res.c=cMatrix
    res.r=wMatrix.T.dot(np.linalg.inv(pMatrix.dot(wMatrix.T))).T
    res.LVN=LVN
    
    return res

    #Below is the definition of what a PLSModelResult class is.
class PLSModelResults:
    def __init__(self):
        self.LVN = int
        self.t = np.matrix
        self.w = np.matrix
        self.r = np.matrix
        self.p = np.matrix
        self.u = np.matrix
        self.c = np.matrix
        self.predTrain=np.matrix
        self.xT2Matrix=np.matrix
    
    def Predict(self, Xmatrix):
        
        return Xmatrix.dot(self.r.T).dot(self.c.T)
    def T2x(self,tMatrix="auto"):
        if tMatrix=="auto":
            T=self.t
            
        tVar=np.std(T,axis=0)
        tOverVarianceSquared=np.power(np.divide(T,tVar),2)
        result=np.matrix
        for i in range(1,T.shape[1]+1):                
            if i == 1 : result=np.matrix(np.sum(tOverVarianceSquared[:,0:i],
                                                axis=1))
            else: 
                result=np.insert(result,result.shape[1],
                                 np.matrix(np.sum(tOverVarianceSquared[:,0:i],axis=1)).T,
                                 axis=1)       
        return result

    def Value_over_threshold_T2_y(self,threshold=0.95,numberOfComponents="auto"): 
        if numberOfComponents=="auto" : numberOfComponents=self.LVN
        return 0
    def Value_over_threshold_SPE_y(self,threshold=0.95,numberOfComponents="auto"):
        if numberOfComponents=="auto" : numberOfComponents=self.LVN  
        return 0
    def Value_over_threshold_T2_x(self,threshold=0.95,numberOfComponents="auto"): 
        if numberOfComponents=="auto" : numberOfComponents=self.LVN
        return 0
    def Value_over_threshold_SPE_x(self,threshold=0.95,numberOfComponents="auto"):
        if numberOfComponents=="auto" : numberOfComponents=self.LVN  
        return 0
