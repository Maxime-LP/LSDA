#X = data = 
from matplotlib.colors import Colormap
import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.linalg import eig
import matplotlib.pyplot as plt

X = np.array([
    [1,2,3],
    [1,4,1],
    [2,4,7],
    [2,5,1],
    [0,4,6],
    [-1,50,12],
    [20,30,15]
])
Y = [1,1,0,1,0,0,1]

def get_W(X,K):
    A = kneighbors_graph(X,K).toarray()
    tmp = A+A.T
    tmp = np.array(tmp)
    return np.ceil((A+A.T)/2)

def Wwb(X,Y,K):
    W = get_W(X,K)
    n = np.shape(W)[0]
    Ww = np.zeros((n,n))
    Wb = np.zeros((n,n))

    for i in range(0,n):
        for j in range(0,n):
            if W[i,j]==1:
                if Y[i]==Y[j]:
                    Ww[i,j]=1
                else:
                    Wb[i,j]=1
    return Ww,Wb 

def fit(X,Y,K,alpha,dim):
    Ww,Wb = Wwb(X,Y,K)
    Dw = np.diag(np.sum(Ww,axis=1))
    Db = np.diag(np.sum(Wb,axis=1))
    Lb = Db - Wb

    #print(X,'\n',Lb,'\n',Ww,'\n')

    M1 = X.T.dot(alpha*Lb + (1-alpha)*Ww).dot(X) #nxn
    M2 = X.T.dot(Dw).dot(X) #nxn
    eigenValues,eigenVectors = eig(M1,M2)
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    #print(eigenValues,'\n',eigenVectors)
    return X.dot(eigenVectors)[:,:dim]




Z = fit(X,Y,2,0.5,2)
cdict = {0: 'red', 1: 'blue'}

ax1 = plt.subplot(1,2,2,title = 'Après transformation Z = XA')
ax2 = plt.subplot(1,2,1, title = 'Avant transformation')

for g in np.unique(Y):
    ix = np.where(Y == g)
    ax1.scatter(Z[ix,0], Z[ix,1], c = cdict[g], label = g)
    ax2.scatter(X[ix,0], X[ix,1], c = cdict[g], alpha=0.6, label = g)
ax2.legend()
plt.show()
