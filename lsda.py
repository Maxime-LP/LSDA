import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import kneighbors_graph
from scipy.linalg import eig, norm


def get_W(X, y, n_neighbors):
        A = kneighbors_graph(X, n_neighbors).toarray()
        tmp = A + A.T
        tmp = np.array(tmp)
        W = np.ceil((A + A.T) / 2)

        n = np.shape(W)[0]
        Ww = np.zeros((n, n))
        Wb = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if W[i, j] == 1:
                    if y[i] == y[j]:
                        Ww[i, j] = 1
                    else:
                        Wb[i, j] = 1
        return W, Ww, Wb

class LSDA(BaseEstimator, ClassifierMixin):

    def __init__(self, n_neighbors, n_components=2, alpha=0.5):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.alpha = alpha

    def fit(self, X, y, scale=False):
        
        # Checks that X and Y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        # Computes W, Ww, Wb, Db and Lb
        self.W, self.Ww, self.Wb = get_W(self.X_,self.y_,self.n_neighbors)
        self.Dw = np.diag(np.sum(self.Ww, axis=1))
        self.Db = np.diag(np.sum(self.Wb, axis=1))
        self.Lb = self.Db - self.Wb

        # Solves general eigenvalues/eigenvectors problem
        left_term = X.T.dot(self.alpha * self.Lb + (1 - self.alpha) * self.Ww).dot(X)  # (n,n) ndarray
        right_term = X.T.dot(self.Dw).dot(X) # (n,n) ndarray
        self.eigen_values, self.eigen_vectors = eig(left_term, right_term)

        # Sorts the eigenvectors according to their eigenvalues
        idx = self.eigen_values.argsort()[::-1]
        self.eigen_values = self.eigen_values[idx]
        self.eigen_vectors = self.eigen_vectors[:, idx]

        return self

    def transform(self, X):
        
        # Checks if fit has been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Projects X on a subspace
        projected_X = X.dot(self.eigen_vectors)[:, :self.n_components]

        return projected_X
    
    def fit_transform(self,X,y):
        self.fit(X,y)
        projected_X = self.transform(X)
        return projected_X
    
   


    
