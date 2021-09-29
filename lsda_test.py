import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.linalg import eig
import matplotlib.pyplot as plt


def get_W(X, Y, K):
    A = kneighbors_graph(X, K).toarray()
    tmp = A + A.T
    tmp = np.array(tmp)
    W = np.ceil((A + A.T) / 2)

    n = np.shape(W)[0]
    Ww = np.zeros((n, n))
    Wb = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            if W[i, j] == 1:
                if Y[i] == Y[j]:
                    Ww[i, j] = 1
                else:
                    Wb[i, j] = 1
    return W, Ww, Wb


def fit(X, Y, K, alpha, d):
    W, Ww, Wb = get_W(X, Y, K)
    Dw = np.diag(np.sum(Ww, axis=1))
    Db = np.diag(np.sum(Wb, axis=1))
    Lb = Db - Wb

    # Solves general eigenvalues/eigenvectors problem
    left_mat = X.T.dot(alpha * Lb + (1 - alpha) * Ww).dot(X)  # nxn
    right_mat = X.T.dot(Dw).dot(X)  # nxn
    eigen_values, eigen_vectors = eig(left_mat, right_mat)

    # Sorts the eigenvectors according to their eigenvalues
    idx = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]

    # Projects X on a subspace
    projected_X = X.dot(eigen_vectors)[:, :d]

    return projected_X


X = np.array([
    [1, 2, 3],
    [2, 4, 7],
    [1, 4, 1],
    [2, 5, 1],
    [0, 4, 6],
    [-1, 50, 12],
    [20, 30, 15]
])

Y = [1, 1, 0, 1, 0, 0, 1]

Z = fit(X, Y, 2, 0.5, 2)
cdict = {0: 'red', 1: 'blue'}

ax1 = plt.subplot(1, 2, 2, title='Après transformation Z = XA')
ax2 = plt.subplot(1, 2, 1, title='Avant transformation')

for g in np.unique(Y):
    ix = np.where(Y == g)
    ax1.scatter(Z[ix, 0], Z[ix, 1], c=cdict[g], label=g)
    ax2.scatter(X[ix, 0], X[ix, 1], c=cdict[g], alpha=0.6, label=g)
ax2.legend()
plt.show()
