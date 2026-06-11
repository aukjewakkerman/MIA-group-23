import numpy as np


def mypca(X):
    # Rotates the data X such that the dimensions of rotated data Xpca
    # are uncorrelated and sorted by variance.
    # Input:
    # X - Nxk feature matrix
    # Output:
    # X_pca - Nxk rotated feature matrix
    # U - kxk matrix of eigenvectors
    # Lambda - kx1 vector of eigenvalues
    # fraction_variance - kx1 vector which stores how much variance
    #                     is retained in the k components

    X = X - np.mean(X, axis=0)

    # ------------------------------------------------------------------#
    # TODO: Calculate covariance matrix of X, find eigenvalues and eigenvectors,
    # sort them, and rotate X using the eigenvectors
    # ------------------------------------------------------------------#
    #!studentstart
    # Calculates covariance matrix of X, find eigenvalues and eigenvectors,
    # sort them, and rotate X using the eigenvectors
    # Calculate covariance matrix of X
    sigma = np.cov(X, rowvar=False)
    # Find eigenvalues and eigenvectors of covariance matrix
    # - the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i]
    w, v = np.linalg.eig(sigma)
    # Sort eigenvalues and eigenvectors
    # Find ordering of eigenvalues
    ix = np.argsort(w)[::-1]
    # Reorder eigenvalues
    w = w[ix]
    # Reorder eigenvectors
    v = v[:, ix]
    # Rotate X using the eigenvectors
    X_pca = v.T.dot(X.T)
    X_pca = X_pca.T
    #!studentend

    # Return fraction of variance
    fraction_variance = np.zeros((X_pca.shape[1], 1))
    for i in np.arange(X_pca.shape[1]):
        fraction_variance[i] = np.sum(w[: i + 1]) / np.sum(w)

    return X_pca, v, w, fraction_variance
