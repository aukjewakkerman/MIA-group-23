"""
Segmentation module main code.
"""

import numpy as np
import scipy 
from sklearn.neighbors import KNeighborsClassifier
import registration as reg
import segmentation_util as util


# SECTION 1. Segmentation in feature space


def generate_gaussian_data(N=100, mu1=[0,0], mu2=[2,0], sigma1=[[1,0],[0,1]], sigma2=[[1,0],[0,1]]):
    # Generates a 2D toy dataset with 2 classes, N samples per class. 
    # Class 1 is Gaussian distributed with mu1 and sigma2
    # Class 2 is Gaussian distributed with mu2 and sigma2.
    # Input:
    # N - Number of samples per class (2N in total)
    # mu1 - 1x2 vector, mean of class 1
    # mu2 - 1x2 vector, mean of class 2
    # sigma1 - 2x2 matrix, covariance of class 1
    # sigma2 - 2x2 matrix, covariance of class 2
    
    # Generate class 1
    # Rotate data according to covariance matrix (must be positive
    # definite), and add the mean
    A = np.linalg.cholesky(sigma1)
    data1 = np.random.randn(N,2).dot(A) + mu1

    # Generate class 2
    B = np.linalg.cholesky(sigma2)
    data2 = np.random.randn(N,2).dot(B) + mu2
    
    # Put the data together
    X = np.concatenate((data1, data2), axis=0)

    # Create labels
    Y = np.concatenate((np.zeros((N,1)), np.ones((N,1))), axis=0)

    return X, Y


def extract_coordinate_feature(im):
    # Creates a coordinate feature, which encodes how far a pixel is
    # from the center of the image.
    # Input:
    # im - An NxM image
    # Output:
    # c - A (N*M)x1 vector which encodes how far each pixel is from
    # the center of the image

    # Get the image size
    n_rows, n_cols = im.shape   
    
    # Find image center
    x_center = np.floor(n_rows/2)
    y_center = np.floor(n_cols/2)
    
    # Generate coordinate images
    ar = np.arange(n_cols).reshape(1,-1)
    x_coord = np.tile(ar, (n_rows, 1))      # rows --> y
    ar = ar.T
    y_coord = np.tile(ar, (1, n_cols))      # columns --> x
    
    #------------------------------------------------------------------#
    # TODO: Use the above variables to create an image coord_im
    # that combines the information from x_coord and y_coord 
    # afstand tot centrum
    coord_im = np.sqrt((x_coord - y_center)**2 + (y_coord - x_center)**2)

    #------------------------------------------------------------------#
    
    # Create a feature from the coordinate image
    c = coord_im.flatten().T
    c = c.reshape(-1, 1)

    return c, coord_im


def normalize_data(train_data, test_data=None):
    # Normalizes data train_data (and optionally, test_data), by
    # subtracting the mean of train_data, and dividing by the standard
    # deviation of train_data.
    # Input:
    # train_data - num_train x k dataset with Ntrain samples and k features
    # test_data - (Optional input) num_test x k dataset with Ntest samples
    #             and k features
    # Output:
    # train_data - num_train x k dataset with Ntrain samples and k features,
    #              that has been normalized by trainX
    # test_data - (Optional output) num_test x k dataset with Ntest samples
    #             and k features, that has been normalized by trainX
    
    #Find mean and standard deviation of trainX
    mean_train = np.mean(train_data,axis=0)
    std_train = np.std(train_data,axis=0)

    # Subtract mean
    train_data = train_data - mean_train

    # Divide by standard deviation
    train_data = train_data / std_train

    # (Optional) If testX needs to be normalized also - note it is normalized
    # by the mean and variance of trainX, not testX!
    if test_data is not None:
        test_data = test_data - mean_train
        test_data = test_data / std_train

    return train_data, test_data


def cost_kmeans(X, w_vector):
    # Computes the cost of assigning data in X to clusters in w_vector 
    
    # Get the data dimensions
    n, m = X.shape

    # Number of clusters
    K = int(len(w_vector)/m)

    # Reshape cluster centers into dataset format
    W = w_vector.reshape(K, m)

    #------------------------------------------------------------------#
    # TODO: Find distance of each point to each cluster center
    # Then find the minimum distances min_dist and indices min_index
    # Then calculate the cost

    # Compute distances and find closest center
    D = scipy.spatial.distance.cdist(X, W, metric='euclidean')
    min_dist = np.min(D, axis=1)

    # Sum of squared distances (cost)
    J = np.sum(min_dist ** 2)

    #------------------------------------------------------------------#

    return J

def ngradient(fun, x, h=1e-3):
    # Computes the derivative of a function with numerical differentiation.
    # Input:
    # fun - function for which the gradient is computed
    # x - vector of parameter values at which to compute the gradient
    # h - a small positive number used in the finite difference formula
    # Output:
    # g - vector of partial derivatives (gradient) of fun

    #------------------------------------------------------------------#
    # TODO: Implement the  computation of the partial derivatives of
    # the function at x with numerical differentiation.
    # g[k] should store the partial derivative w.r.t. the k-th parameter
    #!studentstart
    g = np.zeros_like(x)
    for k in range(x.size):
        xh1 = x.copy()
        xh2 = x.copy()
        xh1[k] = xh1[k] + h/2
        xh2[k] = xh2[k] - h/2
        a = fun(xh1)
        b = fun(xh2)
        if isinstance(a, tuple):
            g[k] = (a[0] - b[0])/h
        else:
            g[k] = (a - b)/h
    #!studentend
    #------------------------------------------------------------------#

    return g

def kmeans_clustering(test_data, K=2):
    # Returns the labels for test_data, predicted by the kMeans
    # classifier which assumes that clusters are ordered by intensity
    # Input:
    # test_data - num_test x p matrix with features for the test data
    # k - Number of clusters to take into account (2 by default)
    # Output:
    # predicted_labels - num_test x 1 predicted vector with labels for
    #                    the test data

    # Link to the cost function of kMeans
    fun = lambda w: cost_kmeans(test_data, w)

    # the learning rate
    mu = 0.01

    # iterations
    num_iter = 100

    #------------------------------------------------------------------#
    # TODO: Initialize cluster centers and store them in w_initial
    #!studentstart
    N, M = test_data.shape
    idx = np.random.randint(N, size=K)
    w_initial = test_data[idx,:]
    #!studentend
    #------------------------------------------------------------------#

    #Reshape centers to a vector (needed by ngradient)
    w_vector = w_initial.reshape(K*M, 1)

    for i in np.arange(num_iter):
        # gradient ascent
        w_vector = w_vector - mu*reg.ngradient(fun,w_vector)

    #Reshape back to dataset
    w_final = w_vector.reshape(K, M)

    #------------------------------------------------------------------#
    # TODO: Find distance of each point to each cluster center
    # Then find the minimum distances min_dist and indices min_index
    
    d = scipy.spatial.distance.cdist(test_data, w_final, metric='euclidean')
    min_index = np.argmin(d, axis=1)
    min_dist = np.empty(*min_index.shape)
    for i in np.arange(len(d)):
        min_dist[i] = d[i, min_index[i]]
    
    #------------------------------------------------------------------#

    # Sort by intensity of cluster center
    sorted_order = np.argsort(w_final[:,0], axis=0)

    # Update the cluster indices based on the sorted order and return
    # results in predicted_labels
    predicted_labels = np.empty(*min_index.shape)
    predicted_labels[:] = np.nan

    for i in np.arange(len(sorted_order)):
        predicted_labels[min_index==sorted_order[i]] = i

    return predicted_labels



def nn_classifier(train_data, train_labels, test_data):
    # Returns the labels for test_data, predicted by the 1-NN
    # classifier trained on train_data and train_labels
    # Input:
    # train_data - num_train x p matrix with features for the training data
    # train_labels - num_train x 1 vector with labels for the training data
    # test_data - num_test x p matrix with features for the test data
    # Output:
    # predicted_labels - num_test x 1 predicted vector with labels for
    #                    the test data

    #------------------------------------------------------------------#
    # TODO: Implement missing functionality

    num_test = test_data.shape[0]
    predicted_labels = np.empty((num_test, 1))
    D = scipy.spatial.distance.cdist(test_data, train_data, metric='euclidean')
    min_index = np.argmin(D, axis=1)
    predicted_labels = train_labels[min_index]

    #------------------------------------------------------------------#
    return predicted_labels


def knn_classifier(train_data, train_labels, test_data, k):
    # Returns the labels for test_data, predicted by the k-NN
    # clasifier trained on train_data and train_labels
    # Input:
    # train_data - num_train x p matrix with features for the training data
    # train_labels - num_train x 1 vector with labels for the training data
    # test_data - num_test x p matrix with features for the test data
    # k - Number of neighbors to take into account (1 by default)
    # Output:
    # predicted_labels - num_test x 1 predicted vector with labels for the test data
    
    D = scipy.spatial.distance.cdist(test_data, train_data, metric='euclidean')
    sort_ix = np.argsort(D, axis=1)
    sort_ix_k = sort_ix[:,:k] # Get the k smallest distances
    train_labels = np.array(train_labels).astype(int)
    predicted_labels = train_labels[sort_ix_k]
    predicted_labels = scipy.stats.mode(predicted_labels, axis=1)[0]

    return predicted_labels


# SECTION 2. Generalization and overfitting


def mypca(X, feature_labels=None, threshold=0.3, visualize=True):
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


    # Return fraction of variance
    fraction_variance = np.cumsum(w) / np.sum(w)
    fraction_variance = fraction_variance.reshape(-1, 1)

    # Step 3: Feature selection based on loadings
    num_components = np.argmax(fraction_variance >= 0.95)

    if visualize == True:
        for i in range(num_components):
            print(f"Component {i+1} explains {fraction_variance[i,0]:.2%} of variance")
            loadings = v[:, i]
            for feature_index, loading in enumerate(loadings):
                print(f"  Feature {feature_labels[feature_index]} loading: {loading:.3f}")
            print()
    
    important_features = None
    if feature_labels is not None:
        important_features = set()
        for i in range(num_components):
            loadings = v[:, i]
            for idx, loading in enumerate(loadings):
                if abs(loading) >= threshold:
                    important_features.add(feature_labels[idx])
        important_features = list(important_features)

    return X_pca, v, w, fraction_variance, important_features
    
# SECTION 3. Atlases and active shapes


def segmentation_combined_atlas(train_labels_matrix, combining='mode'):
    # Segments the image defined based only on the labels/atlases of
    # the other subjects
    # Input:
    # train_labels - num_train x num_atlases training labels vector
    # combining - String corresponding to combining type: 'mode', 'min'
    #             (only binary labels), 'max' (only binary labels)
    # Output:
    # predicted_labels - Predicted labels for the test slice

    r, c = train_labels_matrix.shape

    # Segment the test subject by each individual atlas
    predicted_labels = np.empty([r,c])
    predicted_labels[:] = np.nan

    for i in np.arange(c):
        predicted_labels[:,i] = segmentation_atlas(None, train_labels_matrix[:,i], None)

    # Combine labels
    # Option 1: Most frequent label
    if combining == 'mode':
        predicted_labels = scipy.stats.mode(predicted_labels, axis=1)[0]
    
    #------------------------------------------------------------------#
    # TODO: Add options for combining with min and max
    if combining == 'min':
        predicted_labels = scipy.stats.min(predicted_labels, axis=1)[0]
    if combining == 'max':
        predicted_labels = scipy.stats.max(predicted_labels, axis=1)[0]
    #------------------------------------------------------------------#
    else:
        raise ValueError("No such combining type exists")

    return predicted_labels.astype(bool)


def segmentation_atlas(train_data, train_labels, test_data):
    # Segments the image defined by test_subject and test_slice,
    # based only on the labels/atlases of the other subjects
    # Input:
    # train_labels - num_train x 1 training labels vector
    # Output:
    # predicted_labels - Predicted labels for the test slice
    # Note that train_data and test_data are not used here because
    # we assume the images are registered. But in practice, we would
    # want to first do registration on the image intensity

    #Assume predicted labels are the atlas labels
    predicted_labels = train_labels

    return predicted_labels


def segmentation_combined_knn(train_data_matrix, train_labels_matrix, test_data, k=1):
    # Segments the image defined by test_data based on
    # kNN classifiers trained on data in train_data_matrix and
    # train_labels_matrix
    # Input:
    # train_data_matrix - num_pixels x num_features x num_subjects matrix of
    # features
    # train_labels_matrix - num_pixels x num_subjects matrix of labels
    # test_data - num_pixels x num_features test data
    # k - Number of neighbors
    # Output:
    # predicted_labels - Predicted labels for the test slice

    r, c = train_labels_matrix.shape

    predicted_labels = np.empty([r,c])
    predicted_labels[:] = np.nan

    for i in np.arange(c):
        predicted_labels[:,i] = segmentation_knn(train_data_matrix[:,:,i], train_labels_matrix[:,i], test_data, k)

    #Combine labels, majority voting (take the most common label per pixel)
    predicted_labels = scipy.stats.mode(predicted_labels, axis=1)[0]

    return predicted_labels.astype(bool)


def segmentation_knn(train_data, train_labels, test_data, k=1):
    # Segments the image using a knn classsifier trained on
    # train_data and train_labels
    # Input:
    # train_data - num_train x num_features training data matrix
    # train_labels - num_train x 1 training labels vector
    # test_data - num_test x num_features test data matrix
    # k - Number of neighbors
    # Output:
    # predicted_labels - Predicted labels for the test slice
    
    # Subsample training data for efficiency
    num_samples=3000
    ix = np.random.randint(train_data.shape[0], size=num_samples)

    subset_train_data = train_data[ix,:]
    subset_train_labels = train_labels[ix]

    # Normalize
    [train_data_norm, test_data_norm] = normalize_data(subset_train_data, test_data);

    # Train and apply kNN classifier

    # Option 1: The implementation we made in this course (slower)
    # predicted_labels = knn_classifier(train_data_norm, subset_train_labels, test_data_norm, k)

    # Option 2: The implementation of sklearn (faster)
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(train_data_norm, subset_train_labels)
    predicted_labels = neigh.predict(test_data_norm)

    return predicted_labels

