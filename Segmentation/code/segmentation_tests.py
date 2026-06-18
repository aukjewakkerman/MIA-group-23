"""
Test code for segmentation.
"""

import numpy as np
import segmentation_util as util
import matplotlib.pyplot as plt
import segmentation as seg
from scipy import ndimage, stats
import scipy
import scipy.io
from sklearn.neighbors import KNeighborsClassifier
import timeit
from IPython.display import display, clear_output
plt.rcParams['image.cmap'] = 'gray'


# Helper: classification error (not in segmentation_util)
def classification_error(true_labels, predicted_labels):
    true_labels = np.array(true_labels).flatten()
    predicted_labels = np.array(predicted_labels).flatten()
    return np.mean(true_labels != predicted_labels)


# SECTION 1. Segmentation in feature space


def scatter_data_test(showFigs=True):
    I = plt.imread('../data/dataset_brains/1_1_t1.tif')
    X1 = I.flatten().T
    X1 = X1.reshape(-1, 1)
    GT = plt.imread('../data/dataset_brains/1_1_gt.tif')
    gt_mask = GT > 0
    Y = gt_mask.flatten()  # labels

    I_blurred = ndimage.gaussian_filter(I, sigma=2)
    X2 = I_blurred.flatten().T
    X2 = X2.reshape(-1, 1)

    # Additional features: gradient magnitude and LoG
    grad = ndimage.gaussian_gradient_magnitude(I, sigma=1)
    X3 = grad.flatten().reshape(-1, 1)

    log = ndimage.gaussian_laplace(I, sigma=2)
    X4 = log.flatten().reshape(-1, 1)

    X_data = np.concatenate((X1, X2, X3, X4), axis=1)

    # Keep track of features you added
    features = ('T1 intensity', 'T1 gauss sigma=2', 'T1 gradient mag', 'T1 LoG sigma=2')

    if showFigs:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Feature scatter plots (subject 1, slice 1, brain task)')
        util.scatter_data(X_data, Y, 0, 1, ax=axes[0])
        axes[0].set_title(f'{features[0]} vs {features[1]}')
        util.scatter_data(X_data, Y, 0, 2, ax=axes[1])
        axes[1].set_title(f'{features[0]} vs {features[2]}')
        util.scatter_data(X_data, Y, 0, 3, ax=axes[2])
        axes[2].set_title(f'{features[0]} vs {features[3]}')
        plt.tight_layout()
        plt.show()

    return X_data, Y


def scatter_t2_test(showFigs=True):
    I1 = plt.imread('../data/dataset_brains/1_1_t1.tif')
    X1 = I1.flatten().T.reshape(-1, 1)
    I2 = plt.imread('../data/dataset_brains/1_1_t2.tif')
    X2 = I2.flatten().T.reshape(-1, 1)

    GT = plt.imread('../data/dataset_brains/1_1_gt.tif')
    gt_mask = GT > 0
    Y = gt_mask.flatten()

    # T1 features
    I1_blurred = ndimage.gaussian_filter(I1, sigma=4)
    X1_blur = I1_blurred.flatten().reshape(-1, 1)

    # T2 features: raw, blurred, gradient magnitude
    I2_blurred = ndimage.gaussian_filter(I2, sigma=4)
    X2_blur = I2_blurred.flatten().reshape(-1, 1)
    grad_t2 = ndimage.gaussian_gradient_magnitude(I2, sigma=1)
    X2_grad = grad_t2.flatten().reshape(-1, 1)

    X_data = np.concatenate((X1, X1_blur, X2, X2_blur, X2_grad), axis=1)
    features = ('T1 raw', 'T1 gauss sigma=4', 'T2 raw', 'T2 gauss sigma=4', 'T2 grad mag')

    if showFigs:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('T1 vs T2 feature scatter plots (subject 1, slice 1)')
        # T1 raw vs T2 raw
        util.scatter_data(X_data, Y, 0, 2, ax=axes[0])
        axes[0].set_title(f'{features[0]} vs {features[2]}')
        # T1 raw vs T2 blurred
        util.scatter_data(X_data, Y, 0, 3, ax=axes[1])
        axes[1].set_title(f'{features[0]} vs {features[3]}')
        # T1 blurred vs T2 gradient
        util.scatter_data(X_data, Y, 1, 4, ax=axes[2])
        axes[2].set_title(f'{features[1]} vs {features[4]}')
        plt.tight_layout()
        plt.show()

    return X_data, Y


def extract_coordinate_feature_test():
    I = plt.imread('../data/dataset_brains/1_1_t1.tif')
    c, coord_im = seg.extract_coordinate_feature(I)
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(121)
    ax1.imshow(I)
    ax1.set_title('T1 image')
    ax2 = fig.add_subplot(122)
    ax2.imshow(coord_im)
    ax2.set_title('Coordinate feature (distance from center)')
    plt.show()


def feature_stats_test():
    X, Y = scatter_data_test(showFigs=False)
    I = plt.imread('../data/dataset_brains/1_1_t1.tif')
    c, coord_im = seg.extract_coordinate_feature(I)
    X_data = np.concatenate((X, c), axis=1)

    # Examine mean and standard deviation per feature
    feature_names = ['T1 raw', 'T1 gauss sigma=2', 'T1 grad mag', 'T1 LoG sigma=2', 'coord dist']
    print(f"{'Feature':<22} {'Mean':>10} {'Std':>10}")
    print("-" * 44)
    for i, name in enumerate(feature_names):
        print(f"{name:<22} {np.mean(X_data[:, i]):>10.4f} {np.std(X_data[:, i]):>10.4f}")

    print("\nObservation: features have very different scales — normalisation is needed before classification.")


def normalized_stats_test():
    X, Y = scatter_data_test(showFigs=False)
    I = plt.imread('../data/dataset_brains/1_1_t1.tif')
    c, coord_im = seg.extract_coordinate_feature(I)
    X_data = np.concatenate((X, c), axis=1)

    # Normalise using training-set statistics (z-score)
    X_norm, _ = seg.normalize_data(X_data)

    feature_names = ['T1 raw', 'T1 gauss sigma=2', 'T1 grad mag', 'T1 LoG sigma=2', 'coord dist']
    print("After normalisation:")
    print(f"{'Feature':<22} {'Mean':>10} {'Std':>10}")
    print("-" * 44)
    for i, name in enumerate(feature_names):
        print(f"{name:<22} {np.mean(X_norm[:, i]):>10.4f} {np.std(X_norm[:, i]):>10.4f}")

    print("\nAll features now have mean ≈ 0 and std ≈ 1, so no single feature dominates the distance metric.")


def distance_test():
    # Generate a Gaussian dataset with 100 samples per class
    X, Y = seg.generate_gaussian_data(N=100)

    # Compute pairwise Euclidean distance matrix
    D = scipy.spatial.distance.cdist(X, X, metric='euclidean')

    # Visualise as image
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(X[:100, 0], X[:100, 1], label='Class 0', alpha=0.5)
    axes[0].scatter(X[100:, 0], X[100:, 1], label='Class 1', alpha=0.5)
    axes[0].set_title('Gaussian dataset (200 samples)')
    axes[0].legend()
    axes[0].grid(True)

    im = axes[1].imshow(D, cmap='viridis')
    axes[1].set_title('Pairwise distance matrix')
    axes[1].set_xlabel('Sample index')
    axes[1].set_ylabel('Sample index')
    plt.colorbar(im, ax=axes[1])
    plt.tight_layout()
    plt.show()

    print(f"Distance matrix shape: {D.shape}")
    print(f"Min distance (non-zero): {D[D > 0].min():.4f}")
    print(f"Max distance: {D.max():.4f}")
    print(f"Mean distance: {D.mean():.4f}")


def small_samples_distance_test():
    # Generate a small Gaussian dataset X (10 samples per class)
    X, Y = seg.generate_gaussian_data(N=10)

    # Create dataset C: 5 random points spread over the data range
    np.random.seed(42)
    C = np.random.uniform(X.min(), X.max(), size=(5, 2))

    # Calculate distances between X and C
    D = scipy.spatial.distance.cdist(X, C, metric='euclidean')

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(X[:10, 0], X[:10, 1], c='blue', label='X class 0')
    axes[0].scatter(X[10:, 0], X[10:, 1], c='orange', label='X class 1')
    axes[0].scatter(C[:, 0], C[:, 1], c='red', marker='*', s=200, label='C (query points)')
    axes[0].set_title('Datasets X and C')
    axes[0].legend()
    axes[0].grid(True)

    im = axes[1].imshow(D, cmap='viridis', aspect='auto')
    axes[1].set_title('Distance matrix (X vs C)')
    axes[1].set_xlabel('C index')
    axes[1].set_ylabel('X index')
    plt.colorbar(im, ax=axes[1])
    plt.tight_layout()
    plt.show()

    print(f"D shape: {D.shape}  (X has {len(X)} samples, C has {len(C)} points)")


def minimum_distance_test(X, Y, C, D):
    # Plot datasets on top of each other
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(X[:, 0], X[:, 1], c='blue', alpha=0.5, label='X')
    axes[0].scatter(C[:, 0], C[:, 1], c='red', marker='*', s=200, label='C')
    axes[0].set_title('X and C datasets')
    axes[0].legend()
    axes[0].grid(True)

    # Order distances min to max per row (each X point to all C points)
    sorted_idx = np.argsort(D, axis=1)
    sorted_D = np.sort(D, axis=1)

    # Count how many X samples are closest to each C point
    nearest_C = sorted_idx[:, 0]  # index of nearest C for each X
    counts = np.bincount(nearest_C, minlength=len(C))
    print("Number of X samples closest to each C point:")
    for i, count in enumerate(counts):
        print(f"  C[{i}]: {count} samples")

    axes[1].bar(range(len(C)), counts)
    axes[1].set_xlabel('C point index')
    axes[1].set_ylabel('Number of nearest X samples')
    axes[1].set_title('X samples closest to each C point')
    axes[1].grid(True, axis='y')
    plt.tight_layout()
    plt.show()


def distance_classification_test():
    # Generate training and test data
    train_data, train_labels = seg.generate_gaussian_data(N=100)
    test_data, test_labels = seg.generate_gaussian_data(N=100)

    # Normalise
    train_data_n, test_data_n = seg.normalize_data(train_data, test_data)

    # Classify test points based on nearest training point (1-NN)
    D = scipy.spatial.distance.cdist(test_data_n, train_data_n, metric='euclidean')
    nearest_idx = np.argmin(D, axis=1)
    predicted_labels = train_labels[nearest_idx].flatten()

    err = classification_error(test_labels.flatten(), predicted_labels)
    print(f"1-NN distance classification error: {err:.4f}")

    # Visualise
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(test_data[test_labels.flatten() == 0, 0],
                    test_data[test_labels.flatten() == 0, 1], alpha=0.5, label='True class 0')
    axes[0].scatter(test_data[test_labels.flatten() == 1, 0],
                    test_data[test_labels.flatten() == 1, 1], alpha=0.5, label='True class 1')
    axes[0].set_title('True labels')
    axes[0].legend(); axes[0].grid(True)

    axes[1].scatter(test_data[predicted_labels == 0, 0],
                    test_data[predicted_labels == 0, 1], alpha=0.5, label='Predicted class 0')
    axes[1].scatter(test_data[predicted_labels == 1, 0],
                    test_data[predicted_labels == 1, 1], alpha=0.5, label='Predicted class 1')
    axes[1].set_title(f'Predicted labels (error={err:.3f})')
    axes[1].legend(); axes[1].grid(True)
    plt.tight_layout()
    plt.show()


def funX(X):
    return lambda w: seg.cost_kmeans(X, w)


def kmeans_demo():
    ## Define some data and parameters
    n = 100
    X1 = np.random.randn(n, 2)
    X2 = np.random.randn(n, 2) + 5
    X = np.concatenate((X1, X2), axis=0)
    Y = np.concatenate((np.zeros((n, 1)), np.ones((n, 1))), axis=0)
    N, M = X.shape

    clusters = 2
    mu = 1
    num_iter = 100
    fun = funX(X)

    idx = np.random.randint(N, size=clusters)
    initial_w = X[idx, :]
    w_draw = initial_w
    print(w_draw)

    w_vector = initial_w.reshape(clusters * M, 1)

    xx = np.linspace(1, num_iter, num_iter)
    kmeans_cost = np.empty(*xx.shape)
    kmeans_cost[:] = np.nan

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121)
    ax1.scatter(X[:n, 0], X[:n, 1], label='X-class0')
    ax1.scatter(X[n:, 0], X[n:, 1], label='X-class1')
    line1, = ax1.plot(w_draw[:, 0], w_draw[:, 1], "or", markersize=5, label='W-vector')
    ax1.grid()
    ax2 = fig.add_subplot(122, xlim=(0, num_iter), ylim=(0, 10))
    text_str = 'k={}, g={:.2f}\ncost={:.2f}'.format(0, 0, 0)
    txt2 = ax2.text(0.3, 0.95, text_str, bbox={'facecolor': 'green', 'alpha': 0.4, 'pad': 10},
                    transform=ax2.transAxes)
    line2, = ax2.plot(xx, kmeans_cost, lw=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Cost')
    ax2.grid()

    for k in np.arange(num_iter):
        g = util.ngradient(fun, w_vector)
        w_vector = w_vector - mu * g
        kmeans_cost[k] = fun(w_vector)
        text_str = 'k={}, cost={:.2f}'.format(k, kmeans_cost[k])
        txt2.set_text(text_str)
        line2.set_ydata(kmeans_cost)
        w_draw_new = w_vector.reshape(clusters, M)
        line1.set_data(w_draw_new[:, 0], w_draw_new[:, 1])
        display(fig)
        clear_output(wait=True)
        plt.pause(.005)

    return kmeans_cost


def kmeans_clustering_test():
    # Generate a 2-class Gaussian dataset
    X, Y = seg.generate_gaussian_data(N=200)
    Y = Y.flatten()

    # Run k-means clustering
    predicted_labels = seg.kmeans_clustering(X, K=2)
    predicted_labels = predicted_labels.flatten()

    # Compute error (accounting for possible label swap)
    err1 = classification_error(Y, predicted_labels)
    err2 = classification_error(Y, 1 - predicted_labels)
    err = min(err1, err2)
    print(f"k-means clustering error (best label assignment): {err:.4f}")

    # Visualise
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(X[Y == 0, 0], X[Y == 0, 1], alpha=0.5, label='True class 0')
    axes[0].scatter(X[Y == 1, 0], X[Y == 1, 1], alpha=0.5, label='True class 1')
    axes[0].set_title('Ground truth'); axes[0].legend(); axes[0].grid(True)

    axes[1].scatter(X[predicted_labels == 0, 0], X[predicted_labels == 0, 1], alpha=0.5, label='Cluster 0')
    axes[1].scatter(X[predicted_labels == 1, 0], X[predicted_labels == 1, 1], alpha=0.5, label='Cluster 1')
    axes[1].set_title(f'k-means clusters (error={err:.3f})'); axes[1].legend(); axes[1].grid(True)
    plt.tight_layout()
    plt.show()


def nn_classifier_test_samples():
    train_data, train_labels = seg.generate_gaussian_data(2)
    test_data, test_labels = seg.generate_gaussian_data(1)
    predicted_labels = seg.nn_classifier(train_data, train_labels, test_data)

    err = classification_error(test_labels, predicted_labels)
    print('True labels:\n{}'.format(test_labels))
    print('Predicted labels:\n{}'.format(predicted_labels))
    print('Error:\n{}'.format(err))


def generate_train_test(N, task):
    if task == 'easy':
        # Well-separated classes: means far apart, low variance
        mu1 = [0, 0]
        mu2 = [6, 6]
        sigma1 = [[1, 0], [0, 1]]
        sigma2 = [[1, 0], [0, 1]]

    if task == 'hard':
        # Heavily overlapping classes: means close together, high variance
        mu1 = [0, 0]
        mu2 = [1, 1]
        sigma1 = [[3, 1], [1, 3]]
        sigma2 = [[3, 1], [1, 3]]

    trainX, trainY = seg.generate_gaussian_data(N, mu1, mu2, sigma1, sigma2)
    testX, testY = seg.generate_gaussian_data(N, mu1, mu2, sigma1, sigma2)

    return trainX, trainY, testX, testY


def easy_hard_data_classifier_test():
    N = 100

    for task in ['easy', 'hard']:
        trainX, trainY, testX, testY = generate_train_test(N, task)
        predicted = seg.nn_classifier(trainX, trainY.flatten(), testX)
        err = classification_error(testY.flatten(), predicted)
        print(f"Task: {task:5s}  |  1-NN classification error: {err:.4f}")

    print("\nExpected: easy dataset has low error; hard dataset has higher error due to class overlap.")


def nn_classifier_test_brains(testDice=False):
    X, Y, feature_labels_train = util.create_dataset(1, 1, 'brain')
    N = 1000
    ix = np.random.randint(len(X), size=N)
    train_data = X[ix, :]
    train_labels = Y[ix, :].flatten()

    test_data, test_labels, feature_labels_test = util.create_dataset(3, 1, 'brain')
    test_labels = test_labels.flatten()

    predicted_labels = seg.nn_classifier(train_data, train_labels, test_data)
    predicted_labels = predicted_labels.astype(bool)
    test_labels_bool = test_labels.astype(bool)
    err = classification_error(test_labels_bool, predicted_labels)
    print('Error:\n{}'.format(err))

    if testDice:
        dice = util.dice_overlap(test_labels_bool, predicted_labels)
        print('Dice coefficient:\n{}'.format(dice))
    else:
        I = plt.imread('../data/dataset_brains/3_1_t1.tif')
        GT = plt.imread('../data/dataset_brains/3_1_gt.tif')
        gt_mask = GT > 0
        predicted_mask = predicted_labels.reshape(I.shape)
        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(131); ax1.imshow(I); ax1.set_title('T1 image')
        ax2 = fig.add_subplot(132); ax2.imshow(predicted_mask); ax2.set_title('Predicted')
        ax3 = fig.add_subplot(133); ax3.imshow(gt_mask); ax3.set_title('Ground truth')
        plt.show()


def knn_curve():
    train_data, train_labels, train_feature_labels = util.create_dataset(1, 1, 'brain')
    test_data, test_labels, test_feature_labels = util.create_dataset(2, 1, 'brain')
    train_labels = train_labels.flatten()
    test_labels = test_labels.flatten()

    train_data, test_data = seg.normalize_data(train_data, test_data)

    num_iter = 3
    train_size = 100
    k = np.array([1, 3, 5, 9, 15, 25, 100])

    test_error = np.empty([len(k), num_iter])
    test_error[:] = np.nan
    dice = np.empty([len(k), num_iter])
    dice[:] = np.nan

    for i in np.arange(len(k)):
        for j in np.arange(num_iter):
            print('k = {}, iter = {}'.format(k[i], j))
            ix = np.random.randint(len(train_data), size=train_size)
            subset_train_data = train_data[ix, :]
            subset_train_labels = train_labels[ix]

            predicted_test_labels = seg.knn_classifier(subset_train_data, subset_train_labels, test_data, k[i])
            predicted_test_labels = predicted_test_labels.flatten().astype(bool)
            test_labels_bool = test_labels.astype(bool)

            test_error[i, j] = classification_error(test_labels_bool, predicted_test_labels)
            dice[i, j] = util.dice_overlap(test_labels_bool, predicted_test_labels)

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)
    ax1.plot(k, np.mean(test_error, 1), 'r', label='error')
    ax1.plot(k, np.mean(dice, 1), 'k', label='dice')
    ax1.set_xlabel('k')
    ax1.set_ylabel('error')
    ax1.grid()
    ax1.legend()
    plt.show()


# SECTION 2. Generalization and overfitting


def learning_curve():
    train_data, train_labels = seg.generate_gaussian_data(1000)
    test_data, test_labels = seg.generate_gaussian_data(1000)
    [train_data, test_data] = seg.normalize_data(train_data, test_data)

    train_sizes = np.array([1, 3, 10, 30, 100, 300])
    k = 1
    num_iter = 3

    test_error = np.empty([len(train_sizes), num_iter]); test_error[:] = np.nan
    test_dice  = np.empty([len(train_sizes), num_iter]); test_dice[:] = np.nan
    # Store errors for training data
    train_error = np.empty([len(train_sizes), num_iter]); train_error[:] = np.nan

    for i in np.arange(len(train_sizes)):
        for j in np.arange(num_iter):
            print('train_size = {}, iter = {}'.format(train_sizes[i], j))
            ix = np.random.randint(len(train_data), size=train_sizes[i])
            subset_train_data = train_data[ix, :]
            subset_train_labels = train_labels[ix, :]

            neigh = KNeighborsClassifier(n_neighbors=k)
            neigh.fit(subset_train_data, subset_train_labels.ravel())

            predicted_test_labels = neigh.predict(test_data)
            test_labels_bool = test_labels.flatten().astype(bool)
            predicted_test_labels_bool = predicted_test_labels.astype(bool)

            test_error[i, j] = classification_error(test_labels_bool, predicted_test_labels_bool)
            test_dice[i, j]  = util.dice_overlap(test_labels_bool, predicted_test_labels_bool)

            # Predict training labels and evaluate
            predicted_train_labels = neigh.predict(subset_train_data)
            train_labels_bool = subset_train_labels.flatten().astype(bool)
            predicted_train_labels_bool = predicted_train_labels.astype(bool)
            train_error[i, j] = classification_error(train_labels_bool, predicted_train_labels_bool)

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)
    x = np.log(train_sizes)
    y_test  = np.mean(test_error, 1)
    yerr_test = np.std(test_error, 1)
    ax1.errorbar(x, y_test, yerr=yerr_test, label='Test error')

    # Plot training error
    y_train = np.mean(train_error, 1)
    yerr_train = np.std(train_error, 1)
    ax1.errorbar(x, y_train, yerr=yerr_train, label='Train error', linestyle='--')

    ax1.set_xlabel('Number of training samples (k)')
    ax1.set_ylabel('error')
    ticks = list(x)
    ax1.set_xticks(ticks)
    ax1.set_xticklabels([str(i) for i in train_sizes])
    ax1.grid()
    ax1.legend()
    plt.show()


def feature_curve(use_random=False):
    train_data, train_labels, train_feature_labels = util.create_dataset(1, 1, 'brain')
    test_data, test_labels, test_feature_labels    = util.create_dataset(2, 1, 'brain')

    if use_random:
        # Replace features by random numbers of the same size
        train_data = np.random.rand(*train_data.shape)
        test_data  = np.random.rand(*test_data.shape)

    train_data, test_data = seg.normalize_data(train_data, test_data)

    feature_sizes = np.arange(train_data.shape[1]) + 1
    train_size = 10
    k = 3
    num_iter = 5

    test_error  = np.empty([len(feature_sizes), num_iter]); test_error[:] = np.nan
    train_error = np.empty([len(feature_sizes), num_iter]); train_error[:] = np.nan

    for i in np.arange(len(feature_sizes)):
        for j in np.arange(num_iter):
            print('feature size = {}, iter = {}'.format(feature_sizes[i], j))
            ix = np.random.randint(len(train_data), size=train_size)
            subset_train_data   = train_data[ix, :]
            subset_train_labels = train_labels[ix, :]

            neigh = KNeighborsClassifier(n_neighbors=k)
            neigh.fit(subset_train_data[:, :feature_sizes[i]], subset_train_labels.ravel())

            predicted_test_labels  = neigh.predict(test_data[:, :feature_sizes[i]])
            predicted_train_labels = neigh.predict(subset_train_data[:, :feature_sizes[i]])

            test_error[i, j]  = classification_error(test_labels.flatten(),  predicted_test_labels)
            train_error[i, j] = classification_error(subset_train_labels.flatten(), predicted_train_labels)

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)
    x = feature_sizes
    ax1.errorbar(x, np.mean(test_error, 1),  yerr=np.std(test_error, 1),  label='Test error')
    # Plot training error
    ax1.errorbar(x, np.mean(train_error, 1), yerr=np.std(train_error, 1), label='Train error', linestyle='--')

    ax1.set_xlabel('Number of features')
    ax1.set_ylabel('Error')
    ax1.grid()
    ax1.legend()
    plt.show()


def high_dimensions_demo():
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)

    X1 = np.random.randn(100, 2)
    mn1, mx1, mns1 = high_dimensions_output(X1, ax1, 20)
    print('2D Gaussian distribution')
    print('Mean = {:.4f}, Max = {:.4f}, Mean nn = {:.4f}'.format(mn1, mx1, mns1))

    X2 = np.random.randn(100, 1000)
    mn2, mx2, mns2 = high_dimensions_output(X2, ax2, 20)
    print('1000D Gaussian distribution')
    print('Mean = {:.4f}, Max = {:.4f}, Mean nn = {:.4f}'.format(mn2, mx2, mns2))

    n = 10
    k = 0.01
    frac = np.empty(n); frac[:] = None
    for i in np.arange(n):
        frac[i] = k ** (1 / (i + 1))

    ax3 = fig.add_subplot(133)
    ax3.plot(frac)
    ax3.set_xlabel('dimensions')
    ax3.set_ylabel('fraction to travel per dimension')
    plt.tight_layout()
    plt.show()


def high_dimensions_output(X, ax, n_bins=20):
    D = scipy.spatial.distance.cdist(X, X, metric='euclidean')

    ax.hist(D.flatten(), bins=n_bins)
    ax.set_xlabel('Distance')
    ax.set_ylabel('Count')
    ax.set_title(f'{X.shape[1]}D distances')

    # Mean distance, maximum distance, average nearest-neighbour distance
    mn = np.mean(D)
    mx = np.max(D)
    # Nearest neighbour: for each point, find min distance to any other point
    D_no_diag = D.copy()
    np.fill_diagonal(D_no_diag, np.inf)
    mns = np.mean(np.min(D_no_diag, axis=1))

    return mn, mx, mns


def covariance_matrix_test():
    N = 100
    mu1 = [0, 0]
    mu2 = [0, 0]
    sigma1 = [[3, 1], [1, 1]]
    sigma2 = [[3, 1], [1, 1]]
    X, Y = seg.generate_gaussian_data(N, mu1, mu2, sigma1, sigma2)

    # Calculate mean and covariance matrix of the data
    data_mean = np.mean(X, axis=0)
    data_cov  = np.cov(X, rowvar=False)

    print("Input parameters:")
    print(f"  mu1 = {mu1},  mu2 = {mu2}")
    print(f"  sigma1 =\n{np.array(sigma1)}")
    print()
    print("Estimated from data (both classes pooled):")
    print(f"  mean = {data_mean.round(3)}")
    print(f"  covariance =\n{data_cov.round(3)}")
    print()
    print("Note: the pooled estimate mixes both classes at [0,0], so it approximates sigma well.")


def eigen_vecval_test(sigma):
    # Compute eigenvectors and eigenvalues of the covariance matrix
    w, v = np.linalg.eigh(sigma)

    # Sort descending
    idx = np.argsort(w)[::-1]
    w = w[idx]; v = v[:, idx]

    print("Eigenvalues:", w.round(4))
    print("Eigenvectors (columns):\n", v.round(4))

    # Property 1: eigenvectors are orthogonal — verify via dot product
    dot = np.dot(v[:, 0], v[:, 1])
    print(f"\nProperty 1 – Orthogonality: v0 · v1 = {dot:.6f}  (should be ≈ 0)")

    # Property 2: eigenvectors are unit vectors — verify via norm
    norms = np.linalg.norm(v, axis=0)
    print(f"Property 2 – Unit length: ||v0|| = {norms[0]:.6f}, ||v1|| = {norms[1]:.6f}  (should be ≈ 1)")

    print(f"\nLargest eigenvalue: {w[0]:.4f}  →  eigenvector {v[:,0].round(4)}")
    print(f"Smallest eigenvalue: {w[-1]:.4f}  →  eigenvector {v[:,-1].round(4)}")

    return w, v


def rotate_using_eigenvectors_test(X, Y, v):
    # Rotate X using the eigenvectors: project onto eigenvector basis
    X_rot = X.dot(v)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    util.scatter_data(X,     Y, ax=axes[0]); axes[0].set_title('Original data');  axes[0].grid(True)
    util.scatter_data(X_rot, Y, ax=axes[1]); axes[1].set_title('Rotated data (eigenvectors)'); axes[1].grid(True)
    plt.tight_layout()
    plt.show()

    print("After rotation, the axes align with the directions of maximum variance (PCA basis).")
    return X_rot


def test_mypca():
    N = 100
    mu1 = [0, 0]; mu2 = [2, 0]
    sigma1 = [[2, 1], [1, 1]]; sigma2 = [[2, 1], [1, 1]]

    XG, YG = seg.generate_gaussian_data(N, mu1, mu2, sigma1, sigma2)

    fig = plt.figure(figsize=(15, 6))

    ax1 = fig.add_subplot(121)
    util.scatter_data(XG, YG, ax=ax1)
    sigma = np.cov(XG, rowvar=False)
    w, v = np.linalg.eig(sigma)
    ax1.plot([0, v[0, 0]], [0, v[1, 0]], c='g', linewidth=3, label='Eigenvector1')
    ax1.plot([0, v[0, 1]], [0, v[1, 1]], c='k', linewidth=3, label='Eigenvector2')
    ax1.set_title('Original data')
    ax_settings(ax1)

    ax2 = fig.add_subplot(122)
    X_pca, v, w, fraction_variance = seg.mypca(XG)
    util.scatter_data(X_pca, YG, ax=ax2)
    sigma2 = np.cov(X_pca, rowvar=False)
    w2, v2 = np.linalg.eig(sigma2)
    ax2.plot([0, v2[0, 0]], [0, v2[1, 0]], c='g', linewidth=3, label='Eigenvector1')
    ax2.plot([0, v2[0, 1]], [0, v2[1, 1]], c='k', linewidth=3, label='Eigenvector2')
    ax2.set_title('My PCA')
    ax_settings(ax2)

    handles, labels = ax2.get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                  bbox_transform=plt.gcf().transFigure, ncol=4)
    plt.show()
    print(fraction_variance)


def ax_settings(ax):
    ax.set_xlim(-7, 7); ax.set_ylim(-7, 7)
    ax.set_aspect('equal', adjustable='box')
    ax.grid()


# SECTION 3. Active shapes

def plot_hand_shapes():
    fn = '../data/dataset_hands/coordinates.txt'
    coordinates = np.loadtxt(fn)

    # Plot four example hand shapes
    fig = plt.figure(figsize=(16, 4))
    for n, pos in enumerate([141, 142, 143, 144]):
        ax = fig.add_subplot(pos)
        lbl = f'hand_{n + 1}'
        ax.plot(coordinates[n, :56], coordinates[n, 56:], label=lbl)
        ax.set_title(lbl)
        ax.set_aspect('equal')
        ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

    # Calculate the mean hand shape and plot in a new figure
    mean_shape = np.mean(coordinates, axis=0)
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.plot(mean_shape[:56], mean_shape[56:], 'b-o', markersize=4, label='Mean shape')
    ax2.set_title('Mean hand shape (averaged over all 40 hands)')
    ax2.set_aspect('equal')
    ax2.invert_yaxis()
    ax2.legend()
    plt.tight_layout()
    plt.show()


def test_mypca_hands():
    fn = '../data/dataset_hands/coordinates.txt'
    coordinates = np.loadtxt(fn)

    # Apply PCA to the hand coordinates
    X_pca, v, w, fraction_variance = seg.mypca(coordinates)

    # Determine how many dimensions explain 95% of the variance
    num_dims = int(np.searchsorted(fraction_variance.flatten(), 0.95) + 1)
    print(f"Dimensions needed to explain 95% of variance: {num_dims}")
    print(f"Fraction of variance explained by first {num_dims} components: {fraction_variance[num_dims-1, 0]:.4f}")

    # Keep only the top num_dims eigenvectors
    v_new = v[:, :num_dims]

    # Plot fraction of variance explained
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, len(fraction_variance) + 1), fraction_variance.flatten(), 'b-o')
    ax.axhline(0.95, color='r', linestyle='--', label='95% threshold')
    ax.axvline(num_dims, color='g', linestyle='--', label=f'{num_dims} dims')
    ax.set_xlabel('Number of PCA components')
    ax.set_ylabel('Fraction of variance explained')
    ax.set_title('Cumulative variance explained by PCA components (hand shapes)')
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.show()

    return num_dims, v_new


def test_remaining_variance():
    fn = '../data/dataset_hands/coordinates.txt'
    coordinates = np.loadtxt(fn)
    mn = np.mean(coordinates, axis=0)
    num_dims, v_new = test_mypca_hands()

    X_pca, v, w, fraction_variance = seg.mypca(coordinates)

    fig = plt.figure(figsize=(16, 4 * num_dims))
    plot_idx = 1

    # Loop through the retained PCA dimensions and show the variation each produces
    for dim in range(num_dims):
        # Vary the score along this dimension by ±3 std deviations
        std = np.sqrt(w[dim])
        for scale, label in [(-3 * std, f'dim {dim+1}: −3σ'), (0, f'dim {dim+1}: mean'), (3 * std, f'dim {dim+1}: +3σ')]:
            scores = np.zeros(num_dims)
            scores[dim] = scale
            shape = mn + scores.dot(v_new.T)
            ax = fig.add_subplot(num_dims, 3, plot_idx)
            ax.plot(shape[:56], shape[56:])
            ax.set_title(label)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            plot_idx += 1

    plt.suptitle('Variation produced by each retained PCA dimension (±3σ)', y=1.01)
    plt.tight_layout()
    plt.show()


def plot_hand_grayscale():
    fn_img = '../data/dataset_hands/test001.jpg'
    fn_coords = '../data/dataset_hands/coordinates.txt'
    img_hand = plt.imread(fn_img)
    coordinates = np.loadtxt(fn_coords)

    # Convert to grayscale: L = R*299/1000 + G*587/1000 + B*114/1000
    img_gray = (img_hand[:, :, 0] * 299.0 / 1000
                + img_hand[:, :, 1] * 587.0 / 1000
                + img_hand[:, :, 2] * 114.0 / 1000)

    # Compute mean hand shape to use as template
    mean_shape = np.mean(coordinates, axis=0)

    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121)
    ax1.imshow(img_hand)
    ax1.set_title('Original colour image')

    ax2 = fig.add_subplot(122)
    ax2.imshow(img_gray, cmap='gray')
    # Plot the mean hand template on top of the grayscale image
    ax2.plot(mean_shape[:56], mean_shape[56:], 'r-o', markersize=3, linewidth=1.5, label='Mean template')
    ax2.set_title('Grayscale image with mean hand template')
    ax2.legend()
    plt.tight_layout()
    plt.show()


def test_transformed_hand():
    fn_img = '../data/dataset_hands/test001.jpg'
    fn_coords = '../data/dataset_hands/coordinates.txt'
    img_hand = plt.imread(fn_img)
    coordinates = np.loadtxt(fn_coords)
    mn = np.mean(coordinates, axis=0)

    # Convert mean shape to 2D format (2 x 56): row 0 = x, row 1 = y
    initialpos = np.concatenate((mn[:56].reshape(1, -1), mn[56:].reshape(1, -1)), axis=0)

    # Define a scaling/rotation/alignment matrix to fit the hand in test001.jpg.
    # The mean shape is centered near the origin; the image hand is roughly at
    # (image_width/2, image_height/2) with a scale factor to match image size.
    img_gray = (img_hand[:, :, 0] * 299.0 / 1000
                + img_hand[:, :, 1] * 587.0 / 1000
                + img_hand[:, :, 2] * 114.0 / 1000)

    H, W = img_gray.shape
    # Estimate scale: ratio of image extent to coordinate extent
    coord_range_x = mn[:56].max() - mn[:56].min()
    coord_range_y = mn[56:].max() - mn[56:].min()
    scale = min(W / coord_range_x, H / coord_range_y) * 0.8

    # Rotation angle (radians) — the hand in test001.jpg appears upright
    theta = 0.0
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # 2x2 similarity transform: scale + rotation
    R = scale * np.array([[cos_t, -sin_t],
                           [sin_t,  cos_t]])

    # Apply transform and translate to image center
    shape_t = R.dot(initialpos)
    shape_t[0, :] += W / 2 - np.mean(shape_t[0, :])
    shape_t[1, :] += H / 2 - np.mean(shape_t[1, :])

    # Plot image and transformed shape
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121)
    ax1.imshow(img_gray, cmap='gray')
    ax1.plot(initialpos[0, :], initialpos[1, :], 'r', label='Initial (mean shape)')
    ax1.set_title('Mean shape before transform')
    ax1.legend()

    ax2 = fig.add_subplot(122)
    ax2.imshow(img_gray, cmap='gray')
    ax2.plot(shape_t[0, :], shape_t[1, :], 'r', label='Transformed shape')
    ax2.set_title('Transformed shape overlaid on image')
    ax2.legend()
    plt.tight_layout()
    plt.show()
