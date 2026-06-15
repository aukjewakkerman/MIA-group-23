"""
Utility functions for segmentation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import segmentation as seg
from scipy import ndimage


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
    pass
    #------------------------------------------------------------------#

    # return g

def scatter_data(X, Y, feature0=0, feature1=1, ax=None):
    # scater_data displays a scatterplot of at most 1000 samples from dataset X, and gives each point
    # a different color based on its label in Y

    k = 1000
    if len(X) > k:
        idx = np.random.randint(len(X), size=k)
        X = X[idx,:]
        Y = Y[idx]

    class_labels, indices1, indices2 = np.unique(Y, return_index=True, return_inverse=True)
    if ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        ax.grid()

    colors = cm.rainbow(np.linspace(0, 1, len(class_labels)))
    for i, c in zip(np.arange(len(class_labels)), colors):
        idx2 = indices2 == class_labels[i]
        lbl = 'Class '+str(class_labels[i])
        ax.scatter(X[idx2,feature0], X[idx2,feature1], color=c, label=lbl)

    # Show legend mapping colors to classes
    ax.legend(title='Classes', loc='best')

    return ax


def create_dataset(image_number, slice_number, task, use_t2=True):
    # create_dataset Creates a dataset for a particular subject (image), slice and task
    # Input:
    # image_number - Number of the subject (scalar)
    # slice_number - Number of the slice (scalar)
    # task        - String corresponding to the task, either 'brain' or 'tissue'
    # Output:
    # X           - Nxk feature matrix, where N is the number of pixels and k is the number of features
    # Y           - Nx1 vector with labels
    # feature_labels - kx1 cell array with descriptions of the k features

    #Extract features from the subject/slice
    X, feature_labels = extract_features(image_number, slice_number, use_t2)

    #Create labels
    Y = create_labels(image_number, slice_number, task)

    return X, Y, feature_labels

def extract_features(image_number, slice_number, use_t2=True):
    base_dir = '../data/dataset_brains/'

    # Load T1
    t1 = plt.imread(base_dir + str(image_number) + '_' + str(slice_number) + '_t1.tif').astype(float)
    
    feats = []
    features = ()

    # --- T1 FEATURES  ---

    # Raw intensity
    feats.append(t1.flatten().reshape(-1, 1))
    features += ('T1 raw',)

    # Gaussian-blurred intensity
    for s in [1, 2, 4]:
        t1_g = ndimage.gaussian_filter(t1, sigma=s).flatten().reshape(-1, 1)
        feats.append(t1_g)
        features += (f'T1 gauss sigma={s}',)

    # Local mean (uniform filter / box filter)
    t1_mean = ndimage.uniform_filter(t1, size=3)
    feats.append(t1_mean.flatten().reshape(-1, 1))
    features += ('T1 local mean 3x3',)

    # Local standard deviation
    t1_mean_sq = ndimage.uniform_filter(t1**2, size=3)
    t1_sq_mean = t1_mean**2
    t1_std = np.sqrt(np.maximum(t1_mean_sq - t1_sq_mean, 0))
    feats.append(t1_std.flatten().reshape(-1, 1))
    features += ('T1 local std 3x3',)

    # Local gradient magnitude
    gx, gy = np.gradient(t1)
    t1_grad = np.sqrt(gx**2 + gy**2).flatten().reshape(-1, 1)
    feats.append(t1_grad)
    features += ('T1 grad mag',)

    # Laplacian of Gaussian 
    t1_log = ndimage.gaussian_laplace(t1, sigma=2).flatten().reshape(-1, 1)
    feats.append(t1_log)
    features += ('T1 LoG sigma=2',)

    # Difference of Gaussians 
    t1_dog = (ndimage.gaussian_filter(t1, sigma=1) - ndimage.gaussian_filter(t1, sigma=4)).flatten().reshape(-1, 1)
    feats.append(t1_dog)
    features += ('T1 DoG (1, 4)',)

    # --- T2 FEATURES ---
    if use_t2:
        t2 = plt.imread(base_dir + str(image_number) + '_' + str(slice_number) + '_t2.tif').astype(float)
        
        # Raw intensity
        feats.append(t2.flatten().reshape(-1, 1))
        features += ('T2 raw',)

        # Gaussian-blurred intensity
        for s in [1, 2, 4]:
            t2_g = ndimage.gaussian_filter(t2, sigma=s).flatten().reshape(-1, 1)
            feats.append(t2_g)
            features += (f'T2 gauss sigma={s}',)

        # Local mean
        t2_mean = ndimage.uniform_filter(t2, size=3)
        feats.append(t2_mean.flatten().reshape(-1, 1))
        features += ('T2 local mean 3x3',)

        # Local standard deviation
        t2_mean_sq = ndimage.uniform_filter(t2**2, size=3)
        t2_sq_mean = t2_mean**2
        t2_std = np.sqrt(np.maximum(t2_mean_sq - t2_sq_mean, 0))
        feats.append(t2_std.flatten().reshape(-1, 1))
        features += ('T2 local std 3x3',)

        # Local gradient magnitude
        gx2, gy2 = np.gradient(t2)
        t2_grad = np.sqrt(gx2**2 + gy2**2).flatten().reshape(-1, 1)
        feats.append(t2_grad)
        features += ('T2 grad mag',)

        # Laplacian of Gaussian (LoG)
        t2_log = ndimage.gaussian_laplace(t2, sigma=2).flatten().reshape(-1, 1)
        feats.append(t2_log)
        features += ('T2 LoG sigma=2',)

        # Difference of Gaussians (DoG)
        t2_dog = (ndimage.gaussian_filter(t2, sigma=1) - ndimage.gaussian_filter(t2, sigma=4)).flatten().reshape(-1, 1)
        feats.append(t2_dog)
        features += ('T2 DoG (1, 4)',)

        # T2/T1 intensity ratio (The cross-modal feature)
        ratio = (t2 / (t1 + 1e-5)).flatten().reshape(-1, 1)
        feats.append(ratio)
        features += ('T2/T1 ratio',)

    X = np.concatenate(feats, axis=1)
    return X, list(features)


def create_labels(image_number, slice_number, task):
    # Creates labels for a particular subject (image), slice and
    # task
    #
    # Input:
    # image_number - Number of the subject (scalar)
    # slice_number - Number of the slice (scalar)
    # task        - String corresponding to the task, either 'brain' or 'tissue'
    #
    # Output:
    # Y           - Nx1 vector with labels
    #
    # Original labels reference:
    # 0 background
    # 1 cerebellum
    # 2 white matter hyperintensities/lesions
    # 3 basal ganglia and thalami
    # 4 ventricles
    # 5 white matter
    # 6 brainstem
    # 7 cortical grey matter
    # 8 cerebrospinal fluid in the extracerebral space

    #Read the ground-truth image
    base_dir = '../data/dataset_brains/'

    I = plt.imread(base_dir + str(image_number) + '_' + str(slice_number) + '_gt.tif')

    if task == 'brain':
        Y = I>0
    elif task == 'tissue':
        
        # sub-binarize
        white_matter = np.isin(I, [2, 5])
        gray_matter = np.isin(I, [3, 7])
        csf = np.isin(I, [4, 8])
        background = np.isin(I, [0, 1, 6])

        # new GT
        Y = np.copy(I)
        Y[background] = 0
        Y[white_matter] = 1
        Y[gray_matter] = 2
        Y[csf] = 3

    
    else:
        print(task)
        raise ValueError("Variable 'task' must be one of two values: 'brain' or 'tissue'")

    Y = Y.flatten().T
    Y = Y.reshape(-1,1)

    return Y

def dice_overlap(true_labels, predicted_labels, smooth=1.):
    # returns the Dice coefficient for two binary label vectors
    # Input:
    # true_labels         Nx1 binary vector with the true labels
    # predicted_labels    Nx1 binary vector with the predicted labels
    # smooth              smoothing factor that prevents division by zero
    # Output:
    # dice          Dice coefficient

    assert true_labels.shape[0] == predicted_labels.shape[0], "Number of labels do not match"

    t = true_labels.flatten()
    p = predicted_labels.flatten()

    intersection = np.sum(t * p)
    dice = (2. * intersection + smooth) / (np.sum(t) + np.sum(p) + smooth)
    
    return dice

def dice_multiclass(true_labels, predicted_labels):
    #dice_multiclass.m returns the Dice coefficient for two label vectors with
    #multiple classses
    #
    # Input:
    # true_labels         Nx1 vector with the true labels
    # predicted_labels    Nx1 vector with the predicted labels
    #
    # Output:
    # dice_score          Dice coefficient

    all_classes, indices1, indices2 = np.unique(true_labels, return_index=True, return_inverse=True)

    dice_score = np.empty((len(all_classes), 1))
    dice_score[:] = np.nan

    #Consider each class as the foreground class
    for i in np.arange(len(all_classes)):
        idx2 = indices2 == all_classes[i]
        lbl = 'X, class '+ str(all_classes[i])
        temp_true = true_labels.copy()
        temp_true[true_labels == all_classes[i]] = 1  #Class i is foreground
        temp_true[true_labels != all_classes[i]] = 0  #Everything else is background

        temp_predicted = predicted_labels.copy();
        #print(temp_predicted.dtype) Aukje denkt dat dit wel weg kan
        temp_predicted[predicted_labels == all_classes[i]] = 1
        temp_predicted[predicted_labels != all_classes[i]] = 0
        dice_score[i] = dice_overlap(temp_true.astype(int), temp_predicted.astype(int))
        print("Dice score for class [{}]: {}".format(i, dice_score[i]))

    dice_score_mean = dice_score.mean()

    return dice_score_mean

def confusion_matrix(true_labels, predicted_labels):
    import pandas as pd
    # confusion_matrix.m returns the confusion matrix for two vectors with labels
    #
    # Input:
    # true_labels         Nx1 vector with the true labels
    # predicted_labels    Nx1 vector with the predicted labels
    #
    # Output:
    # conf_matrix        CxC confusion matrix, where C is the number of classes
    assert true_labels.shape[0] == predicted_labels.shape[0], "Number of labels do not match"
    t = true_labels.flatten()
    p = predicted_labels.flatten()
    all_classes, indices1, indices2 = np.unique(true_labels, return_index=True, return_inverse=True)
    class_names = ['Background', 'CSF', 'GM', 'WM'] #still needs to be generalized for more classes
    conf_matrix = np.zeros((len(all_classes), len(all_classes)), dtype=int)
    for i in range(len(t)):
        true_class = t[i]
        pred_class = p[i]
        conf_matrix[true_class, pred_class] += 1
    conf_matrix = pd.DataFrame(conf_matrix, index=[f"True {name}" for name in class_names], columns=[f"Pred {name}" for name in class_names])

    return conf_matrix


def add_label_legend(ax, class_names=None, num_classes=None, cmap_name='tab10', loc='best'):
    """Add a legend for integer label images (0..C-1) showing class colors.

    - ax: matplotlib Axes where legend will be placed
    - class_names: optional list of class display names (length C)
    - num_classes: optional number of classes; inferred from class_names if not provided
    - cmap_name: matplotlib colormap name to sample colors from
    - loc: legend location
    """
    if class_names is None and num_classes is None:
        return ax
    if num_classes is None:
        num_classes = len(class_names)
    cmap = cm.get_cmap(cmap_name, num_classes)
    colors = cmap(np.arange(num_classes))
    handles = [mpatches.Patch(color=colors[i], label=(class_names[i] if class_names is not None else f'Class {i}')) for i in range(num_classes)]
    ax.legend(handles=handles, loc=loc, title='Classes')
    return ax






