%matplotlib inline
import sys
sys.path.append("../code")
import numpy as np
import segmentation_util as util
import matplotlib.pyplot as plt
import segmentation as seg

## -------------- TESTING FEATURES ON TRAINING DATA ---------------------

X_train, Y_train, feature_labels_train = util.create_dataset(1, 1, 'tissue')
X_test, Y_test, feature_labels_test = util.create_dataset(3, 1, 'tissue')

# based on pca, I select the features that explain 95% of the variance (of training data)

X_pca, v, w, fraction_variance, important_features = seg.mypca(X_train, feature_labels_train, visualize=False)

#print(f"Important features (explain 95% variance):")
#for f in important_features:
#    print(f)

important_indices = [feature_labels_train.index(f) for f in important_features]
X_train_reduced = X_train[:, important_indices]

# use selected features from training on test data
X_test_reduced = X_test[:, important_indices]

# in the task it says 4 classes (including background). But in the previous task we should filter it?!

#---------------TESTING CLASSIFIERS ON TRAINING DATA -----------------------