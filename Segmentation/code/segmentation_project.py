"""
Project code+scripts for 8BE030 course
"""

import numpy as np
import segmentation_util as util
import matplotlib.pyplot as plt
import segmentation as seg


def segmentation_mymethod(train_data_matrix, train_labels_matrix, test_data, task='tissue'):
    # segments the image based on your own method!
    # Input:
    # train_data_matrix   num_pixels x num_features x num_subjects matrix of
    # features
    # train_labels_matrix num_pixels x num_subjects matrix of labels
    # test_data           num_pixels x num_features test data
    # task           String corresponding to the segmentation task: either 'brain' or 'tissue'
    # Output:
    # predicted_labels    Predicted labels for the test slice

    num_pixels, num_features, num_subjects = train_data_matrix.shape

    # Pool all training subjects into one massive 2D array
    train_data = train_data_matrix.transpose(2, 0, 1).reshape(-1, num_features)
    train_labels = train_labels_matrix.transpose(1, 0).reshape(-1)

    # Call the k-NN classifier (k=5 as per your research notebook)
    predicted_labels = seg.segmentation_knn(train_data, train_labels, test_data, k=5)

    return predicted_labels


def segmentation_demo():
    num_subjects = 5
    im_size = [240, 240]
    train_slices = [1,2,3]
    task = 'tissue'  

    all_subjects = np.arange(num_subjects)
    
    all_data_t1 = []
    all_data_t1t2 = []
    all_labels = []

    print(f'Loading data for {num_subjects} subjects with {len(train_slices)} slices...')
    for i in all_subjects:
        sub = i + 1
        print(f"\tSubject {sub}...")

        for j in train_slices:
            slice = j
            print(f"\t\tSlice {slice}...")
            # Load T1-only
            X_t1, Y, _ = util.create_dataset(sub, slice, task, use_t2=False)
            all_data_t1.append(X_t1)
            
            # Load T2 + T1
            X_both, _, _ = util.create_dataset(sub, slice, task, use_t2=True)
            all_data_t1t2.append(X_both)
            all_labels.append(Y.flatten())

    all_data_t1_matrix = np.dstack(all_data_t1)
    all_data_t1t2_matrix = np.dstack(all_data_t1t2)
    all_labels_matrix = np.column_stack(all_labels)

    print('Finished loading data.')
    print(f"\tData shapes - T1: {all_data_t1_matrix.shape}, T1+T2: {all_data_t1t2_matrix.shape}, Labels: {all_labels_matrix.shape}")
    print('\nStarting cross-validation...')

    all_subjects = np.arange(num_subjects)
    all_indices = np.arange(num_subjects * len(train_slices))

    dice_score_list = []

    # Leave-One-Slice-Out Cross-Validation
    for i in all_subjects:
        sub = i + 1

        fig = plt.figure(figsize=(11, 13))
        fig.suptitle(f"Subject {sub}")
        classes = ['background', 'White Matter', 'Grey Matter', 'CSF']

        for j in range(len(train_slices)):

            dice_score_mini_list = []

            slice = j + 1   # For naming the right slice in plots

            slice_index = len(train_slices) * i + j     # Current test slice (excluded from training data)

            # Define Training data (all slices except current slice (slice_index))
            train_subjects = np.delete(all_indices, slice_index)     #Exclude current test slice
            train_data_t1 = all_data_t1_matrix[:, :, train_subjects]
            train_data_t1t2 = all_data_t1t2_matrix[:, :, train_subjects]
            train_labels = all_labels_matrix[:, train_subjects]

            # Define Test data (current slice_index)
            test_data_t1 = all_data_t1_matrix[:, :, slice_index]
            test_data_t1t2 = all_data_t1t2_matrix[:, :, slice_index]
            test_labels = all_labels_matrix[:, slice_index]

            #----------------- T1 ----------------------------------------------
            # Make prediction clusters with T1 MRI scan features only
            print(f"Training Subject {sub} slice {slice} on T1-only...")
            predicted_labels_t1 = segmentation_mymethod(train_data_t1, train_labels, test_data_t1, task)

            # Validate prediction
            err_t1 = util.classification_error(test_labels, predicted_labels_t1)
            dice_t1 = util.dice_multiclass(test_labels, predicted_labels_t1)
            conf_matrix_t1 = util.confusion_matrix(test_labels, predicted_labels_t1)
            print(f"Confusion Matrix for T1-only:\n{conf_matrix_t1}")

            #----------------- T1 + T2 ------------------------------------------
            # Make prediction clusters with both T1 and T2 MRI scan features
            print(f"Training Subject {sub} slice {slice} on T1+T2...")
            predicted_labels_t1t2 = segmentation_mymethod(train_data_t1t2, train_labels, test_data_t1t2, task)

            # Validate prediction
            err_t1t2 = util.classification_error(test_labels, predicted_labels_t1t2)
            dice_t1t2 = util.dice_multiclass(test_labels, predicted_labels_t1t2)
            conf_matrix_t1t2 = util.confusion_matrix(test_labels, predicted_labels_t1t2)
            print(f"Confusion Matrix for T1+T2:\n{conf_matrix_t1t2}")

            dice_score_mini_list.append(float(dice_t1))
            dice_score_mini_list.append(float(dice_t1t2))

            # -------------- Visualization --------------------------------
            # Plot resulting predictions and validations per slice
            # T1 features only
            ax1 = fig.add_subplot(3,3,(1+3*j))
            ax1.imshow(predicted_labels_t1.reshape(im_size[0], im_size[1]), 'viridis')
            ax1.set_title(f'Slice {slice}: T1-Only Baseline')
            ax1.set_xlabel(f'Err {err_t1:.4f}, Mean dice {dice_t1:.4f}')

            # T1 + T2 features
            ax2 = fig.add_subplot(3,3,(2+3*j))
            ax2.imshow(predicted_labels_t1t2.reshape(im_size[0], im_size[1]), 'viridis')
            ax2.set_title(f'Slice {slice}: T1+T2 Proposed')
            ax2.set_xlabel(f'Err {err_t1t2:.4f}, Mean dice {dice_t1t2:.4f}')

            # Ground Truth
            ax3 = fig.add_subplot(3,3,(3+3*j))
            ax3.imshow(test_labels.reshape(im_size[0], im_size[1]), 'viridis')
            ax3.set_title(f'Slice {slice}: Ground Truth')

            dice_score_list.append(dice_score_mini_list)

        # Save plot per Subject before displaying it
        fig.tight_layout(rect=[0, 0, 1, 1])
        fig.savefig(f"Test_{sub}.png", dpi=150, bbox_inches='tight')
        fig.legend(classes)
        plt.show()

    print("All dice scores:", dice_score_list)
    for i in dice_score_list:
        print(i[0], i[1])