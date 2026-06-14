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

    all_slices = np.arange(num_subjects*len(train_slices))

    # Leave-One-Subject-Out Cross-Validation
    #for i in np.arange(num_subjects):
    for i in all_slices:
        sub = i + 1

        fig = plt.figure(figsize=(11, 13))
        fig.suptitle(f"Subject {sub}")

        for j in range(len(train_slices)):
            slice = j + 1   # For naming the right slice in plots

            slice_index = len(train_slices)*i+j     # Current test slice (excluded from training data)

            # Define Training data (all slices except current slice (slice_index))
            train_subjects = np.delete(all_slices, slice_index)     #Exclude current test slice
            train_data_t1 = all_data_t1_matrix[:, :, train_subjects]
            train_data_t1t2 = all_data_t1t2_matrix[:, :, train_subjects]
            train_labels = all_labels_matrix[:, train_subjects]

            # Define Test data (Current slice_index)
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
            print(f"Confusion Matrix for T1-only:\n{conf_matrix_t1t2}")

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

        # Show plot per Subject
        plt.show()
        plt.savefig(f"Test {sub}", format='png')


# def segmentation_demo():

#     train_subject = 1
#     test_subject = 2
#     train_slice = 1
#     test_slice = 1
#     task = 'brain'

#     #Load data
#     train_data, train_labels, train_feature_labels = util.create_dataset(train_subject,train_slice,task)
#     test_data, test_labels, test_feature_labels = util.create_dataset(test_subject,test_slice,task)

#     # predicted_labels = seg.segmentation_knn(None, train_labels, None)

#     # err = util.classification_error(test_labels, predicted_labels)
#     # dice = util.dice_overlap(test_labels, predicted_labels)

#     #Display results
#     # true_mask = test_labels.reshape(240, 240)
#     # predicted_mask = predicted_labels.reshape(240, 240)

#     # fig = plt.figure(figsize=(8,8))
#     # ax1 = fig.add_subplot(111)
#     # ax1.imshow(true_mask, 'gray')
#     # ax1.imshow(predicted_mask, 'viridis', alpha=0.5)
#     # print('Subject {}, slice {}.\nErr {}, dice {}'.format(test_subject, test_slice, err, dice))

#     ## Compare methods
#     num_images = 5
#     num_methods = 3
#     im_size = [240, 240]

#     all_errors = np.empty([num_images,num_methods])
#     all_errors[:] = np.nan
#     all_dice = np.empty([num_images,num_methods])
#     all_dice[:] = np.nan

#     all_subjects = np.arange(num_images)
#     train_slice = 1
#     task = 'brain'
#     all_data_matrix = np.empty([train_data.shape[0],train_data.shape[1],num_images])
#     all_labels_matrix = np.empty([train_labels.size,num_images], dtype=bool)

#     #Load datasets once
#     print('Loading data for ' + str(num_images) + ' subjects...')

#     for i in all_subjects:
#         sub = i+1
#         train_data, train_labels, train_feature_labels = util.create_dataset(sub,train_slice,task)
#         all_data_matrix[:,:,i] = train_data
#         all_labels_matrix[:,i] = train_labels.flatten()

#     print('Finished loading data.\nStarting segmentation...')

#     #Go through each subject, taking i-th subject as the test
#     for i in np.arange(num_images):
#         sub = i+1
#         #Define training subjects as all, except the test subject
#         train_subjects = all_subjects.copy()
#         train_subjects = np.delete(train_subjects, i)

#         train_data_matrix = all_data_matrix[:,:,train_subjects]
#         train_labels_matrix = all_labels_matrix[:,train_subjects]
#         test_data = all_data_matrix[:,:,i]
#         test_labels = all_labels_matrix[:,i]
#         test_shape_1 = test_labels.reshape(im_size[0],im_size[1])

#         fig = plt.figure(figsize=(10,5))

#         predicted_labels = seg.segmentation_combined_knn(train_data_matrix,train_labels_matrix,test_data)
#         all_errors[i,0] = util.classification_error(test_labels, predicted_labels)
#         all_dice[i,0] = util.dice_overlap(test_labels, predicted_labels)
#         predicted_mask_1 = predicted_labels.reshape(im_size[0],im_size[1])
#         ax1 = fig.add_subplot(121)
#         ax1.imshow(test_shape_1, 'gray')
#         ax1.imshow(predicted_mask_1, 'viridis', alpha=0.5)
#         text_str = 'Err {:.4f}, dice {:.4f}'.format(all_errors[i,0], all_dice[i,0])
#         ax1.set_xlabel(text_str)
#         ax1.set_title('Subject {}: Combined k-NN'.format(sub))

#         predicted_labels = segmentation_mymethod(train_data_matrix,train_labels_matrix,test_data,task)
#         all_errors[i,1] = util.classification_error(test_labels, predicted_labels)
#         all_dice[i,1] = util.dice_overlap(test_labels, predicted_labels)
#         predicted_mask_2 = predicted_labels.reshape(im_size[0],im_size[1])
#         ax2 = fig.add_subplot(122)
#         ax2.imshow(test_shape_1, 'gray')
#         ax2.imshow(predicted_mask_2, 'viridis', alpha=0.5)
#         text_str = 'Err {:.4f}, dice {:.4f}'.format(all_errors[i,1], all_dice[i,1])
#         ax2.set_xlabel(text_str)
#         ax2.set_title('Subject {}: My method'.format(sub))