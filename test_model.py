# Author: Narmada Balasooriya #
# Based on: https://blog.sicara.com/keras-tutorial-content-based-image-retrieval-convolutional-denoising-autoencoder-dc91450cc511
# ########################### #
# This code creates an autoencoder model using Keras and trains it in batches for the dataset since the whole dataset cannot be loaded into memory
# ########################### #

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.models import Model
from keras.models import load_model
from keras.callbacks import TensorBoard
from keras.utils.data_utils import get_file
from keras import optimizers, losses

import numpy as np 
import h5py
import tables
from math import ceil
import matplotlib.pyplot as plt 
from random import shuffle
import csv

print('defining the function')

# Function to Retrieve the closest images #
# trained_codes = encoder layer features 
# test_element = test image
# test_label = label of the corresponding test image
# n_samples = retrieves the closest 25 images to the test image
def retrieve_closest_images(trained_codes, test_element, test_label, n_samples=25):
	
	# shape of the trained codes is printed for ease
	print('trained codes: ', trained_codes.shape)
	print('trained code shapes: ', trained_codes.shape[0], trained_codes.shape[1], trained_codes.shape[2], trained_codes.shape[3])
	
	# reshape the trained codes
	trained_codes = trained_codes.reshape(trained_codes.shape[0], trained_codes.shape[1]*trained_codes.shape[2]*trained_codes.shape[3])
	
	# predict the encoder layer codes for the test image
	test_codes = encoder.predict(np.array([test_element]))
	# reshape the tested codes of the test image
	test_codes = test_codes.reshape(test_codes.shape[1]*test_codes.shape[2]*test_codes.shape[3])
	
	# initialize the distance list
	distances = []

	for code in trained_codes:
		# for each code of the training images, compute the Euclidean distance between the train image and test image
		distance = np.linalg.norm(code - test_codes)
		distances.append(distance) # append to the distance list

	nb_elements = trained_codes.shape[0] # get the totale number of images in the training set
	distances = np.array(distances) # convert the distance list to a numpy array
	trained_code_index = np.arange(nb_elements) # creae an index list from 0 - nb_elements
	
	# create a numpy stack with the distances, index_list
	distances_with_index = np.stack((distances, trained_code_index), axis=-1)
	sorted_distance_with_index = distance_with_index[distance_with_index[:,0].argsort()] # sort the stack

	sorted_distances = sorted_distance_with_index[:, 0].astype('float32') # change the datatype
	sorted_indexes = sorted_distance_with_index[:, 1]

	kept_indexes = sorted_indexes[:n_samples] # Get the first 25 indexes of the sorted_indexes list

	return kept_indexes

###############################################
# Retrieve similar images for the test images #
###############################################

print('loading the train and test datasets')

train_path = './dataset/dataset.hdf5' # path for the training dataset
test_path = './dataset/test_dataset.hdf5' # path for the test dataset

train_files = tables.open_file(train_path, mode='r') # read the training datafiles
test_files = tables.open_file(test_path, mode='r') # read the test datafiles

train_size = train_files.root.train_data.shape[0] 
test_size = test_files.root.test_data.shape[0]

train_lbl_size = train_files.root.train_label.shape[0]
test_lbl_size = test_files.root.test_label.shape[0]

# define the batch sizes
train_batch_size = 10000 
test_batch_size = 10000

# define the batch lists with the batch sizes
train_list = list(range(int(ceil(float(train_size) / train_batch_size))))
test_list = list(range(int(ceil(float(test_size) / test_batch_size))))

print('Loading the trained autoencoder')

# load the autoencoder model
autoencoder = load_model('autoencoder.h5')
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output) # get the encoder layer features

trained_codes = np.array([], dtype='float32').reshape(0, 16, 16, 8) # initialize the training codes

print('predict the encoder for the the training dataset')

# predict the encoder layer features for the training dataset
for n, i in enumerate(train_list):
	i_s = i * train_batch_size
	i_e = min([(i + 1) * train_batch_size, train_size])

	train_images = train_files.root.train_data[i_s:i_e]
	train_labels = train_files.root.train_label[i_s:i_e]

	x_train = train_images.astype('float32') / 255.
	y_train = train_labels

	print('predicting for ', i_s, ' to ', i_e, '\n')
	predicted_codes = encoder.predict(x_train)
	trained_codes = np.concatenate((trained_codes, predicted_codes))

# labels of the similar images to the query image are appended to a csv file

with open('similar_images.csv', "w") as csv_file:
	writer = csv.writer(csv_file, delimiter=' ')

	# enumerate over the test images
	for j in range(test_size):
		test_image = test_files.root.test_data[j] # get the j th test image and the label
		test_label = test_files.root.test_label[j]

		x_test = test_image.astype('float32') / 255. # normalize the test image
		y_test = test_label

		similar_indexes = retrieve_closest_images(trained_codes, x_test, y_test) # retrieve the indexes closest images for the give test image

		similar_indexes = similar_indexes.astype('int')
		x_similar_labels = train_files.root.train_label[similar_indexes] # get the labels of the closest images

		final_list = np.insert(x_similar_labels, 0, test_label) # set the test label as the 1st index of the list

		print('writing row: ', j)
		writer.writerow(final_list) # write to the csv

train_files.close()
test_files.close()
# end #
