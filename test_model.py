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

def retrieve_closest_images(trained_codes, test_element, test_label, n_samples=25):
	print('trained codes: ', trained_codes.shape)
	print('trained code shapes: ', trained_codes.shape[0], trained_codes.shape[1], trained_codes.shape[2], trained_codes.shape[3])

	trained_codes = trained_codes.reshape(trained_codes.shape[0], trained_codes.shape[1]*trained_codes.shape[2]*trained_codes.shape[3])

	test_codes = encoder.predict(np.array([test_element]))
	test_codes = test_codes.reshape(test_codes.shape[1]*test_codes.shape[2]*test_codes.shape[3])

	distances = []

	for code in trained_codes:
		distance = np.linalg.norm(code - test_codes)
		distances.append(distance)

	nb_elements = trained_codes.shape[0]
	distances = np.array(distances)
	trained_code_index = np.arange(nb_elements)
	#labels = np.copy(y_train).astype('str')

	distance_with_labels = np.stack((distances, trained_code_index), axis=-1)
	sorted_distance_with_labels = distance_with_labels[distance_with_labels[:,0].argsort()]

	sorted_distances = sorted_distance_with_labels[:, 0].astype('float32')
	#sorted_labels = sorted_distance_with_labels[:, 1]
	sorted_indexes = sorted_distance_with_labels[:, 1]

	kept_indexes = sorted_indexes[:n_samples]

	return kept_indexes

	"""
	original_image = test_element
	cv2.imshow('original image', original_image)

	retrieved_images = x_train[int(kept_indexes[0]), :]

	for i in range(1, n_samples):
		retrieved_images = np.hstack((retrieved_images, x_train[int(kept_indexes[1]), :]))

	cv2.imshow('results: ', retrieved_images)
	cv2.waitKey(0)

	cv2.imwrite('test_results/original_img.jpg', 255 * cv2.resize(original_image, (0,0), fx=3, fy=3))
	cv2.imwrite('test_results/retrieved_img.jpg', 255 * cv2.resize(original_image, (0,0), fx=2, fy=2))

	return kept_indexes
	"""


print('loading the train and test datasets')

train_path = './dataset/dataset.hdf5'
test_path = './dataset/test_dataset.hdf5'

train_files = tables.open_file(train_path, mode='r')
test_files = tables.open_file(test_path, mode='r')

train_size = train_files.root.train_data.shape[0]
test_size = test_files.root.test_data.shape[0]

train_lbl_size = train_files.root.train_label.shape[0]
test_lbl_size = test_files.root.test_label.shape[0]

train_batch_size = 10000
test_batch_size = 10000

train_list = list(range(int(ceil(float(train_size) / train_batch_size))))
test_list = list(range(int(ceil(float(test_size) / test_batch_size))))

print('Loading the trained autoencoder')

autoencoder = load_model('autoencoder.h5')
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output)

trained_codes = np.array([], dtype='float32').reshape(0, 16, 16, 8)

print('predict the encoder for the the training dataset')

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

indexes = retrieve_closest_images(trained_codes, test_files.root.test_data[10], test_files.root.test_label[10])

print(indexes)

with open('sample_sub_v2.csv', "w") as csv_file:
	writer = csv.writer(csv_file, delimiter=' ')

	for j in range(test_size):
		test_image = test_files.root.test_data[j]
		test_label = test_files.root.test_label[j]

		x_test = test_image.astype('float32') / 255.
		y_test = test_label

		similar_indexes = retrieve_closest_images(trained_codes, x_test, y_test)

		similar_indexes = similar_indexes.astype('int')
		x_similar_labels = train_files.root.train_label[similar_indexes]

		final_list = np.insert(x_similar_labels, 0, test_label)

		print('writing row: ', j)
		writer.writerow(final_list)

train_files.close()
test_files.close()
