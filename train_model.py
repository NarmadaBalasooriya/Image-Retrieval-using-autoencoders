# Author: Narmada Balasooriya #
# Based on: https://blog.sicara.com/keras-tutorial-content-based-image-retrieval-convolutional-denoising-autoencoder-dc91450cc511
# ########################### #
# This code creates an autoencoder model using Keras and trains it in batches for the dataset since the whole dataset cannot be loaded into memory
#

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.utils.data_utils import get_file
from keras import optimizers, losses

import numpy as np 
import h5py
import tables
from math import ceil
import matplotlib.pyplot as plt 
from random import shuffle


print('starting the train model')
# the training datafile is in .hdf5 file format
filepath = './dataset/dataset.hdf5'

files = tables.open_file(filepath, mode='r')
# file size = total no.of images in the training data
file_size = files.root.train_data.shape[0]

# defines the batch size
batch_size = 10000

# create batches of batch_size and make a list
batches_list = list(range(int(ceil(float(file_size) / batch_size))))

print('batches list: ', batches_list)

###################################
## Creates the Autoencoder Model ##
###################################
print('Create the autoencoder model')

# Encoder #
input_img = Input(shape=(128, 128, 3)) # input image shape -> width = height = 128, no.of color channels = 3
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same', name='encoder')(x)

# Decoder #
x = UpSampling2D((2,2))(encoded)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

# create the autoencoder model #
autoencoder = Model(input_img, decoded)
# Uses Stochastic gradient descent optimizer
sgd = optimizers.SGD(lr=0.01, clipnorm=1.)

# Loss = Mean Squared Error
autoencoder.compile(optimizer='sgd', loss=losses.mean_squared_error)

print('Autoencoder compilation done')

#################################
# Training the autoencoder mode #
#################################

# Enumerate over the batch list with each batch of size 10,000
for n, i in enumerate(batches_list):
    i_s = i * batch_size
    i_e = min([(i + 1) * batch_size, file_size])

    train_images = files.root.train_data[i_s:i_e] # Get the training images
    train_labels = files.root.train_label[i_s:i_e] # Get the training labels

    x_train = train_images.astype('float32') / 255. # normalize the images
    y_train = train_labels

    noise_factor = 0.5
    #create noise in the images and save it as x_train_noisy
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)

    print('Train the autoencoder from: ', i_s, ' to ', i_e, '\n')
    
    # train the autoencoder with x = x_train_noisy and y = x_train
    autoencoder.fit(x_train_noisy, x_train, batch_size=32, epochs=5)

# save the trained autoencoder
autoencoder.save('autoencoder2.h5')
