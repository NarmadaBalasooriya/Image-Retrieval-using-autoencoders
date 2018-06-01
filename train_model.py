# Author: Narmada Balasooriya #
# Based on: https://blog.sicara.com/keras-tutorial-content-based-image-retrieval-convolutional-denoising-autoencoder-dc91450cc511
# ########################### #


from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
import numpy as np 
import h5py
import tables
from math import ceil
import matplotlib.pyplot as plt 
from random import shuffle
from keras.utils.data_utils import get_file
from keras import optimizers, losses


#width, height = 224, 224
#batch_size = 100

print('starting the train model')
#filepath = 'E:/1-HIGHER STUDIES/2.2 Kaggle Competition/keras_python/mnist.npz'
filepath = './dataset/dataset.hdf5'
#f = np.load(filepath)

#x_train, y_train = f['x_train'], f['y_train']
#x_test, y_test = f['x_test'], f['y_test']


files = tables.open_file(filepath, mode='r')
file_size = files.root.train_data.shape[0]
batch_size = 10000

print('files opened')
batches_list = list(range(int(ceil(float(file_size) / batch_size))))

print('batches list: ', batches_list)
"""
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

noise_factor = 0.5

x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
"""

print('Create the autoencoder model')
input_img = Input(shape=(128, 128, 3))
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same', name='encoder')(x)

x = UpSampling2D((2,2))(encoded)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
#x = Conv2D(16, (3, 3), activation='relu')(x)
#x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)


autoencoder = Model(input_img, decoded)
sgd = optimizers.SGD(lr=0.01, clipnorm=1.)

autoencoder.compile(optimizer='sgd', loss=losses.mean_squared_error)

print('Autoencoder compilation done')

for n, i in enumerate(batches_list):
    i_s = i * batch_size
    i_e = min([(i + 1) * batch_size, file_size])

    train_images = files.root.train_data[i_s:i_e]
    train_labels = files.root.train_label[i_s:i_e]

    x_train = train_images.astype('float32') / 255.
    #x_train = np.reshape(x_train, (len(x_train), 128, 128, 1))
    y_train = train_labels

    noise_factor = 0.5

    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)

    print('Train the autoencoder from: ', i_s, ' to ', i_e, '\n')
    autoencoder.fit(x_train_noisy, x_train, batch_size=32, epochs=5)

autoencoder.save('autoencoder2.h5')

train_model()

f.close()
