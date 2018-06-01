# Author: Narmada Balasooriya #
# Based on: machinelearninguru.com/deep_learning/data_preparation/hdf5/hdf5.html
# ########################### #
# This code was used to create a single hdf5 file for the 1.2 million images given in the challenge
# ########################### #

from random import shuffle
import glob
import numpy as np 
import tables
import cv2

file_path = './dataset/dataset.hdf5' # file path for the hdf5 file
img_path = './data/*.jpg' # path for the images

print('initializing')

width = 128 # or 256x256 recommended size is 256*256
height = 128
n_color_channels = 3 # RGB color image
k = 0
j = 0

# file name of an image is landmark_id,key.jpg
# Retrieves the key which is unique for each image
def get_id(name):
	if name:
		img_name = name.split("/data/")
		img_name = img_name[1].split(",")
		img_name = img_name[1].split(".jpg")
		return img_name[0]
	return None

print('starting')

# get the images 
images = glob.glob(img_path)
print('glob done')
labels = []

for img in images:
	#print('geting id')
	# get the image unique id and set it as the label
	label_id = get_id(img)
	labels.append(label_id)
	print(k, ' th label id ', label_id)
	k += 1


# create arrays
train_images = images[0:int(len(images))]
train_labels = labels[0:int(len(labels))]

# image type -> uint8
img_type = tables.UInt8Atom()
# label type -> string
label_type = tables.StringAtom(itemsize=16)
# shape for images
data_shape = (0, width, height, 3)

print('create path')
# create the file at the file path with write mode
h_file = tables.open_file(file_path, mode="w")

print('create datasets')
# Uses EArrays to store the data
train_data = h_file.create_earray(h_file.root, 'train_data', img_type,
	shape = data_shape)
mean_st = h_file.create_earray(h_file.root, 'train_mean', img_type,
	shape = data_shape)

train_label = h_file.create_earray(h_file.root, 'train_label', label_type, shape=(0,))

mean = np.zeros(data_shape[1:], np.float32)

print('start for loop')
# loop over the images and append each image to the EArray

for i in range(len(train_images)):

	img_addr = train_images[i]
	img = cv2.imread(img_addr)

	print('read image')
	if(img == None):
		# if the image is not readable print error message and continue the loop
		print('error image. replaced with zeros')
		continue
	else:

		print('resizing')
		img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
		img_lbl = get_id(img_addr) # get the unique ID

		print('appending images', img_lbl, type(img_lbl), type(img))
		
		train_data.append(img[None]) # append the images
		img_lbl = np.array([img_lbl]) 
		train_label.append(img_lbl) # append the labels
		
		print('append done')
		mean += img / float(len(train_labels))
		print('done round ', i)

print('done the loop')
mean_st.append(mean[None])             
print('close the file')
h_file.close() # close the file
