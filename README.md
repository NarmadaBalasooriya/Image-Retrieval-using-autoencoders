# Image Retrieval using autoencoders
This is done as part of the Kaggle competition - Google Landmark Retrieval Challenge 2018(https://www.kaggle.com/c/landmark-retrieval-challenge)

# Requirements
1. Keras with Tensorflow as backend (GPU version) -> Installation https://keras.io/#installation
2. tables -> Installation https://www.pytables.org/usersguide/installation.html
3. h5py -> Installation http://docs.h5py.org/en/latest/build.html
4. matplotlib
5. cv2
6. numpy

# Dataset
The dataset was downloaded when it was available at the begining of the competition.

# Execution
1. Make sure all the requirements are met

2. Make sure to have the dataset ready

3. Run the train_model.py
```
$ python train_model.py
```

4. Compute the Euclidean distance between the query image(test images) and the learned features from the autoencoder to retrieve the simialr images. For this run test_model.py
```
$ python test_model.py
```

# Altering the program
The parameters for the autoencoder can be changes according to own dataset
