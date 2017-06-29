from __future__ import print_function
import numpy as np
from numpy import *

np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import numpy

batch_size = 128
nb_classes = 2
nb_epoch = 12


# number of convolutional filters to use
nb_filters1 = 20
nb_filters2 = 50
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 5

# the data, shuffled and split between train and test sets

X_train = np.load('X_train.npy')
Y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
Y_test = np.load('y_test.npy')
X_deploy = np.load('X_deploy.npy')
y_deploy = np.load('y_deploy.npy')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
#Y_train = np_utils.to_categorical(y_train, nb_classes)
#Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(40, input_dim=57, init='uniform', activation='relu')) # sigmoid, relu, tanh, W_regularizer=l2(0.01)
model.add(Dropout(0.25))
model.add(Dense(1, init='uniform', activation='sigmoid'))
model.load_weights('bestModel.hdf5')
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy']) # rmsprop, adam

#n_samples ,_=Y_test.shape


SamplingTesting=500

All_weights=model.get_weights()
All_weights_BUCKUP = model.get_weights()
FC_weights_3=All_weights[0]
row,col= shape(FC_weights_3)
SizeWights=row*col
score = model.evaluate(X_deploy, y_deploy, verbose=0)
#score = model.evaluate(X_test, Y_test, verbose=0)
OldAccuracy = score[1]
print('Test score:', score[0])
print('Test accuracy:', score[1])
