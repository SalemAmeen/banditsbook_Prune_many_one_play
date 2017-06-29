from __future__ import print_function
import numpy as np
from numpy import *

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

batch_size = 128
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters1 = 20
nb_filters2 = 50
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 5

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters1, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters2, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
#model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(500))
model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.load_weights('/home/sal/Dropbox/banditsbook/python/model_weight.hdf5')
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
              
n_samples ,_=Y_test.shape
SamplingTesting=500

All_weights=model.get_weights()
FC_weights_3=All_weights[4]       
row,col= shape(FC_weights_3)
SizeWights=row*col
       
score = model.evaluate(X_test, Y_test, verbose=0)
OldAccuracy = score[1]
print('Test score:', score[0])
print('Test accuracy:', score[1])
num_sims=1 # How many times want to play at one time. With Particulr arm a
           # How many other arms or weights want to check to see if they work
           # well with this arm or weigh.
###############################################
horizon=2000 # Playing times

MaxofPrune = 500
###################################### 
#arms=FC_weights_3
arms=np.arange(col)
#arms=np.ravel(arms)
#rewards = [0.0 for i in range(row,col)]
#rewards= [[0 for x in range(col)] for y in range(row)] 
#rewards=np.zeros((row,col))
rewards=np.zeros(col)
AccuracyAftrerPrune=np.zeros((MaxofPrune))
cumulative_rewards = [0.0 for i in range(horizon)]
FC_weights_3Buck=FC_weights_3



for t in range(500):  
    FC_weights_3[:,t]=0
    All_weights[4]=FC_weights_3 
    model.set_weights(All_weights)      
    #print 'Number of pruning = ', t
    score = model.evaluate(X_test, Y_test, verbose=0)
    AccuracyAftrerPrune[t] = score[1]
    print('Test accuracy after pruning:', score[1]) 