

X_train = np.load('/Users/salemameen/Desktop/banditsbook/python_iris/iris/X_train.npy')
Y_train = np.load('/Users/salemameen/Desktop/banditsbook/python_iris/iris/y_train.npy')
X_test = np.load('/Users/salemameen/Desktop/banditsbook/python_iris/iris/X_test.npy')
Y_test = np.load('/Users/salemameen/Desktop/banditsbook/python_iris/iris/y_test.npy')
X_deploy = np.load('/Users/salemameen/Desktop/banditsbook/python_iris/iris/X_deploy.npy')
y_deploy = np.load('/Users/salemameen/Desktop/banditsbook/python_iris/iris/y_deploy.npy')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
#Y_train = np_utils.to_categorical(y_train, nb_classes)
#Y_test = np_utils.to_categorical(y_test, nb_classes)



import seaborn as sns
import numpy as np
import numpy
from numpy import *

from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2

from keras.utils import np_utils

#
labelsVal = np_utils.to_categorical(Y_test)                                              


#labelsTrain = np_utils.to_categorical(Y_train)
labelsTest = np_utils.to_categorical(y_deploy)                                              
model = Sequential()
model.add(Dense(16,
                input_shape=(4,), 
                activation="relu",
                W_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(3, activation="softmax"))

model.load_weights('/Users/salemameen/Desktop/banditsbook/python_iris/IrisModelbest.hdf5')
model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'],
              optimizer='adam')
# Actual modelling


              
#n_samples ,_=Y_test.shape


SamplingTesting=500

All_weights=model.get_weights()
All_weights_BUCKUP = model.get_weights()
FC_weights_3=All_weights[0]
row,col= shape(FC_weights_3)
SizeWights=row*col
score = model.evaluate(X_test, labelsVal, verbose=0)   
#score = model.evaluate(X_test, Y_test, verbose=0)
OldAccuracy = score[1]
print('Test score:', score[0])
print('Test accuracy:', score[1])
