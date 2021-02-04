import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import MA as microcanonical
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.datasets import mnist

# Variables
input_shape = [None, 28, 28, 1]
number_of_classes = 10

# Load MNIST Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Create Reduced Training Data
# In order to speed up the search process, a reduced sample of 50% of the original training samples are selected randomly; 
# and 10% of this reduced sample is used as the reduced validation set.
np.random.seed(226)
Xtrain = np.empty((int(x_train.shape[0] / 2),28,28,1), dtype='uint8')
Ytrain = np.empty(int(y_train.shape[0] / 2), dtype='uint8')
Xtest = np.empty((int(x_test.shape[0] / 2),28,28,1), dtype='uint8')
Ytest = np.empty((int(y_test.shape[0] / 2)), dtype='uint8')

for i in range(int(x_train.shape[0] / 2)):
    rnd = np.random.randint(0, x_train.shape[0])
    Xtrain[i,:] = x_train[rnd,:]
    Ytrain[i] = y_train[rnd]

for j in range(int(x_test.shape[0] / 2)):
    rnd = np.random.randint(0, y_test.shape[0])
    Xtest[j,:] = x_test[rnd,:]
    Ytest[j] = y_test[rnd]

#Conver output label to one hot vector
Ytrain = to_categorical(Ytrain, number_of_classes)
Ytest = to_categorical(Ytest, number_of_classes)

# 10% of this reduced sample is used as the reduced validation set.
Xtrain, Xvalid, Ytrain, Yvalid = train_test_split(Xtrain, Ytrain, 
                                                test_size = int(Xtrain.shape[0] * 0.1), random_state = 42)

print("Shape of training features: {}".format(Xtrain.shape))
print("Shape of training lables: {}".format(Ytrain.shape))
print("Shape of testing features: {}".format(Xvalid.shape))
print("Shape of testing lables: {}".format(Yvalid.shape))

#Hyper parameters
learning_rate = 0.0001
batch_size = 32

Xtrain = Xtrain.astype('float32')
Xvalid = Xvalid.astype('float32')

# MO (Microcanonical Optimization) Parameters
parameters = {'x_train': Xtrain , 'y_train': Ytrain, 'x_valid': Xvalid, 'y_valid': Yvalid, 'batch_size':batch_size, 'learning_rate':0.0001}

# Start MO Algorithm (MA == MO)
alg = microcanonical.MA(**parameters)
alg.startAlgorithm()

## Outputs: 
 # model_history.txt: The loss and accuracy values (per epoch) of each model produced for training and validation are stored.
 # models.txt: Store iteration number, model_no, #parameters, Flops, train accuracy, validation accuracy and model topology
 # *.json files: Stores information about the topology of solutions on the Pareto front (Keras Model).