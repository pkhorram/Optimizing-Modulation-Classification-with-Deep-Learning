import numpy as np
import pickle 
from utils import *
from model import *
from keras.datasets import cifar10
from keras.utils import np_utils
from keras import metrics
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, UpSampling2D, Dropout
from keras import metrics
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD
import pickle
import matplotlib.pyplot as plt
import numpy as np 
from keras.preprocessing.image import ImageDataGenerator

with open("./dataset", "rb") as p:
    data = pickle.load(p)

for key in data.keys():
    dataset = []
    labels = []
    for values in data[key]:
        labels.append(values[0])
        dataset.append(values[1]) 
        
    #print('Starting training for SNR:', key)
    
    N = len(dataset)
    shuffled_indeces = np.random.permutation(range(N))
    new_dataset = np.array(dataset)[shuffled_indeces,:,:]
    new_labels = np.array(labels)[shuffled_indeces,:]
    
    num_train = int(0.8*N)
    
    x_train = new_dataset[:num_train,:,:]
    y_train = new_labels[:num_train,:]
    
    num_val = int(0.1*len(x_train))
    
    x_val = x_train[:num_val,:,:]
    x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],x_val.shape[2], -1)
    y_val = y_train[:num_val,:]
    
    x_train = x_train[num_val:,:,:]
    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2], -1)
    y_train = y_train[num_val:,:]
    
    x_test = new_dataset[num_train:,:,:]
    x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2], -1)
    y_test = new_labels[num_train:,:] 
    
    #choose model by un-commenting only one of the three:
    models = new_CNN()
    sgd = SGD(lr=0.001, momentum=0.9)
    models.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    num_epochs = 50
    history = models.fit(x_train, y_train, epochs=num_epochs, batch_size=100, validation_data=(x_test, y_test))

    
    
    