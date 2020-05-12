from keras.datasets import cifar10
from keras.utils import np_utils
from keras import metrics
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras import metrics
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD
import pickle
import matplotlib.pyplot as plt
import numpy as np 
from keras.preprocessing.image import ImageDataGenerator
from keras import layers


def new_CNN():
    model = Sequential()
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', input_shape=(2,128,1)))
    model.add(MaxPooling2D(pool_size=(1, 2), padding='valid', data_format=None))
    model.add(layers.Dropout(.3))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(1, 2), padding='valid', data_format=None))
    model.add(layers.Dropout(.3))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(1, 2), padding='valid', data_format=None))
    model.add(layers.Dropout(.3))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(1, 2), padding='valid', data_format=None))
    model.add(layers.Dropout(.3))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dense(11))
    
    return model

    
    