from keras.datasets import cifar10
from keras.utils import np_utils
from keras import metrics
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, LSTM, merge
from keras import metrics
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD
import pickle
import matplotlib.pyplot as plt
import numpy as np 
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras.callbacks import EarlyStopping


def Robust_CNN():
    
    model = Sequential()
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', init='glorot_uniform', input_shape=(2,128,1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 2), padding='valid',  data_format=None))
    model.add(layers.Dropout(.3))
    model.add(Conv2D(128, (3, 3), activation='relu', init='glorot_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 2), padding='valid', data_format=None))
    model.add(layers.Dropout(.3))
    model.add(Conv2D(64, (3, 3), activation='relu', init='glorot_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 2), padding='valid', data_format=None))
    model.add(layers.Dropout(.3))
    model.add(Conv2D(64, (3, 3), activation='relu', init='glorot_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 2), padding='valid', data_format=None))
    model.add(layers.Dropout(.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', init='he_normal'))
    model.add(BatchNormalization())
    model.add(Dense(11, activation='softmax', init='he_normal'))
    
    return model



def CLDNN():

    model = Sequential()
    model.add(Conv2D(256, (1, 3), activation='relu', padding='same', init='glorot_uniform', input_shape=(2, 128, 1)))
    model.add(Conv2D(256, (2, 3), activation='relu', padding='same', init='glorot_uniform'))
    model.add(Conv2D(80, (1, 3), activation='relu', padding='same', init='glorot_uniform'))
    model.add(Conv2D(80, (1, 3), activation='relu', padding='same', init='glorot_uniform'))
    model.add(LSTM(50, activation='tanh'))
    model.add(Dense(128, activation='relu', init='he_normal'))
    model.add(Dense(11, activation='softmax', init='he_normal'))

    return model


def resnet(x):
    y = Conv2D(256, (1, 3), activation='relu', padding='same', init='glorot_uniform')(x)
    y = Conv2D(256, (2, 3), activation='relu', padding='same', init='glorot_uniform')(y)
    z = merge([x,y], mode='sum')
    z = Conv2D(80, (1, 3), activation='relu', padding='same', init='glorot_uniform')(z)
    z = Conv2D(80, (1, 3), activation='relu', padding='same', init='glorot_uniform')(z)
    z = Dense(128, activation='relu', init='he_normal')(z)
    z = Dense(11, activation='softmax', init='he_normal')(z)
    return z

# def densenet(x):
#     y = Conv2D(256, (1, 3), activation='relu', padding='same', init='glorot_uniform')(x)
#     y = Conv2D(256, (2, 3), activation='relu', padding='same', init='glorot_uniform')(y)
