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


infile = open('dataset.pickle','rb')
dataset = pickle.load(infile)
infile.close()

