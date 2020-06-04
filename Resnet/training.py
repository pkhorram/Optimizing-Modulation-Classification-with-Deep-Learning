#import numpy as np
#import pickle
#import tensorflow  as tf
from tensorflow  import  keras
from utils import *
from model import *
from keras.datasets import cifar10
from keras.utils import np_utils
from keras import metrics
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, UpSampling2D, Dropout
from keras import metrics
from sklearn.metrics import confusion_matrix
from keras.losses import categorical_crossentropy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import SGD, Adam
import pickle
import matplotlib.pyplot as plt
import numpy as np 

import pickle 
with open('dataset', 'rb') as file:
    data = pickle.load(file)
confusion_matrices_All = []
accuracies_All = []

for key in data.keys():
    dataset = []
    labels = []

    for values in data[key]:
        labels.append(values[0])
        dataset.append(values[1]) 

        
    print('Starting training for SNR:', key)
    
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
    # models = new_CNN()
    # opt = Adam(learning_rate=0.0001)
    # models.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    #models.summary()

    #models = CLDNN()
    #opt = Adam(learning_rate=0.0001)
    #models.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    #models.summary()

    #inputs = keras.Input(shape = (2,128,1))
    xx_shape = (2,128,1)
    models = resnet(xx_shape)
    opt = Adam(learning_rate=0.001)
    models.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    #models.summary()
    
    num_epochs = 150

#     earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_accuracy')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, verbose=1, epsilon=1e-4, mode='min')
    print(x_val.shape)
    
    history = models.fit(x_train,
                         y_train,
                         epochs=num_epochs,
                         batch_size = 64,
                         callbacks = [reduce_lr_loss,mcp_save],
                         validation_data=(x_val, y_val))
    loss, acc = models.evaluate(x_test,y_test, verbose=2)
    predicted_data = models.predict(x_test)
    accuracies_All.append([acc,key])
    
    res = np.argmax(predicted_data,1)
    y_test_res = np.argmax(y_test,1)
    results = confusion_matrix((y_test_res+1),(res+1))
    confusion_matrices_All.append([results,key])
    
    with open('./history/SNR_{}_history'.format(key), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

outfile = open('Accuracy resutls','wb')
pickle.dump(accuracies_All,outfile)
outfile.close()

outfile = open('confusion_matrix_results','wb')
pickle.dump(confusion_matrices_All,outfile)
outfile.close()
