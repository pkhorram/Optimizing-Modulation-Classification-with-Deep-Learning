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
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import SGD, Adam
import pickle
import matplotlib.pyplot as plt
import numpy as np 

import pickle 
with open('dataset', 'rb') as file:
    data = pickle.load(file)


       
    
c = 0
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
    models = Robust_CNN()
#     models = CLDNN()    
#     models = resnet(x_train)
        
    
    opt = Adam(learning_rate=0.0001)
    models.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    
    num_epochs = 100

    #earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_accuracy')
    #reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, epsilon=1e-4, mode='min')
    #EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    
    history = models.fit(x_train,
                        y_train,
                        epochs=num_epochs,
                        batch_size=10,
                        callbacks = [mcp_save],
                        validation_data=(x_val, y_val))
    
    
    models.load_weights(".mdl_wts.hdf5") 
    loss, acc = models.evaluate(x_test,y_test, verbose=2)
    accuracies_All.append([acc,key])   
    
    prediction = models.predict(x_test)
    labels_pred =np.argmax(prediction, axis = 1)
    
    
    with open('./history/SNR_{}_history.pkl'.format(key), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    
    DICTED = {'1':labels_pred,'2':y_test}
    with open('./history/SNR_{}_prediction.pkl'.format(key), 'wb') as file_pi:
        pickle.dump(DICTED, file_pi)
        
    with open('./SNR_accuracies.pkl', 'wb') as file_pi:
        pickle.dump(accuracies_All, file_pi)
    
        

       
