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


with open('new_model_SNR_test_samples', 'rb') as file:
    test_data = pickle.load(file)

with open('combined_SNR_data', 'rb') as file:
    train_data = pickle.load(file)
       
    
train_data = train_data['combined']    

dataset = []
labels = []
    

for values in train_data:
    labels.append(values[0])
    dataset.append(values[1])

    
N = len(dataset)

shuffled_indeces = np.random.permutation(range(N))
new_dataset = np.array(dataset)[shuffled_indeces,:,:]
new_labels = np.array(labels)[shuffled_indeces,:]

num_train = int(0.9*N)
x_train = new_dataset[:num_train,:,:]
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2], -1)
y_train = new_labels[:num_train,:]

x_val = new_dataset[num_train:,:,:]
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],x_val.shape[2], -1)
y_val = new_labels[num_train:,:]


    
# >>>>>>>>>>>>>>>>>>>>> choose a model by un-commenting only one of the three <<<<<<<<<<<<<<<<<<<<<<<<<<<
xx_shape = (2,128,1)
#models = Robust_CNN()
#models = CLDNN()    
models = resnet(xx_shape)
        
    
opt = Adam(learning_rate=0.0001)
models.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


num_epochs = 100

#earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_accuracy',save_weights_only=True)
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, verbose=1, epsilon=1e-4, mode='min')
#EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')

history = models.fit(x_train,
                    y_train,
                    epochs=num_epochs,
                    batch_size=128,
                    callbacks = [reduce_lr_loss,mcp_save],
                    validation_data=(x_val, y_val))


models.load_weights(".mdl_wts.hdf5") 

accuracies_All = []
confusion_matrices_All = []

for keys in test_data.keys():

    x_test = []
    y_test = []

    for values in test_data[keys]:
        y_test.append(values[0])
        x_test.append(values[1])

    y_test = np.array(y_test)
    x_test = np.array(x_test)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], -1)

    oss, acc = models.evaluate(x_test, y_test, verbose=2)
    accuracies_All.append([acc, keys])
    print('accuracy =', acc)

    prediction = models.predict(x_test)

    res = np.argmax(prediction, 1)
    y_test_res = np.argmax(y_test, 1)
    results = confusion_matrix((y_test_res + 1), (res + 1))
    confusion_matrices_All.append([results, keys])
    


with open('./model_history.pkl', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)


outfile = open('confusion_matrix_results-B', 'wb')
pickle.dump(confusion_matrices_All, outfile)
outfile.close()

outfile = open('accuracy_results-B', 'wb')
pickle.dump(accuracies_All, outfile)
outfile.close()
