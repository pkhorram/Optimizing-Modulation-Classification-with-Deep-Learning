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
import utils

# Be careful: each data is a tuple of class label, and an array of IQ sample of shape (2,128)
infile = open('dataset.pickle','rb')
dataset = pickle.load(infile)
infile.close()

#np.random.shuffle(dataset)
dataset = np.reshape(dataset, (dataset.shape[0], -1))


for i in range(len(dataset)):
    dataset[i][1] = np.reshape(dataset[i][1],(dataset[i][1].shape[0]*dataset[i][1].shape[1],-1)) 

classes = []
data = []
for i in range(len(dataset)):
    classes.append(dataset[i][0])   
    data.append(dataset[i][1])

data = np.reshape(np.array(data), (np.array(data).shape[0],np.array(data).shape[1])) 
    
    
label_dict,  digit_label = utils.digtizer(classes)
classes = utils.onehot_encoder(max(digit_label),  digit_label)    

    
    
# percentage of training dataset
train_len = int(len(dataset)*0.8)

train_x = data[:train_len,:]
train_y = classes[:train_len,:]

test_x = data[train_len:,:]
test_y = classes[train_len:,:]


model = Sequential()
model.add(Dense(128, input_shape=(256,), activation='relu'))
model.add(Dense(11, input_shape=(128,), activation='softmax'))

sgd = SGD(lr=0.0001, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
num_epochs = 50

history = model.fit(train_x, train_y, epochs=num_epochs, batch_size=100, validation_data=(test_x, test_y))
with open('./results', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)












