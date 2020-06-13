from cldnn_model import *
from sklearn.metrics import confusion_matrix
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
import os
import pickle
import numpy as np
# Use this code only if you want to generate 20 different models corresponding to 20 SNR values
with open('dataset', 'rb') as file:
    data = pickle.load(file)
accuracies_All = []
confusion_matrices_All = []

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

    models = CLDNN()
    opt = Adam(learning_rate=0.0001)
    models.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    num_epochs = 300

    # Checkpoint for models
    ckpt_folder = "cldnn_models/"
    ckpt_file_path = 'cldnn_model_SNR_{}'.format(key)
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)
    model_ckpt_callback = ModelCheckpoint(filepath=ckpt_folder+ckpt_file_path,monitor='val_loss', mode='min', save_best_only=True)
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, verbose=1, epsilon=1e-4, mode='min')

    history = models.fit(x_train,
                         y_train,
                         epochs=num_epochs,
                         batch_size=128,
                         callbacks = [reduce_lr_loss, model_ckpt_callback],
                         validation_data=(x_val, y_val))
    loss, acc = models.evaluate(x_test, y_test, verbose=2)
    predicted_data = models.predict(x_test)
    accuracies_All.append([acc, key])
    print('accuracy =', acc)
    res = np.argmax(predicted_data, 1)
    y_test_res = np.argmax(y_test, 1)
    results = confusion_matrix((y_test_res+1), (res+1))
    confusion_matrices_All.append([results, key])
    with open('./history/cldnn/SNR_{}_history'.format(key), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

outfile = open('accuracy_results', 'wb')
pickle.dump(accuracies_All, outfile)
outfile.close()

outfile = open('confusion_matrix_results', 'wb')
pickle.dump(confusion_matrices_All, outfile)
outfile.close()
