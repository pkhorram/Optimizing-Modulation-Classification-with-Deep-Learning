from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, LSTM, Reshape
from keras import layers


def CLDNN():

    model = Sequential()
    model.add(Conv2D(256, (1, 3), activation='relu', padding='same', init='glorot_uniform', input_shape=(2, 128, 1)))
    model.add(MaxPooling2D(pool_size=(1, 2), padding='valid',  data_format=None))
    model.add(layers.Dropout(0.3))

    model.add(Conv2D(256, (2, 3), activation='relu', padding='same', init='glorot_uniform'))
    model.add(MaxPooling2D(pool_size=(1, 2), padding='valid',  data_format=None))
    model.add(layers.Dropout(0.3))

    model.add(Conv2D(80, (1, 3), activation='relu', padding='same', init='glorot_uniform'))
    model.add(MaxPooling2D(pool_size=(1, 2), padding='valid',  data_format=None))
    model.add(layers.Dropout(0.3))

    model.add(Conv2D(80, (1, 3), activation='relu', padding='same', init='glorot_uniform'))
    model.add(MaxPooling2D(pool_size=(1, 2), padding='valid',  data_format=None))
    model.add(layers.Dropout(0.3))

    model.add(Reshape((2, 640)))

    model.add(LSTM(50, activation='relu'))
    model.add(layers.Dropout(0.3))

    model.add(Dense(128, activation='relu', init='he_normal'))
    model.add(layers.Dropout(0.3))

    model.add(Dense(11, activation='softmax', init='he_normal'))

    return model
