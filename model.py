from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Bidirectional
from keras.layers import BatchNormalization as BatchNorm


WEIGHTS_PATH = "weights/weights-5-500-0.0581.hdf5"


def create_network(network_input, n_vocab):
    """ create the structure of the notes neural network """
    model = Sequential()
    model.add(Bidirectional(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    )))
    model.add(Bidirectional(LSTM(512, return_sequences=True, recurrent_dropout=0.3,)))
    model.add(Bidirectional(LSTM(512)))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    if WEIGHTS_PATH is not None:
        model.build(input_shape=network_input.shape)
        model.load_weights(WEIGHTS_PATH)

    return model
