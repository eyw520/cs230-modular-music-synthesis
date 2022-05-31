from keras.callbacks import ModelCheckpoint
from util import get_components, prepare_sequences
from model import create_network


FILEPATH = "weights/weights-5-{epoch:02d}-{loss:.4f}.hdf5"
CURR_EPOCH = 0


def train_network(num_epochs):
    # retrieve components from the midi song library for decomposition.
    notes = get_components()
    pitch_names = sorted(set(item for item in notes))
    n_vocab = len(set(notes))

    # create RNN model with appropriate sequence inputs
    network_input, normalized_input, network_output = prepare_sequences(notes, pitch_names, n_vocab)
    model = create_network(network_input, n_vocab)

    # train model for num_epochs iterations
    train(model, normalized_input, network_output, num_epochs, FILEPATH)


def train(model, network_input, network_output, num_epochs, filepath):
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]
    model.fit(network_input, network_output, epochs=num_epochs, batch_size=64, callbacks=callbacks_list, initial_epoch=CURR_EPOCH)


if __name__ == '__main__':
    train_network(500)
