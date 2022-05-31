import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.utils import np_utils

SEQUENCE_LENGTH = 32


def convert_to_float(frac_str):
    """
    Helper Function: convert fractional strings into float values.
    """
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        try:
            leading, num = num.split(' ')
            whole = float(leading)
        except ValueError:
            whole = 0
        frac = float(num) / float(denom)
        return whole - frac if whole < 0 else whole + frac


def get_components():
    """
    Retrieve all the notes, chords, and rests from the midi files in the ./midi_songs directory.
    """
    notes = []

    for file in glob.glob("midi_songs/*.mid"):
        midi = converter.parse(file)
        print("Parsing %s" % file)

        to_parse = None
        try:  # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            to_parse = s2.parts[0].recurse()
        except:  # file has notes in a flat structure
            to_parse = midi.flat.notes

        for element in to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch) + " " + str(element.duration.quarterLength))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder) + " " + str(element.duration.quarterLength))
            # elif isinstance(element, note.Rest):
            #    notes.append("Rest " + str(element.duration.quarterLength))

    n_vocab = len(notes)
    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)
    print("Retrieved {num} components".format(num=n_vocab))

    return notes


def prepare_sequences(notes, pitch_names, n_vocab):
    """
    Prepare the note, rest, and chord sequences for use by the Neural Network.
    """
    # create a dictionary to map pitch & durations to integers
    note_to_int = dict((elem, number) for number, elem in enumerate(pitch_names))
    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - SEQUENCE_LENGTH, 1):
        sequence_in = notes[i:i + SEQUENCE_LENGTH]
        sequence_out = notes[i + SEQUENCE_LENGTH]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)
    # reshape the input into a format compatible with LSTM layers & normalize
    network_input = numpy.reshape(network_input, (n_patterns, SEQUENCE_LENGTH, 1))
    normalized_input = network_input / float(n_vocab)
    network_output = np_utils.to_categorical(network_output)

    return network_input, normalized_input, network_output
