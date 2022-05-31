import matplotlib.pyplot as plt
import numpy as np
import glob
import pickle


def save_training_data():
    training_data = None
    for weight in glob.glob("weights/*.hdf5"):
        split = weight.split("-")
        it, loss = int(split[2]), float(split[3][:6])
        row = np.asarray([it, loss]).reshape(1, 2)
        if training_data is None:
            training_data = row
        else:
            training_data = np.concatenate((training_data, row), axis=0)
    training_data = training_data[np.argsort(training_data[:, 0])]
    with open('data/training_data', 'wb') as filepath:
        pickle.dump(training_data, filepath)


def update_training_data():
    with open('data/training_data', 'rb') as filepath:
        training_data = pickle.load(filepath)
    for weight in glob.glob("weights/*.hdf5"):
        split = weight.split("-")
        it, loss = int(split[2]), float(split[3][:6])
        row = np.asarray([it, loss]).reshape(1, 2)
        if it not in training_data[:,0]:
            training_data = np.concatenate((training_data, row), axis=0)
    training_data = training_data[np.argsort(training_data[:, 0])]
    with open('data/training_data', 'wb') as filepath:
        pickle.dump(training_data, filepath)


def save_to_txt():
    with open('data/training_data', 'rb') as filepath:
        training_data = pickle.load(filepath)
    np.savetxt("data/training_data_txt", training_data, fmt="%d %f")


def plot_data():
    with open('data/training_data', 'rb') as filepath:
        training_data = pickle.load(filepath)
    plt.plot(training_data[:, 0], training_data[:, 1])
    plt.show()


if __name__ == '__main__':
    save_training_data()
    # update_training_data()
    save_to_txt()
    plot_data()

