
import h5py


def load_data(dataset):
    data = h5py.File(dataset, 'r')
    sequence_code = data['sequences'].value
    label = data['labs'].value
    return [sequence_code, label]


def get_data(data_path):
    data_path = data_path
    test_dataset = data_path + "test.hdf5"
    training_dataset = data_path + "train.hdf5"
    X_test, Y_test = load_data(test_dataset)
    X_train, Y_train = load_data(training_dataset)

    return X_test, Y_test, X_train, Y_train


if __name__ == '__main__':

    pass
