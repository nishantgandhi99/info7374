import tensorflow as tf
import multiprocessing as mp
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import CSVLogger



def get_model(neurons_in_layer, number_of_layers=2, lr=0.1, activation_function='relu', dropout_rates=0.2):
    """
    Design the model with Multilevel Perceptrons
    :return:
    """
    model = Sequential()

    model.add(Dense(neurons_in_layer, activation=activation_function, input_shape=(3072,)))
    model.add(Dropout(dropout_rates))
    for i in range(number_of_layers-1):
        model.add(Dense(neurons_in_layer, activation=activation_function))
        model.add(Dropout(dropout_rates))

    model.add(Dense(num_classes, activation='softmax'))

    sgd = SGD(lr=lr, decay=1e-6, nesterov=True)

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    return model


def get_preprocessed_data(x_train, x_test, y_train, y_test):
    """
    Preprocess images. convert them into vector.
    :return:
    """
    x_train = x_train.reshape(50000, 3072)
    x_test = x_test.reshape(10000, 3072)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    return x_train, x_test, y_train, y_test


def run(epochs, batch_size, neurons_in_layer, number_of_layers, lr, activation_function, dropout_rates,
        x_train, x_test, y_train, y_test, callbacks=[]):
    x_train, x_test, y_train, y_test = get_preprocessed_data(x_train, x_test, y_train, y_test)
    model = get_model(neurons_in_layer, number_of_layers, lr, activation_function, dropout_rates)
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        callbacks=callbacks)

    scores = model.evaluate(x_test, y_test, verbose=0)
    print(scores)

if __name__ == '__main__':

    num_classes = 10

    if K.backend() == 'tensorflow':
        K.set_image_dim_ordering("th")

    core_num = mp.cpu_count()
    config = tf.ConfigProto(
        inter_op_parallelism_threads=core_num,
        intra_op_parallelism_threads=core_num)

    sess = tf.Session(config=config)

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Declare variables
    epochs = 5
    batch_size = 32

    neurons_in_layer = 512
    number_of_layers = 2
    lr = 0.1
    activation_function = 'relu'
    dropout_rates = 0.2

    csv_logger = CSVLogger('training.log')
    callbacks = [csv_logger]
    run(epochs, batch_size, neurons_in_layer, number_of_layers, lr, activation_function, dropout_rates,
        x_train, x_test, y_train, y_test, callbacks)





