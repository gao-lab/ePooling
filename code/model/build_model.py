
from ePooling import *
import keras
from keras import backend as K
import keras.callbacks
from keras.layers import Conv1D, Activation
# build model with different pooling layer


def build_CNN(model_template, number_of_kernel, kernel_length, input_shape,
              pooling="GlobalMax", mode=0, m=1, m_trainable=False, local_window_size=5):
    """
    :param number_of_kernel: the number of kernels
    :param kernel_length:  kernel length
    :param input_shape:  input shape
    :param pooling:  pooling type GlobalMax or GlobalExpect

    :param mode:  0 or 1, 0 reperezent the option structure 1 (calculate the expectation over the average)
    :param m:   the base to calculate the prob
    :param m_trainable:   True or False
    :param local_window_size:  the local window size of the local max pooling before the GlobalExpect
    :return:  model, optimizer
    """

    def relu_advanced(x):
        return K.relu(x, alpha=0.5, max_value=10)

    model_template.add(Conv1D(
        input_shape=input_shape,
        kernel_size=kernel_length,
        filters=number_of_kernel,
        padding='same',
        strides=1))
    model_template.add(Activation(relu_advanced))

    if pooling == 'GlobalMax':
        model_template.add(keras.layers.GlobalMaxPooling1D())
    elif pooling == 'GlobalAverage':
        model_template.add(keras.layers.GlobalAveragePooling1D())
    else:
        # add the Global Expect pooling 1D
        model_template.add(keras.layers.pooling.MaxPooling1D(pool_length=local_window_size,
                                                             stride=None, border_mode='valid'))
        model_template.add(GlobalExpectationPooling1D(mode = mode, m_trainable = m_trainable, m_value = m))

    model_template.add(keras.layers.core.Dense(output_dim=1, name='Dense_l1'))
    model_template.add(keras.layers.Activation("sigmoid"))
    sgd = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    return model_template, sgd


if __name__ == '__main__':

    pass
