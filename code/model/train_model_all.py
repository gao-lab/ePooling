import sys
from build_model import *
from data_load import *
import os
import numpy as np
import pickle
import random
import math
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def mkdir(path):
    isexists = os.path.exists(path)
    if not isexists:
        os.makedirs(path)
        return True
    else:
        return False

def train_CNN_model(number_of_kernel,
                    ker_len, input_shape,
                    batch_size,
                    pooling,
                    epoch_num,
                    data_info,
                    modelsave_output_prefix,
                    random_seed,
                    mode = 0,
                    m_trainable = False,
                    m = 1,
                    local_window_size = 5):
    model = keras.models.Sequential()
    model, sgd = build_CNN(model,
                           number_of_kernel,
                           ker_len,
                           input_shape=input_shape,
                           pooling=pooling,
                           mode = mode,
                           m = m,
                           local_window_size= local_window_size,
                           m_trainable = m_trainable)
    # model = init_mask_final(model, init_ker_len_dict,max_ker_len)
    model.compile(loss='binary_crossentropy', optimizer=sgd)

    # set the result path
    output_path = modelsave_output_prefix + "/" + str(data_info)
    mkdir(output_path)
    output_prefix = output_path + "/" \
                                + "model-KernelNum_" + str(number_of_kernel) \
                                + "_random-seed_" + str(random_seed) \
                                + "_batch-size_" + str(batch_size) \
                                + '_kernel-length_' + str(ker_len) \
                                + '_mode_' + str(mode) \
                                + '_m-trainable_' + str(m_trainable) \
                                + '_m_' + str(m)\
                                + '_localwindow_'+str(local_window_size) \
                                + '_pooling_'+str(pooling)

    modelsave_output_filename = output_prefix + "_checkpointer.hdf5"
    history_output_path = output_prefix + '.history'
    prediction_save_path = output_prefix + '.npy'

    # set the checkpoint and earlystop to save the best model
    earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                 patience=10,
                                                 verbose=1)

    checkpointer = keras.callbacks.ModelCheckpoint(filepath=modelsave_output_filename,
                                                   verbose=1,
                                                   save_best_only=True)

    # train the model, and save the history
    history = model.fit(X_train,
                        Y_train,
                        epochs=epoch_num,
                        batch_size=batch_size,
                        validation_split=0.1,
                        verbose=1,
                        callbacks=[checkpointer, earlystopper])

    # load the best weight
    model.load_weights(modelsave_output_filename)

    # get the prediction of the test data set, and save as .npy
    prediction = model.predict(X_test)
    np.save(prediction_save_path, prediction)

    # save the history
    with open(history_output_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    return history, prediction


# read the hyper-parameters
data_path = sys.argv[1]
result_path = sys.argv[2]
data_info = sys.argv[3]
number_of_kernel = int(sys.argv[4])
random_seed = int(sys.argv[5])
m = float(sys.argv[6])
local_window = int(sys.argv[7])
GPU_SET = sys.argv[8]


# set the hyper-parameters
ker_len = 24
batch_size = 32
epoch_num = 20000

np.random.seed(random_seed)
random.seed(random_seed)
tf.set_random_seed(random_seed)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_SET
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

X_test, Y_test, X_train, Y_train = get_data(data_path)


input_shape = (X_train.shape[1], 4)

mode = 0
m_trainable = False
m = 1
History_expectation, prediction_expectation = train_CNN_model(number_of_kernel=number_of_kernel,
                                                ker_len=ker_len,
                                                input_shape=input_shape,
                                                batch_size=batch_size,
                                                pooling='GlobalExpectPooling',
                                                epoch_num=epoch_num,
                                                data_info=data_info,
                                                modelsave_output_prefix=result_path,
                                                random_seed=random_seed,
                                                m = m,
                                                mode = mode, m_trainable=m_trainable,
                                                local_window_size = local_window)

mode = 1
m_trainable = False
m = 1
History_expectation, prediction_expectation = train_CNN_model(number_of_kernel=number_of_kernel,
                                                ker_len=ker_len,
                                                input_shape=input_shape,
                                                batch_size=batch_size,
                                                pooling='GlobalExpectPooling',
                                                epoch_num=epoch_num,
                                                data_info=data_info,
                                                modelsave_output_prefix=result_path,
                                                random_seed=random_seed,
                                                m = m,
                                                mode = mode, m_trainable=m_trainable,
                                                local_window_size = local_window)


mode = 0
m_trainable = True

History_expectation, prediction_expectation = train_CNN_model(number_of_kernel=number_of_kernel,
                                                ker_len=ker_len,
                                                input_shape=input_shape,
                                                batch_size=batch_size,
                                                pooling='GlobalExpectPooling',
                                                epoch_num=epoch_num,
                                                data_info=data_info,
                                                modelsave_output_prefix=result_path,
                                                random_seed=random_seed,
                                                m = m,
                                                mode = mode, m_trainable=m_trainable,
                                                local_window_size = local_window)

mode = 1
m_trainable = True
m = 1

History_expectation, prediction_expectation = train_CNN_model(number_of_kernel=number_of_kernel,
                                                ker_len=ker_len,
                                                input_shape=input_shape,
                                                batch_size=batch_size,
                                                pooling='GlobalExpectPooling',
                                                epoch_num=epoch_num,
                                                data_info=data_info,
                                                modelsave_output_prefix=result_path,
                                                random_seed=random_seed,
                                                m = m,
                                                mode = mode, m_trainable=m_trainable,
                                                local_window_size = local_window)



History_Max, prediction_Max = train_CNN_model(number_of_kernel=number_of_kernel,
                                             ker_len=ker_len,
                                             input_shape=input_shape,
                                             batch_size=batch_size,
                                             pooling='GlobalMax',
                                             epoch_num=epoch_num,
                                             data_info=data_info,
                                             modelsave_output_prefix=result_path,
                                             random_seed=random_seed)

History_Average, prediction_Average = train_CNN_model(number_of_kernel=number_of_kernel,
                                             ker_len=ker_len,
                                             input_shape=input_shape,
                                             batch_size=batch_size,
                                             pooling='GlobalAverage',
                                             epoch_num=epoch_num,
                                             data_info=data_info,
                                             modelsave_output_prefix=result_path,
                                             random_seed=random_seed)










