import glob
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
from tqdm import tqdm
import pickle
import sklearn
from sklearn.metrics import roc_auc_score, auc, roc_curve
import seaborn as sns


def plot_history_loss_max():
    model_type = 'Maxpooling'
    path = '/home/tuxm/Pycharm/result/long_short_motif_result_loss/'
    result_list = glob.glob(path + '*/*' + model_type + '*history')
    sum_loss = np.zeros((100,))
    sum_val_loss = np.zeros((100,))
    for path in result_list:
        with open(path, mode="rb") as f:
            history = pickle.load(f)
        # print(len(history['val_loss']))
        plt.plot(history['loss'], color='lightgreen', alpha=0.2)
        plt.plot(history['val_loss'], color='lavenderblush')
        sum_loss = np.array(history['loss']) + sum_loss
        sum_val_loss = np.array(history['val_loss']) + sum_val_loss
    sum_loss = sum_loss / (len(result_list))
    sum_val_loss = sum_val_loss / (len(result_list))
    print(sum_loss.shape)
    plt.plot(sum_loss, color='lawngreen', alpha=0.7, linewidth=2, label=model_type + ' average train loss')
    plt.plot(sum_val_loss, color='hotpink', alpha=0.7, linewidth=2, label=model_type + ' average test loss')


def plot_history_loss_Soft():
    model_type = 'Softpooling'
    path = '/home/tuxm/Pycharm/result/long_short_motif_result_loss/'
    result_list = glob.glob(path + '*/*' + model_type + '*history')
    sum_loss = np.zeros((100,))
    sum_val_loss = np.zeros((100,))
    for path in result_list:
        with open(path, mode="rb") as f:
            history = pickle.load(f)
        # print(len(history['val_loss']))
        plt.plot(history['loss'], color='paleturquoise', alpha=0.2)
        plt.plot(history['val_loss'], color='lemonchiffon')
        sum_loss = np.array(history['loss']) + sum_loss
        sum_val_loss = np.array(history['val_loss']) + sum_val_loss
    sum_loss = sum_loss / (len(result_list))
    sum_val_loss = sum_val_loss / (len(result_list))
    print(sum_loss.shape)
    plt.plot(sum_loss, color='deepskyblue', alpha=0.7, linewidth=2, label=model_type + ' average train loss')
    plt.plot(sum_val_loss, color='gold', alpha=0.7, linewidth=2, label=model_type + ' average test loss')


def plot_history_loss_average():
    model_type = 'Averagepooling'
    path = '/home/tuxm/Pycharm/result/long_short_motif_result_loss/'
    result_list = glob.glob(path + '*/*' + model_type + '*history')
    sum_loss = np.zeros((100,))
    sum_val_loss = np.zeros((100,))
    for path in result_list:
        with open(path, mode="rb") as f:
            history = pickle.load(f)
        # print(len(history['val_loss']))
        plt.plot(history['loss'], color='darkgrey', alpha=0.2)
        plt.plot(history['val_loss'], color='linen')
        sum_loss = np.array(history['loss']) + sum_loss
        sum_val_loss = np.array(history['val_loss']) + sum_val_loss
    sum_loss = sum_loss / (len(result_list))
    sum_val_loss = sum_val_loss / (len(result_list))
    print(sum_loss.shape)
    plt.plot(sum_loss, color='black', alpha=0.7, linewidth=2, label=model_type + ' average train loss')
    plt.plot(sum_val_loss, color='r', alpha=0.7, linewidth=2, label=model_type + ' average test loss')



fig, ax = plt.subplots(figsize=(15, 10))
plot_history_loss_max()
plot_history_loss_Soft()
plot_history_loss_average()
plt.rc('legend',**{'fontsize':18})
plt.legend(loc='best')
plt.ylabel('loss',  fontsize=20)
plt.xlabel('epoch',  fontsize=20)
fig.savefig("loss_history.eps", format = 'eps', dpi=1000)


