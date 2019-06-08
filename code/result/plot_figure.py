import glob
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
from tqdm import tqdm
import sklearn
from sklearn.metrics import roc_auc_score, auc, roc_curve
#import seaborn as sns
import pickle


data_all = pd.read_csv('all_kernel_8_16.csv')

random_seed_list = [4, 5, 6, 7]
mode_list = [0]
trainable_list = [False]
local_window_list = [10]
kernel_number_list = [8, 16]
# plt.figure(figsize = (20, 16))
for local_window in local_window_list:
    for kernel_number in kernel_number_list:
        for trainable in trainable_list:
            for random_seed in random_seed_list:
                print('random seed = ', random_seed, ' kernel_number = ', kernel_number, ' trainable=', trainable,
                      'local_window=', local_window)

                data_Soft = get_model_AUC(data_all, local_window=local_window, m=1, mode=0,
                                          model_type='GlobalExpectPooling.npy', random_seed=random_seed,
                                          trainable=trainable, kernel_number=kernel_number)
                data_Max = data_all[(data_all['local window'] == 5) & (data_all['model_type'] == 'GlobalMax.npy') & (
                            data_all['random seed'] == random_seed) & (data_all['kernel_number'] == kernel_number)]
                x = data_Max['AUC']
                y = data_Soft['AUC']
                x = np.array(x)
                y = np.array(y)
                print('the mean of the ePooling:', np.mean(y))
                print('the mean of the MaxPooling:', np.mean(x))
                plt.figure(figsize=(8, 6))
                plt.plot([0.4, 1], [0.4, 1], 'r--')

                print(x.shape, y.shape)

                plt.scatter(x, y, s=1)
                plt.xlim([0.45, 1])
                plt.ylim([0.45, 1])
                plt.xlabel('Max Pooling', fontsize=20)
                plt.ylabel('Expectation Pooling', fontsize=20)
                plt.figure(figsize=(8, 6))

                z = y - x
                z1 = z[z > 0]
                z2 = z[z < 0]
                plt.hist(z1, bins=50, color='r', label=' Expectation pooling better than Maxpooling : %d' % z1.shape[0])
                plt.hist(z2, bins=50, color='gray',
                         label='Expectation pooling worse than Maxpooling : %d' % z2.shape[0])
                plt.rc('legend', **{'fontsize': 12})
                plt.legend()
                plt.ylabel('number of dataset', fontsize=20)
                plt.xlabel('AUC difference', fontsize=20)
                plt.show()
# plt.savefig('all_real_data_diff_distribution_Soft.eps', format='eps', dpi=1000)

# plt.show()

plt.show()