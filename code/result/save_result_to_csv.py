# import all packages
import glob
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
#from tqdm import tqdm
import sklearn
from sklearn.metrics import roc_auc_score, auc, roc_curve
import seaborn as sns
import pickle

def get_label(rec):
    data = h5py.File(rec, 'r')
    sequence_code = data['sequences'].value
    label = data['labs'].value
    return label

def AUC(label, pred):
    roc_auc_score(label, pred)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(label, pred)
    roc_auc = sklearn.metrics.auc(false_positive_rate, true_positive_rate)
    return roc_auc


def get_model_result(result_path, data_path):
    path = result_path
    result_list = glob.glob(path + '*/*.npy')

    auc = []
    random_seed = []
    local_window = []
    model_type_list = []
    data_info_list = []
    m_list = []
    mode_list = []
    trainable_list = []
    kernel_number_list = []

    for rec in tqdm(result_list):
        data_info = rec.split("/")[-2]
        label_path = data_path + str(data_info) + '/test.hdf5'
        label = get_label(label_path)
        label = np.array(label)
        pred = np.load(rec)

        model_type_list.extend([rec.split("_")[-1]])
        local_window.extend([rec.split("_")[-3]])
        m_list.extend([rec.split("_")[-5]])
        trainable_list.extend([rec.split("_")[-7]])
        mode_list.extend([rec.split("_")[-9]])
        random_seed.extend([rec.split("_")[-15]])
        kernel_number_list.extend([rec.split("_")[-17]])

        auc.extend([AUC(label, pred)])
        data_info_list.extend([data_info])
    return auc, random_seed, local_window, m_list, mode_list, trainable_list, model_type_list, data_info_list, kernel_number_list


result_path = "/rd1/tuxm/ePooling/result/real_chip_kernel_128_head_30/"
data_path = "/rd2/lijy/KDD/vCNNFinal/Data/ChIPSeqData/HDF5/"
# initial the list
auc = []
random_seed = []
local_window = []
model_type_list = []
data_info_list = []
m_list = []
mode_list = []
trainable_list = []
kernel_number_list = []
data_info_list = []
# calculate all thing we need
now_auc, now_random_seed, now_local_window, now_m_list, now_mode_list, now_trainable_list,  now_model_type_list, now_data_info_list, now_kernel_number_list = get_model_result(result_path, data_path)

auc.extend(now_auc)
random_seed.extend(now_random_seed)
local_window.extend(now_local_window)
m_list.extend(now_m_list)
mode_list.extend(now_mode_list)
trainable_list.extend(now_trainable_list)
model_type_list.extend(now_model_type_list)
data_info_list.extend(now_data_info_list)
kernel_number_list.extend(now_kernel_number_list)


dic = {
    'data_set': data_info_list,
    'AUC': auc,
    'random seed': random_seed,
    'mode': mode_list,
    'trainable': trainable_list,
    'local window': local_window,
    'kernel_number': kernel_number_list,
    'm': m_list,
    'model_type': model_type_list
}
# list to DataFrame
data_all = pd.DataFrame(dic)
data_all['random seed'] = data_all['random seed'].astype(int)
data_all['local window'] = data_all['local window'].astype(int)
data_all['kernel_number'] = data_all['kernel_number'].astype(int)
data_all['m'] = data_all['m'].astype(float)
data_all['AUC'] = data_all['AUC'].astype(float)

# save the result to csv file
data_all.to_csv('simulation_result_kernel.csv', index=False)