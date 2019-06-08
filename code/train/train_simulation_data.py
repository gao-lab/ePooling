import os
from multiprocessing import Pool
import sys
import glob
import time

def run_data(data_prefix, result_path, data_info, GPU_SET, kernel_number, local_window_size,random_seed,  m = 1):
    """
    :param data_prefix:  data_prefix to get the data
    :param result_path:  the path to save the result
    :param data_info: the data info (name of the data)
    :param GPU_SET:   which GPU to use
    :param kernel_number:  number of kernels
    :param local_window_size:  the window size of the local max pooling
    :param random_seed:  the  random seed
    :param m:  the base
    :return:  model, optimizer
    """
    print()
    cmd = "python ../model/train_model_all.py"
    data_path = data_prefix + data_info + "/"
    set = cmd + " " + data_path + " " +result_path + "  " + data_info + " "+str(kernel_number) + " " + str(random_seed) + " " +str(m) +" " + str(local_window_size)+ " " +GPU_SET
    print(set)
    os.system(set)

if __name__ == '__main__':

    # GPU_SET: which GPU to use
    # start : the start in this running
    # end: the end in this running
    # all these parameters are got from argv
    GPU_SET = sys.argv[1]
    start = int(sys.argv[2])
    end =  int(sys.argv[3])
    # the path of data
    path = "/rd2/lijy/KDD/vCNNFinal/Data/ICSimulation/HDF5/"
    data_prefix = "/rd2/lijy/KDD/vCNNFinal/Data/ICSimulation/HDF5/simu_0"
    result_path = "/rd1/tuxm/ePooling/result/real_simulation_data_all_pooling"
    data_list = ['1', '2', '3']
    start_time =  time.time()
    # pool is max paiallel number of data to run
    pool = Pool(processes  = 6)
    # all hyper-parameter list
    # local window size list
    # random seed list
    # kernel number list
    # m list
    local_window_size_list = [10]
    random_seed_list = range(10, 30)
    kernal_number_list = [128, 64, 32, 16, 8, 4]
    m_list = [1]


    print("start run all models")
    for kernel_number in kernal_number_list:
        for random_seed in random_seed_list:
            for local_window_size in local_window_size_list:
                for data_info in data_list[start:end]:
                    for m in m_list:
                        print(data_info, GPU_SET, local_window_size, random_seed, m)
                        time.sleep(2)
                        pool.apply_async(run_data,(data_prefix, result_path, data_info, GPU_SET, kernel_number, local_window_size, random_seed, m))
    pool.close()
    pool.join()
    print("all model cost ",  time.time() - start_time)