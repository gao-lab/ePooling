import numpy as np
import pandas as pd
import h5py
import glob
import os

"""
transform the sequence to matrix
"""


"""
sequence to matrix
"""
def seq_to_matrix(seq,seq_matrix,seq_order):
    '''
    change target 3D tensor according to sequence and order
    :param seq: DNA sequence
    :param seq_matrix
    :param seq_order
    :return:
    '''
    for i in range(len(seq)):
        if((seq[i]=='A')|(seq[i]=='a')):
            seq_matrix[seq_order,i,0]=1
        if((seq[i]=='C')|(seq[i]=='c')):
            seq_matrix[seq_order,i,1]=1
        if((seq[i]=='G')|(seq[i]=='g')):
            seq_matrix[seq_order,i,2]=1
        if((seq[i]=='T')|(seq[i]=='t')):
            seq_matrix[seq_order,i,3]=1
    return seq_matrix

def genarate_matrix_for_train(seq_shape,seq_series):
    """
    genarate matrix for train
    :param shape: (seq number, sequence_length, 4)
    :param seq_series: dataframe of all sequences
    :return:seq
    """
    seq_matrix = np.zeros(seq_shape)
    for i in range(seq_series.shape[0]):
        seq_tem = seq_series[i]
        seq_matrix = seq_to_matrix(seq_tem, seq_matrix, i)
    return seq_matrix


def mkdir(path):
    """
    make dictionary
    :param path
    :return:
    """
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return False


def generate_dataset_matrix(file_path):
    """
    generate matrix of the data set(the path)
    :param file_path:
    :return:
    """
    filenames = glob.glob(file_path+"/*.data")
    for allFileFa in filenames:
        AllTem = allFileFa.split("/")[-1].split(".")[0]

        output_dir = allFileFa.split(AllTem)[0].replace("motif_discovery", "HDF5")
        # positive sample
        SeqLen = 101
        ChipSeqlFileFa = pd.read_csv(allFileFa, sep=' ', header=None, index_col=None)
        seq_series = np.asarray(ChipSeqlFileFa.ix[:, 1])
        seq_name = np.asarray(ChipSeqlFileFa.ix[:, 0]).astype("string")
        seq_matrix_out = genarate_matrix_for_train((seq_series.shape[0], SeqLen, 4), seq_series)
        seq_label_out = np.asarray(ChipSeqlFileFa.ix[:, 2])
        mkdir(output_dir)
        f = h5py.File(output_dir + AllTem +".hdf5")
        f.create_dataset("sequences",data=seq_matrix_out)
        f.create_dataset("labs",data=seq_label_out)
        f.create_dataset("seq_idx",data=seq_name)
        f.close()
        print(output_dir)



if __name__ == '__main__':
    base = {0:"A",1:"C",2:"G",3:"T"}
    allFileFaList = glob.glob("./Data/ChIPSeqData/motif_discovery/wgEncodeAwg*")

    for FilePath in allFileFaList:
        generate_dataset_matrix(FilePath)