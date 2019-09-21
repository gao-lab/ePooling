from generate_data import *
import random
from tqdm import tqdm
import glob

#You need first download the meme file to local. And transfor to the txt file.

def to_PWM(file_path, path):
    data = []
    # #读取
    with open(file_path,encoding='utf-8',) as txtfile:
        line=txtfile.readlines()
        for i,rows in enumerate(line):
            if i in range(11,len(line)-2) :
                #print(rows)
                data.append(rows)
    #print("length",len(data))
    txtfile.close()
    with open(path,"w",) as f:
        for i in range(len(data)):
            f.writelines(data[i])
    f.close()



all_motif_path = '/home/tuxm/Github/ePooling-dev/data/data_generator/All_motif'
motifTxt = glob.glob(all_motif_path + "/*.meme")
output_path = '/rd1/tuxm/ePooling/simulation_data-random/HDF5/simulation'
random.seed(10)
Part_motif = random.sample(motifTxt, 150)
result  =  glob.glob(output_path + "*/*train*")


for file in tqdm(Part_motif):
    name = file.split('/')[-1][0:6] + '.txt'
    i = int(file.split('/')[-1][2:6])
    mkdir(output_path + str(i))
    now_path = output_path + str(i) + '/' + name
    to_PWM(file, now_path)
