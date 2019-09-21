import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def LoadMotif(path):
    motifTxt = glob.glob(path + "/*.txt")
    motiflist = []
    for file in motifTxt:
        temMotif = np.loadtxt(file)
        #print(temMotif.shape)
        motiflist.append(temMotif.shape[0])
    return motiflist[0]


# You need change the path
DataDirlist = glob.glob("/home/tuxm/Github/ePooling-dev/data/data_generator/Realmotif-random/*")
data_simualtion = pd.read_csv('revise-simulation_data_different_windowsize_20randomeseeds_150case_random.csv')


simulation_to_motif_length = {}
for dataPath in DataDirlist:
    data_set_info = int(dataPath.split('/')[-1][10:])
    motif_length = LoadMotif(dataPath)

    simulation_to_motif_length[data_set_info] = motif_length

data_simualtion['motif_length'] = data_simualtion['data_set'].apply(lambda x: simulation_to_motif_length[x] )


plt.figure(figsize = (10,10))
sns.lineplot(x="local window", y="AUC", hue="motif_length",
             data=data_simualtion[data_simualtion['model_type'] == 'GlobalExpectPooling.npy'], legend = 'full')
plt.legend(loc='lower right')
plt.savefig('simulation_100different_motif_window_size-random-150.pdf', format='pdf', dpi = 1000)


plt.figure(figsize = (12,8))
sns.lineplot(x="motif_length", y="AUC", hue="model_type",
             data=data_simualtion[(data_simualtion['local window'] == 15) | (data_simualtion['model_type'] == 'GlobalMax.npy') ])
plt.legend(['Expectation Pooling', 'Max Pooling'])
plt.savefig('simulation_100different_motif-random-150.pdf', format='pdf', dpi = 1000)


s = pd.Series(list(simulation_to_motif_length.values()))

sns.distplot(s, bins = 17, hist = True, kde = True, norm_hist = False,
            rug = False, vertical = False,
             kde_kws={"color": "k", "lw": 3},
            color = 'b', label = 'distplot', axlabel = 'Motif length')
plt.savefig('distribution_motif_length.pdf', format = 'pdf')