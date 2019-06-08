# Expectation pooling: An effective and interpretable method of pooling for predicting DNA-protein binding
This repo contains all core code in our paper [Expectation pooling: An effective and interpretable method of
pooling for predicting DNA-protein binding](https://www.biorxiv.org/content/10.1101/658427v1)

### Motivation
Convolutional neural networks (CNNs) have outperformed conventional methods in modeling the sequence specificity of DNA-protein binding. While previous studies have built a connection between CNNs and probabilistic models, simple models of CNNs cannot achieve sufficient accuracy on this problem. Recently, some methods of neural networks have increased performance using complex neural networks whose results cannot be directly interpreted. However, it is difficult to combine probabilistic models and CNNs effectively to improve DNA-protein binding predictions.

### Result
In this paper, we present a novel global pooling method: expectation pooling for predicting DNA-protein binding. Our pooling method stems naturally from the EM algorithm, and its benefits can be interpreted both statistically and via deep learning theory. Through experiments, we demonstrate that our pooling method improves the prediction performance DNA-protein binding. Our interpretable pooling method combines probabilistic ideas with global pooling by taking the expectations of inputs without increasing the number of parameters. We also analyze the hyperparameters in our method and propose optional structures to help fit different datasets. We explore how to effectively utilize these novel pooling methods and show that combining statistical methods with deep learning is highly beneficial, which is promising and meaningful for future studies in this field.

### File Structure
```
ePooling
│   README.md    
│
└───Code
│   │
│   └───moodel
│   │   │   build_model.py
│   │   │   data_load.py
│   │   │   ePooling.py
│   │   │   train_model.py
│   │   │   train_model_all.py
│   │
│   └───result
│   │   │   plot_figure.py
│   │   │   plot_history.py
│   │   │   save_result_to_csv.py
│   │   
│   └───train
│       │   train_chip_data.py
│       │   train_simulation_data.py   
│   
└───Demo
│   │   demo.ipynb
│   
└───Data
    │   data_process.py
    │   get_data.py
    │   readme.md
  
```

### Demo
We provide a [jupyter notebook](https://github.com/gao-lab/ePooling-dev/blob/master/demo/demo.ipynb) to help you use the new Global Expectation pooling layer to replace the Global Max pooling Layer.
And you can also see the better performance of expectation pooling than max pooling and average pooling in the simulation data.
### Dependency

* Keras-gpu=2.1.6
* Tensorflow-gpu=1.8.0
* pool
* tqdm

### Train the model
We use the data from http://cnn.csail.mit.edu/ 

**Download data**
All the data to reproduce the paper can be downloaded from the [website](http://cnn.csail.mit.edu/motif_discovery/)
And we also provide a script  ```get_data.py``` to download all the data, which will take several quarters. After downloading the raw data, you should run the ```data_process.py``` to convert the raw data to ```HDF5 format```.

```commandline
python get_data.py
python data_process.py
```


**Data Structure and Demanstrate**

* The real data contains 690 datasets to reproduce the results in real data section of the paper.
* The simulation data contains 3 datasets to reproduce the results in simulated data section of the paper.

**Train models in the simulation data**
```commandline
python train_simulation_data.py 0 0 3
```

**Train models in the real data**
```commandline
python train_chip_data.py 0 0 175
python train_chip_data.py 1 175 350
python train_chip_data.py 2 350 520
python train_chip_data.py 3 520 690
```

Please note that running all results from raw real data will take a long time about **several days** in 4 * GPU(Ti 1080) with six parallel models in each GPU.


### Contact Us
If there is any question, you can send e-mails to xinmingtu@pku.edu.cn
