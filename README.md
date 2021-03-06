# CNN-Microcanonical-Optimization-Algorithm
Hyper-Parameter Selection in Convolutional Neural Networks Using Microcanonical Optimization Algorithm

Ayla Gülcü ([@aylagulcu]), Zeki Kuş
Dept. of Computer Science, Fatih Sultan Mehmet University, Istanbul, Turkey

This repository contains code for the paper: [Hyper-Parameter Selection in Convolutional Neural Networks Using Microcanonical Optimization Algorithm](https://ieeexplore.ieee.org/abstract/document/9037322)

## Citation
If you find our code useful, please consider citing our work using the bibtex:
```
@article{gulcu2020hyper,
  title={Hyper-parameter selection in convolutional neural networks using microcanonical optimization algorithm},
  author={G{\"u}lc{\"u}, Ayla and KU{\c{s}}, Zeki},
  journal={IEEE Access},
  volume={8},
  pages={52528--52540},
  year={2020},
  publisher={IEEE}
}
```

## Enviroment
 - Python3
 - Keras
 
 
**Datasets**
- [EMNIST](https://drive.google.com/drive/folders/1AMmm_c48epiNfAu66N7VHz4e2Bm2-_DK?usp=sharing)
 
## Train
Run
CIFAR1O: ```python ./cifar10.py```
MNIST: ```python ./mnist.py```
EMNIST-Datasets: ```python ./emnist.py```
FashionMNIST: ```python ./fashionmnist.py```

### Overview
The success of Convolutional Neural Networks is highly dependent on the selected architecture and the hyper-parameters. The need for the automatic design of the networks is especially important for complex architectures where the parameter space is so large that trying all possible combinations is computationally infeasible. In this study, Microcanonical Optimization algorithm which is a variant of Simulated Annealing method is used for hyper-parameter optimization and architecture selection for Convolutional Neural Networks. To the best of our knowledge, our study provides a first attempt at applying Microcanonical Optimization for this task. The networks generated by the proposed method is compared to the networks generated by Simulated Annealing method in terms of both accuracy and size using six widely-used image recognition datasets. Moreover, a performance comparison using Tree Parzen Estimator which is a Bayesion optimization-based approach is also presented. It is shown that the proposed method is able to achieve competitive classification results with the state-of-the-art architectures. When the size of the networks is also taken into account, one can see that the networks generated by Microcanonical Optimization method contain far less parameters than the state-of-the-art architectures. Therefore, the proposed method can be preferred for automatically tuning the networks especially in situations where fast training is as important as the accuracy.
