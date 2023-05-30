# Conformal Prediction with Missing Values

This repository contains the code to reproduce the experiments of the paper _Conformal Prediction with Missing Values_, M. Zaffran, A. Dieuleveut, J. Josse and Y. Romano, ICML 2023.

The notebook ``CP_NA_Synthetic.ipynb`` allows to reproduce the synthetic experiments, while ``CP_NA_Semi-synthetic.ipynb`` focuses on the semi-synthetic experiments.  
The corresponding ``_Plots`` notebooks contain the code for displaying the results in the same format as in the paper.

The core code for the algorithms CP-MDA-Exact and CP-MDA-Nested can be found in the ```prediciton.py``` file.

``imputation.py`` contains the functions used for imputation of the data sets.  
``generation.py`` allows to generate synthetic data (outcome and features, but also missing values).  
``files.py`` handles the file names, files writing and loading.  
``utils.py`` contains some useful functions like computing the metrics associated to interval predictions, combinatorics on patterns etc.  
``datasets.py`` pre-process the real data sets used in the semi-synthetic experiments.  

Note that, as mentioned in the ```.py``` files, some piece of code are taken from other GitHub repositories, namely:  
+ CQR (Romano et al., 2019) repository, available [here](https://github.com/yromano/cqr), for the (cleaning of the) data sets used in the semi-synthetic experiments; 
+ CHR (Sesia and Romano, 2021) repository, available [here](https://github.com/msesia/chr), for the Quantile Neural Network architecture.

This repository will be updated in the next few days.

## License

[MIT](LICENSE) © Margaux Zaffran