import pickle
import lzma
import numpy as np
import os

def get_setting(dim=3, params_reg={'regression':'Linear'}, params_noise={'noise':'Gaussian'}, params_missing={}):

    regression  = params_reg['regression']
    assert regression in ['Linear'], 'regression must be Linear.'

    if 'mean' in params_reg:
        mean = params_reg['mean']
    else:
        mean = 1
    if 'scale' in params_reg:
        scale = params_reg['scale']
    else:
        scale = 1
    if params_reg['beta'] is not None:
        beta = params_reg['beta']
    else:
        beta = np.full(dim,1)
    if dim < 10:
        name = 'Linear_d_'+str(dim)+'_beta_'+'_'.join(str(x) for x in beta)+'_Gaussian_Mean_'+str(mean)
    else:
        name = 'Linear_d_'+str(dim)+'_beta_varies_Gaussian_Mean_'+str(mean)+'_Scale_'+str(scale)

    if 'prob_missing' in params_missing:
        prob_missing = params_missing['prob_missing']
    else:
        prob_missing = 0.2

    if 'mechanism' in params_missing:
        if params_missing['mechanism'] == 'MNAR_mask_quantiles':
            name = name + '_' + params_missing['mechanism'] + '_q_' + str(params_missing['q']) + '_'
        else:
            name = name + '_' + params_missing['mechanism'] + '_'
            if 'id_setting' in params_missing:
                name = name + 'id_'+str(params_missing['id_setting'])+'_'
    else:
        name = name + '_MCAR_'

    name = name + str(prob_missing)

    return name

def get_name_data(train_size, cal_size, params_test, dim=3, params_reg={}, params_noise={}, dataset=None, params_missing={}, seed=1):
    """
    Parameters
    ----------
    n : experiment sample size
    dim : dimension of the covariates (i.e. X lies in R^dim)
    regression : regression model, should be Linear
    noise : noise type, can be Gaussian
    params_reg : parameters of the regression part
    params_noise : parameters of the noise, e.g. a dictionary {'ar': [1, ar1], 'ma':[1]}
                   to generate an AR(1) noise with coefficient -ar1
    seed : random seed for reproducibility used in the experiment

    Returns
    -------
    name : name of the file containing (if existing)
    the generated data with the given parameters of simulations
    """

    max_test_size = np.max(params_test['test_size'])

    if dataset is None:

        regression = params_reg['regression']

        assert regression in ['Linear'], 'regression must be Linear.'

        name = get_setting(dim=dim, params_reg=params_reg, params_noise=params_noise, params_missing=params_missing)

    else:
        name = dataset

    name = name + '_seed_' + str(seed) + '_train_' + str(train_size) + '_cal_' + str(cal_size) + '_test_' + str(max_test_size)

    if 'prob_missing' in list(params_missing.keys()):
        name = name + '_prob_' + str(params_missing['prob_missing'])

    return name

def get_name_data_imputed(train_size, cal_size, params_test, imputation,
                          dim=3, params_reg={}, params_noise={}, dataset=None, params_missing={}, seed=1):
    """
    Parameters
    ----------
    n : experiment sample size
    dim : dimension of the covariates (i.e. X lies in R^dim)
    regression : regression model, should be Linear
    noise : noise type, can be Gaussian
    params_reg : parameters of the regression part
    params_noise : parameters of the noise, e.g. a dictionary {'ar': [1, ar1], 'ma':[1]}
                   to generate an AR(1) noise with coefficient -ar1
    seed : random seed for reproducibility used in the experiment

    Returns
    -------
    name : name of the file containing (if existing)
    the generated data with the given parameters of simulations
    """

    name = get_name_data(train_size, cal_size, params_test, dim=dim,
                         params_reg=params_reg, params_noise=params_noise, dataset=dataset, params_missing=params_missing, seed=seed)

    if imputation is not None:
        name = name + '_imputation_' + imputation

    return name

def get_name_results(pipeline, train_size, cal_size, n_rep, imputation=None, d=3,
                     params_reg={}, params_noise={}, dataset=None, params_missing={}):
    """ ...
    Parameters
    ----------
    pipeline :
    params_method :
    Returns
    -------
    name :
    """

    # Results file name, depending on the method

    if pipeline != 'Oracle':
        name_method = pipeline+'_Imp_'+imputation
    else:
        name_method = pipeline

    # Results directory name, depending on the data simulation

    if dataset is not None:
        name_directory = dataset
    else:
        name_directory = get_setting(dim=d, params_reg=params_reg, params_noise=params_noise, params_missing=params_missing)
    if 'prob_missing' in list(params_missing.keys()):
        name_directory = name_directory + '_train_' + str(train_size) + '_cal_' + str(cal_size) + '_prob_' + str(params_missing['prob_missing']) + '_rep_' + str(n_rep)
    else:
        name_directory = name_directory + '_train_' + str(train_size) + '_cal_' + str(cal_size) + '_rep_' + str(n_rep)

    return name_directory, name_method

def load_file(parent, name, ext):
    """ ...
    Parameters
    ----------
    parent :
    name :
    ext :
    Returns
    -------
    file :
    """
    assert ext in ['pkl', 'xz'], 'ext must be pkl or xz.'
    path = parent + '/' + name + '.' + ext
    if ext == 'pkl':
        with open(path,'rb') as f:
            file = pickle.load(f)
    elif ext == 'xz':
        with lzma.open(path,'rb') as f:
            file = pickle.load(f)

    return file

def write_file(parent, name, ext, file):
    """ ...
    Parameters
    ----------
    parent :
    name :
    ext :
    file :
    Returns
    -------
    """

    assert ext in ['pkl', 'xz'], 'ext must be pkl or xz.'
    path = parent + '/' + name + '.' + ext
    if ext == 'pkl':
        if not os.path.isdir(parent):
            os.makedirs(parent)
        with open(path,'wb') as f:
            pickle.dump(file, f)
    elif ext == 'xz':
        if not os.path.isdir(parent):
            os.makedirs(parent)
        with lzma.open(path,'wb') as f:
            pickle.dump(file, f)

def get_name_method(method, basemodel=None, mask='No', protection='No', subset=False):
    if subset == True:
        assert method == 'CQR_Masking_Cal', 'With subsetting calibration you should be masking.'
        method = method + '_subset'
    if method == 'Oracle':
        name = method
    elif method == 'Oracle_mean' and protection=='No':
        name = method
    elif method == 'Oracle_mean' and protection!='No':
        name = '_'.join([method, protection])
    elif protection == 'No' and mask == 'No':
        name = '_'.join([method, basemodel])
    elif method in ['QR', 'QR_TrainCal', 'CQR_Masking_Cal', 'CQR_Masking_Cal_subset'] and mask == 'No':
        name = '_'.join([method, basemodel])
    elif method in ['QR', 'QR_TrainCal', 'CQR_Masking_Cal', 'CQR_Masking_Cal_subset'] and mask == 'Yes':
        name = '_'.join([method, basemodel, 'Mask'])
    elif protection == 'No':
        name = '_'.join([method, basemodel, 'Mask'])
    elif mask == 'No':
        name = '_'.join([method, basemodel, protection])
    else:
        name = '_'.join([method, basemodel, 'Mask', protection])
    return name
