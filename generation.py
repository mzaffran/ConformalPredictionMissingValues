import random
import numpy as np
import pandas as pd
import copy
import utils
from tqdm.autonotebook import tqdm

def generate_data(n, dim=3, params_reg={'regression':'Linear'}, params_noise={'noise':'Gaussian'}, seed=1):
    """
    Parameters
    ----------
    n : sample size to generate
    dim : dimension of the covariates (i.e. X lies in R^dim)
    regression : regression model, should be Linear
    noise : noise type, can be Gaussian
    params_reg : parameters for the regression part
    params_noise : parameters for the noise, e.g. a dictionary {'ar': [1, ar1], 'ma':[1]}
                   to generate an AR(1) noise with coefficient -ar1
    seed : random seed for reproducibility

    Returns
    -------
    X : covariates values, array of size n x dim
    Y : response values, array of size n
    """

    random.seed(seed)
    np.random.seed(seed)

    regression = params_reg['regression']
    assert regression in ['Linear'], 'regression must be Linear.'

    noise = params_noise['noise']

    d = dim

    if 'mean' in params_reg:
        mean = params_reg['mean']
    else:
        mean = 1
    if 'phi' in params_reg:
        phi = params_reg['phi']
    else:
        phi = 0.8
    mean = np.full(d, mean)
    cov = np.full((d,d),phi)+(1-phi)*np.eye(d)
    X = np.random.multivariate_normal(mean, cov, size=n)
    if 'beta' not in params_reg or params_reg['beta'] is None:
        beta = np.full(d,1)
    else:
        beta = params_reg['beta']
    Y_reg = X.dot(beta)

    assert noise in ['Gaussian'], 'noise must be Gaussian.'
    if noise == 'Gaussian':
        if 'mean' in params_noise:
            mean = params_noise['mean']
        else:
            mean = 0
        if 'scale' in params_noise:
            scale = params_noise['scale']
        else:
            scale = 1
        eps = np.random.normal(loc=mean,scale=scale,size=(n))

    Y = Y_reg + eps

    data = {'X': X, 'Y': Y}

    return data

def generate_split(train_size, cal_size, params_test, data):

    X = data['X']
    X_train = X[:train_size,:]
    X_cal = X[train_size:(train_size+cal_size),:]

    Y = data['Y']
    Y_train = Y[:train_size]
    Y_cal = Y[train_size:(train_size+cal_size)]

    test_size = params_test['test_size']

    mechanisms_test = params_test['mechanisms_test']

    #if test_size is list:
    X_test = dict.fromkeys(test_size)
    Y_test = dict.fromkeys(test_size)
    for n_test in test_size:
        if (train_size+cal_size+n_test) <= X.shape[0]:
            X_test[n_test] = X[(train_size+cal_size):(train_size+cal_size+n_test),:]
            Y_test[n_test] = Y[(train_size+cal_size):(train_size+cal_size+n_test)]
        else:
            if 'iid' in mechanisms_test:
                assert params_test['iid']['test_size'] != n_test
            for extreme in ['worst_pattern', 'best_pattern']:
                if extreme in mechanisms_test:
                    assert params_test[extreme]['test_size'] != n_test
            for fixed in ['fixed_nb_sample_pattern','fixed_nb_sample_pattern_size']:
                if fixed in mechanisms_test and params_test[fixed]['test_size'] == n_test:
                    assert (train_size + cal_size + params_test[fixed]['nb_sample_pattern']) <= X.shape[0]

            X_test_created = np.empty((n_test,X.shape[1]))
            Y_test_created = np.empty((n_test))

            X_to_shuffle = copy.deepcopy(X[(train_size+cal_size):,:])
            Y_to_shuffle = copy.deepcopy(Y[(train_size+cal_size):])
            n_shuffle = X_to_shuffle.shape[0]
            nb_exact_shuffle = n_test//n_shuffle
            nb_rest = n_test%n_shuffle
            for k in range(nb_exact_shuffle):
                ido = random.sample(range(n_shuffle), n_shuffle)
                X_test_created[(k * n_shuffle):((k+1) * n_shuffle), :] = X_to_shuffle[ido, :]
                Y_test_created[(k * n_shuffle):((k + 1) * n_shuffle)] = Y_to_shuffle[ido]
            ido = random.sample(range(n_shuffle), nb_rest)
            X_test_created[((k+1) * n_shuffle):, :] = X_to_shuffle[ido, :]
            Y_test_created[((k+1) * n_shuffle):] = Y_to_shuffle[ido]
            X_test[n_test] = X_test_created
            Y_test[n_test] = Y_test_created

    X_split = {'Train': X_train, 'Cal': X_cal, 'Test': X_test}
    Y_split = {'Train': Y_train, 'Cal': Y_cal, 'Test': Y_test}

    return X_split, Y_split

def generate_MCAR(X, params_test, params_missing={}, seed=1):

    """
    Parameters
    ----------
    X : data array (of shape n x dim) which will suffer missing values
    prob_missing : probability of being missing
    var_missing : binary vector of length dim, containing 1 if the variables can suffer from missing values, 0 otherwise
                  (e.g. [1,1,0] indicates that X_3 can not have missing values but X_1 and X_2 can)

    Returns
    -------
    X_mcar : covariates values (observed or missing, nan in this case), array of size  n x dim
    M_mcar : Mask array of size n x dim, containing 1 if the realization is missing, 0 otherwise
    """

    random.seed(seed)
    np.random.seed(seed)

    d = X['Train'].shape[1]

    if 'prob_missing' in params_missing:
        prob_missing = params_missing['prob_missing']
    else:
        prob_missing = 0.2
    if 'var_missing' in params_missing:
        var_missing = params_missing['var_missing']
    else:
        var_missing = np.full(d, 1)

    nb_var_missing = np.sum(var_missing)

    train_size = X['Train'].shape[0]
    cal_size = X['Cal'].shape[0]

    M_mcar_train = np.full(X['Train'].shape, False)
    X_mcar_train = copy.deepcopy(X['Train'])

    M_mcar_cal = np.full(X['Cal'].shape, False)
    X_mcar_cal = copy.deepcopy(X['Cal'])

    M_mcar_train[:,np.where(np.array(var_missing) == 1)[0]] = (np.random.uniform(low=0,high=1,size=(train_size,nb_var_missing)) <= (prob_missing))
    X_mcar_train[M_mcar_train] = np.nan
    M_mcar_cal[:,np.where(np.array(var_missing) == 1)[0]] = (np.random.uniform(low=0,high=1,size=(cal_size,nb_var_missing)) <= (prob_missing))
    X_mcar_cal[M_mcar_cal] = np.nan

    mechanisms_test = params_test['mechanisms_test']

    M_mcar = {'Train': M_mcar_train, 'Cal': M_mcar_cal}
    M_mcar_test = dict.fromkeys(mechanisms_test)

    X_mcar = {'Train': X_mcar_train, 'Cal': X_mcar_cal}
    X_mcar_test = dict.fromkeys(mechanisms_test)

    if 'iid' in mechanisms_test:
        test_size = params_test['iid']['test_size']
        M_mcar_iid = np.full((test_size, d), False)
        M_mcar_iid[:,np.where(np.array(var_missing) == 1)[0]] = (np.random.uniform(low=0,high=1,size=(test_size,nb_var_missing)) <= (prob_missing))
        M_mcar_test['iid'] = M_mcar_iid
        X_mcar_iid = copy.deepcopy(X['Test'][test_size])
        X_mcar_iid[M_mcar_iid] = np.nan
        X_mcar_test['iid'] = X_mcar_iid
    for extreme in ['worst_pattern', 'best_pattern']:
        if extreme in mechanisms_test:
            test_size = params_test[extreme]['test_size']
            test_pattern = params_test[extreme]['pattern']
            M_mcar_extreme = np.full((test_size, d), False)
            M_mcar_extreme[:,np.where(np.array(test_pattern) == 1)[0]] = 1
            M_mcar_test[extreme] = M_mcar_extreme
            X_mcar_extreme = copy.deepcopy(X['Test'][test_size])
            X_mcar_extreme[M_mcar_extreme] = np.nan
            X_mcar_test[extreme] = X_mcar_extreme
    if 'fixed_nb_sample_pattern' in mechanisms_test:
        list_patterns = utils.create_patterns(d, var_missing)
        test_size = params_test['fixed_nb_sample_pattern']['test_size']
        nb_sample_pattern = params_test['fixed_nb_sample_pattern']['nb_sample_pattern']
        M_mcar_fixed_sample_pattern = np.full((test_size, d), False)
        X_mcar_fixed_sample_pattern = copy.deepcopy(X['Test'][test_size])
        for idp, pattern in enumerate(list_patterns):
            M_mcar_fixed_sample_pattern[(idp*nb_sample_pattern):((idp+1)*nb_sample_pattern),np.where(np.array(pattern) == 1)[0]] = 1
        X_mcar_fixed_sample_pattern[M_mcar_fixed_sample_pattern] = np.nan
        M_mcar_test['fixed_nb_sample_pattern'] = M_mcar_fixed_sample_pattern
        X_mcar_test['fixed_nb_sample_pattern'] = X_mcar_fixed_sample_pattern
    if 'fixed_nb_sample_pattern_size' in mechanisms_test:
        list_pattern_sizes = np.arange(np.sum(var_missing))
        test_size = params_test['fixed_nb_sample_pattern_size']['test_size']
        nb_sample_pattern_size = params_test['fixed_nb_sample_pattern_size']['nb_sample_pattern']
        M_mcar_fixed_sample_pattern_size = np.full((test_size, d), False)
        X_mcar_fixed_sample_pattern_size = copy.deepcopy(X['Test'][test_size])

        list_patterns = utils.create_patterns(d, var_missing)
        size_to_ids = dict.fromkeys(np.arange(0, d))
        for k in np.arange(0, d):
            size_to_ids[k] = []
        for pattern in list_patterns:
            key_pattern = utils.pattern_to_id(pattern)
            size_pattern = utils.pattern_to_size(pattern)
            size_to_ids[size_pattern] = np.append(size_to_ids[size_pattern], key_pattern)

        for idp, pattern_size in enumerate(list_pattern_sizes):
            keys = random.choices(size_to_ids[pattern_size], k=nb_sample_pattern_size)
            unique_keys, count_keys = np.unique(keys, return_counts=True)
            min_ind = idp * nb_sample_pattern_size
            for idps, key in enumerate(unique_keys):
                nb_sample_pattern = count_keys[idps]
                pattern = utils.bin_to_vec(bin(int(key)), d)
                M_mcar_fixed_sample_pattern_size[min_ind:(min_ind+nb_sample_pattern),np.where(np.array(pattern) == 1)[0]] = 1
                min_ind = min_ind + nb_sample_pattern
        X_mcar_fixed_sample_pattern_size[M_mcar_fixed_sample_pattern_size] = np.nan
        M_mcar_test['fixed_nb_sample_pattern_size'] = M_mcar_fixed_sample_pattern_size
        X_mcar_test['fixed_nb_sample_pattern_size'] = X_mcar_fixed_sample_pattern_size

    X_mcar['Test'] = X_mcar_test
    M_mcar['Test'] = M_mcar_test

    return X_mcar, M_mcar

def process_test(params_test, d, params_missing={}):

    test_sizes = []
    mechanisms_test = []

    for mechanism in list(params_test.keys()):
        assert mechanism in ['iid', 'worst_pattern', 'best_pattern', 'test_pattern', 'fixed_nb_sample_pattern', 'fixed_nb_sample_pattern_size'], 'Test mechanism should be among iid, worst_pattern, best_pattern, test_pattern, fixed_nb_sample_pattern, fixed_nb_sample_pattern_size.'
        mechanisms_test = np.append(mechanisms_test, mechanism)
        if mechanism not in ['fixed_nb_sample_pattern', 'fixed_nb_sample_pattern_size']:
            assert 'test_size' in list(params_test[mechanism].keys()), 'test_size should be provided for each test mechanism.'
            test_sizes = np.append(test_sizes, int(params_test[mechanism]['test_size']))
        else:
            assert 'nb_sample_pattern' in list(params_test[mechanism].keys()), 'nb_sample_pattern should be provided for fixed_nb_sample_pattern mechanism.'
            nb_sample_pattern = params_test[mechanism]['nb_sample_pattern']

            if 'var_missing' in params_missing:
                var_missing = params_missing['var_missing']
            else:
                var_missing = np.full(d, 1)

            if mechanism == 'fixed_nb_sample_pattern':

                list_patterns = utils.create_patterns(d, var_missing)
                nb_pattern = len(list_patterns)
                test_size = nb_sample_pattern*nb_pattern
                test_sizes = np.append(test_sizes, int(test_size))
                params_test[mechanism]['test_size'] = test_size

            else:

                nb_pattern_size = np.sum(var_missing)
                test_size = nb_sample_pattern * nb_pattern_size
                test_sizes = np.append(test_sizes, int(test_size))
                params_test[mechanism]['test_size'] = test_size

    test_sizes = np.unique(test_sizes).astype(int)

    params_test['test_size'] = test_sizes
    params_test['mechanisms_test'] = mechanisms_test

    return params_test

def generate_multiple_data(train_size, cal_size, params_test, n_rep, dim=3,
                           params_reg={'regression':'Linear'}, params_noise={'noise':'Gaussian'},
                           params_missing={'mechanism':'MCAR'}):
    """
    Parameters
    ----------
    n : sample size to generate
    dim : dimension of the covariates (i.e. X lies in R^dim)
    regression : regression model, should be Linear
    noise : noise type, can be Gaussian
    params_reg : parameters for the regression part
    params_noise : parameters for the noise, e.g. a dictionary {'ar': [1, ar1], 'ma':[1]}
                   to generate an AR(1) noise with coefficient -ar1
    seed_max : random seeds for reproducibility, will generate seed_max data-sets, of seeds 0 to seed_max-1

    Returns
    -------
    X : covariates values, array of size seedmax x n x dim
    Y : response values, array of size seedmax x n
    """

    sets = ['Train', 'Cal', 'Test']
    mechanisms_test = params_test['mechanisms_test']
    max_test_size = np.max(params_test['test_size'])

    n = train_size + cal_size + max_test_size

    X = dict.fromkeys(sets)
    X_missing = dict.fromkeys(sets)
    M = dict.fromkeys(sets)
    Y = dict.fromkeys(sets)

    for k in tqdm(range(n_rep)):
        data = generate_data(n, dim=dim, params_reg=params_reg, params_noise=params_noise, seed=k)
        Xk, Yk = generate_split(train_size, cal_size, params_test, data)
        Xk_missing, Mk_missing = generate_MCAR(Xk, params_test, params_missing, seed=k)

        for set in ['Train', 'Cal']:
            if k == 0:
                X[set] = np.expand_dims(Xk[set], axis=0)
                X_missing[set] = np.expand_dims(Xk_missing[set], axis=0)
                M[set] = np.expand_dims(Mk_missing[set], axis=0)
                Y[set] = Yk[set]
            else:
                X[set] = np.vstack((X[set],np.expand_dims(Xk[set], axis=0)))
                X_missing[set] = np.vstack((X_missing[set],np.expand_dims(Xk_missing[set], axis=0)))
                M[set] = np.vstack((M[set],np.expand_dims(Mk_missing[set], axis=0)))
                Y[set] = np.vstack((Y[set],np.array(Yk[set])))

        set = 'Test'
        if k == 0:
            X[set] = dict.fromkeys(mechanisms_test)
            X_missing[set] = dict.fromkeys(mechanisms_test)
            M[set] = dict.fromkeys(mechanisms_test)
            Y[set] = dict.fromkeys(mechanisms_test)
            for key in mechanisms_test:
                n_test = params_test[key]['test_size']
                X[set][key] = np.expand_dims(Xk[set][n_test], axis=0)
                Y[set][key] = Yk[set][n_test]
                X_missing[set][key] = np.expand_dims(Xk_missing[set][key], axis=0)
                M[set][key] = np.expand_dims(Mk_missing[set][key], axis=0)

        else:
            for key in mechanisms_test:
                n_test = params_test[key]['test_size']
                X[set][key] = np.vstack((X[set][key],np.expand_dims(Xk[set][n_test], axis=0)))
                Y[set][key] = np.vstack((Y[set][key], np.array(Yk[set][n_test])))
                X_missing[set][key] = np.vstack((X_missing[set][key], np.expand_dims(Xk_missing[set][key], axis=0)))
                M[set][key] = np.vstack((M[set][key], np.expand_dims(Mk_missing[set][key], axis=0)))


    return X, X_missing, M, Y, params_missing

# Real data

def real_generate_multiple_split(dataframe, target, prob_test=0.2, seed_max=1):

    data_features = dataframe.loc[:, dataframe.columns != target]
    response = dataframe.loc[:, target]
    n = dataframe.shape[0]
    d = data_features.shape[1]

    test_size = int(n*prob_test)
    train_cal_size = int(n-test_size)
    train_size = int(2*(train_cal_size//3) + train_cal_size%3)
    cal_size = int(train_cal_size//3)

    sizes = {'Train': train_size, 'Cal': cal_size, 'Test':test_size}

    mask_original = data_features.isnull().replace({True: 1, False: 0})

    vars_categ = data_features.select_dtypes("object").columns

    data_features_categ = data_features[vars_categ]

    vars_categ = data_features.select_dtypes("object").columns
    vars_quant = set(data_features.columns).difference(set(vars_categ))
    mask_features = data_features[vars_quant].isnull().replace({True: 1,False: 0})

    data_features_categ_na = data_features_categ.fillna("-2")
    data_features_categ_encoded = pd.DataFrame(index=data_features_categ_na.index)
    for var in vars_categ:
        if np.sum(data_features_categ_na[var]=="1") > 0:
            data_features_categ_encoded[str(var)+"_1"] = data_features_categ_na[var]=="1"
        if np.sum(data_features_categ_na[var]=="0") > 0:
            data_features_categ_encoded[str(var)+"_0"] = data_features_categ_na[var]=="0"
        if np.sum(data_features_categ_na[var]=="-1") > 0:
            data_features_categ_encoded[str(var)+"_-1"] = data_features_categ_na[var]=="-1"
        if np.sum(data_features_categ_na[var]=="-2") > 0:
            data_features_categ_encoded[str(var)+"_-2"] = data_features_categ_na[var]=="-2"
    data_features_categ_encoded = data_features_categ_encoded.replace({True:1, False:0})
    data_features = data_features[vars_quant].merge(data_features_categ_encoded, left_index=True, right_index=True)

    mask = data_features.isnull().replace({True: 1, False: 0})

    col_features = list(data_features.columns)

    d_quant = mask_features.shape[1]
    d_aug = data_features.shape[1]

    X_missing = np.empty((seed_max,n,d_aug))
    M_original = np.empty((seed_max,n,d))
    M = np.empty((seed_max, n, d_aug))
    M_quant = np.empty((seed_max,n,d_quant))
    Y = np.empty((seed_max,n))

    for k in range(seed_max):

        random.seed(k)
        np.random.seed(k)

        ido = random.sample(range(n), n)

        X_missing[k,:,:] = data_features.iloc[ido,:]
        M_original[k,:,:] = mask_original.iloc[ido,:]
        M[k, :, :] = mask.iloc[ido, :]
        M_quant[k,:,:] = mask_features.iloc[ido,:]
        Y[k,:] = response[ido]

    data = {'X_missing':X_missing, 'M_original':M_original,'M':M, 'M_quant':M_quant, 'Y':Y}

    keys = ['X_missing', 'M_original', 'M', 'M_quant']
    for key in keys:
        arr = data[key]
        arr_train = arr[:,:train_size,:]
        arr_cal = arr[:,train_size:(train_size+cal_size),:]
        arr_test = arr[:,(n-test_size):n,:]
        globals()[key+'_split'] = {'Train': arr_train, 'Cal': arr_cal, 'Test': {'iid': arr_test}}

    Y = data['Y']
    Y_train = Y[:,:train_size]
    Y_cal = Y[:,train_size:(train_size+cal_size)]
    Y_test = Y[:,(n-test_size):n]
    Y_split = {'Train': Y_train, 'Cal': Y_cal, 'Test':{'iid': Y_test}}

    return X_missing_split, M_original_split, M_split, M_quant_split, Y_split, col_features, sizes

def real_generate_multiple_split_holdout(dataframe, target, prob_test=0.2):

    n = dataframe.shape[0]
    ido = random.sample(range(n), n)
    dataframe = dataframe.iloc[ido,:]
    dataframe = dataframe.reset_index(drop=True)

    data_features = dataframe.loc[:, dataframe.columns != target]
    response = dataframe.loc[:, target]

    d = data_features.shape[1]

    test_size = int(n*prob_test)
    train_cal_size = int(n-test_size)
    train_size = int(2*(train_cal_size//3) + train_cal_size%3)
    cal_size = int(train_cal_size//3)

    sizes = {'Train': train_size, 'Cal': cal_size, 'Test':test_size}

    mask_original = data_features.isnull().replace({True: 1, False: 0})

    vars_categ = data_features.select_dtypes("object").columns

    data_features_categ = data_features[vars_categ]

    vars_categ = data_features.select_dtypes("object").columns
    vars_quant = set(data_features.columns).difference(set(vars_categ))
    mask_features = data_features[vars_quant].isnull().replace({True: 1,False: 0})

    data_features_categ_na = data_features_categ.fillna("-2")
    data_features_categ_encoded = pd.DataFrame(index=data_features_categ_na.index)
    for var in vars_categ:
        if np.sum(data_features_categ_na[var]=="1") > 0:
            data_features_categ_encoded[str(var)+"_1"] = data_features_categ_na[var]=="1"
        if np.sum(data_features_categ_na[var]=="0") > 0:
            data_features_categ_encoded[str(var)+"_0"] = data_features_categ_na[var]=="0"
        if np.sum(data_features_categ_na[var]=="-1") > 0:
            data_features_categ_encoded[str(var)+"_-1"] = data_features_categ_na[var]=="-1"
        if np.sum(data_features_categ_na[var]=="-2") > 0:
            data_features_categ_encoded[str(var)+"_-2"] = data_features_categ_na[var]=="-2"
    data_features_categ_encoded = data_features_categ_encoded.replace({True:1, False:0})
    data_features = data_features[vars_quant].merge(data_features_categ_encoded, left_index=True, right_index=True)

    mask = data_features.isnull().replace({True: 1, False: 0})

    col_features = list(data_features.columns)

    d_quant = mask_features.shape[1]
    d_aug = data_features.shape[1]

    nb_split = n//(test_size)

    X_missing_train = np.empty((nb_split, sizes['Train'], d_aug))
    M_train = np.empty((nb_split, sizes['Train'], d_aug))
    M_original_train = np.empty((nb_split, sizes['Train'], d))
    M_quant_train = np.empty((nb_split, sizes['Train'], d_quant))
    Y_train = np.empty((nb_split, sizes['Train']))

    X_missing_cal = np.empty((nb_split, sizes['Cal'], d_aug))
    M_cal = np.empty((nb_split, sizes['Cal'], d_aug))
    M_original_cal = np.empty((nb_split, sizes['Cal'], d))
    M_quant_cal = np.empty((nb_split, sizes['Cal'], d_quant))
    Y_cal = np.empty((nb_split, sizes['Cal']))

    X_missing_test = np.empty((nb_split, sizes['Test'], d_aug))
    M_test = np.empty((nb_split, sizes['Test'], d_aug))
    M_original_test = np.empty((nb_split, sizes['Test'], d))
    M_quant_test = np.empty((nb_split, sizes['Test'], d_quant))
    Y_test = np.empty((nb_split, sizes['Test']))

    idx = np.array(list(np.arange(n)))

    for k in range(nb_split):

        id_test = idx[(k*sizes['Test']):((k+1)*sizes['Test'])]
        idbool = np.full(len(idx), True, dtype=bool)
        idbool[id_test] = False
        test = list(idx[~idbool])
        traincal = list(idx[idbool])
        train = traincal[:train_size]
        cal = traincal[train_size:]

        X_missing_train[k, :, :] = data_features.iloc[train, :]
        X_missing_cal[k, :, :] = data_features.iloc[cal, :]
        X_missing_test[k,:,:] = data_features.iloc[test, :]
        M_train[k, :, :] = mask.iloc[train, :]
        M_cal[k, :, :] = mask.iloc[cal, :]
        M_test[k, :, :] = mask.iloc[test, :]
        M_original_train[k, :, :] = mask_original.iloc[train, :]
        M_original_cal[k, :, :] = mask_original.iloc[cal, :]
        M_original_test[k, :, :] = mask_original.iloc[test, :]
        M_quant_train[k, :, :] = mask_features.iloc[train, :]
        M_quant_cal[k, :, :] = mask_features.iloc[cal, :]
        M_quant_test[k, :, :] = mask_features.iloc[test, :]
        Y_train[k, :] = response[train]
        Y_cal[k, :] = response[cal]
        Y_test[k, :] = response[test]

    X_missing = {'Train': X_missing_train, 'Cal': X_missing_cal, 'Test': {'iid': X_missing_test}}
    M = {'Train': M_train, 'Cal': M_cal, 'Test': {'iid': M_test}}
    M_original = {'Train': M_original_train, 'Cal': M_original_cal, 'Test': {'iid': M_original_test}}
    M_quant = {'Train': M_quant_train, 'Cal': M_quant_cal, 'Test': {'iid': M_quant_test}}
    Y = {'Train': Y_train, 'Cal': Y_cal, 'Test': {'iid': Y_test}}

    return X_missing, M_original, M, M_quant, Y, col_features, sizes


def generate_multiple_real_data_MCAR(dataframe, target, train_size, cal_size, params_test, params_missing={}, seed_max=1):
    """
    Parameters
    ----------

    seed_max : random seeds for reproducibility, will generate seed_max data-sets, of seeds 0 to seed_max-1

    Returns
    -------
    X : covariates values, array of size seedmax x n x dim
    Y : response values, array of size seedmax x n
    """

    data_features = dataframe.loc[:, dataframe.columns != target]
    response = dataframe.loc[:, target]

    sets = ['Train', 'Cal', 'Test']
    mechanisms_test = params_test['mechanisms_test']
    max_test_size = np.max(params_test['test_size'])

    n = dataframe.shape[0]

    X = dict.fromkeys(sets)
    X_missing = dict.fromkeys(sets)
    M = dict.fromkeys(sets)
    Y = dict.fromkeys(sets)

    for k in range(seed_max):

        random.seed(k)
        np.random.seed(k)

        ido = random.sample(range(n), n)

        Xk = np.array(data_features.iloc[ido,:])
        Yk = np.array(response[ido])

        data = {'X': Xk, 'Y':Yk}

        Xk, Yk = generate_split(train_size, cal_size, params_test, data)
        Xk_missing, Mk_missing = generate_MCAR(Xk, params_test, params_missing, seed=k)

        for set in ['Train', 'Cal']:
            if k == 0:
                X[set] = np.expand_dims(Xk[set], axis=0)
                X_missing[set] = np.expand_dims(Xk_missing[set], axis=0)
                M[set] = np.expand_dims(Mk_missing[set], axis=0)
                Y[set] = Yk[set]
            else:
                X[set] = np.vstack((X[set], np.expand_dims(Xk[set], axis=0)))
                X_missing[set] = np.vstack((X_missing[set], np.expand_dims(Xk_missing[set], axis=0)))
                M[set] = np.vstack((M[set], np.expand_dims(Mk_missing[set], axis=0)))
                Y[set] = np.vstack((Y[set], np.array(Yk[set])))

        set = 'Test'
        if k == 0:
            X[set] = dict.fromkeys(mechanisms_test)
            X_missing[set] = dict.fromkeys(mechanisms_test)
            M[set] = dict.fromkeys(mechanisms_test)
            Y[set] = dict.fromkeys(mechanisms_test)
            for key in mechanisms_test:
                n_test = params_test[key]['test_size']
                X[set][key] = np.expand_dims(Xk[set][n_test], axis=0)
                X_missing[set][key] = np.expand_dims(Xk_missing[set][key], axis=0)
                M[set][key] = np.expand_dims(Mk_missing[set][key], axis=0)
                Y[set][key] = Yk[set][n_test]
        else:
            for key in mechanisms_test:
                n_test = params_test[key]['test_size']
                X[set][key] = np.vstack((X[set][key], np.expand_dims(Xk[set][n_test], axis=0)))
                X_missing[set][key] = np.vstack((X_missing[set][key], np.expand_dims(Xk_missing[set][key], axis=0)))
                M[set][key] = np.vstack((M[set][key], np.expand_dims(Mk_missing[set][key], axis=0)))
                Y[set][key] = np.vstack((Y[set][key], np.array(Yk[set][n_test])))

    return X, X_missing, M, Y
