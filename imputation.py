import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, IterativeImputer

def impute(data, imputation):

    assert imputation in ['mean', 'constant', 'MICE', 'iterative_ridge'], 'imputation must be constant, mean, iterative_ridge or MICE.'

    X_missing = data['X_missing']

    if imputation in ['mean', 'constant']:
        imputer = SimpleImputer(missing_values=np.nan, strategy=imputation)
    elif imputation == 'MICE':
        imputer = IterativeImputer(missing_values=np.nan, sample_posterior=True)
    elif imputation == 'iterative_ridge':
        imputer = IterativeImputer(missing_values=np.nan, sample_posterior=False)

    n_rep = X_missing['Train'].shape[0]

    X_train_imp = np.empty(X_missing['Train'].shape)
    X_cal_imp = np.empty(X_missing['Cal'].shape)
    if type(X_missing['Test']) is dict:
        multiple_test = True
        keys_test = list(X_missing['Test'].keys())
        X_test_imp = dict.fromkeys(keys_test)
        for key in keys_test:
            X_test_imp[key] = np.empty(X_missing['Test'][key].shape)
    else:
        multiple_test = False
        X_test_imp = np.empty(X_missing['Test'].shape)

    for k in range(n_rep):

        imputer.fit(X_missing['Train'][k,:,:])

        X_train_imp[k,:,:] = imputer.transform(X_missing['Train'][k,:,:])
        X_cal_imp[k,:,:] = imputer.transform(X_missing['Cal'][k,:,:])
        if multiple_test:
            for key in keys_test:
                X_test_imp[key][k,:,:] = imputer.transform(X_missing['Test'][key][k,:,:])
        else:
            X_test_imp[k,:,:] = imputer.transform(X_missing['Test'][k,:,:])

    X_imputed = {'Train': X_train_imp, 'Cal': X_cal_imp, 'Test': X_test_imp}

    return X_imputed

def impute_imputer(X, imputation):

    assert imputation in ['mean', 'constant', 'MICE', 'iterative_ridge'], 'imputation must be constant, mean, iterative_ridge or MICE.'

    if imputation in ['mean', 'constant']:
        imputer = SimpleImputer(missing_values=np.nan, strategy=imputation)
    elif imputation == 'MICE':
        imputer = IterativeImputer(missing_values=np.nan, sample_posterior=True)
    elif imputation == 'iterative_ridge':
        imputer = IterativeImputer(missing_values=np.nan, sample_posterior=False)

    imputer.fit(X)

    return imputer
