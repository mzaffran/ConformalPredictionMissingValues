import numpy as np
import files
import functools

def pattern_to_id(m):
    return(int(''.join(map(str,m)), 2))

def pattern_to_id_float(m):
    return(float(int(''.join(map(str,m)), 2)))

def pattern_to_size(m):
    return(int(np.sum(m)))

def bin_to_vec(bin_pattern, d, var_missing=None):
    bin_pattern = bin_pattern[2:]
    l = len(bin_pattern)
    if var_missing is None:
        nb_missing = d
    else:
        nb_missing = np.sum(var_missing)
    if l < nb_missing:
        for i in range(nb_missing-l):
            bin_pattern = '0'+bin_pattern
    vec_bin = [int(x) for x in bin_pattern]
    if nb_missing < d:
        vec_pattern = [0] * d
        vec_pattern = np.array(vec_pattern)
        vec_pattern[list(np.where(var_missing == 1)[0])] = vec_bin
        vec_pattern = list(vec_pattern)
    else:
        vec_pattern = vec_bin
    return(vec_pattern)

def create_patterns(d, var_missing):
    nb_var_missing = np.sum(var_missing)
    if nb_var_missing == d:
        keys_patterns = np.arange(0, 2**d-1)
        bin_patterns = list(map(bin, keys_patterns))
        vec_patterns = list(map(functools.partial(bin_to_vec, d=d), bin_patterns))
    else:
        keys_patterns = np.arange(0, 2**(nb_var_missing))
        bin_patterns = list(map(bin, keys_patterns))
        vec_patterns = list(map(functools.partial(bin_to_vec, d=d, var_missing=var_missing), bin_patterns))
    return(vec_patterns)


def get_data_results(method, train_size, cal_size, params_test, n_rep, imputation, d=3,
                     params_reg={}, params_noise={}, dataset=None, params_missing={},
                     parent_results='results', parent_data='data', extension='pkl'):

   name_dir, name_method = files.get_name_results(method, train_size, cal_size, n_rep,
                                                  imputation=imputation, d=d,
                                                  params_reg=params_reg, params_noise=params_noise,
                                                  dataset=dataset,
                                                  params_missing=params_missing)
   results = files.load_file(parent_results+'/'+name_dir, name_method, extension)

   name_data = files.get_name_data(train_size, cal_size, params_test, dim=d,
                                   params_reg=params_reg, params_noise=params_noise,
                                   dataset=dataset,
                                   params_missing=params_missing, seed=n_rep)
   data = files.load_file(parent_data, name_data, extension)

   return data, results

def compute_PI_metrics(data, results, mechanism_test):

    contains = (data['Y']['Test'][mechanism_test] <= results[mechanism_test]['Y_sup']) & (data['Y']['Test'][mechanism_test] >= results[mechanism_test]['Y_inf'])
    lengths = results[mechanism_test]['Y_sup'] - results[mechanism_test]['Y_inf']

    return contains, lengths#,

def compute_metrics_cond(n_rep, data, results, mechanism_test, cond='Pattern', replace_inf=False):

    contains, lengths = compute_PI_metrics(data, results, mechanism_test)


    if replace_inf:
        max_y_train = np.max(data['Y']['Train'], axis=1)
        max_y_cal = np.max(data['Y']['Cal'], axis=1)
        min_y_train = np.min(data['Y']['Train'], axis=1)
        min_y_cal = np.min(data['Y']['Cal'], axis=1)
        max_length_traincal = np.maximum(max_y_train, max_y_cal) - np.minimum(min_y_train, min_y_cal)

    M_test = data['M']['Test'][mechanism_test]

    if cond == 'Pattern':
        groups = np.apply_along_axis(pattern_to_id_float, 2, M_test.astype(int))
        test_patterns_id = np.unique(groups)
    elif cond == 'Pattern_Size':
        groups = np.apply_along_axis(pattern_to_size, 2, M_test.astype(int))
        test_patterns_id = np.unique(groups)

    metrics = dict.fromkeys(test_patterns_id)

    for pattern_id in test_patterns_id:

        avg_cov = []
        avg_len = []
        nb_samples = []

        for k in range(n_rep):
            current_lens  = lengths[k,groups[k,:] == pattern_id]

            temp_cov = np.nanmean(contains[k,groups[k,:] == pattern_id])
            temp_nb = np.sum(groups[k,:] == pattern_id)

            if replace_inf:
                idx_inf = np.where(np.isinf(current_lens))
                if len(idx_inf) > 0:
                    current_lens[idx_inf] = max_length_traincal[k]

            temp_len = np.nanmean(current_lens)

            avg_cov = np.append(avg_cov, temp_cov)
            avg_len = np.append(avg_len, temp_len)
            nb_samples = np.append(nb_samples, temp_nb)

        metrics[pattern_id] = {'avg_cov': avg_cov, 'avg_len': avg_len, 'nb_sample': nb_samples}

    return metrics

def name_tick(name_method):
    if name_method[-4:] == 'Mask':
        name_tick = '+ mask'
    else:
        name_tick = re.search(r"[a-zA-Z]*", name_method).group()
        if name_tick != 'MICE':
            name_tick = name_tick.capitalize()
    return name_tick