import files
import utils
import imputation as imp
from tqdm.autonotebook import tqdm
import numpy as np
np.warnings.filterwarnings('ignore')

from scipy.stats import norm
import functools

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import QuantileRegressor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

import six
import sys
sys.modules['sklearn.externals.six'] = six

import quantile_forest as qf


import copy

### The following lines of code are copied from CHR (Sesia and Romano, 2021) public GitHub.`
### https://github.com/msesia/chr

class RegressionDataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = torch.from_numpy(X_data).float()
        self.y_data = torch.from_numpy(y_data).float()

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__ (self):
        return len(self.X_data)

class NNet(nn.Module):
    """ Conditional quantile estimator, formulated as neural net
    """
    def __init__(self, quantiles, num_features, num_hidden=64, dropout=0.1, no_crossing=False):
        """ Initialization
        Parameters
        ----------
        quantiles : numpy array of quantile levels (q), each in the range (0,1)
        num_features : integer, input signal dimension (p)
        num_hidden : integer, hidden layer dimension
        dropout : float, dropout rate
        no_crossing: boolean, whether to explicitly prevent quantile crossovers
        """
        super(NNet, self).__init__()

        self.no_crossing = no_crossing

        self.num_quantiles = len(quantiles)

        # Construct base network
        self.base_model = nn.Sequential(
            nn.Linear(num_features, num_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_hidden, self.num_quantiles),
        )
        self.init_weights()

    def init_weights(self):
        """ Initialize the network parameters
        """
        for m in self.base_model:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """ Run forward pass
        """
        x = self.base_model(x)
        if self.no_crossing:
            y,_ = torch.sort(x,1)
        else:
            y = x
        return y

class AllQuantileLoss(nn.Module):
    """ Pinball loss function
    """
    def __init__(self, quantiles):
        """ Initialize
        Parameters
        ----------
        quantiles : pytorch vector of quantile levels, each in the range (0,1)
        """
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        """ Compute the pinball loss
        Parameters
        ----------
        preds : pytorch tensor of estimated labels (n)
        target : pytorch tensor of true labels (n)
        Returns
        -------
        loss : cost function value
        """
        #assert not target.requires_grad
        #assert preds.size(0) == target.size(0)

        errors = target.unsqueeze(1)-preds
        Q = self.quantiles.unsqueeze(0)
        loss = torch.max((Q-1.0)*errors, Q*errors).mean()

        return loss


class QNet:
    """ Fit a neural network (conditional quantile) to training data
    """
    def __init__(self, quantiles, num_features, no_crossing=False, dropout=0.2, learning_rate=0.001,
                 num_epochs=100, batch_size=16, num_hidden=64, random_state=0, calibrate=0, verbose=False):
        """ Initialization
        Parameters
        ----------
        quantiles : numpy array of quantile levels (q), each in the range (0,1)
        num_features : integer, input signal dimension (p)
        learning_rate : learning rate
        random_state : integer, seed used in CV when splitting to train-test
        """

        # Detect whether CUDA is available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Store input (sort the quantiles)
        quantiles = np.sort(quantiles)
        self.quantiles = torch.from_numpy(quantiles).float().to(self.device)
        self.num_features = num_features

        # Define NNet model
        self.model = NNet(self.quantiles, self.num_features, num_hidden=num_hidden, dropout=dropout, no_crossing=no_crossing)
        self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Initialize loss function
        self.loss_func = AllQuantileLoss(self.quantiles)

        # Store variables
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.calibrate = int(calibrate)

        # Initialize training logs
        self.loss_history = []
        self.test_loss_history = []
        self.full_loss_history = []

        # Validation
        self.val_period = 10

        self.verbose = verbose

    def fit(self, X, Y, return_loss=False):

        self.scaler = StandardScaler()
        self.scaler.fit(X)
        X = self.scaler.transform(X)

        Y = Y.flatten().astype(np.float32)
        X = X.astype(np.float32)

        dataset = RegressionDataset(X, Y)
        num_epochs = self.num_epochs
        if self.calibrate>0:
            # Train with 80% of samples
            n_valid = int(np.round(0.2*X.shape[0]))
            loss_stats = []
            for b in range(self.calibrate):
                X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=n_valid, random_state=self.random_state+b)
                train_dataset = RegressionDataset(X_train, Y_train)
                val_dataset = RegressionDataset(X_valid, Y_valid)
                loss_stats_tmp = self._fit(train_dataset, num_epochs, val_dataset=val_dataset)
                loss_stats.append([loss_stats_tmp['val']])
                # Reset model
                self.model.init_weights()

            loss_stats = np.matrix(np.concatenate(loss_stats,0)).T

            loss_stats = np.median(loss_stats,1).flatten()
            # Find optimal number of epochs
            num_epochs = self.val_period*(np.argmin(loss_stats)+1)
            loss_stats_cal = loss_stats

        # Train with all samples
        loss_stats = self._fit(dataset, num_epochs)
        if self.calibrate:
            loss_stats = loss_stats_cal

        #if return_loss:
        return self

    def _fit(self, train_dataset, num_epochs, val_dataset=None):
        batch_size = self.batch_size

        # Initialize data loaders
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
        if val_dataset is not None:
            val_loader = DataLoader(dataset=val_dataset, batch_size=1)

        num_samples, num_features = train_dataset.X_data.shape
        print("Training with {} samples and {} features.". \
              format(num_samples, num_features))

        loss_stats = {'train': [], "val": []}

        X_train_batch = train_dataset.X_data.to(self.device)
        y_train_batch = train_dataset.y_data.to(self.device)

        for e in tqdm(range(1, num_epochs+1)):

            # TRAINING
            train_epoch_loss = 0
            self.model.train()

            if batch_size<500:

              for X_train_batch, y_train_batch in train_loader:
                  X_train_batch, y_train_batch = X_train_batch.to(self.device), y_train_batch.to(self.device)
                  self.optimizer.zero_grad()

                  y_train_pred = self.model(X_train_batch).to(self.device)

                  train_loss = self.loss_func(y_train_pred, y_train_batch)

                  train_loss.backward()
                  self.optimizer.step()

                  train_epoch_loss += train_loss.item()

            else:
                self.optimizer.zero_grad()

                y_train_pred = self.model(X_train_batch).to(self.device)

                train_loss = self.loss_func(y_train_pred, y_train_batch)

                train_loss.backward()
                self.optimizer.step()

                train_epoch_loss += train_loss.item()

            # VALIDATION
            if val_dataset is not None:
                if e % self.val_period == 0:
                    self.model.eval()
                    with torch.no_grad():
                        val_epoch_loss = 0
                        for X_val_batch, y_val_batch in val_loader:
                            X_val_batch, y_val_batch = X_val_batch.to(self.device), y_val_batch.to(self.device)
                            y_val_pred = self.model(X_val_batch).to(self.device)
                            val_loss = self.loss_func(y_val_pred, y_val_batch)
                            val_epoch_loss += val_loss.item()

                    loss_stats['val'].append(val_epoch_loss/len(val_loader))
                    self.model.train()

            else:
                loss_stats['val'].append(0)

            if e % self.val_period == 0:
                loss_stats['train'].append(train_epoch_loss/len(train_loader))

            if (e % 10 == 0) and (self.verbose):
                if val_dataset is not None:
                    print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | ', end='')
                    print(f'Val Loss: {val_epoch_loss/len(val_loader):.5f} | ', flush=True)
                else:
                    print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | ', flush=True)

        return loss_stats

    def predict(self, X):
        """ Estimate the label given the features
        Parameters
        ----------
        x : numpy array of training features (nXp)
        Returns
        -------
        ret_val : numpy array of predicted labels (n)
        """
        X = self.scaler.transform(X)
        self.model.eval()
        ret_val = self.model(torch.from_numpy(X).to(self.device).float().requires_grad_(False))
        return ret_val.cpu().detach().numpy()

    def get_quantiles(self):
        return self.quantiles.cpu().numpy()

### Here ends the code from CHR and start the new code.

def fit_basemodel(X_train, Y_train, target='Mean', basemodel='Linear', alpha=0.1, params_basemodel={}):

    assert target in ['Mean', 'Quantiles'], 'regression must be Mean or Quantiles.'
    assert basemodel in ['Linear', 'RF', 'NNet', 'XGBoost'], 'regression must be Linear, RF or NNet.'

    cores = params_basemodel['cores']

    if basemodel == 'RF':
        n_estimators = params_basemodel['n_estimators']
        min_samples_leaf = params_basemodel['min_samples_leaf']
        max_features = params_basemodel['max_features']

    if target == 'Mean':
        if basemodel == 'Linear':
            trained_model = LinearRegression(n_jobs=cores).fit(X_train,Y_train)
        elif basemodel == 'RF':
            trained_model = RandomForestRegressor(n_jobs=cores,n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                                  max_features=max_features, random_state=1).fit(X_train,Y_train)
    elif target == 'Quantiles':
        a_low = alpha/2
        a_high = 1-alpha/2
        if basemodel == 'Linear':
            trained_model = {'q_low': QuantileRegressor(quantile=a_low, solver='highs', alpha=0).fit(X_train,Y_train),
                             'q_high': QuantileRegressor(quantile=a_high, solver='highs', alpha=0).fit(X_train,Y_train)}
        elif basemodel == 'RF':
            trained_model = qf.RandomForestQuantileRegressor(random_state=1, min_samples_leaf=min_samples_leaf,
                                                             n_estimators=n_estimators, max_features=max_features).fit(X_train,Y_train)
        elif basemodel == 'XGBoost':
            trained_model = {'q_low': GradientBoostingRegressor(loss="quantile", alpha=a_low, n_estimators=25).fit(X_train,Y_train),
                             'q_high': GradientBoostingRegressor(loss="quantile", alpha=a_high, n_estimators=25).fit(X_train,Y_train)}

        elif basemodel == 'NNet':

            n_train = len(Y_train)
            n_features = X_train.shape[1]
            epochs = 2000
            lr = 0.0005
            batch_size = n_train
            dropout = 0.1

            grid_quantiles = [alpha/2, 1-alpha/2]
            trained_model = QNet(grid_quantiles, n_features, no_crossing=True, batch_size=batch_size,
                                 dropout=dropout, num_epochs=epochs, learning_rate=lr, calibrate=0,
                                 verbose=False).fit(X_train, Y_train)

    return trained_model

def predict_basemodel(fitted_basemodel, X_test, target='Mean', basemodel='Linear', alpha=0.1):

    assert target in ['Mean', 'Quantiles'], 'regression must be Mean or Quantiles.'
    assert basemodel in ['Linear', 'RF', 'NNet', 'XGBoost'], 'regression must be Linear, RF or NNet.'

    if target == 'Mean':
        predictions = fitted_basemodel.predict(X_test)
    elif target == 'Quantiles':
        a_low = alpha/2
        a_high = 1-alpha/2
        if basemodel == 'Linear':
            predictions = {'y_inf': fitted_basemodel['q_low'].predict(X_test),
                           'y_sup': fitted_basemodel['q_high'].predict(X_test)}
        elif basemodel == 'RF':
            both_pred = fitted_basemodel.predict(X_test, quantiles=[a_low, a_high])
            predictions = {'y_inf': both_pred[:, 0],
                           'y_sup': both_pred[:, 1]}
        elif basemodel == 'XGBoost':
            predictions = {'y_inf': fitted_basemodel['q_low'].predict(X_test),
                           'y_sup': fitted_basemodel['q_high'].predict(X_test)}
        elif basemodel == 'NNet':
            both_pred = fitted_basemodel.predict(X_test)
            predictions = {'y_inf': both_pred[:, 0],
                           'y_sup': both_pred[:, 1]}

    return predictions


def quantile_corrected(x, alpha):
    n_x = len(x)
    if (1-alpha)*(1+1/n_x) > 1:
        return np.inf
    else:
        return np.quantile(x, (1-alpha)*(1+1/n_x))

def calibrate_predict_intervals(pred_cal, Y_cal, pred_test, groups_cal=None, groups_test=None, target='Mean', basemodel='Linear', alpha=0.1):

    assert target in ['Mean', 'Quantiles'], 'regression must be Mean or Quantiles.'
    assert basemodel in ['Oracle', 'Linear', 'RF', 'NNet', 'XGBoost'], 'regression must be Linear, RF or NNet.'

    if groups_cal == None:
        if target == 'Mean':
            scores = np.abs(Y_cal-pred_cal)
            q_scores = quantile_corrected(scores, alpha)
            interval_predictions = {'y_inf': pred_test-q_scores,
                                    'y_sup': pred_test+q_scores}
        elif target == 'Quantiles':
            scores = np.maximum(pred_cal['y_inf']-Y_cal, Y_cal-pred_cal['y_sup'])
            q_scores = quantile_corrected(scores, alpha)
            interval_predictions = {'y_inf': pred_test['y_inf']-q_scores,
                                    'y_sup': pred_test['y_sup']+q_scores}
    else:
        if target == 'Mean':
            scores = np.abs(Y_cal-pred_cal)
        elif target == 'Quantiles':
            scores = np.maximum(pred_cal['y_inf']-Y_cal, Y_cal-pred_cal['y_sup'])

        scores_sorted = np.array(scores)[np.array(groups_cal).argsort()]
        ids = np.unique(np.array(groups_cal)[np.array(groups_cal).argsort()], return_index=True)[0]
        inds = np.unique(np.array(groups_cal)[np.array(groups_cal).argsort()], return_index=True)[1]
        scores_splitted = np.split(scores_sorted, inds)[1:]

        q_scores_cal = list(map(functools.partial(quantile_corrected, alpha=alpha), scores_splitted))

        missing_groups = np.array(groups_test)[~np.isin(groups_test, ids)]
        if (len(missing_groups) > 0):
            ids = np.concatenate((ids, missing_groups))
            q_scores_cal = np.concatenate((q_scores_cal, np.full(len(missing_groups),np.inf)))

        inds_test = list(map(list(ids).index, groups_test))

        q_scores_test = np.array(q_scores_cal)[np.array(inds_test)]

        if target == 'Mean':
            interval_predictions = {'y_inf': pred_test-q_scores_test,
                                    'y_sup': pred_test+q_scores_test}
        elif target == 'Quantiles':
            interval_predictions = {'y_inf': pred_test['y_inf']-q_scores_test,
                                    'y_sup': pred_test['y_sup']+q_scores_test}

    return interval_predictions

def calibrate_masking_predict_intervals(fitted_basemodel, imputer, X_cal, M_cal, Y_cal,
                                        X_mis_test, features_test, M_test, mask,
                                        groups_test, subset=True, target='Quantiles',
                                        basemodel='Linear', alpha=0.1):

    assert target in ['Quantiles'], 'regression must be Quantiles.'
    assert basemodel in ['Linear', 'RF', 'NNet', 'XGBoost'], 'regression must be Linear, RF or NNet.'

    patterns = np.unique(M_test, axis=0)
    ids = list(map(utils.pattern_to_id, patterns.astype(int)))

    n_test = features_test.shape[0]
    q_scores_test = np.empty(n_test)

    if subset == False:
        pred_test = {'y_inf': np.empty(n_test),
                     'y_sup': np.empty(n_test)}

    for idp, id_pattern in enumerate(ids):

        X_imp_cal_masking = copy.deepcopy(X_cal)
        M_cal_masking = copy.deepcopy(M_cal)
        Y_cal_masking = copy.deepcopy(Y_cal)

        pattern = patterns[idp]

        empty = False

        if subset == True:
            ind_subsample = np.all(M_cal[:, pattern == 0] == 0, axis=1)
            if np.sum(ind_subsample) == 0:
                empty = True
            X_imp_cal_masking = X_imp_cal_masking[ind_subsample, :]
            M_cal_masking = M_cal_masking[ind_subsample, :]
            Y_cal_masking = Y_cal_masking[ind_subsample]

        if X_imp_cal_masking.shape[1] > len(pattern):
            nb = X_imp_cal_masking.shape[1] - len(pattern)
            pattern_ext = np.append(pattern, np.full(nb,0))
        else:
            pattern_ext = pattern

        X_imp_cal_masking[:, pattern_ext == 1] = np.nan
        M_cal_masking[:, pattern == 1] = 1

        if not empty:
            X_imp_cal_masking = imputer.transform(X_imp_cal_masking)

            if mask == 'Yes':
                features_cal = np.concatenate((X_imp_cal_masking, M_cal_masking), axis=1)
            else:
                features_cal = X_imp_cal_masking

            cal_predictions = predict_basemodel(fitted_basemodel, features_cal, target, basemodel, alpha)

            scores = np.maximum(cal_predictions['y_inf']-Y_cal_masking, Y_cal_masking-cal_predictions['y_sup'])

        if subset == True:

            if not empty:

                q_scores_cal = quantile_corrected(scores, alpha=alpha)

                q_scores_test[(np.array(groups_test) == id_pattern).flatten()] = q_scores_cal

            else:
                q_scores_test[(np.array(groups_test) == id_pattern).flatten()] = np.inf

        else:

            X_to_pred = copy.deepcopy(X_mis_test[(np.array(groups_test) == id_pattern).flatten(), :])
            M_to_pred = copy.deepcopy(M_test[(np.array(groups_test) == id_pattern).flatten(), :])

            n_current = X_to_pred.shape[0]

            patterns_cal = np.unique(M_cal_masking, axis=0)
            ids_cal = list(map(utils.pattern_to_id, patterns_cal.astype(int)))

            groups_cal_masking = list(map(utils.pattern_to_id, M_cal_masking.astype(int)))

            nb_cal = len(scores)

            all_preds = {'y_inf': np.empty((n_current, nb_cal)),
                         'y_sup': np.empty((n_current, nb_cal))}

            for idp_cal, id_pattern_cal in enumerate(ids_cal):

                idx_cal_masking = (np.array(groups_cal_masking) == id_pattern_cal).flatten()
                nb_mask = np.sum(idx_cal_masking)

                all_preds['y_inf'][:, idx_cal_masking] = np.tile(scores[idx_cal_masking], (n_current, 1))
                all_preds['y_sup'][:, idx_cal_masking] = np.tile(scores[idx_cal_masking], (n_current, 1))

                pattern_masking = patterns_cal[idp_cal]

                X_to_pred_masking = copy.deepcopy(X_to_pred)
                M_to_pred_masking = copy.deepcopy(M_to_pred)
                X_to_pred_masking[:, pattern_masking == 1] = np.nan
                M_to_pred_masking[:, pattern_masking == 1] = 1

                X_to_pred_masking = imputer.transform(X_to_pred_masking)

                if mask == 'Yes':
                    features_test_pattern = np.concatenate((X_to_pred_masking, M_to_pred_masking), axis=1)
                else:
                    features_test_pattern = X_to_pred_masking

                preds_k = predict_basemodel(fitted_basemodel, features_test_pattern, target, basemodel, alpha)

                all_preds['y_inf'][:, idx_cal_masking] = -np.tile(preds_k['y_inf'], (nb_mask, 1)).T + all_preds['y_inf'][:, idx_cal_masking]
                all_preds['y_sup'][:, idx_cal_masking] = np.tile(preds_k['y_sup'], (nb_mask, 1)).T + all_preds['y_sup'][:, idx_cal_masking]


            if (1 - alpha) * (1 + 1 / nb_cal) > 1:
                pred_test['y_inf'][(np.array(groups_test) == id_pattern).flatten()] = [-np.inf] * n_current
                pred_test['y_sup'][(np.array(groups_test) == id_pattern).flatten()] = [np.inf] * n_current
            else:
                pred_test['y_inf'][(np.array(groups_test) == id_pattern).flatten()] = -np.quantile(all_preds['y_inf'], (1 - alpha) * (1 + 1 / nb_cal), axis=1)
                pred_test['y_sup'][(np.array(groups_test) == id_pattern).flatten()] = np.quantile(all_preds['y_sup'], (1 - alpha) * (1 + 1 / nb_cal), axis=1)

    if subset == True:
        pred_test = predict_basemodel(fitted_basemodel, features_test, target, basemodel, alpha)
        interval_predictions = {'y_inf': pred_test['y_inf']-q_scores_test,
                                'y_sup': pred_test['y_sup']+q_scores_test}
    else:
        interval_predictions = {'y_inf': pred_test['y_inf'],
                                'y_sup': pred_test['y_sup']}

    return interval_predictions

def compute_mean_mis_given_obs(X_obs_in_mis, mean_mis, cov_mis_obs, cov_obs_inv, mean_obs):
    return mean_mis + np.dot(cov_mis_obs,np.dot(cov_obs_inv, X_obs_in_mis - mean_obs))

def oracle_pattern(pattern, X_test, M_test, beta, mean, cov, alpha=0.1):

    a_low = alpha/2
    a_high = 1-alpha/2

    pattern_id = utils.pattern_to_id(pattern.astype(int))
    M_test_id = list(map(utils.pattern_to_id, M_test.astype(int)))
    X_pattern = X_test[np.where(np.array(M_test_id) == pattern_id)]

    pattern = np.array(list(map(bool, pattern)))

    beta_mis = beta[pattern]
    beta_obs = beta[~pattern]

    mean_mis = mean[pattern]
    mean_obs = mean[~pattern]

    X_obs_in_mis = X_pattern[:,~pattern]

    cov_obs = cov[~pattern][:,~pattern]
    cov_obs_inv = np.linalg.pinv(cov_obs)

    cov_mis = cov[pattern][:,pattern]
    cov_mis_obs = cov[pattern][:,~pattern]

    mean_mis_given_obs = np.array(list(map(functools.partial(compute_mean_mis_given_obs,
                                                             mean_mis=mean_mis, cov_mis_obs=cov_mis_obs,
                                                             cov_obs_inv=cov_obs_inv, mean_obs=mean_obs), X_obs_in_mis)))

    beta_mis_mean_mis = np.array(list(map(functools.partial(np.dot, beta_mis), mean_mis_given_obs)))
    beta_obs_X_obs = np.array(list(map(functools.partial(np.dot, beta_obs), X_obs_in_mis)))

    cov_mis_given_obs = cov_mis - np.dot(cov_mis_obs,np.dot(cov_obs_inv, cov_mis_obs.T))

    q_low = beta_obs_X_obs + beta_mis_mean_mis + norm.ppf(a_low)*np.sqrt(np.dot(beta_mis, np.dot(cov_mis_given_obs , beta_mis.T))+1)
    q_high = beta_obs_X_obs + beta_mis_mean_mis + norm.ppf(a_high)*np.sqrt(np.dot(beta_mis, np.dot(cov_mis_given_obs , beta_mis.T))+1)

    return {'q_low': q_low, 'q_high': q_high}

def oracle(M_test, X_test, beta, mean, cov, alpha=0.1):

    n_test = X_test.shape[0]

    interval_predictions = {'y_inf': np.empty(n_test),
                            'y_sup': np.empty(n_test)}

    patterns = np.unique(M_test, axis=0)

    oracles_intervals_per_pattern = list(map(functools.partial(oracle_pattern,
                                                               X_test=X_test, M_test=M_test, beta=beta,
                                                               mean=mean, cov=cov, alpha=alpha), patterns))

    for idp, pattern in enumerate(patterns):

            pattern_id = utils.pattern_to_id(pattern.astype(int))
            M_test_id = list(map(utils.pattern_to_id, M_test.astype(int)))
            interval_predictions['y_inf'][np.where(np.array(M_test_id) == pattern_id)] = oracles_intervals_per_pattern[idp]['q_low']
            interval_predictions['y_sup'][np.where(np.array(M_test_id) == pattern_id)] = oracles_intervals_per_pattern[idp]['q_high']

    return interval_predictions

def oracle_len_pattern(pattern, beta, cov, alpha=0.1):

    pattern = np.array(list(map(bool, pattern)))

    beta_mis = beta[pattern]

    cov_obs = cov[~pattern][:,~pattern]
    cov_obs_inv = np.linalg.pinv(cov_obs)

    cov_mis = cov[pattern][:,pattern]
    cov_mis_obs = cov[pattern][:,~pattern]

    cov_mis_given_obs = cov_mis - np.dot(cov_mis_obs,np.dot(cov_obs_inv, cov_mis_obs.T))

    length = 2 * norm.ppf(1-alpha/2) * np.sqrt(np.dot(beta_mis, np.dot(cov_mis_given_obs, beta_mis.T)) + 1)

    return length

def oracle_mean_pattern(pattern, X_test, M_test, beta, mean, cov):

    pattern_id = utils.pattern_to_id(pattern.astype(int))
    M_test_id = list(map(utils.pattern_to_id, M_test.astype(int)))
    X_pattern = X_test[np.where(np.array(M_test_id) == pattern_id)]

    pattern = np.array(list(map(bool, pattern)))

    beta_mis = beta[pattern]
    beta_obs = beta[~pattern]

    mean_mis = mean[pattern]
    mean_obs = mean[~pattern]

    X_obs_in_mis = X_pattern[:,~pattern]

    cov_obs = cov[~pattern][:,~pattern]
    cov_obs_inv = np.linalg.pinv(cov_obs)

    cov_mis = cov[pattern][:,pattern]
    cov_mis_obs = cov[pattern][:,~pattern]

    mean_mis_given_obs = np.array(list(map(functools.partial(compute_mean_mis_given_obs,
                                                             mean_mis=mean_mis, cov_mis_obs=cov_mis_obs,
                                                             cov_obs_inv=cov_obs_inv, mean_obs=mean_obs), X_obs_in_mis)))

    beta_mis_mean_mis = np.array(list(map(functools.partial(np.dot, beta_mis), mean_mis_given_obs)))
    beta_obs_X_obs = np.array(list(map(functools.partial(np.dot, beta_obs), X_obs_in_mis)))

    mean_pattern = beta_obs_X_obs + beta_mis_mean_mis

    return mean_pattern

def oracle_mean(M_test, X_test, beta, mean, cov):

    n_test = X_test.shape[0]

    predictions = np.empty(n_test)

    patterns = np.unique(M_test, axis=0)

    oracles_mean_per_pattern = list(map(functools.partial(oracle_mean_pattern,
                                                          X_test=X_test, M_test=M_test, beta=beta,
                                                          mean=mean, cov=cov), patterns))

    for idp, pattern in enumerate(patterns):

            pattern_id = utils.pattern_to_id(pattern.astype(int))
            M_test_id = list(map(utils.pattern_to_id, M_test.astype(int)))
            predictions[np.where(np.array(M_test_id) == pattern_id)] = oracles_mean_per_pattern[idp]

    return predictions

def run_experiments(data, alpha, methods, basemodels, params_basemodel, masks, protections, subsets=['False'], imputation=None,
                    params_reg={}, params_noise={},
                    parent_results='results'):

    d = data['X_missing']['Train'].shape[2]
    n_rep = data['X_missing']['Train'].shape[0]

    name_pipeline = []
    for method in methods:
        for basemodel in basemodels:
            for mask in masks:
                for protection in protections:
                    if method == 'CQR_Masking_Cal':
                        for subset in subsets:
                            name_temp = files.get_name_method(method, basemodel, mask, protection, subset)
                            if not name_temp in name_pipeline:
                                name_pipeline.append(name_temp)
                    else:
                        name_temp = files.get_name_method(method, basemodel, mask, protection)
                        if not name_temp in name_pipeline:
                            name_pipeline.append(name_temp)

    results_methods = dict.fromkeys(name_pipeline)

    for k in tqdm(range(n_rep)):

        if 'X' in list(data.keys()):
            X_train = data['X']['Train'][k,:,:]
            X_cal = data['X']['Cal'][k,:,:]
        X_mis_train = data['X_missing']['Train'][k,:,:]
        X_mis_cal = data['X_missing']['Cal'][k,:,:]
        X_imp_train = data['X_imp']['Train'][k,:,:]
        X_imp_cal = data['X_imp']['Cal'][k,:,:]
        M_train = data['M']['Train'][k,:,:]
        M_cal = data['M']['Cal'][k,:,:]
        Y_train = data['Y']['Train'][k,:]
        Y_cal = data['Y']['Cal'][k,:]

        keys_test = list(data['X_missing']['Test'].keys())
        X_test = dict.fromkeys(keys_test)
        X_mis_test = dict.fromkeys(keys_test)
        X_imp_test = dict.fromkeys(keys_test)
        M_test = dict.fromkeys(keys_test)
        Y_test = dict.fromkeys(keys_test)
        for key in keys_test:
            if 'X' in list(data.keys()):
                X_test[key] = data['X']['Test'][key][k,:,:]
            X_mis_test[key] = data['X_missing']['Test'][key][k,:,:]
            X_imp_test[key] = data['X_imp']['Test'][key][k,:,:]
            M_test[key] = data['M']['Test'][key][k,:,:]
            Y_test[key] = data['Y']['Test'][key][k,:]

        trained_models = {}

        for method in methods:

            if method in ['Oracle', 'Oracle_mean']:
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
                if 'beta' not in params_reg or params_reg['beta'] is None:
                    beta = np.full(d,1)
                else:
                    beta = params_reg['beta']

                if method == 'Oracle':
                    preds = dict.fromkeys(keys_test)
                    for key in keys_test:
                        pred = oracle(M_test[key], X_test[key], beta, mean, cov, alpha=alpha)
                        preds[key] = pred

                elif method == 'Oracle_mean':
                    cal_predictions = oracle_mean(M_cal, X_cal, beta, mean, cov)

                    test_predictions = dict.fromkeys(keys_test)
                    for key in keys_test:
                        test_predictions[key] = oracle_mean(M_test[key], X_test[key], beta, mean, cov)

                    for protection in protections:
                        pipeline = files.get_name_method(method, basemodel=None, mask='No', protection=protection)
                        if protection == 'No':
                            groups_cal = None
                            groups_test = dict.fromkeys(keys_test)
                            for key in keys_test:
                                groups_test[key] = None
                        elif protection == 'Pattern':
                            groups_cal = list(map(utils.pattern_to_id, M_cal.astype(int)))
                            groups_test = dict.fromkeys(keys_test)
                            for key in keys_test:
                                groups_test[key] = list(map(utils.pattern_to_id, M_test[key].astype(int)))
                        elif protection == 'Pattern_Size':
                            groups_cal = list(map(utils.pattern_to_size, M_cal.astype(int)))
                            groups_test = dict.fromkeys(keys_test)
                            for key in keys_test:
                                groups_test[key] = list(map(utils.pattern_to_size, M_test[key].astype(int)))
                        preds = dict.fromkeys(keys_test)
                        for key in keys_test:

                            preds[key] = calibrate_predict_intervals(cal_predictions, Y_cal, test_predictions[key],
                                                                     groups_cal=groups_cal, groups_test=groups_test[key],
                                                                     target='Mean',
                                                                     basemodel='Oracle', alpha=alpha)

                        results = results_methods[pipeline]
                        if results_methods[pipeline] == None:
                            results = dict.fromkeys(keys_test)
                            for key in keys_test:
                                results[key] = {'Y_inf': np.array(preds[key]['y_inf']),
                                                'Y_sup': np.array(preds[key]['y_sup'])}
                        else:
                            for key in keys_test:
                                results[key]['Y_inf'] = np.vstack(
                                    (results[key]['Y_inf'], np.array(preds[key]['y_inf'])))
                                results[key]['Y_sup'] = np.vstack(
                                    (results[key]['Y_sup'], np.array(preds[key]['y_sup'])))
                        results_methods[pipeline] = results

            elif method == 'CQR_Masking_Cal':

                assert imputation is not None, "imputation must be specified for Masking"

                target = 'Quantiles'

                imputer_masking = imp.impute_imputer(X_mis_train, imputation)

                X_imp_train_masking = imputer_masking.transform(X_mis_train)
                X_imp_test_masking = dict.fromkeys(keys_test)

                for key in keys_test:
                    X_imp_test_masking[key] = imputer_masking.transform(X_mis_test[key])

                if target in trained_models.keys():
                    trained_models_target = trained_models[target]
                else:
                    trained_models[target] = {}
                    trained_models_target = None

                for basemodel in basemodels:

                    if trained_models_target is not None and basemodel in trained_models_target.keys():
                        trained_models_target_basemodel = trained_models_target[basemodel]
                    else:
                        trained_models[target][basemodel] = {}
                        trained_models_target_basemodel = None

                    for mask in masks:

                        if mask == 'Yes':
                            name_mask = 'mask'
                            features_train = np.concatenate((X_imp_train_masking, M_train), axis=1)
                            features_test = dict.fromkeys(keys_test)
                            for key in keys_test:
                                features_test[key] = np.concatenate((X_imp_test_masking[key], M_test[key]), axis=1)
                        else:
                            name_mask = 'no_mask'
                            features_train = X_imp_train_masking
                            features_test = X_imp_test_masking

                        if trained_models_target_basemodel is not None and name_mask in trained_models_target_basemodel.keys():
                            trained_models_target_basemodel_mask = trained_models_target_basemodel[name_mask]
                        else:
                            trained_models[target][basemodel][name_mask] = {}
                            trained_models_target_basemodel_mask = None

                        if trained_models_target_basemodel_mask is None:
                            trained_model = fit_basemodel(features_train, Y_train, target=target, basemodel=basemodel, alpha=alpha,
                                                          params_basemodel=params_basemodel)
                            trained_models[target][basemodel][name_mask] = trained_model
                        else:
                            trained_model = trained_models_target_basemodel_mask

                        groups_test = dict.fromkeys(keys_test)
                        for key in keys_test:
                            groups_test[key] = list(map(utils.pattern_to_id, M_test[key].astype(int)))

                        for subset in subsets:
                            pipeline = files.get_name_method(method, basemodel, mask, subset=subset)

                            preds = dict.fromkeys(keys_test)
                            for key in keys_test:
                                pred = calibrate_masking_predict_intervals(trained_model, imputer_masking,
                                                                           X_mis_cal, M_cal, Y_cal,
                                                                           X_mis_test[key], features_test[key], M_test[key], mask,
                                                                           groups_test=groups_test[key], subset=subset,
                                                                           target=target, basemodel=basemodel, alpha=alpha)
                                preds[key] = pred

                            results = results_methods[pipeline]

                            if results_methods[pipeline] == None:
                                results = dict.fromkeys(keys_test)
                                for key in keys_test:
                                    results[key] = {'Y_inf': np.array(preds[key]['y_inf']), 'Y_sup': np.array(preds[key]['y_sup'])}
                            else:
                                for key in keys_test:
                                    results[key]['Y_inf'] = np.vstack((results[key]['Y_inf'],np.array(preds[key]['y_inf'])))
                                    results[key]['Y_sup'] = np.vstack((results[key]['Y_sup'],np.array(preds[key]['y_sup'])))
                            results_methods[pipeline] = results

            else:

                if method == 'SCP':
                    target = 'Mean'
                elif method in ['CQR', 'QR', 'QR_TrainCal']:
                    target = 'Quantiles'

                if method in ['QR', 'QR_TrainCal']:
                    conformalized = False
                else:
                    conformalized = True

                if target in trained_models.keys():
                    trained_models_target = trained_models[target]
                else:
                    trained_models[target] = {}
                    trained_models_target = None

                for basemodel in basemodels:

                    if method != 'QR_TrainCal' and trained_models_target is not None and basemodel in trained_models_target.keys():
                        trained_models_target_basemodel = trained_models_target[basemodel]
                    elif method != 'QR_TrainCal':
                        trained_models[target][basemodel] = {}
                        trained_models_target_basemodel = None

                    for mask in masks:

                        if mask == 'Yes':
                            name_mask = 'mask'
                            features_train = np.concatenate((X_imp_train, M_train), axis=1)
                            features_cal = np.concatenate((X_imp_cal, M_cal), axis=1)
                            features_test = dict.fromkeys(keys_test)
                            for key in keys_test:
                                features_test[key] = np.concatenate((X_imp_test[key], M_test[key]), axis=1)
                        else:
                            name_mask = 'no_mask'
                            features_train = X_imp_train
                            features_cal = X_imp_cal
                            features_test = X_imp_test

                        if method == 'QR_TrainCal':
                            features_train = np.concatenate((features_train, features_cal), axis=0)
                            Y_traincal = np.concatenate((Y_train, Y_cal), axis=0)
                            trained_model = fit_basemodel(features_train, Y_traincal, target=target, basemodel=basemodel,
                                                          alpha=alpha,
                                                          params_basemodel=params_basemodel)
                        else:
                            if trained_models_target_basemodel is not None and name_mask in trained_models_target_basemodel.keys():
                                trained_models_target_basemodel_mask = trained_models_target_basemodel[name_mask]
                            else:
                                trained_models[target][basemodel][name_mask] = {}
                                trained_models_target_basemodel_mask = None

                            if trained_models_target_basemodel_mask is None:
                                trained_model = fit_basemodel(features_train, Y_train, target=target, basemodel=basemodel, alpha=alpha,
                                                              params_basemodel=params_basemodel)
                                trained_models[target][basemodel][name_mask] = trained_model
                            else:
                                trained_model = trained_models_target_basemodel_mask

                        cal_predictions = predict_basemodel(trained_model, features_cal, target, basemodel, alpha)

                        test_predictions = dict.fromkeys(keys_test)
                        for key in keys_test:
                            test_predictions[key] = predict_basemodel(trained_model, features_test[key], target, basemodel, alpha)

                        if conformalized:

                            for protection in protections:
                                pipeline = files.get_name_method(method, basemodel, mask, protection)
                                if protection == 'No':
                                    groups_cal = None
                                    groups_test = dict.fromkeys(keys_test)
                                    for key in keys_test:
                                        groups_test[key] = None
                                elif protection == 'Pattern':
                                    groups_cal = list(map(utils.pattern_to_id, M_cal.astype(int)))
                                    groups_test = dict.fromkeys(keys_test)
                                    for key in keys_test:
                                        groups_test[key] = list(map(utils.pattern_to_id, M_test[key].astype(int)))
                                elif protection == 'Pattern_Size':
                                    groups_cal = list(map(utils.pattern_to_size, M_cal.astype(int)))
                                    groups_test = dict.fromkeys(keys_test)
                                    for key in keys_test:
                                        groups_test[key] = list(map(utils.pattern_to_size, M_test[key].astype(int)))
                                preds = dict.fromkeys(keys_test)
                                for key in keys_test:
                                    preds[key] = calibrate_predict_intervals(cal_predictions, Y_cal, test_predictions[key],
                                                                             groups_cal=groups_cal, groups_test=groups_test[key],
                                                                             target=target,
                                                                             basemodel=basemodel, alpha=alpha)

                                results = results_methods[pipeline]
                                if results_methods[pipeline] == None:
                                    results = dict.fromkeys(keys_test)
                                    for key in keys_test:
                                        results[key] = {'Y_inf': np.array(preds[key]['y_inf']), 'Y_sup': np.array(preds[key]['y_sup'])}
                                else:
                                    for key in keys_test:
                                        results[key]['Y_inf'] = np.vstack((results[key]['Y_inf'],np.array(preds[key]['y_inf'])))
                                        results[key]['Y_sup'] = np.vstack((results[key]['Y_sup'],np.array(preds[key]['y_sup'])))
                                results_methods[pipeline] = results

                        else:
                            interval_predictions = dict.fromkeys(keys_test)
                            for key in keys_test:
                                interval_predictions[key] = {'y_inf': test_predictions[key]['y_inf'],
                                                             'y_sup': test_predictions[key]['y_sup']}
                            pipeline = files.get_name_method(method, basemodel, mask, conformalized)
                            results = results_methods[pipeline]
                            if results_methods[pipeline] == None:
                                results = dict.fromkeys(keys_test)
                                for key in keys_test:
                                    results[key] = {'Y_inf': np.array(interval_predictions[key]['y_inf']),
                                                    'Y_sup': np.array(interval_predictions[key]['y_sup'])}
                            else:
                                for key in keys_test:
                                    results[key]['Y_inf'] = np.vstack((results[key]['Y_inf'],np.array(interval_predictions[key]['y_inf'])))
                                    results[key]['Y_sup'] = np.vstack((results[key]['Y_sup'],np.array(interval_predictions[key]['y_sup'])))
                            results_methods[pipeline] = results

    return results_methods, name_pipeline

def run_real_experiments(data, alpha, methods, basemodels, params_basemodel, masks, conformalized, protections,
                         n_rep, parent_results='results', imputation=None, data_missing=None, subset=True):

    test_size = len(data['Y']['Test'][0,:])
    d = data['X_imp']['Train'].shape[2]

    name_pipeline = []
    for method in methods:
        for basemodel in basemodels:
            for mask in masks:
                for protection in protections:
                    name_temp = files.get_name_method(method, basemodel, mask, protection, conformalized, subset)
                    if not name_temp in name_pipeline:
                        name_pipeline.append(name_temp)

    results_methods = dict.fromkeys(name_pipeline)

    if 'M_original' in data.keys():
        mask_original = True
    else:
        mask_original = False

    for k in tqdm(range(n_rep)):

        X_imp_train = data['X_imp']['Train'][k,:,:]
        X_imp_cal = data['X_imp']['Cal'][k,:,:]
        X_imp_test = data['X_imp']['Test'][k,:,:]
        if mask_original:
            M_original_train = data['M_original']['Train'][k,:,:]
            M_original_cal = data['M_original']['Cal'][k,:,:]
            M_original_test = data['M_original']['Test'][k,:,:]
        M_train = data['M']['Train'][k,:,:]
        M_cal = data['M']['Cal'][k,:,:]
        M_test = data['M']['Test'][k,:,:]
        Y_train = data['Y']['Train'][k,:]
        Y_cal = data['Y']['Cal'][k,:]
        Y_test = data['Y']['Test'][k,:]

        for method in methods:
            if method == 'CQR_Masking_Cal':

                assert imputation is not None, "imputation must be specified for Masking"

                target = 'Quantiles'

                X_mis_train = data_missing['X_missing']['Train'][k,:,:]
                X_mis_cal = data_missing['X_missing']['Cal'][k,:,:]
                X_mis_test = data_missing['X_missing']['Test'][k,:,:]

                imputer_masking = imp.impute_imputer(X_mis_train, imputation)

                X_imp_train_masking = imputer_masking.transform(X_mis_train)
                X_imp_test_masking = imputer_masking.transform(X_mis_test)

                for basemodel in basemodels:
                    for mask in masks:
                        if mask == 'Yes':
                            features_train = np.concatenate((X_imp_train_masking, M_train), axis=1)
                            features_test = np.concatenate((X_imp_test_masking, M_test), axis=1)
                        else:
                            features_train = X_imp_train_masking
                            features_test = X_imp_test_masking

                        trained_model = fit_basemodel(features_train, Y_train, target=target, basemodel=basemodel, alpha=alpha,
                                                      params_basemodel=params_basemodel)
                        pipeline = files.get_name_method(method, basemodel, mask, subset=subset)
                        groups_test = list(map(utils.pattern_to_id, M_test.astype(int)))

                        pred = calibrate_masking_predict_intervals(trained_model, imputer_masking,
                                                                   X_mis_cal, M_cal, Y_cal, features_test,
                                                                   M_test, mask,
                                                                   groups_test=groups_test, subset=subset, target=target,
                                                                   basemodel=basemodel, alpha=alpha)
                        results = results_methods[pipeline]
                        if results_methods[pipeline] == None:
                            results = {'Y_inf': np.array(pred['y_inf']), 'Y_sup': np.array(pred['y_sup'])}
                        else:
                            results['Y_inf'] = np.vstack((results['Y_inf'],np.array(pred['y_inf'])))
                            results['Y_sup'] = np.vstack((results['Y_sup'],np.array(pred['y_sup'])))
                        results_methods[pipeline] = results
            else:
                if method == 'SCP':
                    target = 'Mean'
                elif method in ['CQR', 'QR']:
                    target = 'Quantiles'
                for basemodel in basemodels:
                    for mask in masks:
                        if mask == 'Yes':
                            features_train = np.concatenate((X_imp_train, M_train), axis=1)
                            features_cal = np.concatenate((X_imp_cal, M_cal), axis=1)
                            features_test = np.concatenate((X_imp_test, M_test), axis=1)
                        else:
                            features_train = X_imp_train
                            features_cal = X_imp_cal
                            features_test = X_imp_test

                        trained_model = fit_basemodel(features_train, Y_train, target=target, basemodel=basemodel, alpha=alpha,
                                                      params_basemodel=params_basemodel)

                        cal_predictions = predict_basemodel(trained_model, features_cal, target, basemodel, alpha)
                        test_predictions = predict_basemodel(trained_model, features_test, target, basemodel, alpha)

                        if conformalized:
                            for protection in protections:
                                pipeline = files.get_name_method(method, basemodel, mask, protection)
                                if protection == 'No':
                                    groups_cal = None
                                    groups_test = None
                                elif protection == 'Pattern':
                                    if mask_original:
                                        groups_cal = list(map(utils.pattern_to_id, M_original_cal.astype(int)))
                                        groups_test = list(map(utils.pattern_to_id, M_original_test.astype(int)))
                                    else:
                                        groups_cal = list(map(utils.pattern_to_id, M_cal.astype(int)))
                                        groups_test = list(map(utils.pattern_to_id, M_test.astype(int)))
                                elif protection == 'Pattern_Size':
                                    if mask_original:
                                        groups_cal = list(map(utils.pattern_to_size, M_original_cal.astype(int)))
                                        groups_test = list(map(utils.pattern_to_size, M_original_test.astype(int)))
                                    else:
                                        groups_cal = list(map(utils.pattern_to_size, M_cal.astype(int)))
                                        groups_test = list(map(utils.pattern_to_size, M_test.astype(int)))
                                pred = calibrate_predict_intervals(cal_predictions, Y_cal, test_predictions,
                                                                   groups_cal=groups_cal, groups_test=groups_test, target=target,
                                                                   basemodel=basemodel, alpha=alpha)
                                results = results_methods[pipeline]
                                if results_methods[pipeline] == None:
                                    results = {'Y_inf': np.array(pred['y_inf']), 'Y_sup': np.array(pred['y_sup'])}
                                else:
                                    results['Y_inf'] = np.vstack((results['Y_inf'],np.array(pred['y_inf'])))
                                    results['Y_sup'] = np.vstack((results['Y_sup'],np.array(pred['y_sup'])))
                                results_methods[pipeline] = results
                        else:
                            interval_predictions = {'y_inf': test_predictions['y_inf'],
                                                    'y_sup': test_predictions['y_sup']}
                            pipeline = files.get_name_method(method, basemodel, mask, protection, conformalized)
                            results = results_methods[pipeline]
                            if results_methods[pipeline] == None:
                                results = {'Y_inf': np.array(interval_predictions['y_inf']),
                                           'Y_sup': np.array(interval_predictions['y_sup'])}
                            else:
                                results['Y_inf'] = np.vstack((results['Y_inf'],np.array(interval_predictions['y_inf'])))
                                results['Y_sup'] = np.vstack((results['Y_sup'],np.array(interval_predictions['y_sup'])))
                            results_methods[pipeline] = results

    return results_methods, name_pipeline
