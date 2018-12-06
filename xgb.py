#!/usr/bin/env python3

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import KFold
from sklearn import metrics, preprocessing, linear_model
from sklearn.metrics import log_loss, accuracy_score, mean_squared_error
import xgboost as xgb
import random


# xgboost parameters
params = {'n_estimators': 1000,
		 'learning_rate': 0.3,
		 'max_depth':2,
         'silent':1,
         'subsample': 0.50,
         'colsample_bytree': 0.60,
         'objective':'reg:linear',
         'reg_lambda': 1,
         'reg_alpha': 1,
         'seed': 2018,
         'nthread': 8, # number of cores that are used
         }


def main():
    # Set seed for reproducibility
    np.random.seed(890890890)

    print("Loading data...")
    # Load the data from the pickle files
    oil_dic = pickle.load(open('data/oil.pickle','rb'))
    snp_dic = pickle.load(open('data/snp.pickle','rb'))
    oil_vector = []
    snp_matrix = []

    random_keys = list(snp_dic.keys())
    random.shuffle(random_keys)
    size = int(len(random_keys) * 0.1) # currently using 1/10 of samples for testing purposes
    for k in random_keys[:size]: # getting labels and features 
        oil_vector.append(oil_dic[k])
        snp_matrix.append(snp_dic[k][:100]) # using first 100 SNPs for testing purposes; remember to change this to include all features
    Y = np.asarray(oil_vector, dtype=np.float32) # cast them to numpy data format
    X = np.asarray(snp_matrix, dtype=np.int8)

    del oil_dic; del snp_dic; del oil_vector; del snp_matrix # delete large objects that will no longer be used to save memory

    # This is your model that will learn to predict
    model = xgb.XGBClassifier(**params)

    # cross validation
    print("Cross-validation:")
    K = 5
    i = 0
    mses = [] # stores MSE of each fold
    kf = KFold(n_splits=K, shuffle=True, random_state=2018)
    for train, val in kf.split(X):
        i += 1
        model.fit(X[train, :], Y[train])
        pred = model.predict(X[val, :])
        mse = mean_squared_error(Y[val], pred)
        mses.append(mse)
        print("MSE %d/%d:" % (i, K), mse)
        pass
    print('average MSE: {}'.format(np.mean(mses)))

if __name__ == '__main__':
    main()
