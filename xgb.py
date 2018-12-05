#!/usr/bin/env python3

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import KFold
from sklearn import metrics, preprocessing, linear_model
from sklearn.metrics import log_loss, accuracy_score
import xgboost as xgb
import random

params = {'n_estimators': 1000,
		 'learning_rate': 0.04,
		 'max_depth':2,
         'silent':1,
         'subsample': 0.50,
         'colsample_bytree': 0.60,
         'objective':'multi:softmax',
         'reg_lambda': 1,
         'reg_alpha': 1,
         'seed': 2018,
         'nthread': 8,
         'tree_method': 'gpu_hist',
         }


def main():
    # Set seed for reproducibility
    np.random.seed(4439854)

    print("Loading data...")
    # Load the data from the pickle files
    oil_dic = pickle.load(open('data/oil.pickle','rb'))
    snp_dic = pickle.load(open('data/snp.pickle','rb'))
    oil_vector = []
    snp_matrix = []
    random_keys = list(snp_dic.keys())
    random.shuffle(random_keys)
    for k in random_keys:
        oil_vector.append(int(oil_dic[k]//1))
        snp_matrix.append(snp_dic[k][:50])
    Y = np.asarray(oil_vector, dtype=np.int8)
    X = np.asarray(snp_matrix, dtype=np.int8)
    
    num_class = len(np.unique(Y))
    params['num_class'] = num_class
    print(num_class)
    del oil_dic; del snp_dic; del oil_vector; del snp_matrix

    # This is your model that will learn to predict
    #model = linear_model.LogisticRegression(n_jobs=-1)
    model = xgb.XGBClassifier(**params)

    # cross validation
    print("Cross-validation:")
    K = 5
    i = 0
    accs = []
    kf = KFold(n_splits=K, shuffle=True, random_state=2018)
    for train, val in kf.split(X):
        i += 1 
        model.fit(X[train, :], Y[train])
        pred_prob = model.predict_proba(X[val, :])
        pred = np.argmax(pred_prob, axis=1)
        print(pred)
        acc = accuracy_score(Y[val], pred)
        accs.append(acc)
        print("accuracy %d/%d:" % (i, K), acc)
        pass


if __name__ == '__main__':
    main()
