# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name  :    train
   Author     :    雨住风停松子落
   E-mail     :
   date       :    2019/4/25
   Description:
-------------------------------------------------
   Change Activity:
                   2019/4/25:
-------------------------------------------------
"""
__author__ = '雨住风停松子落'

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.externals import joblib
import warnings

warnings.filterwarnings("ignore")


def build_model(data, mode='train'):
    '''
    初始化模型，包含各模型使用的特征下标
    :param data: 数据集
    :param mode: train or test
    :return:
    '''
    filter_features_1 = [data.columns[idx] for idx in
                         [0, 1, 2, 3, 4, 5, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 26, 27, 28, 29,
                          31, 33, 34, 35, 36, 37, 38, 39, 42]]
    filter_features_2 = [data.columns[idx] for idx in
                         [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27,
                          28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43]]
    filter_features_3 = [data.columns[idx] for idx in
                         [0, 1, 2, 3, 4, 5, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 26, 27, 28, 29,
                          31, 33, 34, 35, 36, 37, 38, 39, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]]
    filter_features_4 = [data.columns[idx] for idx in
                         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                          26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 68, 69, 70, 71, 72,
                          73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83]]
    if mode == 'train':
        ada_clf = AdaBoostClassifier(
            RandomForestClassifier(n_estimators=60, max_depth=13, min_samples_split=5,
                                   min_samples_leaf=20, oob_score=True, random_state=0,
                                   class_weight='balanced'),
            algorithm="SAMME", n_estimators=1200, learning_rate=0.05
        )
        bag_clf = BaggingClassifier(
            base_estimator=RandomForestClassifier(n_estimators=60, max_depth=13, min_samples_split=5,
                                                  min_samples_leaf=20, oob_score=True, random_state=0,
                                                  class_weight='balanced'),
            n_estimators=500, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=1,
            random_state=0
        )
        gbdt_clf = GradientBoostingClassifier(
            n_estimators=1200, learning_rate=0.05, min_samples_leaf=60, max_depth=10, min_samples_split=5,
            subsample=0.7, random_state=0, loss='deviance'
        )
        lgbm_clf = LGBMClassifier(
            boosting_type='gbdt', num_leaves=51, max_depth=-1, learning_rate=0.05, n_estimators=600,
            subsample_for_bin=1000, objective='binary', class_weight='balanced', min_split_gain=0.0,
            min_child_weight=0.001, min_child_samples=20, subsample=1.0, subsample_freq=0, colsample_bytree=1.0,
            reg_alpha=3.5, reg_lambda=1.0, random_state=None, importance_type='gain', silent=True
        )
        mlp_clf = MLPClassifier(
            solver='lbfgs', activation='logistic', alpha=1e-2, learning_rate='adaptive',
            hidden_layer_sizes=(2048, 67, 2), random_state=1, max_iter=5000, verbose=10, learning_rate_init=0.01
        )
        vote_clf = VotingClassifier(
            estimators=[
                (
                    'LogisticRegression',
                    LogisticRegression(penalty='l2', solver='lbfgs', verbose=0, class_weight='balanced')
                ),
                (
                    'DecisionTreeClassifier', DecisionTreeClassifier(class_weight='balanced')
                ),
                (
                    'RandomForestClassifier', RandomForestClassifier(
                        n_estimators=60, max_depth=13, min_samples_split=5, min_samples_leaf=20,
                        oob_score=True, random_state=0, class_weight='balanced')
                ),
                ('GradientBoostingClassifier', GradientBoostingClassifier(n_estimators=1200, verbose=0)),
                ('GaussianNB', GaussianNB()),
                ('KNeighborsClassifier', KNeighborsClassifier())
            ],
            voting='soft'
        )
        xgb_clf = XGBClassifier(
            learning_rate=0.01, n_estimators=5000, max_depth=6, min_child_weight=1, gamma=0., subsample=0.8,
            colsample_btree=0.8, objective='binary:logistic', scale_pos_weight=1, min_samples_split=5,
            min_samples_leaf=60, seed=27, reg_alpha=0.005, random_state=0
        )

        clfs = {
            'AdaBoostClassifier': [ada_clf, filter_features_1],
            'BaggingClassifier': [bag_clf, filter_features_1],
            'GradientBoostingClassifier': [gbdt_clf, filter_features_2],
            'LGBMClassifier': [lgbm_clf, filter_features_2],
            'MLPClassifier': [mlp_clf, filter_features_3],
            'VotingClassifier': [vote_clf, filter_features_1],
            'XGBClassifier': [xgb_clf, filter_features_4],
        }
    elif mode == 'test':
        clfs = {
            'AdaBoostClassifier': filter_features_1,
            'BaggingClassifier': filter_features_1,
            'GradientBoostingClassifier': filter_features_2,
            'LGBMClassifier': filter_features_2,
            'MLPClassifier': filter_features_3,
            'VotingClassifier': filter_features_1,
            'XGBClassifier': filter_features_4,
        }
    else:
        raise Exception("mode must be train or test !!")

    return clfs


def train(train_data, savedPath):
    '''
    训练
    :param train_data:
    :param savedPath:
    :return:
    '''
    clfs = build_model(train_data, mode='train')
    for clf_name, [clf, filter_features] in clfs.items():
        clf.fit(train_data[filter_features], train_data[train_data.columns[85]])

        print(clf_name, '\t', clf.score(train_data[filter_features], train_data[train_data.columns[85]]))
        joblib.dump(clf, savedPath + clf_name + '.pkl')


def test(test_data, modelPath, savedPath, threshold):
    '''
    测试
    :param test_data:
    :param modelPath:
    :param savedPath:
    :param threshold: 不同方法传入不同阈值
    :return:
    '''
    clfs = build_model(test_data, mode='test')
    result = []
    idx = 0
    for clf_name, filter_features in clfs.items():
        clf = joblib.load(modelPath + clf_name + '.pkl')
        p_proba = clf.predict_proba(test_data[filter_features])
        p_list = [1 if p[1] > threshold[clf_name] else 0 for p in p_proba]
        if idx == 0:
            result = p_list
        else:
            result = [1 if (result[i] == 1 and p_list[i] == 1) else 0 for i in range(5738)]
        idx = idx + 1

    res_df = pd.DataFrame(data=np.column_stack(
        [np.reshape(test_data[test_data.columns[84]], int(test_data[test_data.columns[0]].count())), result]),
        columns=['idx', 'result'])

    res_df.to_csv(savedPath, index=False)
