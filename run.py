# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name  :    run
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

from transform import *
from motor import *
from sklearn.utils import shuffle

if __name__ == '__main__':
    thod = {
        'AdaBoostClassifier': 0.25,
        'BaggingClassifier': 0.25,
        'GradientBoostingClassifier': 0.25,
        'LGBMClassifier': 0.25,
        'MLPClassifier': 0.25,
        'VotingClassifier': 0.25,
        'XGBClassifier': 0.25,
    }

    TRAIN_DATA_PATH = 'D:/workspace/MotorData/Motor_tain/'
    TEST_DATA_PATH = 'D:/workspace/MotorData/Motor_testP/'
    TRAIN_FEATURES_PATH = './Features/train_features.csv'
    TEST_FEATURES_PATH = './Features/test_features.csv'
    MIDEL_SAVED_PATH = './Model/'
    FEATURE_PATH = './Features/'

    SUBMISSION_PATH = 'submission.csv'

    if len(os.listdir(MIDEL_SAVED_PATH)) != 7:
        if not os.path.isfile(TRAIN_FEATURES_PATH):
            transform_all_data(TRAIN_DATA_PATH, FEATURE_PATH, mode='train')

        train_df = pd.read_csv(TRAIN_FEATURES_PATH, header=0)
        train_df.dropna(inplace=True)
        train_df = shuffle(train_df)
        train(train_df, MIDEL_SAVED_PATH)

    if not os.path.isfile(TEST_FEATURES_PATH):
        transform_all_data(TEST_DATA_PATH, FEATURE_PATH, mode='test')

    test_df = pd.read_csv(TEST_FEATURES_PATH, header=0)
    test(test_df, MIDEL_SAVED_PATH, SUBMISSION_PATH, thod)
