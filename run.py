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

if __name__ == '__main__':
    thod = {
        'AdaBoostClassifier': 0.55,
        'BaggingClassifier': 0.55,
        'GradientBoostingClassifier': 0.55,
        'LGBMClassifier': 0.55,
        'VotingClassifier': 0.5,
        'XGBClassifier': 0.39,
    }

    upsampling = {
        'AdaBoostClassifier': 1,
        'BaggingClassifier': 1,
        'GradientBoostingClassifier': 16,
        'LGBMClassifier': 16,
        'VotingClassifier': 16,
        'XGBClassifier': 1,
    }

    TRAIN_DATA_PATH = 'D:/workspace/MotorData/Motor_tain/'
    TEST_DATA_PATH = 'D:/workspace/MotorData/Motor_testP/'
    TRAIN_FEATURES_PATH = './Features/train_features.csv'
    TEST_FEATURES_PATH = './Features/test_features.csv'
    MIDEL_SAVED_PATH = './Model/'
    FEATURE_PATH = './Features/'

    SUBMISSION_PATH = 'submission.csv'

    if len(os.listdir(MIDEL_SAVED_PATH)) != 6:
        if not os.path.isfile(TRAIN_FEATURES_PATH):
            transform_all_data(TRAIN_DATA_PATH, FEATURE_PATH, mode='train')

        train_df = pd.read_csv(TRAIN_FEATURES_PATH, header=0)
        train_df.dropna(inplace=True)
        train(train_df, MIDEL_SAVED_PATH, upsampling)

    if not os.path.isfile(TEST_FEATURES_PATH):
        transform_all_data(TEST_DATA_PATH, FEATURE_PATH, mode='test')

    test_df = pd.read_csv(TEST_FEATURES_PATH, header=0)
    test(test_df, MIDEL_SAVED_PATH, SUBMISSION_PATH, thod)
