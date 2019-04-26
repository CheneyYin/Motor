# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name  :    transform
   Author     :    雨住风停松子落
   E-mail     :
   date       :    2019/4/25
   Description:
                包含两个函数：extract_feature 和 transform_all_data，详细注释见函数内
-------------------------------------------------
   Change Activity:
                   2019/4/25:
-------------------------------------------------
"""
__author__ = '雨住风停松子落'

import numpy as np
import pandas as pd
import os
import time
import tsfresh.feature_extraction.feature_calculators as feature_cal

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def extract_feature(filename, isPositive=None):
    '''
    提取单个样本
    :param filename: 单个电机的F方向数据名
    :param isPositive: Node 未知， True 正样本， False 负样本
    :return: 0-43 常规特征44个, 44-47 number_peaks特征4个, 48-51 agg_autocorrelation特征4个,
             52-67 ratio_beyond_r_sigma特征16个, 68-83 fft_coefficient特征16个, 84 电机编号, 85 target
             从效率考虑，去除了44之后的特征
    '''

    dfF = pd.read_csv(filename, header=0, names=['ai1f', 'ai2f'])
    dfB = pd.read_csv(filename.replace('_F', '_B'), header=0, names=['ai1b', 'ai2b'])
    df = pd.concat([dfB, dfF], axis=1)

    feature = []

    for col in df.columns:
        t_df = df[col]
        feature.append(t_df.mean())
        feature.append(t_df.std())
        feature.append(t_df.median())
        feature.append(t_df.mad())
        feature.append(t_df.skew())
        feature.append(t_df.kurtosis())
        feature.append(t_df.max() - t_df.min())
        feature.append(feature_cal.absolute_sum_of_changes(t_df))
        feature.append(feature_cal.autocorrelation(t_df, 100))
        feature.append(feature_cal.binned_entropy(t_df, 100))
        feature.append(feature_cal.abs_energy(t_df))

    # for col in df.columns:
    #     feature.append(feature_cal.number_peaks(df[col], 1000))

    # param = [{'f_agg': 'mean', 'maxlag': 2}]
    # for col in df.columns:
    #     feature.append(feature_cal.agg_autocorrelation(df[col], param)[0][1])
    #
    # for pp in range(2, 6):
    #     for col in df.columns:
    #         feature.append(feature_cal.ratio_beyond_r_sigma(df[col], pp))
    #
    # param = [{"coeff": 10, 'attr': 'real'}, {"coeff": 10, "attr": "imag"},
    #         {"coeff": 10, "attr": "abs"}, {"coeff": 10, "attr": "angle"}]
    #
    # for col in df.columns:
    #     fft = feature_cal.fft_coefficient(df[col], param)
    #     for i in range(4):
    #         feature.append(fft[i][1])

    if isPositive is None:
        target = None
    else:
        if isPositive:
            target = 1
        else:
            target = 0

    feature.append(filename[-42: -6])
    feature.append(target)

    return feature


def transform_all_data(dataPath, destPath, mode='train'):
    '''
    提取所有样本
    :param dataPath: 数据源文件夹
    :param destPath: 提取的特征保存文件夹
    :param mode:     train or test
    :return: None
    '''
    features = []
    idx = 0

    saved_path = 'train_features.csv' if mode == 'train' else 'test_features.csv'

    for fpathe, dirs, fs in os.walk(dataPath):
        for f in fs:
            if f[-6:] == '_F.csv':
                t0 = time.time()
                filename = os.path.join(fpathe, f).replace('\\', '/')
                if mode == 'train':
                    features.append(extract_feature(filename, isPositive=('Positive' in filename)))
                else:
                    features.append(extract_feature(filename))
                t1 = time.time()
                print(idx, filename, 'time: ', t1 - t0)
                idx = idx + 1

    df = pd.DataFrame(data=np.array(features))
    df.to_csv(os.path.join(destPath, saved_path).replace('\\', '/'), index=False)
