# -*- coding: cp936 -*-
import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

train_N_path = '../data/Features/train_N_D.csv'
train_P_path = '../data/Features/train_P.csv'
test_path = '../data/Features/test.csv'

#train_N_path = '../data/Features_New/train_N1.csv'
#train_P_path = '../data/Features_New/train_P.csv'
#test_path = '../data/Features_New/test.csv'
sub_path = '../dataR/Features/sub/sub4.csv'
sub_path_rate = '../data/Features/sub_XGBOOST_rate.csv'

train_N_df = pd.read_csv(train_N_path, header = 0)
train_P_df = pd.read_csv(train_P_path, header = 0)
test_df = pd.read_csv(test_path, header = 0)
test_size = int(test_df[test_df.columns[0]].count())

train_df = pd.concat([train_N_df, train_P_df], axis = 0)

#for i in range(16):
#        train_df = pd.concat([train_df, train_P_df], axis = 0)


#print train_df.isnull().sum()
train_df.dropna(inplace=True)
#train_df.info()
#train_df [train_df.columns[0:43]] = train_df[train_df.columns[0:43]].astype(np.float32)
#train_df [train_df.columns[45]] = train_df[train_df.columns[45]].astype(np.float32)
#print np.isinf(train_df).all()

#filter_features = [train_N_df.columns[idx] for idx in [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43]]
filter_features = [train_N_df.columns[idx] for idx in [0,
                                                       1,
                                                       2,
                                                       3,
                                                       4,
                                                       5,
                                                       #6,
                                                       7,
                                                       9,
                                                       #10,
                                                       11,
                                                       12,
                                                       13,
                                                       14,
                                                       15,
                                                       16,
                                                       17,
                                                       18,
                                                       20,
                                                       21,
                                                       22,
                                                       23,
                                                       #24,
                                                       #25,
                                                       26,
                                                       27,
                                                       28,
                                                       29,
                                                       31,
                                                       #32,
                                                       33,
                                                       34,
                                                       35,
                                                       36,
                                                       37,
                                                       38,
                                                       39,
                                                       #40,
                                                       42,
                                                       #43
                                                       ]]



#est = XGBClassifier(learning_rate=0.01,
#                       n_estimators=5000,         # 树的个数--1200棵树建立xgboost
#                       max_depth=6,               # 树的深度
#                       min_child_weight = 1,      # 叶子节点最小权重
#                       gamma=0.,                  # 惩罚项中叶子结点个数前的参数
#                       subsample=0.8,             # 随机选择80%样本建立决策树
#                       colsample_btree=0.8,       # 随机选择80%特征建立决策树
#                       objective='binary:logistic',  # 指定损失函数
#                       scale_pos_weight=1,        # 解决样本个数不平衡的问题
#                       min_samples_split = 5,
#                       min_samples_leaf = 60,
#                       seed=27,
#                       reg_alpha=0.005,
                       #num_class = 2,
#                       random_state=0             # 随机数
#                       )
est = AdaBoostClassifier(RandomForestClassifier(n_estimators= 60, max_depth=13, min_samples_split=5,
                                  min_samples_leaf=20,oob_score=True, random_state=0),
                         algorithm="SAMME",
                         n_estimators=1200,
                         learning_rate=0.05
                         )
train_df = shuffle(train_df)

#est.fit(train_df[train_df.columns[0:63]], train_df[train_df.columns[65]])


est.fit(train_df[filter_features], train_df[train_df.columns[45]])


#fig,ax = plt.subplots(figsize=(15,15))
#plot_importance(est,
#                height=0.5,
#                ax=ax,
#               max_num_features=128)
#plt.show()



print est.score(train_df[filter_features], train_df[train_df.columns[45]])
#print est.score(train_df[train_df.columns[0:63]], train_df[train_df.columns[65]])
#test_df.info()

p_proba = est.predict_proba(test_df[filter_features])


print p_proba
#pp = []
#for p in p_proba:
    #pp.append(p[1])
    #print p[1]
#result_df = pd.DataFrame(pp)
#plt.show(result_df.plot(kind = 'kde'))
#plt.show()
#1.98166168718218e-06
#2.12271060222556E-06
#2.48371253016666E-06 fail
#0.0000023521932672728 fail
#2.24972111973169e-06 fail
#2.20142131807966e-06 320ef9ff-04e1-400f-b099-de091d3bc9d0
#2.21653095183105E-06 fail 71023692-b500-4d0d-b940-0bf278b78e42


#print int(len(p_list_mid)*0.3688)
#print p_list_mid[int(len(p_list_mid)*0.3688)]
p_list_midl = [p[1] for p in p_proba]
#print p_list_midl
p_list_midl.sort(reverse = True)
#print p_list_midl
print p_list_midl[int(len(p_list_midl)*0.3)]

p_list_rate = [p[1] for p in p_proba]



p_list = [1 if p[1] > p_list_midl[int(len(p_list_midl)*0.5)] else 0 for p in p_proba]
#p_list = [p[1] for p in p_proba]
#print p_list
#test_pred_df = clf_lgbm.predict(test_df[test_df.columns[0:44]])
#res = pd.DataFrame(data = np.column_stack([np.reshape(test_df[test_df.columns[44]], test_size), test_pred_df]), columns = ['idx','result'])
res = pd.DataFrame(data = np.column_stack([np.reshape(test_df[test_df.columns[44]], test_size), p_list]), columns = ['idx','result'])
res.to_csv(sub_path, index = False)

res = pd.DataFrame(data = np.column_stack([np.reshape(test_df[test_df.columns[44]], test_size), p_list_rate]), columns = ['idx','result'])
res.to_csv(sub_path_rate, index = False)
