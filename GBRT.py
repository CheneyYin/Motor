import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import warnings

warnings.filterwarnings("ignore")

train_N_path = '../MotorData/Features/train_N.csv'
train_P_path = '../MotorData/Features/train_P.csv'
test_path = '../MotorData/Features/test.csv'
sub_path = '../MotorData/Features/sub_voting_new.csv'

train_N_df = pd.read_csv(train_N_path, header=0)
train_P_df = pd.read_csv(train_P_path, header=0)
test_df = pd.read_csv(test_path, header=0)
test_size = int(test_df[test_df.columns[0]].count())

train_df = pd.concat([train_N_df, train_P_df], axis=0)

for i in range(14):
    train_df = pd.concat([train_df, train_P_df], axis=0)

# print train_df.isnull().sum()
# train_df.dropna(inplace=False)
print train_df.isnull().sum().sum()
train_df = train_df.fillna(0)
print train_df.isnull().sum().sum()
train_df = shuffle(train_df)

# z_score = []
# for i in range(44):
#     mean_col = train_df[str(i)].std()
#     std_col = train_df[str(i)].std()
#     z_score.append([mean_col, std_col])
#     train_df[str(i)] = (train_df[str(i)] - mean_col) * 1.0 / std_col
# train_df.info()
# train_df [train_df.columns[0:43]] = train_df[train_df.columns[0:43]].astype(np.float32)
# train_df [train_df.columns[45]] = train_df[train_df.columns[45]].astype(np.float32)
# print np.isinf(train_df).all()

filter_features = [train_N_df.columns[idx] for idx in range(44)]
# [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28,
#  29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43]]

# est = GradientBoostingClassifier(n_estimators=1200,
#                                  learning_rate=0.05,
#                                  min_samples_leaf=60,
#                                  max_depth=10,
#                                  min_samples_split=5,
#                                  # max_features=9,
#                                  subsample=0.7,
#                                  random_state=0,
#                                  loss='deviance')

est = VotingClassifier(estimators=[
    ('LogisticRegression', LogisticRegression(penalty='l2', solver='lbfgs', verbose=0, class_weight='balanced')),
    ('DecisionTreeClassifier', DecisionTreeClassifier(class_weight='balanced')),
    ('RandomForestClassifier', RandomForestClassifier(n_estimators=1200, verbose=0, class_weight='balanced')),
    ('GradientBoostingClassifier', GradientBoostingClassifier(n_estimators=1200, verbose=0, learning_rate=0.05)),
    ('GaussianNB', GaussianNB()),
    ('KNeighborsClassifier', KNeighborsClassifier())
], voting='soft', weights=[1, 2, 2, 2, 1, 1])

est.fit(train_df[filter_features], train_df[train_df.columns[45]])
print est.score(train_df[filter_features], train_df[train_df.columns[45]])

# test_df.info()

# for i in range(44):
#     test_df[str(i)] = (test_df[str(i)] - z_score[i][0]) * 1.0 / z_score[i][1]

test_df = test_df.fillna(0)

p_proba = est.predict_proba(test_df[filter_features])

# print p_proba
# pp = []
# for p in p_proba:
#     pp.append(p[1])
#     # print p[1]
# result_df = pd.DataFrame(pp)
# result_df.plot(kind='kde')
# plt.show()
p_list = [1 if p[1] > 0.01 else 0 for p in p_proba]
# p_list = [p[1] for p in p_proba]
# print p_list
# test_pred_df = clf_lgbm.predict(test_df[test_df.columns[0:44]])
# res = pd.DataFrame(data = np.column_stack([np.reshape(test_df[test_df.columns[44]], test_size), test_pred_df]), columns = ['idx','result'])
res = pd.DataFrame(data=np.column_stack([np.reshape(test_df[test_df.columns[44]], test_size), p_list]),
                   columns=['idx', 'result'])
res.to_csv('../MotorData/Features/sub_voting_2.csv', index=False)
