import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, ShuffleSplit, cross_validate
from sklearn.utils import shuffle

train_N_path = 'D:\\train_N.csv'
train_P_path = 'D:\\train_P.csv'
test_path = 'D:\\test.csv'
sub_path = 'D:\\sub.csv'

train_N_df = pd.read_csv(train_N_path, header = 0)
train_P_df = pd.read_csv(train_P_path, header = 0)
test_df = pd.read_csv(test_path, header = 0)
test_size = int(test_df[test_df.columns[0]].count())

train_df = pd.concat([train_N_df, train_P_df], axis = 0)
for i in range(16):
        train_df = pd.concat([train_df, train_P_df], axis = 0)

#filter_features = [train_N_df.columns[idx] for idx in [2, 5, 8, 10, 11, 21, 22, 24, 29]]
shuffle(train_df)

clf_lgbm = LGBMClassifier(boosting_type='gbdt', \
                          num_leaves=51, \
                          max_depth=-1, \
                          learning_rate=0.05, \
                          n_estimators=600, \
                          subsample_for_bin=1000, \
                          objective='binary', \
                          class_weight='balanced', \
                          min_split_gain=0.0, \
                          min_child_weight=0.001, \
                          min_child_samples=20, \
                          subsample=1.0, \
                          subsample_freq=0, \
                          colsample_bytree=1.0, \
                          reg_alpha=3.5, \
                          reg_lambda=1.0, \
                          random_state=None, \
                          importance_type='gain', \
                          silent=True)
sscv = ShuffleSplit(n_splits = 10, test_size = 0.25, random_state = None)
clf_lgbm.fit(train_df[train_df.columns[0:48]], train_df[train_df.columns[49]])
print clf_lgbm.score(train_df[train_df.columns[0:48]], train_df[train_df.columns[49]])

p_proba = clf_lgbm.predict_proba(test_df[test_df.columns[0:48]])
print p_proba

#0.005
p_list = [1 if p[1] > 0.0056 else 0 for p in p_proba]
print p_list
#test_pred_df = clf_lgbm.predict(test_df[test_df.columns[0:48]])
#res = pd.DataFrame(data = np.column_stack([np.reshape(test_df[test_df.columns[48]], test_size), test_pred_df]), columns = ['idx','result'])
res = pd.DataFrame(data = np.column_stack([np.reshape(test_df[test_df.columns[48]], test_size), p_list]), columns = ['idx','result'])
res.to_csv(sub_path, index = False)

scores = cross_validate(clf_lgbm, train_df[train_df.columns[0:48]], train_df[train_df.columns[49]], cv=sscv, scoring=['precision_macro', 'recall_macro'], return_train_score=False)
print scores
#print clf_lgbm.predict(train_P_df[train_P_df.columns[0:40]])
#print clf_lgbm.predict(train_N_df[train_N_df.columns[0:40]])
#p_proba = clf_lgbm.predict_proba(train_P_df[train_P_df.columns[0:40]])
#p_list = [1 if p[0] < 0.8 else 0 for p in p_proba]
#print p_list
#p_proba = clf_lgbm.predict_proba(train_N_df[train_N_df.columns[0:40]])
#p_list = [1 if p[0] < 0.8 else 0 for p in p_proba]
#print p_list

f_imps = clf_lgbm.feature_importances_
ft_imps = [(idx, f_imps[idx]) for idx in range(len(f_imps))]
ft_imps.sort(key = lambda x: x[1], reverse=True)

for ft in ft_imps:
    print ft
