import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, ShuffleSplit, cross_validate
train_N_path = './train_N.csv'
train_P_path = './train_P.csv'
test_path = './test.csv'
sub_path = './sub.csv'

train_N_df = pd.read_csv(train_N_path, header = 0)
train_P_df = pd.read_csv(train_P_path, header = 0)
test_df = pd.read_csv(test_path, header = 0)
test_size = int(test_df[test_df.columns[0]].count())

train_df = pd.concat([train_N_df, train_P_df], axis = 0)
#for i in range(2):
#        train_df = pd.concat([train_df, train_P_df], axis = 0)


clf_lgbm = LGBMClassifier(boosting_type='gbdt', \
                          num_leaves=51, \
                          max_depth=-1, \
                          learning_rate=0.05, \
                          n_estimators=600, \
                          subsample_for_bin=200, \
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
sscv = ShuffleSplit(n_splits = 10, test_size = 0.45, random_state = None)
clf_lgbm.fit(train_df[train_df.columns[0:32]], train_df[train_df.columns[33]])
print clf_lgbm.score(train_df[train_df.columns[0:32]], train_df[train_df.columns[33]])


#test_pred_df = clf_lgbm.predict(test_df[test_df.columns[0:32]])
#res = pd.DataFrame(data = np.column_stack([np.reshape(test_df[test_df.columns[32]], test_size), test_pred_df]), columns = ['idx','result'])
#res.to_csv(sub_path, index = False)

scores = cross_validate(clf_lgbm, train_df[train_df.columns[0:32]], train_df[train_df.columns[33]], cv=sscv, scoring=['precision_macro', 'recall_macro'], return_train_score=False)
print scores
