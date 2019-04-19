import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

train_N_path = '../data/Features/train_N_D.csv'
train_P_path = '../data/Features/train_P.csv'
test_path = '../data/Features/test.csv'
sub_path = '../dataR/Features/sub/sub2.csv'

train_N_df = pd.read_csv(train_N_path, header = 0)
train_P_df = pd.read_csv(train_P_path, header = 0)
test_df = pd.read_csv(test_path, header = 0)
test_size = int(test_df[test_df.columns[0]].count())

train_df = pd.concat([train_N_df, train_P_df], axis = 0)

for i in range(16):
        train_df = pd.concat([train_df, train_P_df], axis = 0)


#print train_df.isnull().sum()
train_df.dropna(inplace=True)
#train_df.info()
#train_df [train_df.columns[0:43]] = train_df[train_df.columns[0:43]].astype(np.float32)
#train_df [train_df.columns[45]] = train_df[train_df.columns[45]].astype(np.float32)
#print np.isinf(train_df).all()

filter_features = [train_N_df.columns[idx] for idx in [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43]]



est = GradientBoostingClassifier(n_estimators = 1200,
                                learning_rate = 0.05,
                                min_samples_leaf = 60, 
                                max_depth = 10,
                                min_samples_split = 5,
                                #max_features=9,
                                subsample = 0.7,
                                random_state = 0,
                                loss = 'deviance')
train_df = shuffle(train_df)
est.fit(train_df[filter_features], train_df[train_df.columns[45]])
print est.score(train_df[filter_features], train_df[train_df.columns[45]])

test_df.info()

p_proba = est.predict_proba(test_df[filter_features])


#print p_proba
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


p_list_midl = [p[1] for p in p_proba]
#print p_list_midl
p_list_midl.sort(reverse = True)
#print p_list_midl
print p_list_midl[int(len(p_list_midl)*0.3)]


p_list = [1 if p[1] > p_list_midl[int(len(p_list_midl)*0.55)] else 0 for p in p_proba]

#p_list = [p[1] for p in p_proba]
#print p_list
#test_pred_df = clf_lgbm.predict(test_df[test_df.columns[0:44]])
#res = pd.DataFrame(data = np.column_stack([np.reshape(test_df[test_df.columns[44]], test_size), test_pred_df]), columns = ['idx','result'])
res = pd.DataFrame(data = np.column_stack([np.reshape(test_df[test_df.columns[44]], test_size), p_list]), columns = ['idx','result'])
res.to_csv(sub_path, index = False)
