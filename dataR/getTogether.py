import pandas as pd
import numpy as np

sub1 = pd.read_csv('../dataR/Features/sub/sub1.csv', header = 0)
sub2 = pd.read_csv('../dataR/Features/sub/sub2.csv', header = 0)
sub3 = pd.read_csv('../dataR/Features/sub/sub3.csv', header = 0)
sub4 = pd.read_csv('../dataR/Features/sub/sub4.csv', header = 0)
#print sub1['result'][2]

#sub_list = [1 if (sub1['result'][i] == 1 and sub2['result'][i] == 1) else 0 for i in range(5738)]
sub_list = [1 if (sub1['result'][i] == 1 and sub2['result'][i] == 1) else 0 for i in range(5738)]

res = pd.DataFrame(data=np.column_stack([sub1['idx'], sub_list]), columns=['idx', 'result'])

sub_list = [1 if (res['result'][i] == 1 and sub3['result'][i] == 1) else 0 for i in range(5738)]

res = pd.DataFrame(data=np.column_stack([sub1['idx'], sub_list]), columns=['idx', 'result'])

sub_list = [1 if (res['result'][i] == 1 and sub4['result'][i] == 1) else 0 for i in range(5738)]


res = pd.DataFrame(data=np.column_stack([sub1['idx'], sub_list]), columns=['idx', 'result'])
res.to_csv('../dataR/Features/sub/Submission.csv', index = False)
