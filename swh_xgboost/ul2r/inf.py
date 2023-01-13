'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2023-01-07 19:11:13
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2023-01-09 22:39:52
FilePath: /xgboost_test/dev_split/train.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
#导入所需要的包
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV #网格搜索
import matplotlib.pyplot as plt#可视化
import json
from collections import defaultdict
import pickle
# import seaborn as sns#绘图包
# 忽略警告
import warnings
warnings.filterwarnings("ignore")
# read in data

dtrain = xgb.DMatrix('/home/lht/wsdm_cup/lightgbm_ensemble_ultr/data/train/finetune_train.txt')
# dtrain = xgb.DMatrix('/home/lht/wsdm_cup/lightgbm_ensemble_ultr/data/train/finetune_train_w_vaild.txt')

ddev = xgb.DMatrix('/home/lht/wsdm_cup/lightgbm_ensemble_ultr/data/valid/valid_f.txt')

dtest = xgb.DMatrix('/home/lht/wsdm_cup/lightgbm_ensemble_ultr/data/test/test.txt')

bst = pickle.load(open("/home/lht/swh_xgboost/new_l2r/model/xgb_l2r_7_480_1000.pkl", "rb"))

# make prediction
preds = bst.predict(dtrain)
print(len(preds))

writer_res = open('/home/lht/swh_xgboost/new_l2r/xgboost_res/train','w')

for item in preds:
    writer_res.write(f'{item}\n')

exit(9)

path = '/home/lht/lic2022/xgboost_test/dev_split/test.csv'
data = pd.read_csv(path,header=None,names=['qid','pid'],sep='\t')
outputf = 'dev_test_top50.json'
q_dic = defaultdict(list)
i = 0


for index,row in data.iterrows():
    qid = row['qid']
    pid = row['pid']
    s = preds[i]
    q_dic[qid].append((s, pid))
    i = i + 1

print(i)

output = []
for q in q_dic:
    rank = 0
    cands = q_dic[q]
    cands.sort(reverse=True)
    for cand in cands:
        rank += 1
        output.append([q, cand[1], rank])
        if rank > 49:
            break

with open(outputf, 'w') as f:
    res = dict()
    for line in output:
        qid, pid, rank = line
        if qid not in res:
            res[qid] = [0] * 50
        res[qid][int(rank) - 1] = pid
    json.dump(res, f, ensure_ascii=False, indent='\t')




outputf = 'dev_test_top500_score_t.tsv'
output = []
for q in q_dic:
    rank = 0
    cands = q_dic[q]
    cands.sort(reverse=True)
    for cand in cands:
        rank += 1
        output.append([q, cand[0], cand[1], rank])
        if rank > 499:
            break


with open(outputf, 'w') as f:
    res = dict()
    for line in output:
        qid, pid, score,rank = line
        f.write(f"{qid}\t{pid}\t{score}\t{rank}\n")
