'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2023-01-07 19:11:13
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2023-01-09 22:07:21
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

dtrain = xgb.DMatrix('/home/lht/wsdm_cup/lightgbm_ensemble_ultr/data/train/finetune_train_w_vaild.txt')
dtest = xgb.DMatrix('/home/lht/wsdm_cup/lightgbm_ensemble_ultr/data/valid/valid_f.txt')
# dtrain = xgb.DMatrix('/home/lht/swh_xgboost/swh/finetune_train_3features.txt')
# dtest = xgb.DMatrix('/home/lht/swh_xgboost/swh/finetune_dev_3features.txt')

def get_dcg(ordered_labels):
    return np.sum((2 ** ordered_labels - 1) / np.log2(np.arange(ordered_labels.shape[0]) + 2))

def calc_dcg(query_list, K=10, prefix=''):
    """ discounted cumulative gain """
    dcg_10 = []
    for item in query_list:
        pred, label = zip(*item)
        label = np.array(label)
        ranking = np.argsort(pred)[::-1]
     
        topk_rankings = ranking[:K]
        ordered_label = label[topk_rankings]
        dcg_10.append(get_dcg(ordered_label)) 

    return np.mean(dcg_10)

def dcg_10(preds, train_data):
    
    y_train = train_data.get_label()
    group = train_data.get_group()

    all_query = []
    tmp = []
    index = 0
    # print(len(preds))
    # print(len(y_train))
    for i in group:
        tmp = []
        for j in range(int(i)):
            tmp.append([preds[index], y_train[index]])
            index = index + 1
        all_query.append(tmp)


    dcg_all = calc_dcg(all_query, prefix='all')
    
 
    # preds = 1. / (1. + np.exp(-preds))
    return 'dcg_10', dcg_all 
    # return 'dcg_10', dcg_all , True


watchlist  = [(dtest,'eval'), (dtrain,'train')]
# param = {'objective':'binary:logistic'}
# param = {'max_depth':20, 'eta':0.05, 'objective':'rank:pairwise','gamma': 0.1,'min_child_weight':2,'lambda': 1,'subsample':0.7}
# param = {'max_depth':7, 'eta':0.05, 'objective':'rank:pairwise','gamma': 0.1,'min_child_weight':2,'lambda': 1,'subsample':0.7,'eval_metric':'ndcg'}
param = {'max_depth':7, 'eta':0.05, 'objective':'rank:pairwise','gamma': 0.3,'min_child_weight':2,'lambda': 1,'subsample':0.7}
num_round = 450
bst = xgb.train(param, dtrain, num_round,watchlist,early_stopping_rounds=1000,feval=dcg_10)
# xgb.cv(param, dtrain, num_round, nfold=5,metrics={'error'}, seed = 0)

pickle.dump(bst, open("/home/lht/swh_xgboost/new_l2r/model/xgb_l2r_6_450_1000.pkl", "wb"))

bst = pickle.load(open("/home/lht/swh_xgboost/new_l2r/model/xgb_l2r_6_450_1000.pkl", "rb"))
# make prediction
preds = bst.predict(dtest)
print(len(preds))

writer_res = open('/home/lht/swh_xgboost/swh/res_ul2r_pairwise_6_453level_1000.tsv','w')

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
