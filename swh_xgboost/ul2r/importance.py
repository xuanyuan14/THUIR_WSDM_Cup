'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2023-01-07 19:11:13
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2023-01-09 23:25:56
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



gbm =xgb.Booster(model_file="/home/lht/swh_xgboost/new_l2r/model/xgb_l2r_7_480_1000.pkl")

importances = gbm.feature_importance(importance_type='split')
feature_names = gbm.feature_name()

sum = 0.
for value in importances:
    sum += value

for feature_name, importance in zip(feature_names, importances):
    if importance != 0:
        feat_id = int(feature_name.split('_')[1]) + 1
        print('{} : {} : {} : {}'.format(feat_id,  importance, importance / sum))