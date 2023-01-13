'''
Author: lihaitao
Date: 2023-01-05 01:12:37
LastEditors: Do not edit
LastEditTime: 2023-01-05 12:35:51
FilePath: /lht/wsdm_cup/paddle_load.py
'''
import numpy as np
import warnings
import sys
from metrics import *
from Transformer4Ranking.model import *
from paddle.io import DataLoader
from dataloader import *
from args import config
from datetime import datetime
# import paddlehub as hub




# model = hub.Module(name='ernie_tiny', task='seq-cls', num_classes=2)

# load pretrained model

ptm = paddle.load('/home/lht/lic2022/DuReader-Retrieval-Baseline/pretrained-models/ernie_3.0_base_ch_dir/model.meta')
print(ptm.keys())
for k, v in ptm.items():
    print(k)
