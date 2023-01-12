'''
Author: lihaitao
Date: 2023-01-02 00:53:40
LastEditors: Do not edit
LastEditTime: 2023-01-03 21:45:56
FilePath: /lht/wsdm_cup/bm25_metric.py
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

test_annotate_dataset = TestDataset('/home/lht/wsdm_cup/data/vaild/valid_data_f.txt', max_seq_len=128, data_type='annotate')

total_scores_1 = []
total_scores_2 = []

total_scores = []
score_path = f'/home/lht/wsdm_cup/utils/valid_bm25_bigram/result_bm25.tsv'
score_file = open(score_path, 'r')
for line in score_file.readlines():
    total_scores.append(float(line))
    # print(len(total_scores))
    # print(len(test_annotate_dataset.total_labels))
    # print(len(test_annotate_dataset.total_qids))

result_dict_ann = evaluate_all_metric(
    qid_list=test_annotate_dataset.total_qids,
    label_list=test_annotate_dataset.total_labels,
    score_list=total_scores,
    freq_list=test_annotate_dataset.total_freqs
)
print(
    f'valid annotate | '
    f'dcg@10: all {result_dict_ann["all_dcg@10"]:.6f} | '
    f'high {result_dict_ann["high_dcg@10"]:.6f} | '
    f'mid {result_dict_ann["mid_dcg@10"]:.6f} | '
    f'low {result_dict_ann["low_dcg@10"]:.6f} | '
    f'pnr {result_dict_ann["pnr"]:.6f}'
)