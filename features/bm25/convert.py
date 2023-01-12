'''
Author: lihaitao
Date: 2023-01-02 16:12:29
LastEditors: Do not edit
LastEditTime: 2023-01-05 23:44:24
FilePath: /lht/wsdm_cup/utils/pyserini_bm25/convert.py
'''
import json
from tqdm import tqdm

qid2pid = '/home/lht/wsdm_cup/data/annotate_data/qid2pid_label.json' ##

qid2pid_dict = json.load(open(qid2pid,'r'))


output_path = '/home/lht/wsdm_cup/utils/pyserini_bm25/output_coil.tsv'
output_file = open(output_path, 'r')


save_dict = {}
for line in tqdm(output_file.readlines()):
    line = line.strip('\n').split(' ')
    qid = str(line[0])
    pid = str(line[2])
    score = float(line[3])    
    if qid not in save_dict:
        save_dict[qid] = {}
    if pid in qid2pid_dict[qid]:
        save_dict[qid][pid]=score


with open('/home/lht/wsdm_cup/utils/pyserini_bm25/qld_filter_ranker.json','w',encoding='utf-8') as fp:
    json.dump(save_dict,fp,ensure_ascii=False)


# qid_idx_writer = open('/home/lht/wsdm_cup/utils/pyserini_bm25/final_bm25_ranker.tsv','w')
#
#
# for qid in tqdm(qid2pid_dict.keys()):
#     for pid in qid2pid_dict[qid]:
#         try:
#             qid_idx_writer.write(f'{save_dict[qid][pid]}\n')
#         except:
#             qid_idx_writer.write(f'{10000}\n')