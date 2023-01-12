import json
import sys
__author__ = 'chenjia'


model = sys.argv[1]  # bm25, qld
data = sys.argv[2]  # pyserini, valid, test
fw = open(f'/home/lht/wsdm_cup/utils/{data}_bm25/result_{model}.tsv', 'w')

with open(f'/home/lht/wsdm_cup/utils/{data}_bm25/q2pids.json') as f1:
    q2pids = json.load(f1)

score_dict = {}
with open(f'/home/lht/wsdm_cup/utils/{data}_bm25/output_{model}.tsv') as f2:
    for line in f2:
        es = line.strip().split()
        qid, docid, score = es[0].strip(), es[2].strip(), es[4].strip()
        if qid not in score_dict:
            score_dict[qid] = {}
        if docid not in score_dict[qid]:
            score_dict[qid][docid] = score

filename_dict = {'pyserini': 'finetune_data-1.txt', 'valid': 'valid_data_f.txt', 'test': 'wsdm_test_2_all.txt'}
file = filename_dict[data]
qid_dict = {}
with open(f'/home/chenjia/CM2BERT/baidu_data/WSDMCUP_BaiduPLM_Paddle-main/data/non_stop/{file}', 'rb') as f3:
    for line in f3:
        line_list = line.strip(b'\n').split(b'\t')
        qid, query, title, content, label, freq = line_list
        qid = qid.decode()
        if qid not in qid_dict:
            qid_dict[qid] = 0
        docid = str(q2pids[qid][qid_dict[qid]])

        if qid in score_dict and docid in score_dict[qid]:
            score = score_dict[qid][docid]
        else:
            score = 0.0
        qid_dict[qid] += 1
        fw.write(f'{score}\n')