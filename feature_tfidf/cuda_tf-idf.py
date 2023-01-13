import os
os.environ["CUDA_VISIBLE_DEVICES"] = '9'
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.device_count())
print(torch.cuda.current_device())
from tqdm import tqdm
import json
import random
import os
from collections import defaultdict
import numpy as np
max = 0

idf = np.zeros(21865,dtype=np.float)

# input = '/home/lht/wsdm_cup/data/annotate_data/finetune_data-1.txt'       # all
# input = '/home/lht/wsdm_cup/data/vaild/valid_data_f.txt'                  # dev
input = '/home/lht/wsdm_cup/data/test/wsdm_test_2_all.txt'                  # test
writer = open('/home/swh/legal/project/wsdmcup/new_tfidf/all/tfidf/test_content_title','w')


with open(input) as f:
    lines = f.read().split('\n')
    print(len(lines))
    for line in tqdm(lines):
        if not line:continue
        qid,qry,title,doc,label,bucket = line.split('\t')
        doc = doc.split('\x01')
        title = title.split('\x01')
        words = doc + title
        words = set(words)
        qry = qry.split('\x01')
        # qry = ','.join(qry)


        for word in words:
            word = int(word)
            idf[word] += 1

import math
for i in range(len(idf)):
    num = idf[i]
    idf[i] = math.log(397573/(1+num))
   
idf_vec = torch.from_numpy(idf)
idf_vec.to(device=device)


    

cnt = 0

with open(input) as f:
    lines = f.read().split('\n')
    print(len(lines))
    for line in tqdm(lines):
        if not line:continue
        qid,qry,title,doc,label,bucket = line.split('\t')

        qry = qry.split('\x01')


        qry_np = np.zeros(21865,dtype=np.float)
        for word in qry:
            word = int(word)
            # qry_np[word] = 1  # idf
            qry_np[word] += 1
        
        qry_vec = torch.from_numpy(qry_np)
        qry_vec.to(device=device)



        doc = doc.split('\x01')
        title = title.split('\x01')
        content_words = doc + title

        content_np = np.zeros(21865,dtype=np.float)

        for word in content_words:
            word = int(word)
            # content_np[word] = 1
            content_np[word] += 1
        
        content_vec = torch.from_numpy(content_np)

        # res = torch.dot(qry_vec,content_vec)
        # res = torch.dot(tmp,idf_vec)    
        # res = torch.mm(qry_vec,idf_vec)

        tmp = torch.mul(qry_vec,content_vec)
        res = torch.dot(tmp,idf_vec)

        writer.write(f'{res}\n')

print(cnt)