#!/bin/bash
###
 # @Author: lihaitao
 # @Date: 2022-12-21 14:37:37
 # @LastEditors: error: git config user.name && git config user.email & please set dead value or install git
 # @LastEditTime: 2023-01-10 22:47:19
 # @FilePath: /lht/wsdm_cup/finetune.sh
### 
export PYTHONUNBUFFERED=1
python /home/lht/swh_finetune/finetune_listwise_3.py \
--gpu_device 7 \
--lr 5e-5 \
--emb_dim 768 \
--nlayer 12 \
--nhead 12 \
--finetune_epoch 100 \
--dropout 0.1 \
--eval_batch_size 128 \
--save_step 2000 \
--eval_step 2000 \
--init_parameters /home/lht/wsdm_cup/model/ckpt-2022-12-22_12/save_steps38000_6.31402.model \
--valid_annotate_path /home/swh/legal/project/new_pre_train/data/listwise_all.txt \
--test_annotate_path /home/lht/wsdm_cup/data/vaild/valid_data_f.txt \

