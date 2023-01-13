# -*- encoding: utf-8 -*-
'''
@Time    :   2022/06/12 14:49:28
@Author  :   Chu Xiaokai 
@Contact :   xiaokaichu@gmail.com
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


def process_data(query, title, content, max_seq_len):
    """ process [query, title, content] into a tensor
        [CLS] + query + [SEP] + title + [SEP] + content + [SEP] + [PAD]
    """
    data = [config._CLS_]
    segment = [0]

    data = data + [int(item) + 10 for item in query.split(b'\x01')]  # query
    data = data + [config._SEP_]
    segment = segment + [0] * (len(query.split(b'\x01')) + 1)

    data = data + [int(item) + 10 for item in title.split(b'\x01')]  # content
    data = data + [config._SEP_]  # sep defined as 1
    segment = segment + [1] * (len(title.split(b'\x01')) + 1)

    data = data + [int(item) + 10 for item in content.split(b'\x01')]  # content
    data = data + [config._SEP_]
    segment = segment + [1] * (len(content.split(b'\x01')) + 1)

    # padding
    padding_mask = [False] * len(data)
    if len(data) < max_seq_len:
        padding_mask += [True] * (max_seq_len - len(data))
        data += [config._PAD_] * (max_seq_len - len(data))
    else:
        padding_mask = padding_mask[:max_seq_len]
        data = data[:max_seq_len]

    # segment id
    if len(segment) < max_seq_len:
        segment += [1] * (max_seq_len - len(segment))
    else:
        segment = segment[:max_seq_len]
    padding_mask = paddle.to_tensor(padding_mask, dtype='int32')
    data = paddle.to_tensor(data, dtype="int32")
    segment = paddle.to_tensor(segment, dtype="int32")
    return data, segment, padding_mask


# buffer = []
# fpath = config.valid_annotate_path

# with open(fpath, 'rb') as f:
#     for i in tqdm(range(129584+100)):
#         line = f.readline()
#         if not line:continue
#         items = line.split(b'S')
#         # print('---------------------------------------')
#         # print(len(items))
#         # print('---------------------------------------')
#         # print(line)
#         pos_line = items[0]
#         negs = items[1:-1]
#         if len(negs) < 5:continue
#         '''process pos'''
#         line_list = pos_line.strip(b'\n').split(b'\t')
#         qid, query, title, content, label, freq = line_list
#         src_input, src_segment, src_padding_mask = process_data(query, title, content, 128)
#         buffer.append([src_input, src_segment, src_padding_mask, label])
    

#         '''process negs'''
#         if len(negs) >=7:
#             negs = random.sample(negs,k=7)
#         else:
#             negs = random.choices(negs,k=7)
        
#         if len(negs) != 7:
#             print('!!!error')

#         for neg_line in negs:
#             line_list = neg_line.strip(b'\n').split(b'\t')
#             qid, query, title, content, label, freq = line_list
#             src_input, src_segment, src_padding_mask = process_data(query, title, content, 128)
#             buffer.append([src_input, src_segment, src_padding_mask, label])
#         # print(len(buffer))
#     leng = len(buffer)

#     print(len(buffer))
#     print(float(leng)/8.0)

# # if shuffle:
# #     np.random.shuffle(buffer)
# length = int(len(buffer)/240)

# the_end = length * 240

# print('-----',len(buffer))
# print('the end:',the_end)

# buffer = buffer[0:the_end]





random.seed(config.seed+1)
random.seed(config.seed)
np.random.seed(config.seed)
paddle.set_device(f"gpu:{config.gpu_device}")
paddle.seed(config.seed)
print(config)
exp_settings = config.exp_settings

dt_string = datetime.today().strftime('%Y-%m-%d')

model = TransformerModel(
    ntoken=config.ntokens,
    hidden=config.emb_dim,
    nhead=config.nhead,
    nlayers=config.nlayers,
    dropout=config.dropout,
    mode='finetune'
)

# model = hub.Module(name='ernie_tiny', task='seq-cls', num_classes=2)

# load pretrained model
if config.init_parameters != "":
    print('load warm up model ', config.init_parameters)
    ptm = paddle.load(config.init_parameters)
    for k, v in model.state_dict().items():
        if not k in ptm:
            pass
            print("warning: not loading " + k)
        else:
            print("loading " + k)
            v.set_value(ptm[k])

# 优化器设置
scheduler = get_linear_schedule_with_warmup(config.lr, config.warmup_steps,
                                        config.max_steps)
decay_params = [
    p.name for n, p in model.named_parameters()
    if not any(nd in n for nd in ["bias", "norm"])
]

optimizer = paddle.optimizer.AdamW(
    learning_rate=scheduler,
    parameters=model.parameters(),
    weight_decay=config.weight_decay,
    apply_decay_param_fun=lambda x: x in decay_params,
    grad_clip=nn.ClipGradByNorm(clip_norm=0.5)
)
criterion = nn.BCEWithLogitsLoss()

vaild_annotate_dataset = TestDataset_swh(config.valid_annotate_path, max_seq_len=config.max_seq_len, data_type='finetune')
vaild_annotate_loader = DataLoader(vaild_annotate_dataset, batch_size=config.eval_batch_size)
test_annotate_dataset = TestDataset(config.test_annotate_path, max_seq_len=config.max_seq_len, data_type='annotate')
test_annotate_loader = DataLoader(test_annotate_dataset, batch_size=256)



# import math
# def swh_loss(scores):
#     x1 = math.exp(scores[0])
#     x2 = 0

#     for i in len(scores):
#         x2 += math.exp(scores[i])

#     x3 = x1/x2
#     x4 = math.log(x3) * (-1)
#     return x4

    
from paddle.nn.functional import softmax_with_cross_entropy
from tqdm import tqdm

# loss_function = softmax_with_cross_entropy()

idx = 0

a = [0]
all_label = paddle.to_tensor(a, dtype=paddle.int64)


score_path = '/home/lht/swh_finetune/res/score_batch_128.csv'
loss_path = '/home/lht/swh_finetune/res/loss.csv'
with open(score_path,'w') as f:
    pass

save_idx = 0


import paddle
class MyLoss(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
 
    def forward(self, x):
        # 使用paddle内置的cross_entropy算子实现算法
        x1 = paddle.exp(x)
        x2 = x1[0]
        x3 = paddle.sum(x1)
        x4 = x2/x3
        loss = paddle.log(x4) * (-1)
        return loss


myloss = MyLoss()


for i in range(config.finetune_epoch):
    print(f'Start epoch {i}')
    for valid_data_batch in vaild_annotate_loader:
        model.train()
        optimizer.clear_grad()
        
        src_input, src_segment, src_padding_mask, label = valid_data_batch
       

        score = model(
            src=src_input,
            src_segment=src_segment,
            src_padding_mask=src_padding_mask,
        )

        # b = paddle.reshape(b,[4,16])

        if score.shape[0] != 128:
            print('error')
            continue

        score = paddle.reshape(score,[16,8])

        ctr_loss = 0
        for i in range(16):
            ctr_loss += myloss(score[i])
        # ctr_loss = criterion(score, paddle.to_tensor(label, dtype=paddle.float32))
        ctr_loss.backward()
        optimizer.step()
        scheduler.step()

        if idx % config.log_interval == 0:
            print(f'{idx:5d}th step | loss {ctr_loss.item():5.6f}')

        if idx % config.eval_step == 0:
            save_idx += 1
            model.eval()
            # ------------   evaluate on annotated data -------------- #
            total_scores = []
            for test_data_batch in tqdm(test_annotate_loader):
                src_input, src_segment, src_padding_mask, label = test_data_batch
                score = model(
                    src=src_input,
                    src_segment=src_segment,
                    src_padding_mask=src_padding_mask,
                )
                score = score.cpu().detach().numpy().tolist()
                total_scores += score

            result_dict_ann = evaluate_all_metric(
                qid_list=test_annotate_dataset.total_qids,
                label_list=test_annotate_dataset.total_labels,
                score_list=total_scores,
                freq_list=test_annotate_dataset.total_freqs
            )
            print(
                f'{idx}th step valid annotate | '
                f'dcg@10: all {result_dict_ann["all_dcg@10"]:.6f} | '
                f'high {result_dict_ann["high_dcg@10"]:.6f} | '
                f'mid {result_dict_ann["mid_dcg@10"]:.6f} | '
                f'low {result_dict_ann["low_dcg@10"]:.6f} | '
                f'pnr {result_dict_ann["pnr"]:.6f}'
            )

            with open(score_path,'a') as f:
                f.write(str(save_idx) + f',{result_dict_ann["all_dcg@10"]:.6f} | ' + '\n')

            if idx % config.save_step == 0 and idx > 0:
                paddle.save(model.state_dict(),f'/home/lht/swh_finetune/model/batch_128/{save_idx}-{result_dict_ann["all_dcg@10"]:.6f}.model')
                    
        idx += 1