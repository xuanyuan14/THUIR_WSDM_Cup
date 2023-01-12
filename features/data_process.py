import numpy as np
import paddle
import sys
import gzip
import os
from tqdm import tqdm
import json
from collections import Counter
import paddle.nn as nn
from Transformer4Ranking.model import TransformerModel

__author__ = 'chenjia'


def extract_tgt_queries():
    fpath = './data/wsdm_test_2_all.txt'
    tgt_qdict = {}
    for line in open(fpath, 'rb'):
        line_list = line.strip(b'\n').split(b'\t')
        qid, query, title, content, label, freq = line_list
        query = query.decode()
        if query not in tgt_qdict:
            tgt_qdict[query] = 1

    print(len(tgt_qdict))
    with open('./data/tgt_qdict.json', 'w') as fw:
        json.dump(tgt_qdict, fw)


paddle.set_device(f"gpu:1")
_CLS_ = 0
_SEP_ = 1
_PAD_ = 2
_MASK_ = 3
def process_data(content, max_seq_len):
    """ process [query] into a tensor
        [CLS] + content + [SEP] + [PAD]
    """
    data = [_CLS_]
    segment = [0]

    data = data + [int(item) + 10 for item in content.split(b'\x01')]  # content
    data = data + [_SEP_]
    segment = segment + [1] * (len(content.split(b'\x01')) + 1)

    # padding
    padding_mask = [False] * len(data)
    if len(data) < max_seq_len:
        padding_mask += [True] * (max_seq_len - len(data))
        data += [_PAD_] * (max_seq_len - len(data))
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


def filter_valid_with_tgt_queries():
    with open('./data/tgt_qdict.json') as fr:
        tgt_qdict = json.load(fr)

    fw = open('./data/annotate_data/valid_data_f.txt', "wb")

    model = TransformerModel(
        ntoken=22000,
        hidden=768,
        nhead=12,
        nlayers=12,
        dropout=0.0,
        mode='finetune'
    )

    ptm = paddle.load('./model/ckpt-2022-12-29/save_steps81000_10.06159.model')
    for k, v in model.state_dict().items():
        if not k in ptm:
            pass
            # print("warning: not loading " + k)
        else:
            # print("loading " + k)
            v.set_value(ptm[k])

    model.eval()
    tgt_vec_list = []
    for tgt_q in tgt_qdict:
        tgt_q_byte = tgt_q.encode()
        data, segment, padding_mask = process_data(tgt_q_byte, max_seq_len=20)
        data, segment, padding_mask = paddle.unsqueeze(data, axis=0), paddle.unsqueeze(segment, axis=0), paddle.unsqueeze(padding_mask, axis=0)
        # print('data', data)

        with paddle.no_grad():
            vec = model(
                src=data,
                src_segment=segment,
                src_padding_mask=padding_mask,
                generate_vec=True
            )
            vec = vec.squeeze()  # 768
            tgt_vec_list.append(vec)

    tgt_vec_mat = paddle.stack(tgt_vec_list, axis=0)  # 7700 * 768
    print(tgt_vec_mat)

    valid_file_list = ['./data/annotate_data/finetune_data-1.txt']
    sim_q_dict, line_cnt, process_cnt = {}, 0, 0
    for file in valid_file_list:
        for line in open(file, 'rb'):
            line_list = line.strip(b'\n').split(b'\t')
            qid, query, title, content, label, freq = line_list
            qid = qid.decode()
            if qid in sim_q_dict:
                if sim_q_dict[qid] == 1:
                    fw.write(line.strip(b'\n') + b'\n')
                    line_cnt += 1

                process_cnt += 1
                if process_cnt % 1000 == 0:
                    print(process_cnt, line_cnt)
                continue

            data, segment, padding_mask = process_data(query, max_seq_len=20)
            data, segment, padding_mask = paddle.unsqueeze(data, axis=0), paddle.unsqueeze(segment, axis=0), paddle.unsqueeze(padding_mask, axis=0)

            with paddle.no_grad():
                vec = model(
                    src=data,
                    src_segment=segment,
                    src_padding_mask=padding_mask,
                    generate_vec=True
                )
                vec = vec.squeeze()  # 768

            # print(qid, vec[:20])
            sim = nn.CosineSimilarity(axis=1)(tgt_vec_mat, paddle.broadcast_to(vec, shape=tgt_vec_mat.shape))
            max_sim = paddle.amax(sim)
            avg_sim = paddle.mean(sim)
            print(qid, max_sim, avg_sim)
            if max_sim >= 0.9962:
                if qid not in sim_q_dict:
                    sim_q_dict[qid] = 1
                fw.write(line.strip(b'\n') + b'\n')
                line_cnt += 1
            else:
                if qid not in sim_q_dict:
                    sim_q_dict[qid] = 0

            process_cnt += 1
            if process_cnt % 1000 == 0:
                print(process_cnt, line_cnt)

    print(f'{len(sim_q_dict)} queries left for validation')
    print(f'{line_cnt} lines used for validation')


def process_valid_f():
    fpath = './data/annotate_data/valid_data_f.txt'
    line_list = []
    with open(fpath, 'rb') as f:
        for line in f:
            if len(line.strip()) > 0:
                line_list.append(line.strip())
    fw = open('./data/annotate_data/valid_data_f1.txt', 'wb')
    fw.write(b'\n'.join(line_list))


def div_list(ls, n):
    ls_len = len(ls)
    j = ls_len // n
    k = ls_len % n
    ls_return = []
    for i in range(0, (n - 1) * j , j):
        ls_return.append(ls[i: i + j])
    # 算上末尾的j+k
    ls_return.append(ls[(n - 1) * j: ])
    return ls_return


def filter_click_data_with_tgt_queries(directory_path, files):
    with open('./data/tgt_qdict.json') as fr:
        tgt_qdict = json.load(fr)
    output_dir = './data/filtered_click_data'
    buffer = []

    # encode all tgt queries
    model = TransformerModel(
        ntoken=22000,
        hidden=768,
        nhead=12,
        nlayers=12,
        dropout=0.1,
        mode='finetune'
    )

    ptm = paddle.load('./model/ckpt-2022-12-06/save_steps35000_10.95890.model')
    for k, v in model.state_dict().items():
        if not k in ptm:
            pass
            # print("warning: not loading " + k)
        else:
            # print("loading " + k)
            v.set_value(ptm[k])

    tgt_vec_list = []
    for tgt_q in tgt_qdict:
        tgt_q_byte = tgt_q.encode()
        data, segment, padding_mask = process_data(tgt_q_byte, max_seq_len=20)
        data, segment, padding_mask = paddle.unsqueeze(data, axis=0), paddle.unsqueeze(segment, axis=0), paddle.unsqueeze(padding_mask, axis=0)
        # print('data', data)

        with paddle.no_grad():
            vec = model(
                src=data,
                src_segment=segment,
                src_padding_mask=padding_mask,
                generate_vec=True
            )
            vec = vec.squeeze()  # 768
            tgt_vec_list.append(vec)

    tgt_vec_mat = paddle.stack(tgt_vec_list, axis=0)  # 7700 * 768
    print(tgt_vec_mat.shape)

    sim_qdict, add_flag = {}, False
    for file in files:
        print(f'Processing {file}...')
        if file[-3:] != '.gz':  # part-00000.gz is for evaluation
            continue
        fw = open(f'{output_dir}/{file}', 'wb')
        with gzip.open(os.path.join(directory_path, file), 'rb') as f:
            for line in f.readlines():
                line_list = line.strip(b'\n').split(b'\t')

                if len(line_list) == 3:  # new query
                    qid, cur_query = line_list[0], line_list[1]
                    qid = qid.decode()
                    if qid in sim_qdict:
                        add_flag = True
                        fw.write(line.strip(b'\n') + b'\n')
                    else:
                        data, segment, padding_mask = process_data(cur_query, max_seq_len=20)
                        data, segment, padding_mask = paddle.unsqueeze(data, axis=0), paddle.unsqueeze(segment, axis=0), paddle.unsqueeze(padding_mask, axis=0)

                        with paddle.no_grad():
                            vec = model(
                                src=data,
                                src_segment=segment,
                                src_padding_mask=padding_mask,
                                generate_vec=True
                            )
                            vec = vec.squeeze()  # 768

                        sim = nn.CosineSimilarity(axis=1)(tgt_vec_mat, paddle.broadcast_to(vec, shape=tgt_vec_mat.shape))
                        max_sim = paddle.amax(sim)
                        # print(sim.shape)
                        if max_sim >= 0.915:
                            add_flag = True
                            if qid not in sim_qdict:
                                sim_qdict[qid] = 1
                            fw.write(line.strip(b'\n') + b'\n')
                        else:
                            add_flag = False

                elif len(line_list) > 6:  # urls
                    if add_flag:
                        fw.write(line.strip(b'\n') + b'\n')


def remapping_qid_for_files(fpath):
    q2qid_dict = {}
    fw = open(fpath.split('.')[0] + '-1.txt', 'wb')
    for line in open(fpath, 'rb'):
        line_list = line.strip(b'\n').split(b'\t')
        qid, query, title, content, label, freq = line_list
        query_str = query.decode()
        if query_str not in q2qid_dict:
            q2qid_dict[query_str] = len(q2qid_dict)
        new_qid = str(q2qid_dict[query_str]).encode()
        new_line = b'\t'.join([new_qid, query, title, content, label, freq]) + b'\n'
        fw.write(new_line)


INF = 1000
def extract_axiom_feats(fpath, axiom, mode):
    stopws = {}
    cnt = 0
    with open("./data/stop_words_2000.txt", 'r') as fs:  # 只去掉前50个
        for line in fs:
            cnt += 1
            w = line.strip()
            if w not in stopws:
                stopws[w] = 1
            if cnt >= 50:
                break

    if axiom in ['REG', 'STM-1', 'STM-2', 'STM-3']:  # load word embedding from ptms
        # encode all tgt queries
        model = TransformerModel(
            ntoken=22000,
            hidden=768,
            nhead=12,
            nlayers=12,
            dropout=0.1,
            mode='finetune'
        )

        ptm = paddle.load('/home/lht/wsdm_cup/model/ckpt-2022-12-22_12/save_steps27000_6.31586.model')
        for k, v in model.state_dict().items():
            if not k in ptm:
                pass
                # print("warning: not loading " + k)
            else:
                # print("loading " + k)
                v.set_value(ptm[k])

        model.eval()
        token_encoder = model.token_encoder

    fw = open(f"./data/axiom_feats/{axiom}-{mode}.tsv", 'w')
    for line in open(fpath, 'rb'):
        line_list = line.strip(b'\n').split(b'\t')
        qid, query, title, content, label, freq = line_list
        qws = query.split(b'\x01')
        dws = list(title.split(b'\x01')) + list(content.split(b'\x01'))
        qws = [w.decode() for w in qws]
        dws = [w.decode() for w in dws]

        if axiom == 'PROX-1':
            score, valid_p_cnt = 0., 0
            docws_pos_dict = {}
            for pos, w in enumerate(dws):
                if w not in docws_pos_dict:
                    docws_pos_dict[w] = []
                docws_pos_dict[w].append(pos)

            qw_set_num = len(qws)
            for i in range(qw_set_num):
                for j in range(i + 1, qw_set_num):
                    if qws[i] in docws_pos_dict and qws[j] in docws_pos_dict:
                        valid_p_cnt += 1
                        this_score = 0.
                        i_pos = np.array(docws_pos_dict[qws[i]])
                        j_pos = np.array(docws_pos_dict[qws[j]])

                        if len(i_pos) > len(j_pos):  # swap to save a bit time
                            tmp = i_pos
                            i_pos = j_pos
                            j_pos = tmp

                        for k in range(len(i_pos)):
                            this_score = this_score + np.sum(np.abs(j_pos - i_pos[k]))

                        this_score /= len(i_pos) * len(j_pos)
                        score += this_score

            score = INF if score == 0 else score / valid_p_cnt  # 没有term在doc中，或者只有一个term

        elif axiom == 'PROX-2':
            score, valid_w_cnt = 0., 0
            docws_pos_dict = {}
            for pos, w in enumerate(dws):  # only consider the top positions
                if w not in docws_pos_dict:
                    docws_pos_dict[w] = pos
            qws = list(filter(lambda x: x not in stopws, qws))
            for w in qws:
                if w in docws_pos_dict:
                    score += docws_pos_dict[w]
                    valid_w_cnt += 1

            score = INF if score == 0 else score / valid_w_cnt  # 没有term在doc中

        elif axiom == 'PROX-3':  # close pair <= 5
            qws = list(filter(lambda x: x not in stopws, qws))
            qws_dict = {w: 1 for w in qws}

            doc_len = len(dws)
            score = 0
            for i in range(doc_len):
                for j in range(i+1, min(i + 6, doc_len)):
                    dw1, dw2 = dws[i], dws[j]
                    if dw1 != dw2 and dw1 in qws_dict and dw2 in qws_dict:
                        score += 1

        elif axiom == 'PROX-4':  # close pair <= 10
            qws = list(filter(lambda x: x not in stopws, qws))
            qws_dict = {w: 1 for w in qws}

            doc_len = len(dws)
            score = 0
            for i in range(doc_len):
                for j in range(i + 1, min(i + 11, doc_len)):
                    dw1, dw2 = dws[i], dws[j]
                    if dw1 != dw2 and dw1 in qws_dict and dw2 in qws_dict:
                        score += 1

        elif axiom == 'REG':
            doc_count = {w: v for w, v in Counter(dws).most_common()}
            max_sim, best_w = -INF, ''
            if len(qws) > 1:
                for w in qws:
                    left_qws = qws
                    left_qws.remove(w)
                    this_w, left_w = paddle.to_tensor([w], dtype=paddle.int32), paddle.to_tensor(left_qws, dtype=paddle.int32)
                    vec, left_vecs = token_encoder(this_w), token_encoder(left_w)
                    # print('vec', vec)
                    # print('left_vecs', left_vecs)

                    sim = nn.CosineSimilarity(axis=1)(left_vecs, paddle.broadcast_to(vec, shape=left_vecs.shape))
                    this_sim = paddle.amax(sim)
                    if this_sim > max_sim:
                        max_sim = this_sim
                        best_w = w
            elif len(qws):
                best_w = qws[0]
            score = doc_count[best_w] if best_w in doc_count else 0

        elif axiom == 'STM-1':
            if len(qws) and len(dws):
                q_w, d_w = paddle.to_tensor(qws, dtype=paddle.int32), paddle.to_tensor(dws, dtype=paddle.int32)
                q_vec, d_vec = paddle.max(token_encoder(q_w), axis=0), paddle.max(token_encoder(d_w), axis=0)
                q_vec, d_vec = q_vec.reshape([1, 768]), d_vec.reshape([1, 768])
                # print(q_vec, d_vec)
                score = nn.CosineSimilarity(axis=-1)(q_vec, d_vec)
                score = score.cpu().detach().numpy().tolist()[0]

            else:  # no valid term in q
                score = 0.

        elif axiom == 'STM-2':
            if len(qws) and len(dws):
                q_w, d_w = paddle.to_tensor(qws, dtype=paddle.int32), paddle.to_tensor(dws, dtype=paddle.int32)
                q_vec, d_vec = paddle.mean(token_encoder(q_w), axis=0), paddle.mean(token_encoder(d_w), axis=0)
                q_vec, d_vec = q_vec.reshape([1, 768]), d_vec.reshape([1, 768])
                score = nn.CosineSimilarity(axis=-1)(q_vec, d_vec)
                score = score.cpu().detach().numpy().tolist()[0]

            else:  # no valid term in q
                score = 0.

        elif axiom == 'STM-3':
            threshold, score = 0.3, 0
            d_w = paddle.to_tensor(dws, dtype=paddle.int32)
            d_vec = paddle.max(token_encoder(d_w), axis=0)
            d_vec = d_vec.reshape([1, 768])
            for w in qws:
                w_ = paddle.to_tensor([w], dtype=paddle.int32)
                w_v = token_encoder(w_)
                w_v = w_v.reshape([1, 768])
                sim = nn.CosineSimilarity(axis=-1)(w_v, d_vec)
                if sim >= threshold:
                    score += 1

        fw.write(f'{score}\n')


def get_stop_words():
    fpaths = ['./data/wsdm_test_2_all.txt', './data/annotate_data/finetune_data-1.txt']
    w_dict = {}
    all_freq = 0
    for fpath in fpaths:
        for line in open(fpath, 'rb'):
            line_list = line.strip(b'\n').split(b'\t')
            qid, query, title, content, label, freq = line_list
            ws = list(query.split(b'\x01')) + list(title.split(b'\x01')) + list(content.split(b'\x01'))
            for w, freq in Counter(ws).most_common():
                w_ = w.decode()
                if w_ not in w_dict:
                    w_dict[w_] = 0
                w_dict[w_] += freq
                all_freq += freq

    srop_ws = sorted(w_dict.items(), key=lambda x: x[1], reverse=True)
    fw = open('./data/stop_words_2000.txt', 'w')
    for w, freq in srop_ws:
        fw.write(f'{w}\t{freq}\n')


def filter_stop_ws(fpath):
    stopws = {}
    cnt = 0
    with open("./data/stop_words_2000.txt", 'r') as fs:  # 只去掉前50个
        for line in fs:
            cnt += 1
            w = line.strip()
            if w not in stopws:
                stopws[w] = 1
            if cnt >= 50:
                break

    fname = fpath.split('/')[-1]
    fw = open(f'./data/non_stop/{fname}', 'wb')
    for line in open(fpath, 'rb'):
        line_list = line.strip(b'\n').split(b'\t')
        qid, query, title, content, label, freq = line_list
        qws, tws, dws = query.split(b'\x01'), title.split(b'\x01'), content.split(b'\x01')
        qws, tws, dws = [w.decode() for w in qws], [w.decode() for w in tws], [w.decode() for w in dws]

        qws = list(filter(lambda x: x not in stopws, qws))
        tws = list(filter(lambda x: x not in stopws, tws))
        dws = list(filter(lambda x: x not in stopws, dws))

        qws, tws, dws = [w.encode() for w in qws], [w.encode() for w in tws], [w.encode() for w in dws]

        n_q, n_t, n_d = b'\x01'.join(qws), b'\x01'.join(tws), b'\x01'.join(dws)
        new_line = b'\t'.join([qid, n_q, n_t, n_d, label, freq])
        fw.write(new_line + b'\n')


def bigram_process(fpath):
    bigram_path = './data/bigram_dict.json'
    if not os.path.exists(bigram_path):
        bigram_dict = {}
    else:
        with open(bigram_path) as f:
            bigram_dict = json.load(f)

    fname = fpath.split('/')[-1]
    fw = open(f'./data/bigram/{fname}', 'wb')
    for line in open(fpath, 'rb'):
        line_list = line.strip(b'\n').split(b'\t')
        qid, query, title, content, label, freq = line_list
        qws, tws, dws = query.split(b'\x01'), title.split(b'\x01'), content.split(b'\x01')
        qws, tws, dws = [w.decode() for w in qws], [w.decode() for w in tws], [w.decode() for w in dws]

        q_len, t_len, d_len = len(qws), len(tws), len(dws)
        bi_qws, bi_tws, bi_dws = [], [], []
        for i in range(q_len - 1):
            bigram = qws[i] + ' ' + qws[i + 1]
            if bigram not in bigram_dict:
                bigram_dict[bigram] = len(bigram_dict) + 1  # start from 1
            bi_qws.append(str(bigram_dict[bigram]).encode())
        for i in range(t_len - 1):
            bigram = tws[i] + ' ' + tws[i + 1]
            if bigram not in bigram_dict:
                bigram_dict[bigram] = len(bigram_dict) + 1
            bi_tws.append(str(bigram_dict[bigram]).encode())
        for i in range(d_len - 1):
            bigram = dws[i] + ' ' + dws[i + 1]
            if bigram not in bigram_dict:
                bigram_dict[bigram] = len(bigram_dict) + 1
            bi_dws.append(str(bigram_dict[bigram]).encode())

        bi_query, bi_title, bi_content = b'\x01'.join(bi_qws), b'\x01'.join(bi_tws), b'\x01'.join(bi_dws)
        new_line = b'\t'.join([qid, bi_query, bi_title, bi_content, label, freq])
        fw.write(new_line + b'\n')

    with open(bigram_path, 'w') as fw:
        json.dump(bigram_dict, fw)


def generate_corpus(file, dir_name):
    fw1 = open(f'/home/lht/wsdm_cup/utils/{dir_name}/corpus.json', 'w')
    fw2 = open(f'/home/lht/wsdm_cup/utils/{dir_name}/query.tsv', 'w')

    doc2id = {}
    q_dict = {}
    for line in open(file, 'rb'):
        line_list = line.strip(b'\n').split(b'\t')
        qid, query, title, content, label, freq = line_list
        qws, tws, dws = query.split(b'\x01'), title.split(b'\x01'), content.split(b'\x01')
        qws, tws, dws = [w.decode() for w in qws], [w.decode() for w in tws], [w.decode() for w in dws]

        if len(qws) > 0 and qws[0] != '':
            q_str = ' '.join(qws)
        else:
            print('null query')
            q_str = '0'
        d_str = ' '.join(tws + dws)
        if d_str not in doc2id:
            doc2id[d_str] = len(doc2id)
        docid = doc2id[d_str]
        qid = qid.decode()

        json_line = json.dumps({'id': docid, 'contents': d_str})
        fw1.write(f'{json_line}\n')

        if qid not in q_dict:
            q_dict[qid] = []
            fw2.write(f'{qid}\t{q_str}\n')
        q_dict[qid].append(docid)

    with open(f'/home/lht/wsdm_cup/utils/{dir_name}/q2pids.json', 'w') as fw:
        json.dump(q_dict, fw)


def get_doc2q(f_paths):
    doc2q = {}
    for file in f_paths:
        for line in open(file, 'rb'):
            line_list = line.strip(b'\n').split(b'\t')
            qid, query, title, content, label, freq = line_list
            q_str = query.decode()
            d_str = title.decode()
            if d_str not in doc2q:
                doc2q[d_str] = {"total": {"click": 0, "impression": 0, "top_rank": 1000, "skip": 0, "displayed_cnt_mid": 0}}

            # Skip, Displayed Time， Displayed Time Middle, Displayed Count，Displayed Count Middle
            if q_str not in doc2q[d_str]:
                doc2q[d_str][q_str] = {"click": 0, "impression": 0, "top_rank": 1000, "skip": 0, "displayed_cnt_mid": 0}

    with open("./data/doc2q.json", "w") as fw:
        json.dump(doc2q, fw)


def extract_click_feats(ID, fnames):
    dir = 'data/click_data'
    with open("./data/doc2q.json") as f:
        doc2q = json.load(f)

    for fname in fnames:
        fpath = f'{dir}/part-{fname}.gz'
        print(f'Processing {fpath}...')
        if fpath[-3:] != '.gz':  # part-00000.gz is for evaluation
            continue
        f_len = 0
        with gzip.open(fpath, 'rb') as f:
            for _ in f:
                f_len += 1

        valid_cnt = 0
        with gzip.open(fpath, 'rb') as f:
            for _, line in enumerate(tqdm(f.readlines(), desc=f"Testing progress", total=f_len)):
                line_list = line.strip(b'\n').split(b'\t')

                if len(line_list) == 3:  # new query
                    qid, cur_query = line_list[0], line_list[1]
                    q_str = cur_query.decode()

                elif len(line_list) > 6:  # urls {"click": 0, "impression": 0, "top_rank": 1000, "skip": 0, "displayed_cnt_mid": 0}
                    position, title, content, click_label, skip, dis_cnt_mid = line_list[0], line_list[2], line_list[3], line_list[5], line_list[8], line_list[-2]
                    d_str = title.decode()
                    # print(d_str)
                    if d_str in doc2q:
                        valid_cnt += 1
                        doc2q[d_str]["total"]["impression"] += 1

                        if click_label == b'1':
                            doc2q[d_str]["total"]["click"] += 1

                        if skip == b'1':
                            doc2q[d_str]["total"]["skip"] += 1

                        if dis_cnt_mid != b'0':
                            cnt = int(dis_cnt_mid.decode())
                            doc2q[d_str]["total"]["displayed_cnt_mid"] += cnt

                        position_int = int(position.decode())
                        doc2q[d_str]["total"]["top_rank"] = min(position_int, doc2q[d_str]["total"]["top_rank"])

                        if q_str in doc2q[d_str]:
                            doc2q[d_str][q_str]["impression"] += 1
                            if click_label == b'1':
                                doc2q[d_str][q_str]["click"] += 1

                            if skip == b'1':
                                doc2q[d_str][q_str]["skip"] += 1

                            doc2q[d_str][q_str]["top_rank"] = min(position_int, doc2q[d_str][q_str]["top_rank"])

                            if dis_cnt_mid != b'0':
                                cnt = int(dis_cnt_mid.decode())
                                doc2q[d_str][q_str]["displayed_cnt_mid"] += cnt
        print('valid_cnt ', valid_cnt)

    with open(f"./data/doc2q-{ID}.json", "w") as fw:
        json.dump(doc2q, fw)


def merge_click_feats():
    with open("./data/doc2q.json") as f:
        doc2q = json.load(f)

    for i in range(10):
        with open(f"./data/doc2q-{i}.json") as f_tmp:
            doc2q_tmp = json.load(f_tmp)

        for doc in doc2q_tmp:
            for key in doc2q_tmp[doc]:  # click": 0, "impression": 0, "top_rank": 1000, "skip": 0, "displayed_cnt_mid": 0
                click, impression, top_rank, skip, displayed_cnt_mid = doc2q_tmp[doc][key]["click"], doc2q_tmp[doc][key]["impression"], doc2q_tmp[doc][key]["top_rank"], doc2q_tmp[doc][key]["skip"], doc2q_tmp[doc][key]["displayed_cnt_mid"]
                doc2q[doc][key]["click"] += click
                doc2q[doc][key]["impression"] += impression
                doc2q[doc][key]["top_rank"] = min(top_rank, doc2q[doc][key]["top_rank"])
                doc2q[doc][key]["skip"] += skip
                doc2q[doc][key]["displayed_cnt_mid"] += displayed_cnt_mid

    with open("./data/doc2q-all.json", "w") as fw:
        json.dump(doc2q, fw)


if __name__ == '__main__':
    # extract_tgt_queries()

    # sample tgt domain click data
    # multi-processing 10
    # ID = config.ID  # 0-29
    # directory_path = './data/click_data'
    # files = os.listdir(directory_path)
    # ok_files = os.listdir('./data/filtered_click_data')
    # rest_files = list(set(files) - set(ok_files))
    # file_segs = div_list(rest_files, 30)
    # this_files = file_segs[ID]
    # filter_click_data_with_tgt_queries(directory_path, this_files)
    # filter_valid_with_tgt_queries()

    # remapping_qid_for_files(fpath='/home/chenjia/CM2BERT/baidu_data/WSDMCUP_BaiduPLM_Paddle-main/data/annotate_data/finetune_data.txt')
    # process_valid_f()

    # extract axiom features
    # extract_axiom_feats(fpath='./data/wsdm_test_2_all.txt', axiom='PROX-1', mode='test-nonstop')
    # extract_axiom_feats(fpath='./data/wsdm_test_2_all.txt', axiom='PROX-2', mode='test-nonstop')
    # extract_axiom_feats(fpath='./data/annotate_data/finetune_data-1.txt', axiom='PROX-1', mode='train-nonstop')
    # extract_axiom_feats(fpath='./data/annotate_data/finetune_data-1.txt', axiom='PROX-2', mode='train-nonstop')
    # extract_axiom_feats(fpath='./data/annotate_data/valid_data_f.txt', axiom='PROX-1', mode='valid-nonstop')
    # extract_axiom_feats(fpath='./data/annotate_data/valid_data_f.txt', axiom='PROX-2', mode='valid-nonstop')
    #
    # extract_axiom_feats(fpath='./data/wsdm_test_2_all.txt', axiom='PROX-3', mode='test-nonstop')
    # extract_axiom_feats(fpath='./data/annotate_data/finetune_data-1.txt', axiom='PROX-3', mode='train-nonstop')
    # extract_axiom_feats(fpath='./data/annotate_data/valid_data_f.txt', axiom='PROX-3', mode='valid-nonstop')
    #
    # extract_axiom_feats(fpath='./data/wsdm_test_2_all.txt', axiom='PROX-4', mode='test-nonstop')
    # extract_axiom_feats(fpath='./data/annotate_data/finetune_data-1.txt', axiom='PROX-4', mode='train-nonstop')
    # extract_axiom_feats(fpath='./data/annotate_data/valid_data_f.txt', axiom='PROX-4', mode='valid-nonstop')

    # extract_axiom_feats(fpath='./data/wsdm_test_2_all.txt', axiom='REG', mode='test')
    # extract_axiom_feats(fpath='./data/annotate_data/finetune_data-1.txt', axiom='REG', mode='train')
    # extract_axiom_feats(fpath='./data/annotate_data/valid_data_f.txt', axiom='REG', mode='valid')

    # extract_axiom_feats(fpath='./data/wsdm_test_2_all.txt', axiom='STM-1', mode='test')
    # extract_axiom_feats(fpath='./data/annotate_data/finetune_data-1.txt', axiom='STM-1', mode='train')
    # extract_axiom_feats(fpath='./data/annotate_data/valid_data_f.txt', axiom='STM-1', mode='valid')

    # extract_axiom_feats(fpath='./data/wsdm_test_2_all.txt', axiom='STM-2', mode='test')
    # extract_axiom_feats(fpath='./data/annotate_data/finetune_data-1.txt', axiom='STM-2', mode='train')
    # extract_axiom_feats(fpath='./data/annotate_data/valid_data_f.txt', axiom='STM-2', mode='valid')
    #
    # extract_axiom_feats(fpath='./data/wsdm_test_2_all.txt', axiom='STM-3', mode='test')
    # extract_axiom_feats(fpath='./data/annotate_data/finetune_data-1.txt', axiom='STM-3', mode='train')
    # extract_axiom_feats(fpath='./data/annotate_data/valid_data_f.txt', axiom='STM-3', mode='valid')

    # get_stop_words()

    # filter_stop_ws(fpath='./data/wsdm_test_2_all.txt')
    # filter_stop_ws(fpath='./data/annotate_data/finetune_data-1.txt')
    # filter_stop_ws(fpath='./data/annotate_data/valid_data_f.txt')

    # files = [['/home/chenjia/CM2BERT/baidu_data/WSDMCUP_BaiduPLM_Paddle-main/data/bigram/wsdm_test_2_all.txt', 'test_bm25_bigram'], ['/home/chenjia/CM2BERT/baidu_data/WSDMCUP_BaiduPLM_Paddle-main/data/bigram/finetune_data-1.txt', 'pyserini_bm25_bigram'], ['/home/chenjia/CM2BERT/baidu_data/WSDMCUP_BaiduPLM_Paddle-main/data/bigram/valid_data_f.txt', 'valid_bm25_bigram']]
    # for file, dir_name in files:
    #     generate_corpus(file, dir_name)

    # bigram_process(fpath='./data/wsdm_test_2_all.txt')
    # bigram_process(fpath='./data/annotate_data/finetune_data-1.txt')
    # bigram_process(fpath='./data/annotate_data/valid_data_f.txt')

    # click-based features ==> 过于稀疏
    # 在corpus的总点击次数、在该查询下的点击次数、在corpus的总出现次数、在该查询下的总出现次数、在corpus的最高rank、在该查询下的最高rank
    # f_paths = ['./data/wsdm_test_2_all.txt', './data/annotate_data/finetune_data-1.txt']
    # get_doc2q(f_paths)

    # fpath = './data/click_data/part-00110.gz'
    # with gzip.open(fpath, 'rb') as f:
    #     for line in f.readlines():
    #         line_list = line.strip(b'\n').split(b'\t')
    #         print(line_list)

    # inds = [i for i in range(111)]
    # names = [str(ind).zfill(5) for ind in inds]
    # name_list = div_list(names, 10)
    # ID = int(sys.argv[1])  # 0-9
    # extract_click_feats(ID, name_list[ID])

    # merge_click_feats()

    pass
