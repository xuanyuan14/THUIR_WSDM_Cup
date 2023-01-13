'''
Author: lihaitao
Date: 2022-12-21 14:35:08
LastEditors: Do not edit
LastEditTime: 2022-12-21 20:31:06
FilePath: /lht/wsdm_cup/args.py
'''

import argparse

parser = argparse.ArgumentParser(description='Pipeline commandline argument')

# parameters for dataset settings
parser.add_argument("--train_datadir", type=str, default='./data/train_data/', help="The directory of the training dataset.")  # ?
parser.add_argument("--valid_annotate_path", type=str, default='./data/annotate_data/finetune_data.txt', help="The path of the valid/test annotated data.")
parser.add_argument("--valid_click_path", type=str, default='./data/click_data/part-00000.gz', help="The path of the valid/test click data.")  # ？
parser.add_argument("--num_candidates", type=int, default=30, help="The number of candicating documents for each query in training data.")
parser.add_argument("--ntokens", type=int, default=22000, help="The number of tokens in dictionary.")
parser.add_argument("--seed", type=int, default=666, help="seed")

# parameters for Transformer
parser.add_argument("--max_seq_len", type=int, default=128, help="The max sequence of input for Transformer.")
parser.add_argument("--emb_dim", type=int, default=128, help="The embedding dim.")
parser.add_argument("--nlayers", type=int, default=2, help="The number of Transformer encoder layer.")
parser.add_argument("--nhead", type=int, default=2, help="The number of attention head.")
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--mask_rate", type=float, default=0.4)
parser.add_argument("--mlm_loss_weight", type=float, default=1.0)
parser.add_argument("--ctr_loss_weight", type=float, default=1.0)
parser.add_argument("--n_queries_for_each_gpu", type=int, default=5,
                    help='The number of training queries for each GPU. The size of training batch is based on this.')
parser.add_argument("--init_parameters", type=str, default='', help='The warmup model for Transformer.')
parser.add_argument("--train_batch_size", type=int, default=2000, help='The batchsize of training.')
parser.add_argument("--eval_batch_size", type=int, default=2000, help='The batchsize of evaluation.')
# parser.add_argument("--loss_fun", type=str, default='BCE', choices=['CE', 'BCE'],  help='the loss function used for fine-tune')

# parameters for training
parser.add_argument("--n_gpus", type=int, default=2, help='The number of GPUs.')
parser.add_argument("--lr", type=float, default=2e-7, help='The max learning rate for pre-training, and the learning rate for finetune.')
parser.add_argument("--max_steps", type=int, default=100000, help='The max number of training steps.')
parser.add_argument("--warmup_steps", type=int, default=4000)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--gpu_device", type=int, default=0, help='GPU device ID that use')  # TODO: multi-gpu training

# parameter for logs & save model
parser.add_argument("--buffer_size", type=int, default=500000, help='The size of training buffer. Depends on your memory size.')
parser.add_argument("--log_interval", type=int, default=10, help='The number of interval steps to print logs.')
parser.add_argument("--eval_step", type=int, default=500, help='The number of interval steps to validate.')
parser.add_argument("--save_step", type=int, default=5000, help='The number of interval steps to save the model.')

# parameter for baseline models in finetune
parser.add_argument("--method_name", type=str, default="NavieAlgorithm", help='The name of baseline. candidates: [IPWrank, DLA, RegressionEM, PairDebias, NavieAlgorithm]')

# for submission
parser.add_argument("--test_annotate_path", type=str, default='./data/wsdm_test_2_all.txt', help="The path of the test annotated data.")
parser.add_argument("--evaluate_model_path", type=str, default='', help="The path of saved model.")
parser.add_argument("--result_path", type=str, default='result.csv', help="The path of saving submitting results.")
parser.add_argument("--finetune_epoch", type=int, default=20, help="finetune epoch")

parser.add_argument("--ID", type=int, default=0, help="ID for multi-processing")
parser.add_argument("--k1", type=float, default=1.2, help="bm25 k1")
parser.add_argument("--b", type=float, default=0.75, help="bm25 b")

config = parser.parse_args()

config._CLS_ = 0
config._SEP_ = 1
config._PAD_ = 2
config._MASK_ = 3


""" The size of training batch should be 'ngpus * nqueriy * n_candidates' """
# config.train_batch_size = config.n_gpus * config.n_queries_for_each_gpu * config.num_candidates
# config.train_batch_size = 1* 2* 10

""" The size of test batch is flexible. It depends on your memory. """

# The input-dict for baseline model.
config.exp_settings = {
    'method_name': config.method_name,
    'n_gpus': config.n_gpus,
    'init_parameters': config.init_parameters,
    'lr': config.lr,
    'max_candidate_num': config.num_candidates,
    'selection_bias_cutoff': config.num_candidates,  # same as candidate num
    'feature_size': config.emb_dim,
    'train_input_hparams': "",
    'learning_algorithm_hparams': ""
}
