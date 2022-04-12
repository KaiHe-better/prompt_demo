import torch
from modules.train import My_Train_Framework
from modules.my_model import My_model
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--ID', default='0', type=str, help='temp ID for recording results')
parser.add_argument('--GPU', default='0', type=str, help='which GPU used')
parser.add_argument('--batch_size', default=64, type=int, help='number of examples per batch')
parser.add_argument('--max_len', default=80, help='the max length of sentences')
parser.add_argument('--load_ckpt_path', default=None, help='if load_ckpt_path is not None, then loading the checkpoint')

parser.add_argument('--early_stop_num', default=1000, help='how many epoch for training')
parser.add_argument('--epoch', default=10000, help='how many epoch for training')
parser.add_argument('--warmup_step', default=100, help='how many steps for warmup')
parser.add_argument('--weight_decay', default=3e-5, help='L2')
parser.add_argument('--lr', default=1e-5, help='learning rate')
args = parser.parse_args()


my_model = My_model()
my_training_frame = My_Train_Framework(args, my_model)


input_sentence_list = ["I am so happy !",
                       "I hate mornings. So offensive!"
                       ]

prmopt = "I feel <mask> that"
my_training_frame.infer(input_sentence_list, prmopt)