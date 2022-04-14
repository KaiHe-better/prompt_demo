import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ID', default='0', type=str, help='temp ID for recording results')
parser.add_argument('--GPU', default='0', type=str, help='which GPU used')
parser.add_argument('--batch_size', default=180, type=int, help='number of examples per batch')
parser.add_argument('--max_len', default=80, help='the max length of sentences')
parser.add_argument('--load_ckpt_path', default=None, help='if load_ckpt_path is not None, then loading the checkpoint')

parser.add_argument('--early_stop_num', default=1000, help='how many epoch for training')
parser.add_argument('--epoch', default=10000, help='how many epoch for training')
parser.add_argument('--warmup_step', default=100, help='how many steps for warmup')
parser.add_argument('--weight_decay', default=3e-5, help='L2')
parser.add_argument('--lr', default=1e-5, help='learning rate')
args = parser.parse_args()


import torch
from collections import OrderedDict
from data_loader.data_loader import get_data_loader
from modules.train import My_Train_Framework
from modules.my_model import My_model
import os
import sys

CUDA_LAUNCH_BLOCKING=1
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
# label_ids_map_old = ['loving', 'hate', 'funny', 'angery', 'relieved', 'bored', 'empty', 'sad', 'happy', 'worried', 'enthusiastic', 'neutral', 'surprised']
label_ids_map =  ['hate', 'fine', 'angery', 'cool', 'funny', 'relieved', 'windy', 'bored', 'sad', 'fast', 'quiet', 'warm', 'slow', 'surprised', 'enthusiastic', 'safety', 'happy']

if torch.cuda.is_available():
    print("try to use GPU..")
else:
    print("use CPU !")

def main():
    my_model = My_model(label_ids_map)
    train_data_loader, val_data_loader, test_data_loader = get_data_loader(·args.batch_size, args.max_len, my_model.tokenizer, my_model.label_ids_list)
    my_training_frame = My_Train_Framework(args, my_model, train_data_loader, val_data_loader, test_data_loader)
    
    my_training_frame.train()
    # my_training_frame.eval("Test", 0)
    

if __name__=="__main__":
    main()