import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ID', default='0', type=str, help='temp ID for recording results')
parser.add_argument('--GPU', default='0', type=str, help='which GPU used')
parser.add_argument('--batch_size', default=64, type=int, help='number of examples per batch')
parser.add_argument('--max_len', default=80, help='the max length of sentences')
parser.add_argument('--load_ckpt_path', default=None, help='if load_ckpt_path is not None, then loading the checkpoint')

parser.add_argument('--early_stop_num', default=1000, help='how many epoch for training')
parser.add_argument('--epoch', default=10000, help='how many epoch for training')
parser.add_argument('--warmup_step', default=500, help='how many steps for warmup')
parser.add_argument('--weight_decay', default=3e-5, help='L2')
parser.add_argument('--lr', default=1e-5, help='learning rate')
args = parser.parse_args()


import torch
from data_loader.data_loader import get_data_loader
from modules.train import My_Train_Framework
from modules.my_model import My_model
import os
import sys

CUDA_LAUNCH_BLOCKING=1
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

    
if torch.cuda.is_available():
    print("try to use GPU..")
else:
    print("use CPU !")

def main():
    my_model = My_model()
    train_data_loader, val_data_loader, test_data_loader = get_data_loader(args.batch_size, args.max_len, my_model.tokenizer)
    my_training_frame = My_Train_Framework(args, my_model, train_data_loader, val_data_loader, test_data_loader)
    
    # my_training_frame.train()
    my_training_frame.eval("Test", 0)
    

if __name__=="__main__":
    main()