import torch
from torch.utils.data import Dataset, DataLoader
from numpy.random import seed
import numpy as np
import random
import os

SEED = 123
seed(SEED)



class My_Dataset(object):
    def __init__(self, data, max_len, tokenizer, label_ids_list):
        self.all_data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_ids_list = label_ids_list
        
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, index):
        raw_data = self.all_data[index].strip()
        raw_id, raw_label, raw_sent  = raw_data.split(" || ")
        
        split_tokens = self.tokenizer.tokenize(raw_sent)[:self.max_len-2]
        split_tokens.insert(0, self.tokenizer.bos_token) 
        split_tokens.append(self.tokenizer.eos_token)
        pad_num = (self.max_len - len(split_tokens))
        pad_tokens = [self.tokenizer.pad_token] *pad_num
        all_tokens = split_tokens+pad_tokens
       
        all_tokens_ids = self.tokenizer.convert_tokens_to_ids(all_tokens)
        attention_mask = [1] * len(split_tokens) + [0]*pad_num
        
        labels = [-100]*self.max_len
        mask_indexs = all_tokens_ids.index(50264)
        lab_word = sorted(self.tokenizer.tokenize(raw_label, is_split_into_words=True), key=lambda x:len(x), reverse=True)[0]
        labels[mask_indexs] = self.tokenizer.convert_tokens_to_ids(lab_word)
        
        return index, all_tokens_ids, attention_mask, labels, mask_indexs, raw_sent, raw_label

def collate_fn(data):
    index, all_tokens_ids, attention_mask, label_ids, mask_indexs, raw_sent, raw_label = zip(*data)

    return [torch.LongTensor(index),
            torch.LongTensor(all_tokens_ids),
            torch.LongTensor(attention_mask),
            torch.LongTensor(label_ids), 
            torch.LongTensor(mask_indexs), 
            raw_sent,
            raw_label,
            ]

def data_to_device(tensor_data_list):
    my_input ={ 'index': tensor_data_list[0],
                'all_tokens_ids': tensor_data_list[1],
                'attention_mask': tensor_data_list[2] }
        
    my_target ={'label_ids': tensor_data_list[3],
                'mask_indexs': tensor_data_list[4],
                'raw_sent': tensor_data_list[5], 
                'raw_label': tensor_data_list[6], 
                }
    
    if torch.cuda.is_available():
        for i, v in my_input.items():
            my_input[i] = v.cuda()
        
        for i, v in my_target.items():
            if i == "label_ids" or i == "mask_indexs":
                if v is not None:
                    my_target[i] = v.cuda()
    
    return my_input, my_target

    
def get_data_loader(batch_size, max_len, tokenizer, label_ids_map, num_workers=8):
    train_dev_test_list = ['./data/train.csv', './data/valid.csv', './data/test.csv']
    
    data_set_list = []
    for fname in train_dev_test_list:
        raw_data = open(fname, 'r', encoding='utf-8').readlines()
        data_set = My_Dataset(raw_data, max_len, tokenizer, label_ids_map)
        
        data_loader = DataLoader(dataset=data_set,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    num_workers=num_workers,
                                    collate_fn=collate_fn)
        data_set_list.append(data_loader)
        
    return data_set_list
