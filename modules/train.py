import os
import sys
import json
from tqdm import tqdm
import shutil
import torch
from torch import optim
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F
from data_loader.data_loader import data_to_device

class My_Train_Framework:
    
    def __init__(self, args, my_model, train_data_loader=None, val_data_loader=None, test_data_loader=None):
        super(My_Train_Framework, self).__init__()
        self.args = args
        self.my_model = my_model
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        
        self.res_path ="./results/"+str(sys.argv[1:])
        if os.path.exists(self.res_path):
            shutil.rmtree(self.res_path)
        os.mkdir(self.res_path)
        
        with open(os.path.join(self.res_path, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
                
    def __initialize__(self):
        if torch.cuda.is_available():
            self.my_model.cuda()
        
        # initialize model
        if self.args.load_ckpt_path is not None:
            self.args.load_ckpt_path = "./ckpt/"+str(self.args.load_ckpt_path)
            if os.path.isfile(self.args.load_ckpt_path):
                state_dict = torch.load(self.args.load_ckpt_path)
                print("Successfully loaded checkpoint '%s'" % self.args.load_ckpt_path)
            else:
                raise Exception("No checkpoint found at '%s'" % self.args.load_ckpt_path)
        
            own_state = self.my_model.state_dict()
            for name, param in state_dict.items():
                if name in own_state.keys():
                    own_state[name].copy_(param)
        else:
            print("training form strach ! \n ")
            self.args.load_ckpt_path = "./ckpt/"+self.args.ID
            
        # initialize optim 
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters_to_optimize =   list(self.my_model.named_parameters())   
        parameters_to_optimize = [
            {'params': [p for n, p in parameters_to_optimize if not any(nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay, "lr":self.args.lr},
            {'params': [p for n, p in parameters_to_optimize if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, "lr":self.args.lr} ]
        
        self.optim = optim.AdamW(parameters_to_optimize, lr=self.args.lr)  
        # self.scheduler = get_linear_schedule_with_warmup(self.optim, num_warmup_steps=self.args.warmup_step, 
        #                                                                     num_training_steps=len(self.train_data_loader) if self.train_data_loader else 1)
        
    def train(self):
        self.__initialize__()
        
        print("Start training...")
        best_acc = -1
        best_epoch = -1
        for epoch_num in range(self.args.epoch):
            count= 0
            with tqdm(total=len(self.train_data_loader), desc="Training epoch: {}".format(epoch_num), ncols=100) as pbar:  
                for data_list in self.train_data_loader:
                    epoch_gold_list, epoch_pred_id_list, input_list, target_list = [], [], [], []
                    my_input, my_target = data_to_device(data_list)
                    dic_res = self.my_model(my_input, my_target)
                    
                    epoch_gold_list.append(torch.gather(my_target["label_ids"], 1, my_target["mask_indexs"].unsqueeze(1)).squeeze(1))
                    epoch_pred_id_list.append(dic_res["label_words_id"])
                    
                    self.optim.zero_grad()
                    dic_res["loss"].backward()
                    self.optim.step()
                    # self.scheduler.step()
                    # print(self.optim.param_groups[0]["lr"])
                    
                    postfix= {}
                    postfix['train_loss']=  dic_res["loss"].item()
                    pbar.set_postfix(postfix)
                    pbar.update(1)
                    # break
                
            train_acc = self.get_measure(epoch_gold_list, epoch_pred_id_list, epoch_num)
            valid_acc, input_list, target_list, epoch_pred_list = self.eval("Valid", epoch_num)
            
            print("\n")
            if valid_acc > best_acc:
                best_acc = valid_acc
                best_epoch = epoch_num
                count = 0
                print('Best checkpoint !')
                print("epoch: {0:s}, train acc: {1:.2f}, valid acc {2:.2f}".format(str(best_epoch), train_acc.item(), best_acc.item()))
                print("\n")
                torch.save({'state_dict': self.my_model.state_dict()}, self.args.load_ckpt_path)
                self.recored_res(input_list, target_list, epoch_pred_list)
            
            with open(self.res_path+"/"+"perform.txt", "a") as f:
                f.write(str(epoch_num))
                f.write("\n")
                f.write("loss: {}".format(str(dic_res["loss"].item())))
                f.write("\n")
                f.write("acc: {}".format(str(valid_acc.item())))
                f.write("\n")
                f.write("best epch: {}, best acc {}".format(str(epoch_num), str(round(valid_acc.item(), 2))))
                f.write("\n")
                f.write("\n")
                
            if count < self.args.early_stop_num:
                count+=1
            else:
                break

        print("Finish !")
        return best_epoch, best_dic_F

    def eval(self, train_flag, epoch_num):
        self.my_model.eval()
        if train_flag == "Valid":
            used_data_loader = self.val_data_loader
        else:
            used_data_loader = self.test_data_loader
        
        with torch.no_grad():
            with tqdm(total=len(used_data_loader), desc=train_flag, ncols=100) as pbar: 
                for data_list in used_data_loader:
                    epoch_gold_list, epoch_pred_id_list, epoch_pred_list, input_list, target_list = [], [], [], [], []
                    my_input, my_target = data_to_device(data_list)
                    dic_res = self.my_model(my_input, my_target)
                    
                    epoch_gold_list.append(torch.gather(my_target["label_ids"], 1, my_target["mask_indexs"].unsqueeze(1)).squeeze(1))
                    epoch_pred_list.append(dic_res["label_words"])
                    epoch_pred_id_list.append(dic_res["label_words_id"])
                    input_list.append(my_input)
                    target_list.append(my_target)
                    
                    postfix= {}
                    postfix['valid_loss']= '{0:.4f}'.format(dic_res["loss"])
                    pbar.set_postfix(postfix) 
                    pbar.update(1)
                    break
                    
        if train_flag == "Test":
            self.recored_res(input_list, target_list, epoch_pred_list)  
                  
        acc = self.get_measure(epoch_gold_list, epoch_pred_id_list, epoch_num, train_flag)
        self.my_model.train()
        return acc, input_list, target_list, epoch_pred_list
    
    def recored_res(self, input_list, target_list, epoch_pred_list):
        with open(os.path.join(self.res_path, "res.txt"), "w") as f:
            for my_input, my_target, pred in zip(input_list, target_list, epoch_pred_list):
                for index, sentence, gold_item, pred_item in zip(my_input["index"], my_target["raw_sent"], my_target["raw_label"], pred):
                    f.write("index: {} \n".format(index))
                    f.write("sentence: {} \n".format(sentence))
                    f.write("gold: {} \n".format(gold_item))
                    f.write("pred: {} \n".format(self.my_model.tokenizer.convert_ids_to_tokens([pred_item])[0].replace("Ġ", "")))
                    f.write("\n\n")
                                         
    def get_measure(self, epoch_gold_list, epoch_pred_id_list, epoch_num, train_flag="Train"):
        
        acc = torch.mean((torch.stack(epoch_gold_list).view(-1) == torch.stack(epoch_pred_id_list).view(-1)).type(torch.FloatTensor))
        print("")
        print("{0} res:  epoch: {1:s}, acc: {2:.2f}".format(str(train_flag), str(epoch_num), acc.item()))
        return acc
    
    def infer(self, input_list, prompt):
        self.__initialize__()
        new_data_list = []
        all_tokens_ids = []
        attention_mask = []
        mask_indexs = []
        raw_sent_list=[]
        for sent in input_list:
            raw_sent = prompt + " " +sent
            raw_sent_list.append(raw_sent)
            split_tokens = self.my_model.tokenizer.tokenize(raw_sent)[:self.args.max_len-2]
            split_tokens.insert(0, self.my_model.tokenizer.bos_token) 
            split_tokens.append(self.my_model.tokenizer.eos_token)
            pad_num = (self.args.max_len - len(split_tokens))
            pad_tokens = [self.my_model.tokenizer.pad_token] *pad_num
            all_tokens = split_tokens+pad_tokens
            each_sent_id = self.my_model.tokenizer.convert_tokens_to_ids(all_tokens)
            
            each_attention_mask = [1] * len(split_tokens) + [0]*pad_num
            each_mask_indexs = each_sent_id.index(50264)
        
            new_data_list.append(raw_sent)
            all_tokens_ids.append(each_sent_id)
            attention_mask.append(each_attention_mask)
            mask_indexs.append(each_mask_indexs)
            
        data_list = [torch.LongTensor(0),
                    torch.LongTensor(all_tokens_ids),
                    torch.LongTensor(attention_mask),
                    None, 
                    torch.LongTensor(mask_indexs), 
                    sent,
                    None,
                    ]
        my_input, my_target = data_to_device(data_list)
        dic_res = self.my_model(my_input, my_target)
        
        label_list= []
        for index, sent in enumerate(raw_sent_list) :
            generated_label = self.my_model.tokenizer.convert_ids_to_tokens([dic_res["label_words"][index]])[0].replace("Ġ", "")
            generated_label = self.my_model.tokenizer.decode([dic_res["label_words"][index]])[0].replace("Ġ", "")
            label_list.append(generated_label)
            print(sent + "             <mask> -> "+ generated_label)
            
        return label_list
            
            
                    
                    