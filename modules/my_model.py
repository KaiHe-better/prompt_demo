from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch.nn as nn
import torch



class My_model(nn.Module):

    def __init__(self, intention_prompt, intention_label_word, emotion_prompt, emotion_label_word, first_prompt_dic):
        print("PLM loading ...")
        nn.Module.__init__(self)
        
        self.model = RobertaForMaskedLM.from_pretrained("roberta-base")
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        
        self.intention_prompt = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(intention_prompt))[1:] # [38/100, 236, 7, 28, 50264, 4]
        self.emotion_prompt = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(emotion_prompt))[1:]  # [100, 619, 50264, 14]
        
        self.first_prompt_dic = first_prompt_dic
        self.intention_label_ids_list = []
        self.emotion_label_ids_list = []
        
        all_dic = {}
        for id_token in intention_label_word:
            tok = sorted(self.tokenizer.tokenize(id_token, is_split_into_words=True), key=lambda x:len(x), reverse=True)[0]
            self.intention_label_ids_list.append(self.tokenizer.convert_tokens_to_ids(tok))
            all_dic[tok] = self.tokenizer.convert_tokens_to_ids(tok)
            
        for id_token in emotion_label_word:
            tok = sorted(self.tokenizer.tokenize(id_token, is_split_into_words=True), key=lambda x:len(x), reverse=True)[0]
            self.emotion_label_ids_list.append(self.tokenizer.convert_tokens_to_ids(tok))
            all_dic[tok] = self.tokenizer.convert_tokens_to_ids(tok)
    
        # print(all_dic)
    
    def forward(self, my_input, my_target):
        logits = self.model(my_input['all_tokens_ids'], attention_mask=my_input['attention_mask'], labels=my_target['label_ids'])
        
        # label_words_id = torch.argmax(logits["logits"][:, :, self.label_ids_list], dim=-1)
        # label_words_id = torch.gather(label_words_id, 1, my_target["mask_indexs"].unsqueeze(1)).squeeze(1)
        # label_words_id = torch.tensor([i for i in map(lambda x:self.label_ids_list[x], label_words_id)])
        # label_words_id = label_words_id.cuda() if torch.cuda.is_available() else label_words_id
        
                
        label_words_id = []
        for index, ids in enumerate(my_input['all_tokens_ids']) :
            
            if self.first_prompt_dic["emotion"]:
                emotion_s = ids[2:len(self.emotion_prompt)+2].tolist()  # checked
            else:
                emotion_s = ids[:sum(my_input['attention_mask'][index])][-(len(self.emotion_prompt)+1):-1].tolist() 
                    
            if self.first_prompt_dic["intention"]:
                intention_s = ids[2:len(self.intention_prompt)+2].tolist()  # checked
            else:
                intention_s = ids[:sum(my_input['attention_mask'][index])][-(len(self.intention_prompt)+1):-1].tolist()   # checked
            
            
            if emotion_s == self.emotion_prompt:
                choose_id = torch.argmax(logits["logits"][index][my_target["mask_indexs"][index], self.emotion_label_ids_list])
                label_words_id.append(self.emotion_label_ids_list[choose_id])
            elif intention_s == self.intention_prompt:
                choose_id = torch.argmax(logits["logits"][index][my_target["mask_indexs"][index], self.intention_label_ids_list])
                label_words_id.append(self.intention_label_ids_list[choose_id])
            else:
                raise Exception("no right prompt founded ! if first_prompt = False, need period at last !")
        
        
        # print("")
        # print(" self.emotion_prompt",  self.emotion_prompt)
        # print(" self.intention_prompt",  self.intention_prompt)
        # if emotion_s == self.emotion_prompt:
        #     print(emotion_s)
        #     print(self.tokenizer.convert_ids_to_tokens(emotion_s))
        #     print(self.emotion_label_ids_list)
        #     print(self.tokenizer.convert_ids_to_tokens(self.emotion_label_ids_list))
            
        #     print("logits")
        #     print(logits["logits"][index][my_target["mask_indexs"][index], self.emotion_label_ids_list])
        #     print("choose_id")
        #     print(choose_id)
        
        
        # else:
        #     print(intention_s)
        #     print(self.tokenizer.convert_ids_to_tokens(intention_s))
        #     print(self.intention_label_ids_list)
        #     print(self.tokenizer.convert_ids_to_tokens(self.intention_label_ids_list))
            
        #     print("logits")
        #     print(logits["logits"][index][my_target["mask_indexs"][index], self.intention_label_ids_list])
        #     print("choose_id")
        #     print(choose_id)
            
   
        
        
        return {"label_words_id": torch.tensor(label_words_id).cuda() if torch.cuda.is_available() else torch.tensor(label_words_id),
                "loss": logits["loss"] if my_target['label_ids'] is not None  else None
                }
                     
                     
    
