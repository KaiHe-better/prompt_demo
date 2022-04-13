from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch.nn as nn
import torch


class My_model(nn.Module):

    def __init__(self, label_list):
        print("PLM loading ...")
        nn.Module.__init__(self)
        
        self.model = RobertaForMaskedLM.from_pretrained("roberta-base")
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        
        self.label_ids_list = []
        for id_token in label_list:
            tok = sorted(self.tokenizer.tokenize(id_token, is_split_into_words=True), key=lambda x:len(x), reverse=True)[0]
            self.label_ids_list.append(self.tokenizer.convert_tokens_to_ids(tok))
    
    def forward(self, my_input, my_target):
        logits = self.model(my_input['all_tokens_ids'], attention_mask=my_input['attention_mask'], labels=my_target['label_ids'])
        
        label_words_id = torch.argmax(logits["logits"][:, :, self.label_ids_list], dim=-1)
        label_words_id = torch.gather(label_words_id, 1, my_target["mask_indexs"].unsqueeze(1)).squeeze(1)
        label_words_id = list(map(lambda x:self.label_ids_list[x], label_words_id))
        
        return {"label_words_id":label_words_id,
                "loss": logits["loss"] if my_target['label_ids'] is not None  else None
                }
                     
    