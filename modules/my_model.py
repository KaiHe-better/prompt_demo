from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch.nn as nn
import torch


def to_device():
    if torch.cuda.is_available():
        for i, v in my_input.items():
            my_input[i] = v.cuda()
    return 

class My_model(nn.Module):

    def __init__(self, label_ids):
        print("PLM loading ...")
        nn.Module.__init__(self)
        
        self.model = RobertaForMaskedLM.from_pretrained("roberta-base")
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        # self.model = RobertaForMaskedLM.from_pretrained("bart-base")
        # self.tokenizer = RobertaTokenizer.from_pretrained('bart-base')
        
        self.label_ids = dict(zip(label_ids.values(),label_ids.keys()))
        self.label_ids_map = {}
        for index, id_token in enumerate(label_ids.keys()):
            tok = sorted(self.tokenizer.tokenize(id_token, is_split_into_words=True), key=lambda x:len(x), reverse=True)[0]
            self.label_ids_map[index] = self.tokenizer.convert_tokens_to_ids(tok)
    
    def forward(self, my_input, my_target):
        logits = self.model(my_input['all_tokens_ids'], attention_mask=my_input['attention_mask'], labels=my_target['label_ids'])
        
        label_words_id = torch.argmax(logits["logits"][:, :, list(self.label_ids_map.values())], dim=-1)
        label_words_id = torch.gather(label_words_id, 1, my_target["mask_indexs"].unsqueeze(1)).squeeze(1)
        
        label_words = []
        for i in label_words_id:
            label_words.append(self.label_ids_map[int(i)])
        
        return {"label_words_id":label_words_id,
                "label_words": torch.tensor(label_words).cuda() if torch.cuda.is_available() else torch.tensor(label_words) ,
                "loss": logits["loss"] if my_target['label_ids'] is not None  else None
                }
                     
    
