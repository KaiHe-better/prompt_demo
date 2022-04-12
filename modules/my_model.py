from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch.nn as nn
import torch

class My_model(nn.Module):

    def __init__(self):
        print("PLM loading ...")
        nn.Module.__init__(self)
        self.model = RobertaForMaskedLM.from_pretrained("roberta-base")
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    def forward(self, my_input, my_target):
        logits = self.model(my_input['all_tokens_ids'], attention_mask=my_input['attention_mask'], labels=my_target['label_ids'])
        label_words = torch.gather(torch.argmax(logits["logits"], dim=-1), 1, my_target["mask_indexs"].unsqueeze(1)).squeeze(1)
        
        return {"label_words":label_words, 
                "loss": logits["loss"] if my_target['label_ids'] else None
                }
                     
    
