import pandas as pd

with open("./promt_dataset_emot_intent.csv") as f:
    data = f.readlines()
   
all_num = len(data)
valid_num = 4000  
test_num = 4000

inten_num = 10

val_index = all_num - test_num - valid_num
test_index = all_num - test_num



train_file = "train.csv"
valid_file= "valid.csv"
test_file = "test.csv"

train_list = []
valid_list = []
test_list = []

all_labels = []

def seg(item):
    item = item.strip()
    new = item.split(",")
    all_labels.append(new[1])
    sent = " ".join(new[2:])
    if sent[0] == "\"" and sent[-1] == "\"":
        sent = sent[1:-1]
          
    return new[0]+" || "+new[1]+" || "+ sent.replace("I feel <mask> that ", "").strip()

for index, item in enumerate(data):
    if len(item)>1:
        if index<val_index:
            if index< inten_num*2:
                new_item = seg(item)
                for i in range(1000):
                    train_list.append(new_item)
            else:
                train_list.append(seg(item))
        elif index<test_index:
            valid_list.append(seg(item))
        else:
            test_list.append(seg(item))
        

print(set(all_labels))

with open(train_file, "w") as f:
    for i in train_list:
        f.writelines(i+"\n")
        
with open(valid_file, "w") as f:
    for i in valid_list:
        f.writelines(i+"\n")
        
with open(test_file, "w") as f:
    for i in test_list:
        f.writelines(i+"\n")