import pandas as pd

with open("./tweet_emotions_promts.csv") as f:
    data = f.readlines()
   
valid_num = 1000  
test_num = 2000

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
          
    return new[0]+" || "+new[1]+" || "+ sent

for index, item in enumerate(data):
    if len(item)>1:
        if index<valid_num:
            valid_list.append(seg(item))
        elif index<test_num:
            test_list.append(seg(item))
        else:
            train_list.append(seg(item))
        

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