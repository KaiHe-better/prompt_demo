import torch
from modules.train import My_Train_Framework
from modules.my_model import My_model
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--ID', default='0', type=str, help='temp ID for recording results')
parser.add_argument('--GPU', default='0', type=str, help='which GPU used')
parser.add_argument('--batch_size', default=64, type=int, help='number of examples per batch')
parser.add_argument('--max_len', default=80, help='the max length of sentences')
parser.add_argument('--load_ckpt_path', default="2", help='if load_ckpt_path is not None, then loading the checkpoint')

parser.add_argument('--early_stop_num', default=1000, help='how many epoch for training')
parser.add_argument('--epoch', default=10000, help='how many epoch for training')
parser.add_argument('--warmup_step', default=100, help='how many steps for warmup')
parser.add_argument('--weight_decay', default=3e-5, help='L2')
parser.add_argument('--lr', default=1e-5, help='learning rate')
args = parser.parse_args()


from run import intention_prompt, intention, emotion_prompt, emotion, first_prompt_dic, intention_prompt, emotion_prompt
my_model = My_model(intention_prompt, intention, emotion_prompt, emotion, first_prompt_dic)
my_training_frame = My_Train_Framework(args, my_model)
my_training_frame.__initialize__()
emotion_first_prompt = first_prompt_dic["emotion"]
intention_first_prompt = first_prompt_dic["intention"]


# intention
gold_list = ["fast", "slow", "slow", "cool", "warm", "windy", "quiet",  "safety" , "safety", "safety"]
input_sentence_list = [
    "I want something exciting.",
    "My car uses gasoline to much. ",
    "My kids are in the car.",
    "I feel too warm.",
    "I feel too cold.",
    "I need some fresh air.",
    "Close the window.",
    "The road is muddy." ,
    "It's raining outsides.",
    "It's snowing outside.",
    ]
label_list = my_training_frame.infer(input_sentence_list, intention_prompt, intention_first_prompt, gold_list)


print("\n")

# emotion
gold_list = ["sad", "sad", "enthusiastic", "fine", "sad", "sad", "sad", "sad", "sad", "fine", "sad"]
input_sentence_list = [
    "Layin n bed with a headache ughhhh...waitin on your call...",
    "Funeral ceremony...gloomy friday...",
    "wants to hang out with friends SOON!",
    "We want to trade with someone who has Houston tickets  but no one will.",
    "Re-pinging : why didn't you go to prom? BC my bf didn't like my friends",
    "I should be sleep  but im not! thinking about an old friend who I want. but he's married now. damn  ; he wants me ! scandalous!",
    "Hmmm. is down",
    "Charlene my love. I miss you",
    "I'm sorry at least it's Friday?",
    "cant fall asleep",
    "Choked on her retainers",
    ]
label_list = my_training_frame.infer(input_sentence_list, emotion_prompt, emotion_first_prompt, gold_list)