import torch
import argparse
import tkinter as tk
from tkinter import scrolledtext
from model import *

def generate_text(model,vocab,prompt:str,device,max_length:int=256,temperature:float=0.5):
    model.eval()
    tokens=prompt.lower().split()
    input_id=torch.tensor([vocab[token] for token in tokens],dtype=torch.long).unsqueeze(0).to(device)
    seq=0
    while True:
        output=model(input_id) # input_id:(batch_size=1,sequence_length)
        last=output[0,-1,:] # output:(batch_size=1,sequence_length,vocab_size)
        prob=torch.softmax(last/temperature,dim=-1) # the smaller "temperature", the more conservative the generated sentence
        next_id=torch.multinomial(prob,num_samples=1).to(device)
        input_id=torch.cat((input_id,next_id.reshape(1,1)),dim=1)
        word,_=list(vocab.items())[next_id.item()]
        
        seq+=1
        if seq>max_length or word=='<eos>':
            break
        yield word
        
parser=argparse.ArgumentParser(description='Language Model Text Generator')
parser.add_argument("--model_type",type=str,default="transformer")
parser.add_argument("--max_length",type=int,default=256)
parser.add_argument("--temperature",type=float,default=1.0)

args=parser.parse_args()
vocab_list=torch.load(f"parameters/LMM_{args.model_type}_vocab_list")

device=torch.device('cuda') if torch.cuda.is_available() else 'cpu'

if args.model_type=="transformer":
    model=LMModel_transformer(len(vocab_list),device,dim=256,nhead=8,num_layers=4)
elif args.model_type=="rnn":
    model=LMModel_RNN(len(vocab_list),device,dim=256,hidden_size=256,num_layers=2,dropout=0.5)
elif args.model_type=='lstm':
    model=LMModel_LSTM(len(vocab_list),device,dim=256,hidden_size=256,num_layers=2,dropout=0.5)
else:
    raise NameError("Model type name is not correct! The name should be \"transformer\",\"rnn\" or \"lstm\"")

model.load_state_dict(torch.load(f'parameters/LMM_{args.model_type}_state_dict'))
prompt=input("Prompt:")
words=generate_text(model,vocab_list,prompt,device,args.max_length,args.temperature)
for word in words:
    print(word,end=" ",flush=True)
