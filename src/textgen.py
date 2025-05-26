import torch
import argparse
from collections import defaultdict
from model import *

def generate_text(model,vocab,prompt:str,
                  device,max_length:int=float("inf"),temperature:float=1.5,
                  topk:int=50,dependent_length=16):
    model.eval()
    tokens=prompt.lower().split()
    input_id=torch.tensor([vocab.get(token,vocab["unk"]) for token in tokens]
                          ,dtype=torch.long).unsqueeze(0).to(device)
    if "<unk>" in vocab.keys():
        unk_id=vocab.get("<unk>")
    else:
        unk_id=None
        
    seq,last_id=0,None
    yield prompt
    while True:
        output=model(input_id) # input_id:(batch_size=1,sequence_length)
        last=output[0,-1,:] # output:(batch_size=1,sequence_length,vocab_size)
        prob_raw,ix=torch.topk(torch.softmax(last/temperature,dim=-1),topk) # the smaller "temperature", the more conservative the generated sentence
        prob=torch.zeros(last.size(),device=device)
        prob[ix]=prob_raw
        if unk_id:
            prob[unk_id]=0
        
        if last_id==None:
            next_id=torch.multinomial(prob,num_samples=1).to(device)
        else:
            while next_id==last_id:
                next_id=torch.multinomial(prob,num_samples=1).to(device)
                prob[next_id]=prob[next_id]/2
            
        input_id=torch.cat((input_id,next_id.reshape(1,1)),dim=1)
        preserved_seq=min(dependent_length,input_id.size()[1])
        input_id=input_id[:,-preserved_seq:]
        word,_=list(vocab.items())[next_id.item()]
        
        seq+=1
        last_id=next_id
        if seq>max_length or word=='<eos>':
            break
        yield word
        
parser=argparse.ArgumentParser(description='Language Model Text Generator')
parser.add_argument("--model_type",type=str,default="transformer")
parser.add_argument("--max_length",type=int,default=float("inf"))
parser.add_argument("--topk",type=int,default=50)
parser.add_argument("--dependent_length",type=int,default=8)
parser.add_argument("--temperature",type=float,default=0.5)

args=parser.parse_args()
vocab_list=torch.load(f"Language_model/src/parameters/LMM_{args.model_type}.vocab_list")

device=torch.device('cuda') if torch.cuda.is_available() else 'cpu'

if args.model_type=="transformer":
    model=LMModel_transformer(len(vocab_list),device,dim=256,nhead=8,num_layers=4)
elif args.model_type=="rnn":
    model=LMModel_RNN(len(vocab_list),device,dim=256,hidden_size=256,num_layers=2,dropout=0.5)
elif args.model_type=='lstm':
    model=LMModel_LSTM(len(vocab_list),device,dim=256,hidden_size=256,num_layers=2,dropout=0.5)
else:
    raise ValueError("Model type name is not correct! The name should be \"transformer\",\"rnn\" or \"lstm\"")

model.load_state_dict(torch.load(f'Language_model/src/parameters/LMM_{args.model_type}.state_dict'))

while True:
    prompt=input("\n Prompt:")
    if prompt=="QUIT":
        break
    words=generate_text(model,vocab_list,prompt,device,args.max_length,
                        args.temperature,args.topk,args.dependent_length)
    for word in words:
        print(word,end=" ",flush=True)