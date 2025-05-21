import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNBlock(nn.Module):
    def __init__(self,input_dim,hidden_dim,device):
        super(RNNBlock,self).__init__()
        self.We=nn.Linear(in_features=input_dim,out_features=hidden_dim,bias=True,device=device)
        self.Wh=nn.Linear(in_features=hidden_dim,out_features=hidden_dim,bias=True,device=device)
        self.hidden_dim,self.input_dim,self.device=hidden_dim,input_dim,device
        
    def forward(self,X,h=None): # X:(batch_size,sequence_length,embedding_dim)
        if h==None:
            h=torch.zeros(X.size()[0],self.hidden_dim) # h:(batch_size,hidden_dim)
        h,X=h.to(self.device),X.to(self.device)
        output_h=[]
        for t in range(X.size()[1]):
            x_t=X[:,t,:]
            h=F.relu(self.Wh(h)+self.We(x_t))
            output_h.append(h)
        return torch.stack(output_h,dim=1), h # output:(batch_size,sequence_length,embedding_dim)