import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMBlock(nn.Module):
    def __init__(self,input_dim,hidden_dim,device):
        super(LSTMBlock,self).__init__()
        self.hidden_dim,self.input_dim,self.device=hidden_dim,input_dim,device
        self.Wf=nn.Linear(in_features=self.hidden_dim,out_features=self.hidden_dim,bias=True,device=device)
        self.Uf=nn.Linear(in_features=self.input_dim,out_features=self.hidden_dim,bias=True,device=device)
        self.Wi=nn.Linear(in_features=self.hidden_dim,out_features=self.hidden_dim,bias=True,device=device)
        self.Ui=nn.Linear(in_features=self.input_dim,out_features=self.hidden_dim,bias=True,device=device)
        self.Wo=nn.Linear(in_features=self.hidden_dim,out_features=self.hidden_dim,bias=True,device=device)
        self.Uo=nn.Linear(in_features=self.input_dim,out_features=self.hidden_dim,bias=True,device=device)
        self.Wc=nn.Linear(in_features=self.hidden_dim,out_features=self.hidden_dim,bias=True,device=device)
        self.Uc=nn.Linear(in_features=self.input_dim,out_features=self.hidden_dim,bias=True,device=device)
        
    def forward(self,X,cell=None,hidden=None):
        # X:(batch_size,sequence_length,embedding_dim)
        # cell and hidden:(batch_size,hidden_dim)
        X=X.to(self.device)
        self.cell=(torch.zeros(X.size()[0],self.hidden_dim) if cell==None else cell).to(self.device)
        self.hidden=(torch.zeros(X.size()[0],self.hidden_dim) if hidden==None else hidden).to(self.device)
        output_h=[]
        
        for t in range(X.size()[1]):
            x_t=X[:,t,:] #(batch_size,embed_dim)
            f=F.sigmoid(self.Wf(self.hidden)+self.Uf(x_t))
            i=F.sigmoid(self.Wi(self.hidden)+self.Ui(x_t))
            o=F.sigmoid(self.Wo(self.hidden)+self.Uo(x_t))
            c_=F.tanh(self.Wc(self.hidden)+self.Uc(x_t))
            self.cell=f*self.cell+i*c_
            self.hidden=o*F.tanh(self.cell)
            output_h.append(self.hidden)
            
        return torch.stack(output_h,dim=1)