import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self,dim,n_heads,device):
        super(MultiHeadAttention,self).__init__()
        assert dim % n_heads==0
        self.dim_head=dim//n_heads
        self.embed_dim=dim
        self.n_heads=n_heads
        self.device=device
        self.q_proj=nn.Linear(in_features=dim,out_features=dim,bias=False,device=device)
        self.k_proj=nn.Linear(in_features=dim,out_features=dim,bias=False,device=device)
        self.v_proj=nn.Linear(in_features=dim,out_features=dim,bias=False,device=device)
        self.out_proj=nn.Linear(in_features=dim,out_features=dim,bias=False,device=device)
        
    def forward(self,X,mask):
        X=X.to(self.device)
        # X:(Batch_size,Sequence_length,Embedding_dim)
        Q,K,V=self.q_proj(X),self.k_proj(X),self.v_proj(X)
        Q,K,V=self.qkv_reshape(Q,self.dim_head,self.n_heads),self.qkv_reshape(K,self.dim_head,self.n_heads),self.qkv_reshape(V,self.dim_head,self.n_heads)
        e=torch.matmul(Q,K.transpose(2,3))/math.sqrt(self.dim_head)+mask.unsqueeze(0).unsqueeze(0)
        alpha=torch.softmax(e,dim=3)
        attention=torch.matmul(alpha,V) # attention:(batch_size,num_heads,sequence_length,head_dim)
        attention=attention.transpose(1,2).reshape(X.size()[0],X.size()[1],X.size()[2]) # attention:(batch_size,sequence_length,embedding_dim)
        return self.out_proj(attention)
        
    def qkv_reshape(self,qkv_tensor,dim_head,n_heads): 
        ori_shape=qkv_tensor.size()
        # ori_shape: shape of X:(Batch_size,Sequence_length,Embedding_dim)
        qkv_tensor=qkv_tensor.view(ori_shape[0],ori_shape[1],n_heads,dim_head)
        qkv_tensor=qkv_tensor.transpose(1,2)
        return qkv_tensor # return: (Batch_size,num_heads,Sequence_length,Embedding_dim)
    
class AddNorm(nn.Module):
    def __init__(self,norm_shape,device):
        super(AddNorm,self).__init__()
        self.device=device
        self.layernorm=nn.LayerNorm(norm_shape,device=self.device)
        
    def forward(self,X,Y):
        X,Y=X.to(self.device),Y.to(self.device)
        return (self.layernorm(Y)+X)

class FeedForward(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,device):
        super(FeedForward,self).__init__()
        self.device=device
        self.ffn=nn.Sequential(
            nn.Linear(in_features=input_dim,out_features=hidden_dim,bias=True,device=self.device),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim,out_features=output_dim,bias=True,device=self.device)
        )
        
    def forward(self,X):
        return self.ffn(X.to(self.device))
    
class TransformerDecoderBlock(nn.Module):
    def __init__(self,embed_dim,n_heads,device):
        super(TransformerDecoderBlock,self).__init__()
        self.maskedmultiattention=MultiHeadAttention(embed_dim,n_heads,device)
        self.addnorm1=AddNorm(norm_shape=embed_dim,device=device)
        self.feedforward=FeedForward(embed_dim,2*embed_dim,embed_dim,device)
        self.addnorm2=AddNorm(norm_shape=embed_dim,device=device)
    
    def forward(self,X,mask):
        attn=self.maskedmultiattention(X,mask)
        X=self.addnorm1(X,attn)
        ffn=self.feedforward(X)
        X=self.addnorm2(X,ffn)
        return X
    
def positionencoding(X): # input X:(batch_size,sequence_length,embedding_dim)
    bat,seq,emb=X.size()[0],X.size()[1],X.size()[2]
    assert emb%2==0
    P=torch.zeros((bat,seq,emb),device=X.device)
    index=torch.tensor(list(range(seq)),device=X.device).unsqueeze(1) #(i,0)
    emb_index=torch.exp(-math.log(10000)*torch.arange(0,emb,2,device=X.device)/emb).unsqueeze(0)
    P[:,:,0::2]=torch.sin(index*emb_index)
    P[:,:,1::2]=torch.cos(index*emb_index)
    return X+P