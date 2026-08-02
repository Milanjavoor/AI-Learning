import torch
import torch.nn as nn
import math
vocab_size=50000
dim=512
class Multiheadattention(nn.Module):
  def __init__(self,d_model,num_heads):
    super().__init__()
    self.d_model=d_model
    self.num_heads=num_heads
    self.head_dim=d_model//num_heads
    assert num_heads*self.head_dim==d_model
    self.Wq=nn.Linear(d_model,d_model)
    self.Wk=nn.Linear(d_model,d_model)
    self.Wv=nn.Linear(d_model,d_model)
    self.finalout=nn.Linear(d_model,d_model)
  def forward(self,x):
    batch_size=x.shape[0]
    seq_len=x.shape[1]
    Q=self.Wq(x)
    K=self.Wk(x)
    V=self.Wv(x)
    Q=Q.view(batch_size,seq_len,self.num_heads,self.head_dim)
    K=K.view(batch_size,seq_len,self.num_heads,self.head_dim)
    V=V.view(batch_size,seq_len,self.num_heads,self.head_dim)
    Q.transpose(1,2)
    K.transpose(1,2)
    V.transpose(1,2)
    scores=torch.matmul(Q,K.transpose(-2,-1))
    scores=scores/math.sqrt(self.head_dim)
    attention=torch.softmax(scores,dim=-1)
    output=torch.matmul(attention,V)
    output=output.transpose(1,2)
    output=output.contiguous().view(batch_size,seq_len,self.d_model)
    output=self.finalout(output)
    return output

batchsize=2
seqlen=5
numheads=8

x=torch.randn(batchsize,seqlen,dim)
mha=Multiheadattention(d_model=dim,num_heads=numheads)
out=mha(x)
print("input shape:",x.shape)
print("Output shape:",out.shape)
