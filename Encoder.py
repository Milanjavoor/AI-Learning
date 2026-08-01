import torch 
import torch.nn as nn
vocab_size=50000
dim=512
embedding=nn.Embedding(vocab_size,dim)
token_ids=torch.tensor([15,284,9123])
embedded=embedding(token_ids)
print(embedded.shape)
