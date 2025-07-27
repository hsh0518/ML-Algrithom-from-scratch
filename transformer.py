import torch
import torch.nn as nn
from transformer  import Automodel,AutoTokenizer
from torch.utils.data import TorchDataset, DataLoader
class mh(nn.Module):
    def __init__(self, modelname,head,freeze):
        super().__init__()
        self.bert=Automodel.from_pretrained(modelname)
        self.hidden_d = self.bert.config.hidden_size
        self.head = head
        self.head_d =self.hidden_d//self.head
        if freeze:
            for p in self.bert.embedding.parameters():
                p.require_grad =False
        self.qkv_proj = nn.Linear(self.hidden_d,self.hidden_d*3)
        self.out_proj = nn.Linear(self.hidden_d,self.hidden_d)
    def forward(self,inputtokenid,attnmask):
        with nn.no_grad if not any p.require_grad=True for p in self.bert.embedding.parameters() else nn.enable_grad:
           x = self.bert.embedding(inputtokenid)
        B,T,D = x.shape
        qkv = self.qkv_project(x)
        qkv = qkv.reshape(B,T,3,self.head,self.head_d)
        qkv = qkv.permute(2,0,3,1,4) #3BHThd
        Q,K,V = qkv[0],qkv[1],qkv[2]

        score = Q@K.transpose(-1,-2)/self.head_d**0.5 #BHTT
        score = score.fillmask(attnmask[:,None,None,:]==0,int(-inf))
        attn_w = f.softmax(score)
        context = attn_w@V#BHThd
        context = context().transpose(2,1).reshape(B,T,self.hidden_d)
        context = self.out_proj(context)
        return context,x
class forwordfeedn(nn.Module):
    def __init__(self,d, dropout):
        super().__init__()
        self.hidden_d = d
        self.ffn_d = d*4
        self.fc1 = nn.Linear(self.hidden_d,self.ffn_d )
        self.act = nn.Gule()
        self.fc2 = nn.Linear(self.ffn_d ,self.hidden_d)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        x=self.fc1(x)
        x=self.act(x)
        x=self.dropout(x)
        x=self.fc2(x)
        x=self.dropout(x)
        return x
class residule_layer(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.hidden_d =d
        self.layer = nn.Layernorm(self.hidden_d)

    def forward(self,x,lastoutput):
        x = self.layer(x+lastoutput)
        return x

text = 'xxxx'
tokenizer = AutoTokenizer()
enc = tokenizer(text,return_tokenid = 'pt',padding=True)
inputtokenid = enc[:,:-1]
target = enc[:,1:]

mhfn = mh('bert',8,freeze=True)
ffn = forwordfeedn(768,0.1)
layer =residule_layer(768)
context,x = mhfn(inputtokenid)
x = layer(context,x)
ffnout = ffn(x)
x = layer(ffnout,x)

outputhead=nn.Linear(768, vocab_size)
logit = outputhead(x)

lossfn=torch.CrossEntropyLoss()
optimizer = nn.optim.Adam(model.parameters(),lr,weight_Decay)

loss = lossfn(target, logit)
loss.backward()
optimzer.step()
optimzer.zero_grad()


