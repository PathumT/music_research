import torch
from torch import nn

class Recognizer(nn.Module):
    def __init__(self,kind='baseline'):
        super().__init__(); self.kind=kind
        self.encoder=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2))
        if kind=='attention':
            self.gru=nn.GRU(32,32,batch_first=True,bidirectional=True)
            self.attention=nn.Linear(64,1); self.head=nn.Linear(64,10)
        else: self.head=nn.Linear(32,10)
    def forward(self,x):
        z=self.encoder(x).mean(2).transpose(1,2)
        if self.kind=='attention':
            z,_=self.gru(z); z=(z*torch.softmax(self.attention(z),dim=1)).sum(1)
        else: z=z.mean(1)
        return self.head(z)

class Fusion(nn.Module):
    """Learned per-class convex logit fusion of baseline and attention experts."""
    def __init__(self,baseline=None,attention=None):
        super().__init__()
        self.baseline=baseline or Recognizer('baseline'); self.attention=attention or Recognizer('attention')
        for model in (self.baseline,self.attention):
            for p in model.parameters(): p.requires_grad=False
        self.gate=nn.Parameter(torch.zeros(10)); self.bias=nn.Parameter(torch.zeros(10))
    def train(self,mode=True):
        super().train(mode); self.baseline.eval(); self.attention.eval(); return self
    def forward(self,x):
        a,b=self.baseline(x),self.attention(x)
        # Downweight uncertain experts; trainable prior learned on the training partition only.
        certainty_a=(torch.sigmoid(a)-.5).abs().detach()+.05
        certainty_b=(torch.sigmoid(b)-.5).abs().detach()+.05
        w=torch.sigmoid(self.gate)*certainty_a
        v=(1-torch.sigmoid(self.gate))*certainty_b
        return (w*a+v*b)/(w+v)+self.bias

def create(kind): return Fusion() if kind=='fusion' else Recognizer(kind)
