import torch

a = torch.arange(12).view(4,3)
a = a.float()
print(a.shape[0])
b = torch.mean(a,dim=1).view(a.shape[0])
print(a,b)