import torch

print(f"cuda 0 : {torch.device('cuda:0')}")
print(f"cuda 1 : {torch.device('cuda:1')}")
print(f"cuda 2 : {torch.device('cuda:2')}")

print(torch.cuda.is_available())
print(torch.version.hip)