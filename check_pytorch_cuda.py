import torch

print(torch.cuda._initialized)
torch.cuda._lazy_init()
print(torch.cuda._initialized)

tensor = torch.rand(10)
tensor = tensor.cuda()

print('Done.')