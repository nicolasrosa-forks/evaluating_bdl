from network.ocrnet import HRNet_Mscale
from loss.utils import CrossEntropyLoss2d

criterion = CrossEntropyLoss2d(ignore_index=255).cuda()
model = HRNet_Mscale(num_classes=19, criterion=criterion).cuda()

total_num_params = sum(param.numel() for param in model.parameters())
print("Total number of params: {}\n".format(total_num_params))

print("Done.")
