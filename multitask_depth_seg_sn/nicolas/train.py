import torch
from network.ocrnet import HRNet_Mscale
from loss.utils import CrossEntropyLoss2d
from torchvision.datasets import Cityscapes
import numpy as np
from torchvision import transforms

# ----- Global variables ----- #
batch_size = 1
val_batch_size = 1

num_gpus = torch.cuda.device_count()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def print_image_info(image):
    print(image)
    print(type(image))
    print(image.shape)
    print(f"min: {np.min(image)}\tmax:{np.max(image)}")
    print()

class RandomHorizontalFlip(object):
  def __init__(self, probability=0.5):
    self.flipper = transforms.RandomHorizontalFlip(p=1.0)
    self.p = probability

  def __call__(self, img, ground_truth):
    if np.random.random() < self.p:
      img = self.flipper(img)
      ground_truth = self.flipper(ground_truth)

    return img, ground_truth

# ====== #
#  Main  #
# ====== #
def main():
    # DataAugmentation
    """
    In practice, to obtain the pair of scaled images, we take a single input image and scale it down by a factor of 2, such that we are left with a 1x scale input and an 0.5x
    scaled input, although any scale-down ratio could be selected.
    ...
    Data augmentation: We employ gaussian blur, color augmentation, random horizontal flip and random scaling (0:5x -
    2:0x) on the input images to augment the dataset the training process. We use a crop size of 2048x1024 for Cityscapes
    and 1856x1024 for Mapillary.
    """
    self.hflipper = RandomHorizontalFlip()

    input_transform_ops = transforms.Compose([
        transforms.Resize(512, 1024),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        # transforms.RandomHorizontalFlip(), # TODO: Precisaria flipar o target tbm
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    target_transform_ops = transforms.Compose([

    ])

    img, smnt = self.hflipper(img, smnt)

    # Load Train Dataset
    train_dataset = Cityscapes('/home/lasi/Downloads/datasets/cityscapes/data', split='train', mode='fine', target_type='semantic', transform=input_transform_ops, target_transform=target_transform_ops)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4*num_gpus)
    num_train_steps = len(train_loader)

    # Load Evaluation Dataset
    val_dataset = Cityscapes('/home/lasi/Downloads/datasets/cityscapes/data', split='val', mode='fine', target_type='semantic')
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=4*num_gpus)
    num_val_samples = len(val_loader)

    img, smnt = train_dataset[0]
    # img, smnt = val_dataset[0]

    img = torch.as_tensor(np.asarray(img))  # shape: (1024, 2048, 3)
    smnt = torch.as_tensor(np.asarray(smnt))  # shape: (1024, 2048)

    print_image_info(img)
    print_image_info(smnt)

    # print(type(smnt))
    # print(img.ToTensor())

    print(train_loader)
    print(val_loader)

    input('aki')

    # ----- Network ----- #
    criterion = CrossEntropyLoss2d(ignore_index=255).cuda()
    model = HRNet_Mscale(num_classes=19, criterion=criterion).cuda()

    total_num_params = sum(param.numel() for param in model.parameters())
    print("Total number of params: {}\n".format(total_num_params))

    # ----- Training Looop ----- #
    # TODO

    # ----- Evaluation Loop ----- #
    # TODO

    print("Done.")


if __name__ == '__main__':
    main()
