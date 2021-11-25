import torch
from network.ocrnet import HRNet_Mscale
from loss.utils import CrossEntropyLoss2d
import numpy as np
from torchvision import transforms

# from torchvision.datasets import Cityscapes
from datasets.cityscapes import CityScapes
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import cv2
import cmapy
from glob import glob
import send2trash
import os
from datetime import datetime

# ----- Global variables ----- #
batch_size = 1
val_batch_size = 1
num_epochs = 5

num_gpus = torch.cuda.device_count()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

opt = 'adam'
debug = False

if opt == 'adam':
    LEARN_RATE = 1e-5
    WEIGHT_DECAY = 5e-4
elif opt == 'adabelief':
    LEARN_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
elif opt == 'noam':
    LEARN_RATE = 0
    WEIGHT_DECAY = 5e-4

def print_image_info(image):
		print(image)
		print(type(image))
		print(image.shape)
		# print(f"min: {np.min(image)}\tmax:{np.max(image)}")
		print()

def checkpoints_clean_up(current_run_dir):
    file_paths = glob(current_run_dir + '/*.pth')
    latest_file = max(file_paths, key=os.path.getctime)

    for file_path in file_paths:
        if file_path != latest_file:
            print(f"Deleting {file_path}...")
            send2trash(file_path)

class RandomHorizontalFlip(object):
	def __init__(self, probability=0.5):
		self.flipper = transforms.RandomHorizontalFlip(p=1.0)
		self.p = probability

	def __call__(self, img, ground_truth):
		if np.random.random() < self.p:
			img = self.flipper(img)
			ground_truth = self.flipper(ground_truth)

		return img, ground_truth

def get_lr(optimizer):
    if opt == 'noam':
        optimizer = optimizer.optimizer

    for param_group in optimizer.param_groups:
        return param_group['lr']

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

    # Load Train Dataset
    train_dataset = CityScapes(root='/home/lasi/Downloads/datasets/cityscapes/data', split='train', mode='fine')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4*num_gpus)
    num_train_steps = len(train_loader)
    print(f"num_train_steps: {num_train_steps}")

    # Load Evaluation Dataset
    val_dataset = CityScapes('/home/lasi/Downloads/datasets/cityscapes/data', split='val', mode='fine')
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=4*num_gpus)
    num_val_samples = len(val_loader)
    print(f"num_val_samples: {num_val_samples}")

    img, smnt = train_dataset[0]
    # img, smnt = val_dataset[0]

    # img = torch.as_tensor(np.asarray(img))  # shape: (1024, 2048, 3)
    # smnt = torch.as_tensor(np.asarray(smnt))  # shape: (1024, 2048)

    print_image_info(img)
    print_image_info(smnt)

    # ----- Network ----- #
    criterion = CrossEntropyLoss2d(ignore_index=255).cuda()
    model = HRNet_Mscale(num_classes=19, criterion=criterion).cuda()

    total_num_params = sum(param.numel() for param in model.parameters())
    print("Total number of params: {}\n".format(total_num_params))

    if opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE, weight_decay=WEIGHT_DECAY)
    elif opt == 'adabelief':
        optimizer = AdaBelief(model.parameters(), lr=LEARN_RATE, betas=(0.9, 0.999), eps=1e-8, weight_decay=WEIGHT_DECAY, weight_decouple=True, rectify=False)
    elif opt == 'noam':
        optimizer = NoamOpt(model.total_num_params, 4000, torch.optim.Adam(model.parameters(), lr=LEARN_RATE, betas=(0.9, 0.98), eps=1e-9, weight_decay=WEIGHT_DECAY))

    optimizer.zero_grad()

    # Summary
    dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # current_run_dir = "./runs/{}, M{}, imgs={}, {}, opt={}, bs={}, lr={}, wd={}, {}".format(
    #     dt_string, m, train_dataset.getNumImages(), args.model_act_fn, args.opt,
    #     batch_size, LEARN_RATE, WEIGHT_DECAY, args.num_epochs)
    current_run_dir = "./runs/{}, M{}, imgs={}, opt={}, bs={}, lr={}, wd={}, {}".format(
        dt_string, 0, train_dataset.getNumImages(), opt,
        batch_size, LEARN_RATE, WEIGHT_DECAY, num_epochs)
    writer = SummaryWriter(current_run_dir)

    # ----- Training Looop ----- #
    # TODO
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
    # for epoch in tqdm(range(args.num_epochs), desc="Epochs"):
        model.train()
        
        running_loss = 0
        running_rmse = 0
        tk = tqdm(train_loader, desc="Train")

        for i_iter, batch in enumerate(tk, start=1):
            imgs, labels = batch

            # Input tensors, send to GPU
            imgs = imgs.to(device)      # (shape: (batch_size, 3, h, w))
            labels = labels.to(device)  # (shape: (batch_size, h, w))

            inputs = {'images': imgs, 'gts': labels}

            # .permute(2, 0, 1)

            print('--- main')
            print("inputs['images']: ", imgs.shape)
            print("inputs['gts']: ", labels.shape)
            
            # Outputs tensors
            means = model(inputs)

            # Optimization
            loss = criterion(means, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss_cpu = loss.data.cpu().numpy()
            # rmse_cpu = rmse.data.cpu().numpy()

            # train_batch_losses.append(loss_cpu)
            # train_batch_rmses.append(rmse_cpu)
            
            # print("%d/%d, %d/%d, %d/%d, loss: %g, RMSE: %g" % (m+1, M, epoch+1, args.num_epochs, i_iter+1, num_train_steps, loss_cpu, rmse_cpu))
            
            running_loss += loss.item()

            running_loss_i = running_loss/i_iter

            tk.set_postfix(loss=running_loss_i)

            # Summary
            step = epoch*num_train_steps + i_iter
            # writer.add_scalar('Train/Loss', loss_cpu, step)
            # writer.add_scalar('Train/RMSE', rmse_cpu, step)
            writer.add_scalar('Train/Loss_v2', running_loss_i, step)
            writer.add_scalar('Train/RMSE_v2', running_rmse/i_iter, step)
            writer.add_scalar('Train/LR', get_lr(optimizer), step)
            writer.flush()  # Call flush() method to make sure that all pending events have been written to disk.

            # Visualization
            if args.show_images:
                if debug:
                    np_means_0 = means[0].data.cpu().numpy()
                    np_log_vars_0 = log_vars[0].data.cpu().numpy()
                    np_targets_0 = labels[0].data.cpu().numpy()

                    # Tensors content (meters)
                    print_image_info(np_means_0)
                    print_image_info(np_log_vars_0)
                    print_image_info(np_targets_0)
                    print()

                # Postprocesses tensors (GPU)
                means_uint8 = torch_postprocessing_depth(means, 0.0, 80.0)
                log_vars_uint8 = torch_postprocessing_variance(log_vars)
                targets_uint8 = torch_postprocessing_depth(labels, 0.0, 80.0)

                # Invert colors on GPU
                mask = labels[0] < 0.5

                means_uint8_inv_0 = 255 - means_uint8[0]
                targets_uint8_inv_0 = 255 - targets_uint8[0]
                targets_uint8_inv_0[mask] = 0

                # Train Visualization
                np_imgs_0 = imgs[0].data.cpu().numpy()
                # np_means_uint8_0 = means_uint8[0].data.cpu().numpy()
                np_means_uint8_inv_0 = means_uint8_inv_0.data.cpu().numpy()

                np_log_vars_uint8_0 = log_vars_uint8[0].data.cpu().numpy()
                # np_targets_uint8_0 = targets_uint8[0].data.cpu().numpy()
                np_targets_uint8_inv_0 = targets_uint8_inv_0.data.cpu().numpy()

                # Tensors content (uint8)
                # print_image_info(np_means_uint8_0)
                # print_image_info(np_log_vars_uint8_0)
                # print_image_info(np_targets_uint8_0)
                # print()

                # Colors Maps
                # np_means_inv_0_cmap = cv2.applyColorMap(np_means_inv_0[0, :, :].astype(np.uint8), cmapy.cmap('plasma'))
                np_means_uint8_inv_0_cmap = cv2.applyColorMap(np_means_uint8_inv_0[0, :, :].astype(np.uint8),
                                                            cmapy.cmap('viridis'))
                np_log_vars_uint8_0_cmap = cv2.applyColorMap(np_log_vars_uint8_0[0, :, :].astype(np.uint8),
                                                            cv2.COLORMAP_HOT)
                # np_targets_inv_0_cmap = cv2.applyColorMap(np_targets_inv_0.astype(np.uint8), cmapy.cmap('plasma'))
                np_targets_inv_0_cmap = cv2.applyColorMap(np_targets_uint8_inv_0.astype(np.uint8),
                                                        cmapy.cmap('viridis'))

                cv2.imshow('imgs[0]', np_imgs_0)  # (shape: (h, w, 3))
                # cv2.imshow('means[0]', np_means_0[0, :, :].astype(np.uint8))        # (shape: (1, h, w))
                cv2.imshow('means[0]', np_means_uint8_inv_0_cmap)  # (shape: (1, h, w))
                cv2.imshow('log_vars[0]', np_log_vars_uint8_0_cmap)  # (shape: (1, h, w))
                # cv2.imshow('targets[0]', np_targets_0.astype(np.uint8))             # (shape: (h, w))
                cv2.imshow('targets[0]', np_targets_inv_0_cmap)  # (shape: (h, w))

                # Press 'ESC' on keyboard to exit.
                k = cv2.waitKey(25)
                if k == 27:  # Esc key to stop
                    break
        
        # print('Training loss: {:.6f}'.format(running_loss/num_train_steps))

        # Free GPU memory
        del imgs, labels
        torch.cuda.empty_cache() # FIXME: augusto me falou que não é recomendado utilizar essa função



    # ----- Evaluation Loop ----- #
    # TODO

    print("Done.")


if __name__ == '__main__':
		main()
