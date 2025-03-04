# =========== #
#  Libraries  #
# =========== #
# System Libraries
import math
import argparse

import cmapy
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import os

from datetime import datetime
from glob import glob
from send2trash import send2trash
from tqdm import tqdm

# Torch Libraries
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from adabelief_pytorch import AdaBelief
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

# Custom Libraries
from datasets import DatasetKITTIAugmentation, DatasetKITTIVal
from criterion import MaskedL2Gauss, RMSE
from utils.noam_opt import NoamOpt

from utils.plot import showInMovedWindow

# =========== #
#  Arguments  #
# =========== #
parser = argparse.ArgumentParser(description='Process some integers.')  # TODO
parser.add_argument('--n_models', '-M', type=int, help='set number of models', default=1)
parser.add_argument('--model_act_fn', '-f', type=str, help='set activation function')
parser.add_argument('--opt', '-o', type=str, help='set optimizer')
parser.add_argument('--batch_size', '-b', type=int, help='set batch_size', default=1)
parser.add_argument('--num_epochs', '-e', type=int, help='set number of epochs', default=1)
parser.add_argument('--restore', '-r', type=str, help='set path to restore weights', default=None)
parser.add_argument('--show_images', '-s', action='store_true')
args = parser.parse_args()

# print(args.n_models)
# print(args.batch_size)
# print(args.show_images)
# input("Press 'Enter' to continue...")

# ================== #
#  Global Variables  #
# ================== #
debug = False
model_id = "ensembling"

num_gpus = torch.cuda.device_count()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# kitti_depth_path = "/root/data/kitti_depth"
# kitti_rgb_path = "/root/data/kitti_rgb"

kitti_depth_path = "/home/lasi/Downloads/datasets/kitti/depth/data_depth_annotated"
kitti_rgb_path = "/home/lasi/Downloads/datasets/kitti/raw_data"

# ----- Training params ----- #
M = args.n_models  # TODO: Gustafsson used 33
batch_size = args.batch_size

if args.model_act_fn == 'relu':
    from networks.resnet34_relu import DepthEstimationNet  # ReLU
elif args.model_act_fn == 'elu':
    from networks.resnet34_elu import DepthEstimationNet  # ELU
elif args.model_act_fn == 'selu':
    from networks.resnet34_selu import DepthEstimationNet  # SELU

if args.opt == 'adam':
    LEARN_RATE = 1e-5
    WEIGHT_DECAY = 5e-4
elif args.opt == 'adabelief':
    LEARN_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
elif args.opt == 'noam':
    LEARN_RATE = 0
    WEIGHT_DECAY = 5e-4

# num_epochs = 20
# num_steps = 40000

# ----- Evaluation params ----- #
val_batch_size = 1


# =========== #
#  Functions  #
# =========== #
def print_image_info(image):
    print(image.shape, image.dtype, np.min(image), np.max(image))


def torch_postprocessing_depth(tensor, min, max):
    # tensor -= torch.min(min)

    # Clips to [min, max] range
    # tensor = torch.clamp(tensor, min, max)
    tensor = torch.clamp(tensor, max=max)

    # Normalizes
    tensor /= torch.max(torch.tensor(max))

    # Converts from meters to uint8
    tensor *= 255

    return tensor


def torch_postprocessing_variance(tensor):
    # Normalizes
    tensor -= torch.min(tensor)
    tensor /= torch.max(tensor)

    # Converts from log_vars to uint8
    tensor *= 255

    return tensor

np_means_0 = None
np_log_vars_0 = None

def plotGaussian(event, x, y, flags, param):
    pred_depth_mean = np_means_0[0, y, x]
    pred_depth_var = np.exp(np_log_vars_0[0, y, x])

    if event == cv2.EVENT_LBUTTONDBLCLK:
        mu = pred_depth_mean
        variance = pred_depth_var  # sigma^2
        sigma = math.sqrt(variance)
        x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)

        plt.plot(x, stats.norm.pdf(x, mu, sigma))
        # plt.legend(r'$N(\mu={}, \sigma^2={})$'.format(mu, variance))
        plt.ylim([0, 1.1])
        plt.xlim([0, 90])
        plt.xlabel(r'$\mu$')
        plt.show()

def checkpoints_clean_up(current_run_dir):
    file_paths = glob(current_run_dir + '/*.pth')
    latest_file = max(file_paths, key=os.path.getctime)

    for file_path in file_paths:
        if file_path != latest_file:
            print(f"Deleting {file_path}...")
            send2trash(file_path)

# TODO: mover
if args.show_images:
    showInMovedWindow('imgs[0]', 100, 270)
    showInMovedWindow('means[0]', 580, 270)
    showInMovedWindow('log_vars[0]', 1060, 270)
    showInMovedWindow('targets[0]', 1540, 270)

def get_lr(optimizer):
    if args.opt == 'noam':
        optimizer = optimizer.optimizer

    for param_group in optimizer.param_groups:
        return param_group['lr']

# ====== #
#  Main  #
# ====== #
def main():
    global np_means_0, np_log_vars_0  # Values are updated on the main() for being used on plotGaussian()

    if args.show_images:
        cv2.setMouseCallback('means[0]', plotGaussian)
        cv2.setMouseCallback('log_vars[0]', plotGaussian)

    # Load Train Dataset
    train_dataset = DatasetKITTIAugmentation(kitti_depth_path=kitti_depth_path, kitti_rgb_path=kitti_rgb_path, crop_size=(352, 352))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4*num_gpus)
    num_train_steps = len(train_loader)

    # Load Evaluation Dataset
    val_dataset = DatasetKITTIVal(kitti_depth_path=kitti_depth_path)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=4*num_gpus)
    num_val_samples = len(val_loader)

    # --- Models Loop --- #
    for m in tqdm(range(M), desc="Models"):
        # Network Architecture
        # model = DepthEstimationNet(pretrained=True).cuda()
        model = DepthEstimationNet(pretrained=True).to(device)

        if args.restore is not None:
            model.load_state_dict(torch.load(args.restore))

        model.print_total_num_params()
        # model = torch.nn.DataParallel(model)
        model.train()

        if args.opt == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE, weight_decay=WEIGHT_DECAY)
        elif args.opt == 'adabelief':
            optimizer = AdaBelief(model.parameters(), lr=LEARN_RATE, betas=(0.9, 0.999), eps=1e-8, weight_decay=WEIGHT_DECAY, weight_decouple=True, rectify=False)
        elif args.opt == 'noam':
            optimizer = NoamOpt(model.total_num_params, 4000, torch.optim.Adam(model.parameters(), lr=LEARN_RATE, betas=(0.9, 0.98), eps=1e-9, weight_decay=WEIGHT_DECAY))

        optimizer.zero_grad()

        # Evaluation criteria
        # criterion = MaskedL2Gauss().cuda()
        # rmse_criterion = RMSE().cuda()
        criterion = MaskedL2Gauss().to(device)
        rmse_criterion = RMSE().to(device)

        # Summary
        dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        current_run_dir = "./runs/{}, M{}, imgs={}, {}, opt={}, bs={}, lr={}, wd={}, {}".format(
            dt_string, m, train_dataset.getNumImages(), args.model_act_fn, args.opt,
            batch_size, LEARN_RATE, WEIGHT_DECAY, args.num_epochs)
        
        writer = SummaryWriter(current_run_dir)

        # History
        # train_batch_losses = []
        # train_batch_rmses = []

        last_sum_val_batch_rmses = 1e9

        # --- Training Loop --- # 
        for epoch in tqdm(range(args.num_epochs), desc="Epochs"):
            model.train()
            
            running_loss = 0
            running_loss_rmse = 0
            tk = tqdm(train_loader, desc="Train")

            for i_iter, batch in enumerate(tk, start=1):
                imgs, _, targets, file_ids = batch
                
                # Input tensors (Placeholders)
                # imgs = Variable(imgs).cuda()              # (shape: (batch_size, h, w, 3))
                # targets = Variable(targets).cuda()        # (shape: (batch_size, h, w))
                imgs = imgs.to(device)              # (shape: (batch_size, h, w, 3))
                targets = targets.to(device)        # (shape: (batch_size, h, w))

                # Outputs tensors
                means, log_vars = model(imgs)  # (both will have shape: (batch_size, 1, h, w))

                # Optimization
                loss = criterion(means, log_vars, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                rmse = rmse_criterion(means, targets)

                loss_cpu = loss.data.cpu().numpy()
                rmse_cpu = rmse.data.cpu().numpy()

                # train_batch_losses.append(loss_cpu)
                # train_batch_rmses.append(rmse_cpu)
                
                # print("%d/%d, %d/%d, %d/%d, loss: %g, RMSE: %g" % (m+1, M, epoch+1, args.num_epochs, i_iter+1, num_train_steps, loss_cpu, rmse_cpu))
                
                running_loss += loss.item()
                running_loss_rmse += rmse.item()

                running_loss_i = running_loss/i_iter
                tk.set_postfix(loss=running_loss_i)

                # Summary
                step = epoch*num_train_steps + i_iter
                # writer.add_scalar('Train/Loss', loss_cpu, step)
                # writer.add_scalar('Train/RMSE', rmse_cpu, step)
                writer.add_scalar('Train/Loss_v2', running_loss_i, step)
                writer.add_scalar('Train/RMSE_v2', running_loss_rmse/i_iter, step)
                writer.add_scalar('Train/LR', get_lr(optimizer), step)
                writer.flush()  # Call flush() method to make sure that all pending events have been written to disk.

                # Visualization
                if args.show_images:
                    if debug:
                        np_means_0 = means[0].data.cpu().numpy()
                        np_log_vars_0 = log_vars[0].data.cpu().numpy()
                        np_targets_0 = targets[0].data.cpu().numpy()

                        # Tensors content (meters)
                        print_image_info(np_means_0)
                        print_image_info(np_log_vars_0)
                        print_image_info(np_targets_0)
                        print()

                    # Postprocesses tensors (GPU)
                    means_uint8 = torch_postprocessing_depth(means, 0.0, 80.0)
                    log_vars_uint8 = torch_postprocessing_variance(log_vars)
                    targets_uint8 = torch_postprocessing_depth(targets, 0.0, 80.0)

                    # Invert colors on GPU
                    mask = targets[0] < 0.5

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
            del imgs, targets
            torch.cuda.empty_cache() # FIXME: augusto me falou que não é recomendado utilizar essa função

            # ----- Validation Loop ----- # FIXME: Implementar torch.no_grad()
            with torch.no_grad():
                model.eval()

                sum_val_batch_losses = 0
                sum_val_batch_rmses = 0
                val_tk = tqdm(val_loader, "Val")

                for k_iter, val_batch in enumerate(val_tk, start=1):
                    imgs, _, targets, file_ids = val_batch

                    # Input tensors (Placeholders)
                    # imgs = Variable(imgs).cuda()        # (shape: (batch_size, h, w, 3))
                    # targets = Variable(targets).cuda()  # (shape: (batch_size, h, w))
                    imgs = imgs.to(device)        # (shape: (batch_size, h, w, 3))
                    targets = targets.to(device)  # (shape: (batch_size, h, w))

                    # Outputs tensors
                    means, log_vars = model(imgs)  # (both will have shape: (batch_size, 1, h, w))

                    val_loss = criterion(means, log_vars, targets)
                    val_rmse = rmse_criterion(means, targets)

                    # val_loss_cpu = val_loss.data.cpu().numpy()
                    # val_rmse_cpu = val_rmse.data.cpu().numpy()

                    sum_val_batch_losses += val_loss.item()
                    sum_val_batch_rmses += val_rmse.item()

                    # print("%d/%d, %d/%d, %d/%d, loss: %g, RMSE: %g" % (m+1, M, epoch+1, args.num_epochs, k_iter+1, num_val_samples, val_loss_cpu, val_rmse_cpu))
                    # print(f"{m+1}/{M}, {epoch+1}/{args.num_epochs}, {k_iter+1}/{num_val_samples}, loss: {val_loss_cpu}, RMSE: {val_rmse_cpu}")
                    val_tk.set_postfix(val_loss=sum_val_batch_losses/(k_iter))

                # print('Validation loss: {:.6f}'.format(val_loss/num_val_samples))

                # Summary
                # writer.add_scalar('Val/Loss', sum_val_batch_losses/num_val_samples, epoch)
                # writer.add_scalar('Val/RMSE', sum_val_batch_rmses/num_val_samples, epoch)
                writer.add_scalar('Val/Loss_v2', sum_val_batch_losses/num_val_samples, epoch)
                writer.add_scalar('Val/RMSE_v2', sum_val_batch_rmses/num_val_samples, epoch)
                
                writer.flush()  # Call flush() method to make sure that all pending events have been written to disk.

                if last_sum_val_batch_rmses > sum_val_batch_rmses:
                    print(f"The model improved from {last_sum_val_batch_rmses} to {sum_val_batch_rmses}")
                    # save the model weights to disk:
                    model_checkpoint_filepath = current_run_dir + "/model_M" + str(m) + "_epoch_" + str(epoch+1) + ".pth"
                    torch.save(model.state_dict(), model_checkpoint_filepath)

                last_sum_val_batch_rmses = sum_val_batch_rmses

                # Free GPU memory
                del imgs, targets
                torch.cuda.empty_cache() # FIXME: augusto me falou que não é recomendado utilizar essa função

        # Close SummaryWriter
        writer.close()

        # Checkpoints clean up
        checkpoints_clean_up(current_run_dir)

    # input("Press 'ENTER' to finish...")
    print("Done.")


if __name__ == '__main__':
    main()
