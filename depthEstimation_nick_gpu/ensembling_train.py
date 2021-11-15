# =========== #
#  Libraries  #
# =========== #
# System Libraries
import math
import os
from datetime import datetime

# matplotlib.use("Agg")
import cmapy
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

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

from utils.plot import showInMovedWindow

# Global Variables
showImages = True
debug = False
model_id = "ensembling"

snapshot_dir_base = "./training_logs/%s" % model_id

# kitti_depth_path = "/root/data/kitti_depth"
# kitti_rgb_path = "/root/data/kitti_rgb"

kitti_depth_path = "/home/lasi/Downloads/datasets/kitti/depth/data_depth_annotated"
kitti_rgb_path = "/home/lasi/Downloads/datasets/kitti/raw_data"

# Training params
M = 10  # TODO: Gustafsson used 33
batch_size = 1

selected_model = 'resnet34_relu'
# selected_model = 'resnet34_elu'
# selected_model = 'resnet34_selu'

# selected_optimizer = 'adam'
# learn_rate = 1e-5
# weight_decay = 5e-4

selected_optimizer = 'adabelief'
learn_rate = 1e-3
weight_decay = 1e-2

num_steps = 40000

save_pred_every = 1000  # FIXME: only when the network improve!?

# Evaluation params
val_batch_size = 1


# =========== #
#  Functions  #
# =========== #
def print_image_info(image):
    print(image.shape, image.dtype, np.min(image), np.max(image))


def torch_posprocessing_depth(tensor, min, max):
    # tensor -= torch.min(min)

    # Clips to [min, max] range
    # tensor = torch.clamp(tensor, min, max)
    tensor = torch.clamp(tensor, max=max)

    # Normalizes
    tensor /= torch.max(torch.tensor(max))

    # Converts from meters to uint8
    tensor *= 255

    return tensor


def torch_posprocessing_variance(tensor):
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

# TODO: mover
if showImages:
    showInMovedWindow('imgs[0]', 100, 270)
    showInMovedWindow('means[0]', 580, 270)
    showInMovedWindow('log_vars[0]', 1060, 270)
    showInMovedWindow('targets[0]', 1540, 270)


# ====== #
#  Main  #
# ====== #
def main():
    global np_means_0, np_log_vars_0  # Values are updated on the main() for being used on plotGaussian()

    if showImages:
        cv2.setMouseCallback('means[0]', plotGaussian)
        cv2.setMouseCallback('log_vars[0]', plotGaussian)

    # --- Models Loop --- #
    for i in range(M):
        # Create Snapshot Directory
        snapshot_dir = snapshot_dir_base + "_%d/" % i
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)

        # Load Train Dataset
        train_dataset = DatasetKITTIAugmentation(kitti_depth_path=kitti_depth_path, kitti_rgb_path=kitti_rgb_path, max_iters=num_steps * batch_size, crop_size=(352, 352))
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        writer = SummaryWriter("./runs/{}, M{}, imgs={}, {}, opt={}, bs={}, lr={}, wd={}, {}".format(
            dt_string, i, train_dataset.getNumImages(), selected_model, selected_optimizer,
            batch_size, learn_rate, weight_decay, num_steps)
        )

        # Load Evaluation Dataset
        val_dataset = DatasetKITTIVal(kitti_depth_path=kitti_depth_path)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=1)

        # Network Architecture
        # check_gpu()
        # torch.cuda._lazy_init()

        if selected_model == 'resnet34_relu':
            from networks.resnet34_relu import DepthEstimationNet  # ReLU
        elif selected_model == 'resnet34_elu':
            from networks.resnet34_elu import DepthEstimationNet  # ELU
        elif selected_model == 'resnet34_selu':
            from networks.resnet34_selu import DepthEstimationNet  # SELU

        model = DepthEstimationNet(pretrained=True).cuda()  # GPU
        # model = DepthEstimationNet(pretrained=True)  # CPU

        model.print_total_num_params()
        # model = torch.nn.DataParallel(model)
        model.train()

        if selected_optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=weight_decay)
        elif selected_optimizer == 'adabelief':
            optimizer = AdaBelief(model.parameters(), lr=learn_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay, weight_decouple=True, rectify=False)

        optimizer.zero_grad()

        # Evaluation Criteria
        criterion = MaskedL2Gauss().cuda()
        rmse_criterion = RMSE().cuda()

        # History
        train_losses = []  # epoch_losses?
        train_batch_losses = []
        val_losses = []

        train_rmses = []  # epoch_rmses?
        train_batch_rmses = []
        val_rmses = []

        # --- Training Loop --- #
        for i_iter, batch in enumerate(train_loader):
            imgs, _, targets, file_ids = batch

            # Input Tensors (Placeholders)
            imgs = Variable(imgs).cuda()  # (shape: (batch_size, h, w, 3))
            # sparses = Variable(sparses).cuda()      # (shape: (batch_size, h, w))
            targets = Variable(targets).cuda()  # (shape: (batch_size, h, w))

            # imgs = Variable(imgs)                   # (shape: (batch_size, h, w, 3))
            # sparses = Variable(sparses)             # (shape: (batch_size, h, w))
            # targets = Variable(targets)             # (shape: (batch_size, h, w))

            # Outputs Tensors
            means, log_vars = model(imgs)  # (both will have shape: (batch_size, 1, h, w))

            # Loss
            loss = criterion(means, log_vars, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            rmse = rmse_criterion(means, targets)

            loss_cpu = loss.data.cpu().numpy()
            rmse_cpu = rmse.data.cpu().numpy()

            print("%d/%d, %d/%d, loss: %g, RMSE: %g" % (i, M, i_iter, num_steps, loss_cpu, rmse_cpu))

            train_batch_losses.append(loss_cpu)
            train_batch_rmses.append(rmse_cpu)

            np_means_0 = means[0].data.cpu().numpy()
            # np_sparses_0 = sparses[0].data.cpu().numpy()
            np_log_vars_0 = log_vars[0].data.cpu().numpy()
            np_targets_0 = targets[0].data.cpu().numpy()

            if debug:
                # Tensors content (meters)
                print_image_info(np_means_0)
                # print_image_info(np_sparses_0)
                print_image_info(np_log_vars_0)
                print_image_info(np_targets_0)
                print()

            # Posprocesses tensors (GPU)
            mask = targets[0] < 0.5

            means_uint8 = torch_posprocessing_depth(means, 0.0, 80.0)
            log_vars_uint8 = torch_posprocessing_variance(log_vars)
            targets_uint8 = torch_posprocessing_depth(targets, 0.0, 80.0)

            # Invert colors on GPU
            means_uint8_inv_0 = 255 - means_uint8[0]
            targets_uint8_inv_0 = 255 - targets_uint8[0]
            targets_uint8_inv_0[mask] = 0

            # Train Visualization
            np_imgs_0 = imgs[0].data.cpu().numpy()
            # np_sparses_0 = sparses[0].data.cpu().numpy()
            # np_means_uint8_0 = means_uint8[0].data.cpu().numpy()
            np_means_uint8_inv_0 = means_uint8_inv_0.data.cpu().numpy()

            np_log_vars_uint8_0 = log_vars_uint8[0].data.cpu().numpy()
            # np_targets_uint8_0 = targets_uint8[0].data.cpu().numpy()
            np_targets_uint8_inv_0 = targets_uint8_inv_0.data.cpu().numpy()

            # Tensors content (uint8)
            # print_image_info(np_means_0)
            # # print_image_info(np_sparses_0)
            # print_image_info(np_log_vars_0)
            # print_image_info(np_targets_0)
            # print()

            if showImages:
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
                # cv2.imshow('sparses[0]', np_sparses_0.astype(np.uint8))             # (shape: (h, w))
                # cv2.imshow('means[0]', np_means_0[0, :, :].astype(np.uint8))        # (shape: (1, h, w))
                cv2.imshow('means[0]', np_means_uint8_inv_0_cmap)  # (shape: (1, h, w))
                cv2.imshow('log_vars[0]', np_log_vars_uint8_0_cmap)  # (shape: (1, h, w))
                # cv2.imshow('targets[0]', np_targets_0.astype(np.uint8))             # (shape: (h, w))
                cv2.imshow('targets[0]', np_targets_inv_0_cmap)  # (shape: (h, w))

                # Press 'ESC' on keyboard to exit.
                k = cv2.waitKey(25)
                if k == 27:  # Esc key to stop
                    break

            # Summary
            writer.add_scalar('Train/Loss', loss_cpu, i_iter)
            writer.add_scalar('Train/RMSE', rmse_cpu, i_iter)
            writer.flush()  # Call flush() method to make sure that all pending events have been written to disk.

            # TODO: Implementar Epoch loop, and network training_log
            # epoch_loss = np.mean(batch_losses)
            # epoch_losses_train.append(epoch_loss)
            # with open("%s/epoch_losses_train.pkl" % network.model_dir, "wb") as file:
            #     pickle.dump(epoch_losses_train, file)
            # print ("train loss: %g" % epoch_loss)
            # plt.figure(1)
            # plt.plot(epoch_losses_train, "k^")
            # plt.plot(epoch_losses_train, "k")
            # plt.ylabel("loss")
            # plt.xlabel("epoch")
            # plt.title("train loss per epoch")
            # plt.savefig("%s/epoch_losses_train.png" % network.model_dir)
            # plt.close(1)

            # # save the model weights to disk:
            # checkpoint_path = network.checkpoints_dir + "/model_" + model_id +"_epoch_" + str(epoch+1) + ".pth"
            # torch.save(network.state_dict(), checkpoint_path)

        writer.close()

    input("Press 'ENTER' to finish...")

    print("Done.")


if __name__ == '__main__':
    main()