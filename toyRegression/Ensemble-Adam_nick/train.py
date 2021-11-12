# ======= #
#  Train  #
# ======= #

# Libraries
from datasets import ToyDataset # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
from model import ToyNet

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pickle
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

# NOTE! Change this to not overwrite all log data when you train the model:
model_id = "Ensemble-Adam_1_M1024"
# data_root = "/home/nicolas/Downloads/bnn/"
data_root = "/root/"

# Training Params
num_epochs = 150
batch_size = 32
learning_rate = 0.001
showTrainLossPlot=False

train_dataset = ToyDataset()

num_train_batches = int(len(train_dataset)/batch_size)
print ("num_train_batches: {}\n".format(num_train_batches))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Loss Params
M = 1024
for i in range(M):
    network = ToyNet(model_id + "_%d" % i, project_dir=data_root + "evaluating_bdl/toyRegression").cuda()
    network.train() # (set in training mode, this affects BatchNorm and dropout)

    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    epoch_losses_train = []
    for epoch in range(num_epochs):
        # print ("###########################")
        # print ("######## NEW EPOCH ########")
        # print ("###########################")
        # print ("epoch: %d/%d" % (epoch+1, num_epochs))
        # print ("network: %d/%d" % (i+1, M))

        batch_losses = []
        for step, (x, y) in enumerate(train_loader):
            # Convert to Cuda Tensor and add dimension, [batch_size] -> [batch_size, 1]
            x = Variable(x).cuda().unsqueeze(1)  # (shape: (batch_size, 1)), [?, 1]
            y = Variable(y).cuda().unsqueeze(1)  # (shape: (batch_size, 1)), [?, 1]

            # Predict
            outputs = network(x)
            mean = outputs[0]     # (shape: (batch_size, ))
            log_var = outputs[1]  # (shape: (batch_size, )) (log(sigma^2))

            # Compute the loss
            loss_likelihood = torch.mean(torch.exp(-log_var)*torch.pow(y - mean, 2) + log_var)


            loss = loss_likelihood
            loss_value = loss.data.cpu().numpy()  # batch_loss

            batch_losses.append(loss_value)

            # Optimization Step
            optimizer.zero_grad()  # (reset gradients)
            loss.backward()        # (compute gradients)
            optimizer.step()       # (perform optimization step)

        epoch_loss = np.mean(batch_losses)
        epoch_losses_train.append(epoch_loss)

        with open("%s/epoch_losses_train.pkl" % network.model_dir, "wb") as file:
            pickle.dump(epoch_losses_train, file)

        print ("M: %d/%d, epoch: %d/%d, train_loss: %g" % (i+1, M, epoch+1, num_epochs, epoch_loss))

        # Results
        plt.figure(1)
        plt.plot(epoch_losses_train, "r.-")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.title("Train loss per epoch, M={}/{}".format(i, M))
        plt.savefig("%s/epoch_losses_train.png" % network.model_dir)
        if showTrainLossPlot:
            plt.pause(1e-9)
        else:
            plt.close(1)

        # Save the model weights to disk:
        checkpoint_path = network.checkpoints_dir + "/model_" + model_id +"_epoch_" + str(epoch+1) + ".pth"
        torch.save(network.state_dict(), checkpoint_path)

    if showTrainLossPlot:
        plt.close(1)

print("Done.")
