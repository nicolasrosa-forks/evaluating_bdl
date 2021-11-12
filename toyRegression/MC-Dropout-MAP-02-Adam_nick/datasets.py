# ========== #
#  Datasets  #
# ========== #

# Libraries
import torch
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import matplotlib
# matplotlib.use("Agg")  # I needed to disable it when running on Jupyter
import matplotlib.pyplot as plt

import pickle

class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, data_root="/root/", showPlot=False):
        self.examples = []

        with open(data_root + "evaluating_bdl/toyRegression/x.pkl", "rb") as file: # (needed for python3)
            x = pickle.load(file)

        with open(data_root + "evaluating_bdl/toyRegression/y.pkl", "rb") as file: # (needed for python3)
            y = pickle.load(file)

        if showPlot:
            plt.figure(2)
            plt.plot(x, y, "k.")
            plt.ylabel("y")
            plt.xlabel("x")
            plt.savefig(data_root + "evaluating_bdl/toyRegression/MC-Dropout-MAP-02-SGD/training_data.png")
            # plt.show()
            # plt.pause(1)
            # plt.close(1) # I needed to disable it when running on Jupyter
        
        for i in range(x.shape[0]):
            example = {}
            example["x"] = x[i]
            example["y"] = y[i]
            self.examples.append(example)

        self.num_examples = len(self.examples)

        print("'ToyDataset' object created!")
        print("x:", x.shape)
        print("y:", y.shape)
        print()

    def __getitem__(self, index):
        example = self.examples[index]

        x = example["x"]
        y = example["y"]

        return (x, y)

    def __len__(self):
        return self.num_examples

class ToyDatasetEval(torch.utils.data.Dataset):
    def __init__(self):
        self.examples = []

        x = np.linspace(-7, 7, 1000, dtype=np.float32)

        for i in range(x.shape[0]):
            example = {}
            example["x"] = x[i]
            self.examples.append(example)

        self.num_examples = len(self.examples)

        print("'ToyDatasetEval' object created!")
        print("x:", x.shape)
        print()

    def __getitem__(self, index):
        example = self.examples[index]

        x = example["x"]

        return (x)

    def __len__(self):
        return self.num_examples

# data = ToyDataset(showPlot=True)
# data_eval = ToyDatasetEval()

# print(data[0])  # Test __getitem__
# print(len(data))  # Test __len__

# print(data_eval[0])  # Test __getitem__
# print(len(data_eval))  # Test __len__
