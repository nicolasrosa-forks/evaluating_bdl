# ======= #
#  Model  #
# ======= #

# Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

import os

class ToyNet(nn.Module):
    def __init__(self, model_id, project_dir):
        # super(ToyNet, self).__init__()  # Python2
        super().__init__()  # Python3

        self.model_id = model_id
        self.project_dir = project_dir
        self.create_model_dirs()

        # Network Params
        input_dim = 1
        hidden_dim = 10
        output_dim = 1

        # Applies a linear transformation to the incoming data: y = x.A^T + b (Learnable Weights & bias)
        # Networks 1 - Predicts the mean
        self.fc1_mean = nn.Linear(input_dim, hidden_dim)
        self.fc2_mean = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_mean = nn.Linear(hidden_dim, output_dim)

        # Network 2 - Predicts the Variance
        self.fc1_var = nn.Linear(input_dim, hidden_dim)
        self.fc2_var = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_var = nn.Linear(hidden_dim, output_dim)

        print("'ToyNet' object created!")

    def forward(self, x):
        # (x has shape: (batch_size, input_dim))

        # Networks 1 - Predicts the mean
        mean = F.relu(self.fc1_mean(x))                 # (shape: (batch_size, hidden_dim))
        mean = F.dropout(mean, p=0.2, training=True)      
        mean = F.relu(self.fc2_mean(mean))              # (shape: (batch_size, hidden_dim))
        mean = self.fc3_mean(mean)                      # (shape: (batch_size, output_dim))

        # Network 2 - Predicts the Variance
        var = F.relu(self.fc1_var(x))                   # (shape: (batch_size, hidden_dim))
        var = F.dropout(var, p=0.2, training=True)    
        var = F.relu(self.fc2_var(var))                 # (shape: (batch_size, hidden_dim))
        var = self.fc3_var(var)                         # (shape: (batch_size, output_dim))

        return (mean, var)

    def create_model_dirs(self):
        self.logs_dir = self.project_dir + "/training_logs"
        self.model_dir = self.logs_dir + "/model_%s" % self.model_id
        self.checkpoints_dir = self.model_dir + "/checkpoints"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)
