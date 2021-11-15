# =========== #
#  Criterion  #
# =========== #

# Torch Libraries
import torch
from torch import nn

class MaskedL2Gauss(nn.Module                       ):
    def __init__(self):
        # super(MaskedL2, self).__init__()  # Python2
        super().__init__()                # Python3

        print("'MaskedL2Gauss' object created!")

    def forward(self, means, log_vars, targets):
        # (means has shape: (batch_size, 1, h, w))
        # (log_vars has shape: (batch_size, 1, h, w))
        # (targets has shape: (batch_size, h, w))

        # Add dimension
        targets = torch.unsqueeze(targets, dim=1)  # (shape: (batch_size, 1, h, w))

        # Mask out invalid pixels
        # tensor.detach() creates a tensor that shares storage with tensor that does not require grad. It detaches the 
        # output from the computational graph. So no gradient will be backpropagated along this variable.
        valid_mask = (targets > 0).detach()

        targets = targets[valid_mask]
        means = means[valid_mask]
        log_vars = log_vars[valid_mask]

        # Loss 
        # Check B.2 Implementation details for the Loss Equation
        # NOTE: The following implementation doesn't consider the third term (sum of weights)
        loss = torch.mean(torch.exp(-log_vars)*torch.pow(targets - means, 2) + log_vars)

        return loss


class RMSE(nn.Module):
    def __init__(self):
        # super(MaskedL2, self).__init__()  # Python2
        super().__init__()                # Python3

        print("'RMSE' object created!")

    def forward(self, preds, targets):
        # (preds has shape: (batch_size, 1, h, w))
        # (targets has shape: (batch_size, h, w))

        # Add dimension
        targets = torch.unsqueeze(targets, dim=1)  # (shape: (batch_size, 1, h, w))

        # Mask out invalid pixels
        valid_mask = targets > 0.1

        targets = targets[valid_mask]
        preds = preds[valid_mask]

        # Convert to mm:
        targets = 1000*targets
        preds = 1000*preds

        # Loss
        rmse = torch.sqrt(torch.mean(torch.pow(targets - preds, 2)))

        return rmse

