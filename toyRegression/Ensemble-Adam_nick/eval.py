# ====== #
#  Eval  #
# ====== #

# Libraries
from datasets import ToyDatasetEval # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
from model import ToyNet

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pickle
import matplotlib as mpl
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

# Eval Params
batch_size = 32
M = 64
max_logvar = 2.0

# Load Evaluation Dataset
print("Loading evaluation dataset...")
val_dataset = ToyDatasetEval()
num_val_batches = int(len(val_dataset)/batch_size)

print("num_val_batches: {}\n".format(num_val_batches))

val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# Load Trained Models
networks = []

print("Loading trained models...")
for i in range(M):
    network = ToyNet("eval_Ensemble-Adam_1_M1024", project_dir="/workspace/evaluating_bdl/toyRegression").cuda()
    network.load_state_dict(torch.load("/workspace/evaluating_bdl/toyRegression/training_logs/model_Ensemble-Adam_1_M1024_%d/checkpoints/model_Ensemble-Adam_1_M1024_epoch_150.pth" % i))
    network.eval()   # (set in evaluation mode, this affects BatchNorm and dropout)

    networks.append(network)

M = float(len(networks))
print (M)

# History
x_values = []

final_mean_values = []
final_sigma_tot_values = []   # Total Uncertainty Values
final_sigma_epi_values = []   # Episdemic Uncertainty Values
final_sigma_alea_values = []  # Aleatoric Uncertainty Values

print("Evaluating models...")
for step, (x) in enumerate(val_loader):
    # Convert to Cuda Tensor and add dimension, [batch_size] -> [batch_size, 1]
    x = Variable(x).cuda().unsqueeze(1)  # (shape: (batch_size, 1))

    means = []
    log_vars = []
    for network in networks:
        # Predict
        outputs = network(x)
        mean = outputs[0]     # (shape: (batch_size, ))
        log_var = outputs[1]  # (shape: (batch_size, )) (log(sigma^2))
        log_var = max_logvar - F.relu(max_logvar - log_var)

        means.append(mean)
        log_vars.append(log_var)

    for i in range(x.size(0)):
        x_value = x[i].data.cpu().numpy()[0]

        # Retrieve mean, var values from the GPU tensors
        mean_values = []
        for mean in means:
            mean_value = mean[i].data.cpu().numpy()[0]
            mean_values.append(mean_value)

        sigma_alea_values = []
        for log_var in log_vars:
            sigma_alea_value = torch.exp(log_var[i]).data.cpu().numpy()[0]
            sigma_alea_values.append(sigma_alea_value)

        # Average Mean, Aleatoric and Epistemic Uncertainties values. 
        # Please refer to Appendix A - Approximating a mixture of Gaussian distributions
        # The variables in each one of the following for's will be iterated M times.
        mean_value = 0.0
        for value in mean_values:
            mean_value += value/M  # Average of All the Predicted Mean(x), hat{mu}(x)

        sigma_epi_value = 0.0
        for value in mean_values:
            sigma_epi_value += ((value - mean_value)**2)/M

        sigma_alea_value = 0.0
        for value in sigma_alea_values:
            sigma_alea_value += value/M

        sigma_tot_value = sigma_epi_value + sigma_alea_value

        # print(sigma_tot_value)

        x_values.append(x_value)                            # (1000, )
        final_mean_values.append(mean_value)                # (1000, )
        final_sigma_epi_values.append(sigma_epi_value)      # (1000, )
        final_sigma_alea_values.append(sigma_alea_value)    # (1000, )
        final_sigma_tot_values.append(sigma_tot_value)      # (1000, )

# ============================== #
#  Predictive Uncertainty Plots  #
# ============================== #
print("Preparing Plots...")

plt.figure(1)
plt.plot(x_values, final_mean_values, "r", label=r"Predicted: $\hat{\mu}(x)$")
plt.fill_between(x_values, np.array(final_mean_values) - 2*np.sqrt(np.array(final_sigma_alea_values)), np.array(final_mean_values) + 2*np.sqrt(np.array(final_sigma_alea_values)), color="C4", alpha=0.25, label=r"$Predicted: \hat{\sigma}^2_{alea}(x)$")
plt.plot(x_values, np.sin(np.array(x_values)), "k", label=r"True: $\mu(x)$")
plt.axvline(x=-3.0, linestyle="--", color="0.5")
plt.axvline(x=3.0, linestyle="--", color="0.5")
plt.fill_between(x_values, np.sin(np.array(x_values)) - 2*0.15*(1.0/(1 + np.exp(-np.array(x_values)))), np.sin(np.array(x_values)) + 2*0.15*(1.0/(1 + np.exp(-np.array(x_values)))), color="0.5", alpha=0.25, label=r"$True: \sigma^2(x)$")
plt.ylabel(r"$\mu(x)$")
plt.xlabel("x")
plt.title("predicted vs true mean(x) with aleatoric uncertainty")
plt.xlim([-7, 7])
plt.legend()
plt.savefig("%s/mu_alea_pred_true.png" % network.model_dir)
# plt.close(1)

plt.figure(2)
plt.plot(x_values, np.sin(np.array(x_values)), "k", label=r"True: $\mu(x)$")
plt.axvline(x=-3.0, linestyle="--", color="0.5")
plt.axvline(x=3.0, linestyle="--", color="0.5")
plt.fill_between(x_values, np.sin(np.array(x_values)) - 2*0.15*(1.0/(1 + np.exp(-np.array(x_values)))), np.sin(np.array(x_values)) + 2*0.15*(1.0/(1 + np.exp(-np.array(x_values)))), color="0.5", alpha=0.25, label=r"$True: \sigma^2(x)$")
plt.ylabel(r"$\mu(x)$")
plt.xlabel("x")
plt.title("true mean(x) with aleatoric uncertainty")
plt.xlim([-7, 7])
plt.legend()
plt.savefig("%s/mu_alea_true.png" % network.model_dir)
# plt.close(2)

plt.figure(3)
plt.plot(x_values, final_mean_values, "r", label=r" Predicted: $\hat{\mu}(x)$")
plt.fill_between(x_values, np.array(final_mean_values) - 2*np.sqrt(np.array(final_sigma_alea_values)), np.array(final_mean_values) + 2*np.sqrt(np.array(final_sigma_alea_values)), color="C4", alpha=0.25, label=r"$Predicted: \hat{\sigma}^2_{alea}(x)$")
plt.axvline(x=-3.0, linestyle="--", color="0.5")
plt.axvline(x=3.0, linestyle="--", color="0.5")
plt.ylabel(r"$\mu(x)$")
plt.xlabel("x")
plt.title("predicted mean(x) with aleatoric uncertainty")
plt.xlim([-7, 7])
plt.legend()
plt.savefig("%s/mu_alea_pred.png" % network.model_dir)
# plt.close(3)

plt.figure(4)
plt.plot(x_values, np.sqrt(np.array(final_sigma_alea_values)), "r", label=r"$Predicted: \hat{\sigma}^2_{alea}(x)$")
plt.plot(x_values, 0.15*(1.0/(1 + np.exp(-np.array(x_values)))), "k", label=r"True: $\sigma^2(x)$")
plt.axvline(x=-3.0, linestyle="--", color="0.5")
plt.axvline(x=3.0, linestyle="--", color="0.5")
plt.ylabel(r"$\sigma^2(x)$")
plt.xlabel("x")
plt.title("predicted vs true aleatoric uncertainty")
plt.xlim([-7, 7])
plt.legend()
plt.savefig("%s/alea_pred_true.png" % network.model_dir)
# plt.close(4)

plt.figure(5)
plt.plot(x_values, np.sqrt(np.array(final_sigma_epi_values)), "r", label=r"$Predicted: \hat{\sigma}^2_{epi}(x)$")
plt.axvline(x=-3.0, linestyle="--", color="0.5")
plt.axvline(x=3.0, linestyle="--", color="0.5")
plt.ylabel(r"$\sigma^2(x)$")
plt.xlabel("x")
plt.title("predicted epistemic uncertainty")
plt.xlim([-7, 7])
plt.legend()
plt.savefig("%s/epi_pred.png" % network.model_dir)
# plt.close(5)

plt.figure(6)
plt.plot(x_values, final_mean_values, "r", label=r" Predicted: $\hat{\mu}(x)$")
plt.fill_between(x_values, np.array(final_mean_values) - 2*np.sqrt(np.array(final_sigma_epi_values)), np.array(final_mean_values) + 2*np.sqrt(np.array(final_sigma_epi_values)), color="C1", alpha=0.25, label=r"$Predicted: \hat{\sigma}^2_{epi}(x)$")
plt.plot(x_values, np.sin(np.array(x_values)), "k", label=r"True: $\mu(x)$")
plt.axvline(x=-3.0, linestyle="--", color="0.5")
plt.axvline(x=3.0, linestyle="--", color="0.5")
plt.ylabel(r"$\mu(x)$")
plt.xlabel("x")
plt.title("predicted vs true mean(x) with epistemic uncertainty")
plt.xlim([-7, 7])
plt.legend()
plt.savefig("%s/mu_epi_pred_true.png" % network.model_dir)
# plt.close(6)

plt.figure(7)
plt.plot(x_values, final_mean_values, "r", label=r" Predicted: $\hat{\mu}(x)$")
plt.fill_between(x_values, np.array(final_mean_values) - 2*np.sqrt(np.array(final_sigma_tot_values)), np.array(final_mean_values) + 2*np.sqrt(np.array(final_sigma_tot_values)), color="C3", alpha=0.25, label=r"$Predicted: \hat{\sigma}^2_{tot}(x)$")
plt.plot(x_values, np.sin(np.array(x_values)), "k", label=r"True: $\mu(x)$")
plt.axvline(x=-3.0, linestyle="--", color="0.5")
plt.axvline(x=3.0, linestyle="--", color="0.5")
plt.fill_between(x_values, np.sin(np.array(x_values)) - 2*0.15*(1.0/(1 + np.exp(-np.array(x_values)))), np.sin(np.array(x_values)) + 2*0.15*(1.0/(1 + np.exp(-np.array(x_values)))), color="0.5", alpha=0.25, label=r"$True: \sigma^2(x)$")
plt.ylabel(r"$\mu(x)$")
plt.xlabel("x")
plt.title("predicted vs true mean(x) with total uncertainty")
plt.xlim([-7, 7])
plt.legend()
plt.savefig("%s/mu_tot_pred_true.png" % network.model_dir)
# plt.close(7)

# Load HMC (Hamiltonian Monte Carlo)
with open("/workspace/evaluating_bdl/toyRegression/HMC/x_values.pkl", "rb") as file: # (needed for python3)
    x_values_HMC = pickle.load(file) # (list of 1000 elements)

with open("/workspace/evaluating_bdl/toyRegression/HMC/final_mean_values.pkl", "rb") as file: # (needed for python3)
    mean_values_HMC = pickle.load(file) # (list of 1000 elements)

with open("/workspace/evaluating_bdl/toyRegression/HMC/final_sigma_tot_values.pkl", "rb") as file: # (needed for python3)
    sigma_squared_values_HMC = pickle.load(file) # (list of 1000 elements)

# FIXME: Similar to Figure 2?
plt.figure(8)
plt.plot(x_values, np.sin(np.array(x_values)), "k", label=r"True: $\mu(x)$")
plt.axvline(x=-3.0, linestyle="--", color="0.5")
plt.axvline(x=3.0, linestyle="--", color="0.5")
plt.fill_between(x_values, np.sin(np.array(x_values)) - 2*0.15*(1.0/(1 + np.exp(-np.array(x_values)))), np.sin(np.array(x_values)) + 2*0.15*(1.0/(1 + np.exp(-np.array(x_values)))), color="0.5", alpha=0.25, label=r"$True: \sigma^2(x)$")
plt.xlim([-7, 7])
plt.ylim([-4.25, 4.25])
plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
plt.legend()
plt.savefig("%s/predictive_density_GT_.png" % network.model_dir)
plt.savefig("%s/predictive_density_GT_.pdf" % network.model_dir, dpi=400)
# plt.close(8)

plt.figure(9)
plt.plot(x_values_HMC, mean_values_HMC, "b", label=r" Predicted: $\hat{\mu}_{HMC}(x)$")
plt.fill_between(x_values_HMC, np.array(mean_values_HMC) - 2*np.sqrt(np.array(sigma_squared_values_HMC)), np.array(mean_values_HMC) + 2*np.sqrt(np.array(sigma_squared_values_HMC)), color="C0", alpha=0.25, label=r"$Predicted: \hat{\sigma}^2_{HMC}(x)$")
plt.plot(x_values, np.sin(np.array(x_values)), "k", label=r"True: $\mu(x)$")
plt.fill_between(x_values, np.sin(np.array(x_values)) - 2*0.15*(1.0/(1 + np.exp(-np.array(x_values)))), np.sin(np.array(x_values)) + 2*0.15*(1.0/(1 + np.exp(-np.array(x_values)))), color="0.5", alpha=0.25, label=r"$True: \sigma^2(x)$")
plt.axvline(x=-3.0, linestyle="--", color="0.5")
plt.axvline(x=3.0, linestyle="--", color="0.5")
plt.xlim([-7, 7])
plt.ylim([-4.25, 4.25])
plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
plt.legend()
plt.savefig("%s/predictive_density_HMC_.png" % network.model_dir)
plt.savefig("%s/predictive_density_HMC_.pdf" % network.model_dir, dpi=400)
# plt.close(9)

plt.figure(10)
plt.plot(x_values, final_mean_values, "r", label=r" Predicted: $\hat{\mu}(x)$")
plt.fill_between(x_values, np.array(final_mean_values) - 2*np.sqrt(np.array(final_sigma_tot_values)), np.array(final_mean_values) + 2*np.sqrt(np.array(final_sigma_tot_values)), color="C3", alpha=0.25, label=r"$Predicted: \hat{\sigma}^2_{tot}(x)$")
plt.plot(x_values, np.sin(np.array(x_values)), "k", label=r"True: $\mu(x)$")
plt.axvline(x=-3.0, linestyle="--", color="0.5")
plt.axvline(x=3.0, linestyle="--", color="0.5")
plt.fill_between(x_values, np.sin(np.array(x_values)) - 2*0.15*(1.0/(1 + np.exp(-np.array(x_values)))), np.sin(np.array(x_values)) + 2*0.15*(1.0/(1 + np.exp(-np.array(x_values)))), color="0.5", alpha=0.25, label=r"$True: \sigma^2(x)$")
plt.xlim([-7, 7])
plt.ylim([-4.25, 4.25])
plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
plt.legend()
plt.savefig("%s/predictive_density_.png" % network.model_dir)
plt.savefig("%s/predictive_density_.pdf" % network.model_dir, dpi=400)
# plt.close(10)

with open("/workspace/evaluating_bdl/toyRegression/x.pkl", "rb") as file: # (needed for python3)
    x = pickle.load(file)

with open("/workspace/evaluating_bdl/toyRegression/y.pkl", "rb") as file: # (needed for python3)
    y = pickle.load(file)

plt.figure(11)
# plt.plot(x, y, linestyle="None", marker="2", color="k", markersize=2.5)
plt.plot(x, y, "2k", label='Training Data')
plt.axvline(x=-3.0, linestyle="--", color="0.5")
plt.axvline(x=3.0, linestyle="--", color="0.5")
plt.xlim([-7, 7])
plt.ylim([-4.25, 4.25])
plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
plt.legend()
plt.savefig("%s/training_data.png" % network.model_dir)
plt.savefig("%s/training_data.pdf" % network.model_dir, dpi=400)
# plt.close(11)

plt.show()

# =============================== #
#  Parameters Distribution Plots  #
# =============================== #
M = int(M)

fc1_mean_weight_samples = np.zeros((M, 1, 10, 1))
fc1_mean_bias_samples = np.zeros((M, 1, 10))
fc2_mean_weight_samples = np.zeros((M, 1, 10, 10))
fc2_mean_bias_samples = np.zeros((M, 1, 10))
fc3_mean_weight_samples = np.zeros((M, 1, 1, 10))
fc3_mean_bias_samples = np.zeros((M, 1, 1))

fc1_var_weight_samples = np.zeros((M, 1, 10, 1))
fc1_var_bias_samples = np.zeros((M, 1, 10))
fc2_var_weight_samples = np.zeros((M, 1, 10, 10))
fc2_var_bias_samples = np.zeros((M, 1, 10))
fc3_var_weight_samples = np.zeros((M, 1, 1, 10))
fc3_var_bias_samples = np.zeros((M, 1, 1))

for index, network in enumerate(networks):
    for name, param in network.named_parameters():
        if name == "fc1_mean.weight":
            fc1_mean_weight_samples[index, 0, :] = param.data.cpu().numpy()
        elif name == "fc1_mean.bias":
            fc1_mean_bias_samples[index, 0, :] = param.data.cpu().numpy()
        elif name == "fc2_mean.weight":
            fc2_mean_weight_samples[index, 0, :] = param.data.cpu().numpy()
        elif name == "fc2_mean.bias":
            fc2_mean_bias_samples[index, 0, :] = param.data.cpu().numpy()
        elif name == "fc3_mean.weight":
            fc3_mean_weight_samples[index, 0, :] = param.data.cpu().numpy()
        elif name == "fc3_mean.bias":
            fc3_mean_bias_samples[index, 0, :] = param.data.cpu().numpy()
        elif name == "fc1_var.weight":
            fc1_var_weight_samples[index, 0, :] = param.data.cpu().numpy()
        elif name == "fc1_var.bias":
            fc1_var_bias_samples[index, 0, :] = param.data.cpu().numpy()
        elif name == "fc2_var.weight":
            fc2_var_weight_samples[index, 0, :] = param.data.cpu().numpy()
        elif name == "fc2_var.bias":
            fc2_var_bias_samples[index, 0, :] = param.data.cpu().numpy()
        elif name == "fc3_var.weight":
            fc3_var_weight_samples[index, 0, :] = param.data.cpu().numpy()
        elif name == "fc3_var.bias":
            fc3_var_bias_samples[index, 0, :] = param.data.cpu().numpy()
        else:
            raise Exception("Unknown network parameter!")

import os
if not os.path.exists("%s/param_distributions" % (network.model_dir)):
    os.makedirs("%s/param_distributions" % (network.model_dir))

# (fc1_mean_weight_samples has shape: (M, 1, 10, 1))
for param_index in range(10):
    values = fc1_mean_weight_samples[:, 0, param_index, 0] # (shape: (M, ))
    plt.figure(1)
    plt.hist(np.array(values), bins=100)
    plt.savefig("%s/param_distributions/fc1_mean_weight_%d.png" % (network.model_dir, param_index))
    plt.close(1)

# (fc1_mean_bias_samples has shape: (M, 1, 10))
for param_index in range(10):
    values = fc1_mean_bias_samples[:, 0, param_index] # (shape: (M, ))
    plt.figure(1)
    plt.hist(np.array(values), bins=100)
    plt.savefig("%s/param_distributions/fc1_mean_bias_%d.png" % (network.model_dir, param_index))
    plt.close(1)

# (fc2_mean_weight_samples has shape: (M, 1, 10, 10))
for param_index_i in range(10):
    for param_index_j in range(10):
        values = fc2_mean_weight_samples[:, 0, param_index_i, param_index_j] # (shape: (M, ))
        plt.figure(1)
        plt.hist(np.array(values), bins=100)
        plt.savefig("%s/param_distributions/fc2_mean_weight_%d_%d.png" % (network.model_dir, param_index_i, param_index_j))
        plt.close(1)

# (fc2_mean_bias_samples has shape: (M, 1, 10))
for param_index in range(10):
    values = fc2_mean_bias_samples[:, 0, param_index] # (shape: (M, ))
    plt.figure(1)
    plt.hist(np.array(values), bins=100)
    plt.savefig("%s/param_distributions/fc2_mean_bias_%d.png" % (network.model_dir, param_index))
    plt.close(1)

# (fc3_mean_weight_samples has shape: (M, 1, 1, 10))
for param_index in range(10):
    values = fc3_mean_weight_samples[:, 0, 0, param_index] # (shape: (M, ))
    plt.figure(1)
    plt.hist(np.array(values), bins=100)
    plt.savefig("%s/param_distributions/fc3_mean_weight_%d.png" % (network.model_dir, param_index))
    plt.close(1)

# (fc3_mean_bias_samples has shape: (M, 1, 1))
values = fc3_mean_bias_samples[:, 0, 0] # (shape: (M, ))
plt.figure(1)
plt.hist(np.array(values), bins=100)
plt.savefig("%s/param_distributions/fc3_mean_bias.png" % (network.model_dir))
plt.close(1)


# (fc1_var_weight_samples has shape: (M, 1, 10, 1))
for param_index in range(10):
    values = fc1_var_weight_samples[:, 0, param_index, 0] # (shape: (M, ))
    plt.figure(1)
    plt.hist(np.array(values), bins=100)
    plt.savefig("%s/param_distributions/fc1_var_weight_%d.png" % (network.model_dir, param_index))
    plt.close(1)

# (fc1_var_bias_samples has shape: (M, 1, 10))
for param_index in range(10):
    values = fc1_var_bias_samples[:, 0, param_index] # (shape: (M, ))
    plt.figure(1)
    plt.hist(np.array(values), bins=100)
    plt.savefig("%s/param_distributions/fc1_var_bias_%d.png" % (network.model_dir, param_index))
    plt.close(1)

# (fc2_var_weight_samples has shape: (M, 1, 10, 10))
for param_index_i in range(10):
    for param_index_j in range(10):
        values = fc2_var_weight_samples[:, 0, param_index_i, param_index_j] # (shape: (M, ))
        plt.figure(1)
        plt.hist(np.array(values), bins=100)
        plt.savefig("%s/param_distributions/fc2_var_weight_%d_%d.png" % (network.model_dir, param_index_i, param_index_j))
        plt.close(1)

# (fc2_var_bias_samples has shape: (M, 1, 10))
for param_index in range(10):
    values = fc2_var_bias_samples[:, 0, param_index] # (shape: (M, ))
    plt.figure(1)
    plt.hist(np.array(values), bins=100)
    plt.savefig("%s/param_distributions/fc2_var_bias_%d.png" % (network.model_dir, param_index))
    plt.close(1)

# (fc3_var_weight_samples has shape: (M, 1, 1, 10))
for param_index in range(10):
    values = fc3_var_weight_samples[:, 0, 0, param_index] # (shape: (M, ))
    plt.figure(1)
    plt.hist(np.array(values), bins=100)
    plt.savefig("%s/param_distributions/fc3_var_weight_%d.png" % (network.model_dir, param_index))
    plt.close(1)

# (fc3_var_bias_samples has shape: (M, 1, 1))
values = fc3_var_bias_samples[:, 0, 0] # (shape: (M, ))
plt.figure(1)
plt.hist(np.array(values), bins=100)
plt.savefig("%s/param_distributions/fc3_var_bias.png" % (network.model_dir))
plt.close(1)

plt.show()