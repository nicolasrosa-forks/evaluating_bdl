import torch

print("CUDA available: ", torch.cuda.is_available())

num_gpus = torch.cuda.device_count()

print("Number of devices: ", num_gpus)
print("Current device: ", torch.cuda.current_device())
print()

for id in range(num_gpus):
    print(f"device[{id}].name: ", torch.cuda.get_device_name(id))
    print(f"device[{id}]: ", torch.cuda.device(id))
    print()
