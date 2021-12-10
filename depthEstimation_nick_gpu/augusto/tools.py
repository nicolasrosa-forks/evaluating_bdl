import torch
import torchvision.transforms as T
import imageio
import numpy as np
import os
import cv2

def get_file_lists(dataset_folder, file_extension='.png', img_token='color', depth_token='depth', raw_token='rawDepth'):
  image_files = []
  depth_files = []
  rawDepth_files = []

  for img in os.listdir(dataset_folder):
    if ('-'+str(img_token)+file_extension) in img:
      image_files.append(img)
    elif ('-'+str(depth_token)+file_extension) in img:
      depth_files.append(img)
    elif ('-'+str(raw_token)+file_extension) in img:
      rawDepth_files.append(img)

  image_files = np.sort(image_files)
  depth_files = np.sort(depth_files)
  rawDepth_files = np.sort(rawDepth_files)

  return image_files, rawDepth_files, depth_files

def split_set_indices(file_list, split_ratio, shuffle=True, random_seed=42):
  dataset_size = len(file_list)
  indices = list(range(dataset_size))
  split = int(np.floor(split_ratio * dataset_size))
  if shuffle:
    np.random.seed(random_seed)
    np.random.shuffle(indices)

  train, val = indices[split:], indices[:split]
  return train, val

class RandomChannelSwap(object):
  def __init__(self, probability=0.5):
    from itertools import permutations
    self.p = probability
    self.indices = list(permutations(range(3), 3))

  def __call__(self, img):
    if np.random.random() < self.p:
        img = img[..., list(self.indices[np.random.randint(0, len(self.indices) - 1)])]
    return img

class RandomHorizontalFlip(object):
  def __init__(self, probability=0.5):
    self.flipper = T.RandomHorizontalFlip(p=1.0)
    self.p = probability

  def __call__(self, img, raw, ground_truth):
    if np.random.random() < self.p:
      img = self.flipper(img)
      raw = self.flipper(raw)
      ground_truth = self.flipper(ground_truth)

    return img, raw, ground_truth

class RGBDCompletionDataset(torch.utils.data.Dataset):
  def __init__(self, color_images_files, raw_depth_files, ground_truth_files, image_dir, crop_images=True, apply_augmentation=False):
    assert len(color_images_files) == len(raw_depth_files) == len(ground_truth_files)
    self.img_dir = image_dir
    self.color = color_images_files
    self.raw = raw_depth_files
    self.depth = ground_truth_files

    if apply_augmentation:
      # Data augmentations transforms
      self.hflipper = RandomHorizontalFlip()
      self.cflipper = RandomChannelSwap()

    self.augmentation = apply_augmentation

    ## Cortar imagens manualmente para que tenham o tamanho aproximados
    ## da medicao de profundidade e aplicar o operador morfologico de fechamento
    ## para tapar pequenos buracos (empirismo total aqui)
    if crop_images:
        self.h_start = 63
        self.h_end = 447
        self.w_start = 47
        self.w_end = 591
    else:
        self.h_start = 0
        self.h_end = 480
        self.w_start = 0
        self.w_end = 640

  def __len__(self):
    return len(self.color)

  def __getitem__(self, idx):
    assert idx < self.__len__()

    image_filename = os.path.join(self.img_dir, self.color[idx])
    depth_filename = os.path.join(self.img_dir, self.depth[idx])
    rawDepth_filename = os.path.join(self.img_dir, self.raw[idx])

    np_image = np.array(imageio.imread(image_filename)).astype(np.float32)
    np_image = np_image[self.h_start:self.h_end, self.w_start:self.w_end, :]
    np_depth = np.array(imageio.imread(depth_filename)).astype(np.float32)
    np_depth = np_depth[self.h_start:self.h_end, self.w_start:self.w_end]
    np_rawDepth = np.array(imageio.imread(rawDepth_filename)).astype(np.float32)
    np_rawDepth = np_rawDepth[self.h_start:self.h_end, self.w_start:self.w_end]

    if self.augmentation:
      np_image = self.cflipper(np_image)

    image = torch.from_numpy(np.transpose(np_image, axes=(2, 0, 1)))
    depth = torch.from_numpy(np.expand_dims(np_depth, axis=0))
    rawDepth = torch.from_numpy(np.expand_dims(np_rawDepth, axis=0))

    if self.augmentation:
      image, rawDepth, depth = self.hflipper(image, rawDepth, depth)

    return image.float(), rawDepth.float(), depth.float()

class DepthCompletionDataset(torch.utils.data.Dataset):
  def __init__(self, img_dir, img_token, depth_token, raw_token, file_extension, crop_images=True):
    self.img_dir = img_dir
    self.file_extension = file_extension
    self.input_token = img_token

    self.rawDepth_files = []
    self.depth_files = []
    self.image_files = []

    # Data augmentations transforms
    self.hflipper = RandomHorizontalFlip()
    self.cflipper = RandomChannelSwap()

    for img in os.listdir(self.img_dir):
      if ('-'+str(img_token)+self.file_extension) in img:
        self.image_files.append(img)
      elif ('-'+str(depth_token)+self.file_extension) in img:
        self.depth_files.append(img)
      elif ('-'+str(raw_token)+self.file_extension) in img:
        self.rawDepth_files.append(img)

    self.image_files = np.sort(self.image_files)
    self.depth_files = np.sort(self.depth_files)
    self.rawDepth_files = np.sort(self.rawDepth_files)

    ## Cortar imagens manualmente para que tenham o tamanho aproximados
    ## da medicao de profundidade e aplicar o operador morfologico de fechamento
    ## para tapar pequenos buracos (empirismo total aqui)
    if crop_images:
        self.h_start = 63
        self.h_end = 447
        self.w_start = 47
        self.w_end = 591
    else:
        self.h_start = 0
        self.h_end = 480
        self.w_start = 0
        self.w_end = 640

  def __len__(self):
    img_number = 0
    for img in os.listdir(self.img_dir):
      if ('-'+str(self.input_token)+self.file_extension) in img:
        img_number += 1
    return img_number

  def __getitem__(self, idx):
    idx_len = len(str(idx))
    if idx_len > 5:
      raise Exception('Index out of range')

    image_filename = os.path.join(self.img_dir, self.image_files[idx])
    depth_filename = os.path.join(self.img_dir, self.depth_files[idx])
    rawDepth_filename = os.path.join(self.img_dir, self.rawDepth_files[idx])

    np_image = np.array(imageio.imread(image_filename)).astype(np.float32)
    np_image = np_image[self.h_start:self.h_end, self.w_start:self.w_end, :]
    np_depth = np.array(imageio.imread(depth_filename)).astype(np.float32)
    np_depth = np_depth[self.h_start:self.h_end, self.w_start:self.w_end]
    np_rawDepth = np.array(imageio.imread(rawDepth_filename)).astype(np.float32)
    np_rawDepth = np_rawDepth[self.h_start:self.h_end, self.w_start:self.w_end]

    # # Perform data augmentation
    # np_image = self.cflipper(np_image)
    # np_image, np_rawDepth, np_depth = self.hflipper(np_image, np_rawDepth, np_depth)

    image = torch.from_numpy(np.transpose(np_image, axes=(2, 0, 1)))
    depth = torch.from_numpy(np.expand_dims(np_depth, axis=0))
    rawDepth = torch.from_numpy(np.expand_dims(np_rawDepth, axis=0))

    return image.float(), rawDepth.float(), depth.float()

def train_test_split(dataset, validation_split, batch_size, shuffle_dataset=True, random_seed=42):
  dataset_size = len(dataset)
  indices = list(range(dataset_size))
  split = int(np.floor(validation_split * dataset_size))
  if shuffle_dataset :
      np.random.seed(random_seed)
      np.random.shuffle(indices)
  train_indices, val_indices = indices[split:], indices[:split]

  train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
  valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

  train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                            sampler=train_sampler, shuffle=False)
  validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                  sampler=valid_sampler, shuffle=False)

  return train_loader, validation_loader

def min_max_norm(img, scale_factor=255):
  return scale_factor*(img-np.min(img))/(np.max(img)-np.min(img))

def MAE(output, label, mask=None):
  t1 = output.cpu().detach().numpy()
  t2 = label.cpu().detach().numpy()
  if mask != None:
    t_mask = mask.cpu().detach().numpy()
    t1 = t1[t_mask]
    t2 = t2[t_mask]
  return np.abs(t1-t2).mean()

def RMSE(output, label, mask=None):
  t1 = output.cpu().detach().numpy()
  t2 = label.cpu().detach().numpy()
  if mask != None:
    t_mask = mask.cpu().detach().numpy()
    t1 = t1[t_mask]
    t2 = t2[t_mask]
  return np.sqrt(np.square(t1-t2).mean())

def threshold(output, label, threshold, mask=None):
  y = np.squeeze(output.cpu().detach().numpy())
  y_star = np.squeeze(label.cpu().detach().numpy())
  if mask != None:
    t_mask = np.squeeze(mask.cpu().detach().numpy())
    y = y[t_mask]
    y_star = y_star[t_mask]

  # Calculate delta
  delta = np.maximum(y/y_star, y_star/y)

  # Apply constrain
  constrain = delta < threshold

  # Count the percentage of true values in maks
  if mask == None:
    n = constrain.shape[0]*constrain.shape[1]
  else:
    n = constrain.shape
  return 100*(np.sum(constrain)/(n))

def RelativeError(output, label, mask=None):
  t1 = output.cpu().detach().numpy()
  t2 = label.cpu().detach().numpy()
  if mask != None:
    t_mask = mask.cpu().detach().numpy()
    t1 = t1[t_mask]
    t2 = t2[t_mask]
  return (np.abs(t1-t2)/t2).mean()

def img_gradients(img):
  device = 'cuda' if img.is_cuda else 'cpu'
  _, channel, h, w = img.shape

  # Filtro Sobel para a direcao x:
  kernel_x = torch.from_numpy(np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])).to(device).unsqueeze(0).expand(1, channel, -1, -1).float() # possivelmente errado. Deveria ser (channel, channel/groups, -1, -1)
  gx = torch.nn.functional.conv2d(img, kernel_x, padding=1)

  # Filtro Sobel para a direcao y:
  kernel_y = torch.from_numpy(np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])).to(device).unsqueeze(0).expand(1, channel, -1, -1).float() # possivelmente errado. Deveria ser (channel, channel/groups, -1, -1)
  gy = torch.nn.functional.conv2d(img, kernel_y, padding=1)

  return gx, gy

def gradient_loss(outputs, target):
  gx_outputs, gy_outputs = img_gradients(outputs)
  gx_targets, gy_targets = img_gradients(target)

  return torch.mean(torch.abs(gx_outputs-gx_targets)+torch.abs(gy_outputs-gy_targets))

def gaussian_kernel(size, sigma):
  axis = np.arange((-size // 2)+1.0, (size // 2) + 1.0)
  x, y = np.meshgrid(axis, axis)
  kernel = np.exp(-(1/2)*(np.square(x)+np.square(y))/np.square(sigma))
  return kernel/kernel.sum()

def ssim(img1, img2, window_size, L):
  device = 'cuda' if img1.is_cuda else 'cpu'
  kernel = torch.from_numpy(gaussian_kernel(window_size, 1.5)).to(device)
  _, channels, _, _ = img1.size()
  kernel = kernel.unsqueeze(0).expand(1, channels, -1, -1).float() # possivelmente errado. Deveria ser (channel, channel/groups, -1, -1)

  mu_1 = torch.nn.functional.conv2d(img1, kernel, groups=channels)
  mu_2 = torch.nn.functional.conv2d(img2, kernel, groups=channels)
  mu_squared_1 = torch.pow(mu_1, 2)
  mu_squared_2 = torch.pow(mu_2, 2)

  sigma_squared_1 = torch.nn.functional.conv2d(img1*img1, kernel, groups=channels) - mu_squared_1
  sigma_squared_2 = torch.nn.functional.conv2d(img2*img2, kernel, groups=channels) - mu_squared_2
  covariance = torch.nn.functional.conv2d(img1*img2, kernel, groups=channels) - mu_1*mu_2

  k1 = 0.01
  k2 = 0.03

  c1 = (k1*L)**2
  c2 = (k2*L)**2

  ret = ((2*mu_1*mu_2+c1)*(2*covariance+c2))/((mu_squared_1+mu_squared_2+c1)*(sigma_squared_1+sigma_squared_2+c2))
  return ret

def structural_disparity_loss(outputs, target):
  similarity = ssim(outputs, target, window_size=11, L=7500.0).mean()
  return (1-similarity)/2.0

def tensor_to_rgb(tensor):
  np_img = tensor.cpu().detach().numpy()
  np_img = (255*np_img).astype(np.uint8)
  return np_img[0]

def normals(tensor):
    batch_size, channels, width, height = tensor.shape
    assert channels == 1

    ret = torch.zeros((batch_size, 3, width, height))
    for idx in range(batch_size):
        matrix = torch.nn.functional.pad(tensor[idx, :, :, :], (1, 1), mode='replicate')
        matrix = matrix[0]
        for x in range(1, matrix.shape[0]-1):
            for y in range(1, matrix.shape[1]-1):
                dx = (matrix[x+1, y]-matrix[x-1, y])/2.0
                dy = (matrix[x, y+1]-matrix[x, y-1])/2.0
                z = torch.tensor([[-dx, -dy, 1.0]])
                z = torch.nn.functional.normalize(z)

                ret[idx, :, x-1, y-1] = z

    return ret

def RGBtoGray(tensor):
  _, channels, _, _ = tensor.shape
  assert channels == 3

  return 0.299*tensor[:, 0, :, :]+0.587*tensor[:, 1, :, :]+0.114*tensor[:, 2, :, :]

def canny_edge_detector(batch, sigma=1.0):
  # Retrieve the device currently used for calculation purposes
  device = 'cuda' if batch.is_cuda else 'cpu'

  # Check dimensions
  batch_size, channels, _, _ = batch.shape
  assert channels == 3 or channels == 1

  # # First step: noise reduction
  # kernel = torch.from_numpy(gaussian_kernel(5, sigma=sigma)).to(device)
  # kernel = kernel.unsqueeze(0).expand(channels, 1, -1, -1).float()

  # batch_filtered = torch.nn.functional.conv2d(batch, kernel, groups=channels)

  # # Second step: get image gradients for each image in batch
  # gx, gy = img_gradients(batch_filtered)  
  # grad = torch.sqrt(gx**2+gy**2)/2.0
  # angle = torch.atan2(gy/gx)*(180/np.pi)
  # angle = torch.round(angle/45)*45

  gray = RGBtoGray(batch)
  ret = torch.zeros(gray.shape, dtype=torch.float)
  for i in range(batch_size):
    edges = 1-cv2.Canny(gray[i].cpu().numpy().astype(np.uint8), 100, 200).astype(np.float32)/255.0
    ret[i] = torch.from_numpy(edges)
  
  return ret.to(device)

