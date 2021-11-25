import torch
import numpy as np
import os
import cv2

class CityScapes(torch.utils.data.Dataset):
    def __init__(self, root, mode="fine", split='train', img_shape=(512, 1024)):
        self.shape = img_shape
        self.root = root
        self.mode = 'fine'
        self.split = split

        # Auxiliary variables
        self.path_to_rgb_cities = os.path.join(self.root, 'leftImg8bit', split)
        self.path_to_label_cities = os.path.join(self.root, "gt"+mode.capitalize(), split)

        self.rgb_images = []
        self.label_images = []
        for city in os.listdir(self.path_to_rgb_cities):
            for image in os.listdir(os.path.join(self.path_to_rgb_cities, city)):
                self.rgb_images.append(os.path.join(self.path_to_rgb_cities, city, image))
                img_token = image.replace('leftImg8bit.png', '')
                self.label_images.append(os.path.join(self.path_to_label_cities, city, img_token+"gt"+mode.capitalize()+'_labelIds.png'))

        assert len(self.rgb_images) == len(self.label_images)

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, index):
        image = np.array(cv2.cvtColor(cv2.resize(cv2.imread(self.rgb_images[index]), self.shape), cv2.COLOR_BGR2RGB))
        label = np.array(cv2.resize(cv2.imread(self.label_images[index], cv2.IMREAD_UNCHANGED), self.shape, interpolation=cv2.INTER_NEAREST))

        image = torch.from_numpy(np.transpose(image, axes=(2, 0, 1)))
        # image = torch.from_numpy(np.transpose(image, axes=(2, 1, 0)))
        # image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        
        if self.split == 'val':
            return image.float(), label.long(), self.rgb_images[index]

        return image.float(), label.long()

    def getNumImages(self):
        len(self.rgb_images)
