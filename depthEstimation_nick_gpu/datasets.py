# ========== #
#  Datasets  #
# ========== #

# System Libraries
import cv2
import numpy as np
import os
import random
from tqdm import tqdm

# Torch Libraries
import torch
from torch.utils import data
import pickle
from torchvision import transforms

# DataAugmentation Config
doFlip = True
doRandomCrop = True
train_on_single_image = False

def print_image_info(image):
    print(image.shape, image.dtype, np.min(image), np.max(image))

# ImageNet Normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Excluded sequences
campus_sequences = [
    "2011_09_28_drive_0016_sync",
    "2011_09_28_drive_0021_sync",
    "2011_09_28_drive_0034_sync",
    "2011_09_28_drive_0035_sync",
    "2011_09_28_drive_0037_sync",
    "2011_09_28_drive_0038_sync",
    "2011_09_28_drive_0039_sync",
    "2011_09_28_drive_0043_sync",
    "2011_09_28_drive_0045_sync",
    "2011_09_28_drive_0047_sync",
]

person_sequences = [
    "2011_09_28_drive_0053_sync",
    "2011_09_28_drive_0054_sync",
    "2011_09_28_drive_0057_sync",
    "2011_09_28_drive_0065_sync",
    "2011_09_28_drive_0066_sync",
    "2011_09_28_drive_0068_sync",
    "2011_09_28_drive_0070_sync",
    "2011_09_28_drive_0071_sync",
    "2011_09_28_drive_0075_sync",
    "2011_09_28_drive_0077_sync",
    "2011_09_28_drive_0078_sync",
    "2011_09_28_drive_0080_sync",
    "2011_09_28_drive_0082_sync",
    "2011_09_28_drive_0086_sync",
    "2011_09_28_drive_0087_sync",
    "2011_09_28_drive_0089_sync",
    "2011_09_28_drive_0090_sync",
    "2011_09_28_drive_0094_sync",
    "2011_09_28_drive_0095_sync",
    "2011_09_28_drive_0096_sync",
    "2011_09_28_drive_0098_sync",
    "2011_09_28_drive_0100_sync",
    "2011_09_28_drive_0102_sync",
    "2011_09_28_drive_0103_sync",
    "2011_09_28_drive_0104_sync",
    "2011_09_28_drive_0106_sync",
    "2011_09_28_drive_0108_sync",
    "2011_09_28_drive_0110_sync",
    "2011_09_28_drive_0113_sync",
    "2011_09_28_drive_0117_sync",
    "2011_09_28_drive_0119_sync",
    "2011_09_28_drive_0121_sync",
    "2011_09_28_drive_0122_sync",
    "2011_09_28_drive_0125_sync",
    "2011_09_28_drive_0126_sync",
    "2011_09_28_drive_0128_sync",
    "2011_09_28_drive_0132_sync",
    "2011_09_28_drive_0134_sync",
    "2011_09_28_drive_0135_sync",
    "2011_09_28_drive_0136_sync",
    "2011_09_28_drive_0138_sync",
    "2011_09_28_drive_0141_sync",
    "2011_09_28_drive_0143_sync",
    "2011_09_28_drive_0145_sync",
    "2011_09_28_drive_0146_sync",
    "2011_09_28_drive_0149_sync",
    "2011_09_28_drive_0153_sync",
    "2011_09_28_drive_0154_sync",
    "2011_09_28_drive_0155_sync",
    "2011_09_28_drive_0156_sync",
    "2011_09_28_drive_0160_sync",
    "2011_09_28_drive_0161_sync",
    "2011_09_28_drive_0162_sync",
    "2011_09_28_drive_0165_sync",
    "2011_09_28_drive_0166_sync",
    "2011_09_28_drive_0167_sync",
    "2011_09_28_drive_0168_sync",
    "2011_09_28_drive_0171_sync",
    "2011_09_28_drive_0174_sync",
    "2011_09_28_drive_0177_sync",
    "2011_09_28_drive_0179_sync",
    "2011_09_28_drive_0183_sync",
    "2011_09_28_drive_0184_sync",
    "2011_09_28_drive_0185_sync",
    "2011_09_28_drive_0186_sync",
    "2011_09_28_drive_0187_sync",
    "2011_09_28_drive_0191_sync",
    "2011_09_28_drive_0192_sync",
    "2011_09_28_drive_0195_sync",
    "2011_09_28_drive_0198_sync",
    "2011_09_28_drive_0199_sync",
    "2011_09_28_drive_0201_sync",
    "2011_09_28_drive_0204_sync",
    "2011_09_28_drive_0205_sync",
    "2011_09_28_drive_0208_sync",
    "2011_09_28_drive_0209_sync",
    "2011_09_28_drive_0214_sync",
    "2011_09_28_drive_0216_sync",
    "2011_09_28_drive_0220_sync",
    "2011_09_28_drive_0222_sync",
]

def imagenet_normalization(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0

    # print(img)
    # print_image_info(img)

    for channel in range(3):
        img[channel] = (img[channel] - mean[channel]) / std[channel]
    
    # print(img)
    # print_image_info(img)
    # input()

    return img

def print_example(example):
    print(example["img_path"])
    print(example["sparse_path"])
    print(example["target_path"])
    print(example["file_id"])
    print()

################################################################################
# KITTI:
################################################################################
class DatasetKITTIAugmentation(data.Dataset):  # TODO: Create .txt filename for avoiding online detection of the training files
    def __init__(self, kitti_depth_path, kitti_rgb_path, max_iters=None, crop_size=(352, 352), show_images=False):
        self.crop_h, self.crop_w = crop_size

        self.kitti_depth_train_path = kitti_depth_path + "/train"
        # self.kitti_rgb_train_path = kitti_rgb_path + "/train"
        self.kitti_rgb_train_path = kitti_rgb_path

        self.showImages = show_images
        self.num_images = None

        # print(self.kitti_depth_train_path)
        # print(self.kitti_rgb_train_path)
        # input('aki')

        # Get folders names
        train_dir_names = os.listdir(self.kitti_depth_train_path) # (contains "2011_09_26_drive_0001_sync" and so on)

        # Exclude 'cammpus' sequences
        for seq in campus_sequences:
            try:
                train_dir_names.remove(seq)
            except ValueError:
                print(seq)

        # Exclude 'person' sequences
        for seq in person_sequences:
            try:
                train_dir_names.remove(seq)
            except ValueError:
                print(seq)
     
        # print(train_dir_names)
        # print(len(train_dir_names))

        # Get filenames
        self.examples = []

        train_examples_pickle_path = "./train_examples.pickle"

        if not os.path.exists(train_examples_pickle_path):
            print("Reading 'DatasetKITTIAugmentation' folders:")
            for dir_name in tqdm(train_dir_names):
                for suffix in ["02", "03"]:
                    # Get 'image_0x' filenames
                    groundtruth_dir_path_0x = self.kitti_depth_train_path + "/" + dir_name + "/proj_depth/groundtruth/image_{}".format(suffix)
                    file_ids_0x = os.listdir(groundtruth_dir_path_0x) # (contains e.g. "0000000005.png" and so on)

                    # Get the corresponding RGB, Sparse and Target Depth filenames
                    for file_id in file_ids_0x:
                        target_path = self.kitti_depth_train_path + "/" + dir_name + "/proj_depth/groundtruth/image_{}/".format(suffix) + file_id   # Semi-Dense Depth (11 LiDAR Scans)
                        sparse_path = self.kitti_depth_train_path + "/" + dir_name + "/proj_depth/velodyne_raw/image_{}/".format(suffix) + file_id  # Sparse Depth (1 LiDAR Scan)
                        img_path = self.kitti_rgb_train_path + "/" + dir_name + "/image_{}/data/".format(suffix) + file_id                          # RGB Image

                        example = {}
                        example["img_path"] = img_path
                        example["sparse_path"] = sparse_path
                        example["target_path"] = target_path
                        example["file_id"] = groundtruth_dir_path_0x + "/" + file_id

                        # print_example(example)
                        # input('aki')

                        self.examples.append(example)

            with open(train_examples_pickle_path, 'wb') as f:
                pickle.dump(self.examples, f)
        else:
            with open(train_examples_pickle_path, 'rb') as f:
                self.examples = pickle.load(f)

        if train_on_single_image:
            self.examples = self.examples[100:101]
        
        self.num_images = len(self.examples)

        print ("DatasetKITTIAugmentation - num unique examples: %d" % len(self.examples))
        if max_iters is not None:
            self.examples = self.examples*int(np.ceil(float(max_iters)/len(self.examples)))
        print ("DatasetKITTIAugmentation - num examples: %d\n" % len(self.examples))

        # print("'DatasetKITTIAugmentaiton' object created!")
    
    def __len__(self):
        return len(self.examples)

    def getNumImages(self):
        return self.num_images

    def __getitem__(self, index):
        example = self.examples[index]

        img_path = example["img_path"]
        sparse_path = example["sparse_path"]
        target_path = example["target_path"]
        file_id = example["file_id"]

        img = cv2.imread(img_path, -1) # (shape: (375, 1242, 3), dtype: uint8) (or something close to (375, 1242))
        sparse = cv2.imread(sparse_path, -1) # (shape: (375, 1242), dtype: uint16)
        target = cv2.imread(target_path, -1) # (shape: (375, 1242), dtype: uint16)

        try:
            # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization START
            if self.showImages:
                print (img.shape)
                print (sparse.shape)
                print (target.shape)
                
                cv2.imshow("img (uint8)", img)
                # cv2.waitKey(0)
                
                cv2.imshow("sparse (uint16)", sparse)
                # cv2.waitKey(0)
                
                cv2.imshow("target (uint16)", target)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            # # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization END

            # crop to the bottom center (352, 1216):
            new_img_h = 352
            new_img_w = 1216 # (this is the image size of all images in the selected val/test sets)

            img_h = img.shape[0]
            img_w = img.shape[1]

            img = img[(img_h - new_img_h):img_h, int(img_w/2.0 - new_img_w/2.0):int(img_w/2.0 + new_img_w/2.0)] # (shape: (352, 1216, 3))
            sparse = sparse[(img_h - new_img_h):img_h, int(img_w/2.0 - new_img_w/2.0):int(img_w/2.0 + new_img_w/2.0)] # (shape: (352, 1216))
            target = target[(img_h - new_img_h):img_h, int(img_w/2.0 - new_img_w/2.0):int(img_w/2.0 + new_img_w/2.0)] # (shape: (352, 1216))

            # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization START
            if self.showImages:
                print (img.shape)
                print (sparse.shape)
                print (target.shape)
                
                cv2.imshow("img (uint8, cropped)", img)
                # cv2.waitKey(0)
                
                cv2.imshow("sparse (uint16, cropped)", sparse)
                # cv2.waitKey(0)
                
                cv2.imshow("target (uint16, cropped)", target)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            # # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization END

            # flip img, sparse and target along the vertical axis with 0.5 probability:
            flip = np.random.randint(low=0, high=2)
            if doFlip and flip == 1:
                img = cv2.flip(img, 1)
                sparse = cv2.flip(sparse, 1)
                target = cv2.flip(target, 1)
            
            # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization START
            if self.showImages:
                print (img.shape)
                print (sparse.shape)
                print (target.shape)
                
                cv2.imshow("img (uint8, cropped, flipped)", img)
                # cv2.waitKey(0)
                
                cv2.imshow("sparse (uint8, cropped, flipped)", sparse)
                # cv2.waitKey(0)
                
                cv2.imshow("target (uint8, cropped, flipped)", target)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            # # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization END

            # select a random (crop_h, crop_w) crop:
            if doRandomCrop:
                img_h, img_w = sparse.shape
                h_off = random.randint(0, img_h - self.crop_h)
                w_off = random.randint(0, img_w - self.crop_w)
            else:
                h_off = 0
                w_off = 0

            img = img[h_off:(h_off+self.crop_h), w_off:(w_off+self.crop_w)] # (shape: (crop_h, crop_w, 3))
            sparse = sparse[h_off:(h_off+self.crop_h), w_off:(w_off+self.crop_w)] # (shape: (crop_h, crop_w))
            target = target[h_off:(h_off+self.crop_h), w_off:(w_off+self.crop_w)] # (shape: (crop_h, crop_w))

            # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization START
            if self.showImages:
                print (img.shape)
                print (sparse.shape)
                print (target.shape)
                
                cv2.imshow("img (uint8, cropped, flipped, random cropped)", img)
                # cv2.waitKey(0)
                
                cv2.imshow("sparse (uint16, cropped, flipped, random cropped)", sparse)
                # cv2.waitKey(0)
                
                cv2.imshow("target (uint16, cropped, flipped, random cropped)", target)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            # # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization END

            # Normalize input
            img = imagenet_normalization(img)

            # convert sparse and target to meters:
            sparse = sparse/256.0
            sparse = sparse.astype(np.float32)
            target = target/256.0
            target = target.astype(np.float32)

            # convert img to grayscale:
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # (shape: (352, 1216))

            # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization START
            if self.showImages:
                print (img.shape)
                print (sparse.shape)
                print (target.shape)
                
                cv2.imshow("img (rgb, normalized)", img)
                # cv2.waitKey(0)
                
                cv2.imshow("sparse (meters)", sparse.astype(np.uint8))
                # cv2.waitKey(0)
                
                cv2.imshow("target (meters)", target.astype(np.uint8))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            # # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization END

            # print_image_info(img)
            # img = img.astype(np.float32)
    
            return (img.copy(), sparse.copy(), target.copy(), file_id)
    
        except (AttributeError, TypeError) as e:
            print(e)
            print(file_id)


class DatasetKITTIVal(data.Dataset):
    def __init__(self, kitti_depth_path, show_images=False):
        self.kitti_depth_val_path = kitti_depth_path + "/depth_selection/val_selection_cropped"
        self.showImages = show_images

        img_dir = self.kitti_depth_val_path + "/image"
        sparse_dir = self.kitti_depth_val_path + "/velodyne_raw"
        target_dir = self.kitti_depth_val_path + "/groundtruth_depth"
        
        img_ids = os.listdir(img_dir) # (contains "2011_09_26_drive_0002_sync_image_0000000005_image_02.png" and so on)

        self.examples = []
        val_examples_pickle_path = "./val_examples.pickle"

        if not os.path.exists(val_examples_pickle_path):
            print("Reading 'DatasetKITTIVal' folders:")
            for img_id in tqdm(img_ids):
                # (img_id == "2011_09_26_drive_0002_sync_image_0000000005_image_02.png" (e.g.))

                img_path = img_dir + "/" + img_id

                file_id_start, file_id_end = img_id.split("_sync_image_")
                # (file_id_start == "2011_09_26_drive_0002")
                # (file_id_end == "0000000005_image_02.png")

                sparse_path = sparse_dir + "/" + file_id_start + "_sync_velodyne_raw_" + file_id_end

                target_path = target_dir + "/" + file_id_start + "_sync_groundtruth_depth_" + file_id_end

                example = {}
                example["img_path"] = img_path
                example["sparse_path"] = sparse_path
                example["target_path"] = target_path
                example["file_id"] = img_id
                self.examples.append(example)

            with open(val_examples_pickle_path, 'wb') as f:
                pickle.dump(self.examples, f)
        else:
            with open(val_examples_pickle_path, 'rb') as f:
                self.examples = pickle.load(f)

        print ("DatasetKITTIVal - num examples: %d\n" % len(self.examples))

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        example = self.examples[index]

        img_path = example["img_path"]
        sparse_path = example["sparse_path"]
        target_path = example["target_path"]
        file_id = example["file_id"]

        img = cv2.imread(img_path, -1) # (shape: (352, 1216, 3), dtype: uint8))
        sparse = cv2.imread(sparse_path, -1) # (shape: (352, 1216), dtype: uint16)
        target = cv2.imread(target_path, -1) # (shape: (352, 1216), dtype: uint16)

        try:
            # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization START
            if self.showImages:
                print (img.shape)
                print (sparse.shape)
                print (target.shape)
                
                cv2.imshow("img (uint8)", img)
                # cv2.waitKey(0)
                
                cv2.imshow("sparse (uint16)", sparse)
                # cv2.waitKey(0)
                
                cv2.imshow("target (uint16)", target)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            # # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization END

            # Normalize input
            img = imagenet_normalization(img)

            # convert sparse and target to meters:
            sparse = sparse/256.0
            sparse = sparse.astype(np.float32)
            target = target/256.0
            target = target.astype(np.float32)

            # convert img to grayscale:
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # (shape: (352, 1216))

            # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization START
            if self.showImages:
                print (img.shape)
                print (sparse.shape)
                print (target.shape)
                
                cv2.imshow("img (rgb, normalized)", img)
                # cv2.waitKey(0)
                
                cv2.imshow("sparse (meters)", sparse.astype(np.uint8))
                # cv2.waitKey(0)
                
                cv2.imshow("target (meters)", target.astype(np.uint8))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            # # # # # # # # # # # # # # # # # # # # # # # # # # debug visualization END

            # img = img.astype(np.float32)
        
            return (img.copy(), sparse.copy(), target.copy(), file_id)

        except (AttributeError, TypeError) as e:
            print(e)
            print(file_id)

if __name__ == "__main__":
    # Debug
    # kitti_depth_path = "/root/data/kitti_depth"
    # kitti_rgb_path = "/root/data/kitti_rgb"
    
    kitti_depth_path = "/home/lasi/Downloads/datasets/kitti/depth/data_depth_annotated"
    kitti_rgb_path = "/home/lasi/Downloads/datasets/kitti/raw_data"

    batch_size = 1
    num_steps = 40000

    train_dataset = DatasetKITTIAugmentation(kitti_depth_path=kitti_depth_path, kitti_rgb_path=kitti_rgb_path, max_iters=num_steps*batch_size, crop_size=(352, 352), show_images=False)
    val_dataset = DatasetKITTIVal(kitti_depth_path=kitti_depth_path, show_images=False)

    # Check Train Dataset
    print("Checking 'DatasetKITTIAugmentation' integrity...")
    for i in tqdm(range(len(train_dataset))):
        train_dataset[i]

    # Check Evaluating Dataset
    print("Checking 'DatasetKITTIVal' integrity...")
    for i in tqdm(range(len(val_dataset))):
        val_dataset[i]

    print("Done.")
