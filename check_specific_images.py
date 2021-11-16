from matplotlib import pyplot as plt

image_filenames = [
# '/root/data/kitti_depth/train/2011_09_26_drive_0039_sync/proj_depth/groundtruth/image_02/0000000040.png',
# '/root/data/kitti_depth/train/2011_09_26_drive_0039_sync/proj_depth/groundtruth/image_02/0000000058.png',
# '/root/data/kitti_depth/train/2011_09_26_drive_0039_sync/proj_depth/groundtruth/image_02/0000000076.png',
# '/root/data/kitti_depth/train/2011_09_26_drive_0039_sync/proj_depth/groundtruth/image_02/0000000022.png',
'/home/lasi/Downloads/datasets/kitti/depth/data_depth_annotated/train/2011_09_28_drive_0043_sync/proj_depth/groundtruth/image_02/0000000042.png',
'/home/lasi/Downloads/datasets/kitti/depth/data_depth_annotated/train/2011_09_28_drive_0043_sync/proj_depth/groundtruth/image_02/0000000038.png',
'/home/lasi/Downloads/datasets/kitti/depth/data_depth_annotated/train/2011_09_28_drive_0043_sync/proj_depth/groundtruth/image_02/0000000050.png',
'/home/lasi/Downloads/datasets/kitti/depth/data_depth_annotated/train/2011_09_28_drive_0043_sync/proj_depth/groundtruth/image_02/0000000113.png',
'/home/lasi/Downloads/datasets/kitti/depth/data_depth_annotated/train/2011_09_28_drive_0043_sync/proj_depth/groundtruth/image_02/0000000138.png',
'/home/lasi/Downloads/datasets/kitti/depth/data_depth_annotated/train/2011_09_28_drive_0043_sync/proj_depth/groundtruth/image_02/0000000034.png',
'/home/lasi/Downloads/datasets/kitti/depth/data_depth_annotated/train/2011_09_28_drive_0043_sync/proj_depth/groundtruth/image_02/0000000073.png',
'/home/lasi/Downloads/datasets/kitti/depth/data_depth_annotated/train/2011_09_28_drive_0043_sync/proj_depth/groundtruth/image_02/0000000053.png',
'/home/lasi/Downloads/datasets/kitti/depth/data_depth_annotated/train/2011_09_28_drive_0043_sync/proj_depth/groundtruth/image_02/0000000110.png'
]

for image_filename in image_filenames:

    try:
        img = plt.imread(image_filename)
        
        plt.figure(1)
        plt.imshow(img)
        plt.pause(1)

        print(image_filename, ' ok')

    except FileNotFoundError:
        print(image_filename, ' fail')

print("Done")