#!/bin/bash

# DEFAULT VALUES
GPUIDS="0"
NAME="toyProblems_depthCompletion_GPU"
NV_GPU="$GPUIDS"

# Allows everybody to use your host x server
xhost +

# Run docker
nvidia-docker run -it --rm --shm-size 12G \
        -p 5700:5700\
        --name "$NAME""0" \
        -v /home/olorin/nicolas/bnn:/root/ \
        -v /media/olorin/Documentos/datasets/kitti/depth/depth_prediction/data:/root/data/kitti_depth \
        -v /media/olorin/Documentos/datasets/kitti/raw_data/data:/root/data/kitti_rgb/train \
        -v /tmp/.X11-unix/:/tmp/.X11-unix/ \
        --env "DISPLAY" \
        --env QT_X11_NO_MITSHM=1 \
        fregu856/evaluating_bdl:pytorch_pytorch_0.4_cuda9_cudnn7_evaluating_bdl bash

# -v /media/nicolas/nicolas_seagate/datasets/kitti/depth/depth_prediction/data:/root/data/kitti_depth \
