#!/bin/bash

# DEFAULT VALUES
GPUIDS="0"
NAME="toyProblems_depthEstimation_GPU"
NV_GPU="$GPUIDS"

# Allows everybody to use your host x server
xhost +

# Run docker
nvidia-docker run -it --rm --shm-size 12G \
        -p 5700:5700\
        --name "$NAME""0" \
        -v /home/lasi/Documents/nicolas/bnn:/workspace/ \
        -v /home/lasi/Downloads/datasets/kitti/depth/data_depth_annotated:/root/data/kitti_depth \
        -v /home/lasi/Downloads/datasets/kitti/depth/data_depth_selection/depth_selection/:/root/data/kitti_depth/depth_selection/ \
        -v /home/lasi/Downloads/datasets/kitti/raw_data/:/root/data/kitti_rgb/train \
        -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY -e XAUTHORITY -e NVIDIA_DRIVER_CAPABILITIES=all \
        fregu856/evaluating_bdl:pytorch_pytorch_0.4_cuda9_cudnn7_evaluating_bdl bash
