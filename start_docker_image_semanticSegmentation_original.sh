#!/bin/bash

# DEFAULT VALUES
GPUIDS="0,1"
NAME="segmentation_GPU"
NV_GPU="$GPUIDS"

# Allows everybody to use your host x server
xhost +

# Run docker
nvidia-docker run -it --rm --shm-size 12G \
        -p 5900:5900 \
        --name "$NAME""01" \
        -v /home/nicolas/Downloads/bnn:/home/ \
        -v /media/nicolas/nicolas_seagate/datasets/cityscapes:/home/data/cityscapes \
        -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY -e XAUTHORITY -e NVIDIA_DRIVER_CAPABILITIES=all \
        fregu856/evaluating_bdl:rainbowsecret_pytorch04_20180905_evaluating_bdl bash
