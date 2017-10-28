#!/usr/bin/env bash

# How to configuration the GPU.
# http://philipperemy.github.io/configure_gpu_for_deep_learning/

# Be careful to add 100GB to the instance. 8GB is clearly not enough.

sudo apt-get update
sudo apt-get install -y python3-numpy python3-scipy python3-dev python3-pip python3-nose g++ libopenblas-dev git

# Install Nvidia drivers, CUDA and CUDA toolkit, following some instructions from http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
sudo apt-get update
sudo apt-get install -y cuda

# REBOOT here
sudo reboot

# nvidia-smi command should now work. but nvcc command should not work.

# Add those lines in your ~/.bash_profile then run "source ~/.bash_profile":
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# nvcc command should run.

# Browse to https://developer.nvidia.com/rdp/cudnn-download
# You will have to register.
# Once it's done, hit Download cuDNN v5.1 (Jan 20, 2017), for CUDA 8.0


# PEM is my Amazon private key. If you run on your own server, don't need to run this line!
scp -i ~/Downloads/premy.pem ~/Downloads/cudnn-8.0-linux-x64-v5.1.tgz  ubuntu@ec2-13-114-142-107.ap-northeast-1.compute.amazonaws.com:~/.

tar xvzf cudnn-8.0-linux-x64-v5.1.tgz

sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
sudo cp cuda/lib64/* /usr/local/cuda/lib64/



