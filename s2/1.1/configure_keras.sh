#!/usr/bin/env bash

virtualenv -p python3 venv

source venv/bin/activate

pip3 install tensorflow-gpu # if you run on GPU.
# pip3 install tensorflow # if you run on CPU.

pip3 install keras
pip3 install h5py

git clone https://github.com/fchollet/keras.git

cd keras/examples

export CUDA_VISIBLE_DEVICES=0; python3 mnist_cnn.py