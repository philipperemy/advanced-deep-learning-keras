#!/usr/bin/env bash
virtualenv -p python3 virtual-env
source virtual-env/bin/activate
git clone git@github.com:titu1994/Neural-Style-Transfer.git
cd Neural-Style-Transfer
# But not recommended to run style transfer on CPU.
# # tensorflow instead of tensorflow-gpu if CPU only.
pip3 install tensorflow-gpu
pip3 install scipy keras pillow h5py
python3 Network.py images/inputs/content/Aurea-Luna.jpg \
                   images/inputs/mask/Dawn-Sky-Mask.jpg \
                   output
