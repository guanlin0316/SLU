# An implicit guidance framework for joint Intent detection and Slot Filling

This repository contains the official `PyTorch` implementation of the paper: 

**An implicit guidance framework for joint Intent detection and Slot Filling**

In the following, we will guide you how to use this repository step by step.

Our code is based on PyTorch 1.2 Required python packages:

-   numpy==1.18.1
-   tqdm==4.32.1
-   pytorch==1.2.0
-   python==3.7.3
-   cudatoolkit==9.2

We highly suggest you using [Anaconda](https://www.anaconda.com/) to manage your python environment.

## How to Run it

The script **train.py** acts as a main function to the project, you can run the experiments by the following commands.

# MixSNIPS dataset
python run.py -g -bs=64 -ne=75 -dd=./data/MixSNIPS -lod=./log/MixSNIPS -sd=./save/MixSNIPS -nh=8 -wed=32 -ied=64 -sdhd=64 -ln=MixSNIPS.txt

# SNIPS dataset
python run.py -g -bs=16 -ne=200 -dd=./data/SNIPS -lod=./log/SNIPS -sd=./save/SNIPS -nh=8 -wed=64 -ied=64 -sdhd=64 -ln=SNIPS.txt 

Due to some stochastic factors(*e.g*., GPU and environment), it maybe need to slightly tune the hyper-parameters using grid search to reproduce the results reported in our paper. All the hyper-parameters are in the `utils/config.py` and here are the suggested hyper-parameter settings:

-   Number of attention heads [4, 8]
-   Word Embedding Dim [32, 64]
-   Slot Embedding Dim [32, 64, 128]
-   Decoder Gat Hidden Dim [16, 32, 64]
-   Batch size [16, 32, 64]


> P.S. We just slightly tune the hyper-parameters.

If you have any question, please issue the project or email [me](1551916602@qq.com) and we will reply you soon.
