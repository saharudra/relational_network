# Relational Network
Pytorch implementation for [A Simple Neural Network Module for Relational Reasoning](https://arxiv.org/pdf/1706.01427.pdf).

Work under progress.

Part of the network module code adopted from:
https://github.com/kimhc6028/relational-networks

## How to run the code
Step 1: Download the CLEVR_v1.0 dataset 

Step 2: Point to that location in the `datasets.py` file

Step 3: Run `python datasets.py` file which will create a directory structure as mentioned in the file itsef. This takes a while to complete.

Step 4: Run `python trainer.py --batch_size <batch size> --epochs <epochs>`
        Default batch size and number of epochs are 256 and 800 respectively.


## TODOs:
1) Create a `DataLoader` for CLEVR dataset.

