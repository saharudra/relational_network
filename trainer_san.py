from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

import sys
import argparse
import json
import pprint
import time
from stacked_attention_model import StackedAttentionModel
from data_pipeline import ClevrDataset
from config import cfg, cfg_from_file
from progressbar import ETA, Bar, Percentage, ProgressBar

