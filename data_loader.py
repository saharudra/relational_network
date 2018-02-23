from __future__ import print_function, division

import os
import torch
import numpy as np
import pandas as pd
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json

# Ignore Warnings for now
import warnings
warnings.filterwarnings('ignore')

# Create the dataset class for the clevr dataset that will inherit the Dataset class of torch
class ClevrDataset(Dataset):
    """ CLEVR Dataset """
    # Sample of our dataset will be a dict of the following type
    # {'image': image, 'questions': question, 'answers': answer}
    # As there are multiple questions and answers associated with each of the image,
    # one sample of the dataset will be one combination of the (image, question, answer)
    # tuple.

    def __init__(self, images_root_dir='/media/exx/ssd/DATASETS/CLEVR_v1.0/images',
                 questions_root_dir='/media/exx/ssd/DATASETS/CLEVR_v1.0/questions',
                 data_part='train', transform=None):
        self.images_root_dir = images_root_dir
        self.questions_root_dir = questions_root_dir
        self.data_part = data_part
        self.transform = transform

        # load the question's json file for the corresponding data part in memory
        self.questions_filename = self.questions_root_dir + '/CLEVR_' + data_part + '_questions.json'
        with open(self.questions_filename, 'r') as f:
            self.questions_answers = json.load(f)

    def __len__(self):
        # Returns the length of the dataset for the given data part
        return len(self.questions_answers['questions'])



