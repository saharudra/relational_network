import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from config import cfg

class StackedAttentionModel(nn.Module):

    def __init__(self):
        super(StackedAttentionModel, self).__init__()

        # Define parameters for stacked attention network as used by Satoro et.al
        self.conv_layer_channels = [24, 24, 24, 24]  # Can be substituted by some other file
        self.in_dim = cfg.TRAIN.IMG_DIM  # Working only on CLEVR

        self.question_vector_size = cfg.TRAIN.QUESTION_VECTOR_SIZE
        self.embedding_dim = cfg.TRAIN.EMBEDDING_DIM
        self.vocab_size = cfg.TRAIN.VOCAB_SIZE
        self.answer_size = cfg.TRAIN.ANSWER_SIZE
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.use_cuda = cfg.TRAIN.USE_CUDA  # Can be set using args and thus can be substituted
        self.rnn_type = cfg.TRAIN.RNN_TYPE
        self.n_layers = 1
        self.num_gpus = int(torch.cuda.device_count())
        self.dim_hidden = 512
        self.img_seq_size = 49  # Doing only for CLEVR

        # Define the word embedding for the input questions
        self.question_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)

        # Define the lstm to process the questions
        self.lstm = nn.LSTM(self.embedding_dim, self.question_vector_size, num_layers=self.n_layers)

        # Define layers of the stacked attetion network

    def convolutional_layer(self):
        self.conv1 = nn.Conv2d(self.in_dim, self.conv_layer_channels[0], 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(self.conv_layer_channels[0])
        self.conv2 = nn.Conv2d(self.conv_layer_channels[0], self.conv_layer_channels[1], 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(self.conv_layer_channels[1])
        self.conv3 = nn.Conv2d(self.conv_layer_channels[1], self.conv_layer_channels[2], 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(self.conv_layer_channels[2])
        self.conv4 = nn.Conv2d(self.conv_layer_channels[2], self.conv_layer_channels[3], 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(self.conv_layer_channels[3])

    def apply_convolution(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        return x

    def san(self, conv_feature_map, question_vector):
        # Get image feature vector from feature map
        feature_map_size = conv_feature_map.size()

        # img_feature_vector ---> bs x depth x img_regions; [64 x 24 x 49]
        img_feature_vector = conv_feature_map.view(-1, feature_map_size[1], feature_map_size[2] * feature_map_size[3])

        pass

    def forward(self, image, question_vector):
        question_vector = self.question_embeddings(question_vector)
        question_vector = question_vector.permute(1, 0, 2)

        # Pass the question vector through the lstm
        out_question_vector, out_hidden = self.lstm(question_vector)
        self.lstm.flatten_parameters()
        out_question_vector = out_question_vector[-1]

        # get the convolution feature map
        conv_feature_map = self.apply_convolution(image)

        ans = self.san(conv_feature_map, out_question_vector)

        return ans
