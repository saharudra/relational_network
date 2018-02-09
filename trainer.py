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
from relational_network import RelationalNetwork
from data_pipeline import ClevrDataset
from config import cfg, cfg_from_file
from progressbar import ETA, Bar, Percentage, ProgressBar


# Define the arguments to be parsed for training
parser = argparse.ArgumentParser(description="CLEVR relational network example")
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='input size of a mini-batch for training (default: 64)')
parser.add_argument('--epochs', type=int, default=800, metavar='N',
                    help='number of epochs for training of the network (default: 100)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enable CUDA training')
parser.add_argument('--seed', type=int, default=10, metavar='S',
                    help='random seed (default: 10)')
parser.add_argument('--log_interval', type=int, default=1000, metavar='N', 
                    help='batches to wait for before logging the training status (default: 1000)')
parser.add_argument('--cfg', dest='cfg_file', 
                    help='optional configuration file to set the cfg in config')
parser.add_argument('--gpu', dest='gpu_id', default=0,
                    help='GPU device to perform training on, works with --no_cuda set to False')

# Parse the arguments and see if cuda is available
# Make sure to create a level of cuda check when creating a Variable
# or putting the model on gpu.
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
cfg.TRAIN.USE_CUDA = args.cuda

if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
if args.gpu_id != -1:
    cfg.GPU_ID = args.gpu_id

# Initialize the state using the defined seed
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Define the obj to access the dataset
data_dir = cfg.DATA_DIR
dataset = ClevrDataset(train_data_dir=data_dir + '/train/', val_data_dir=data_dir + '/val')
train_dataset = dataset.get_train_data()
val_dataset = dataset.get_val_data()

# Define variables to access misc dataset
word_to_ix_file = data_dir + '/word_to_ix.json'
answer_to_ix_file = data_dir + '/answer_to_ix.json'

with open(word_to_ix_file, 'r') as vocab_dict, open(answer_to_ix_file, 'r') as answer_dict:
    word_to_ix = json.load(vocab_dict)
    answer_to_ix = json.load(answer_dict)

# Define the parameters that have to be reset for the cfg file
cfg.TRAIN.VOCAB_SIZE = len(word_to_ix) + 1
cfg.TRAIN.ANSWER_SIZE = len(answer_to_ix)
cfg.TRAIN.BATCH_SIZE = args.batch_size

print("Current configuration being used")
pprint.pprint(cfg)


# Define the model and port it to gpu
model = RelationalNetwork()

if torch.cuda.device_count() > 1:
    print("Using ", torch.cuda.device_count(), " GPUs")
    model = nn.DataParallel(model)

if args.cuda:
    model = model.cuda()

# Define the optimizer for training the network
optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE)
# criterion = nn.CrossEntropyLoss()

# Define the train step
def train(updates_per_epoch):
    train_loss = 0.0
    train_accuracy = 0.0
    for iteration in range(updates_per_epoch):
        pbar.update(iteration)
        images, questions, answers = train_dataset.next_batch(args.batch_size)

        # Convert the input images into tensor Variables
        images = Variable(torch.from_numpy(images).permute(0, 3, 1, 2).float())

        # Process the questions and answers separately
        questions = Variable(torch.LongTensor(questions))
        answers = Variable(torch.LongTensor(answers)).view(args.batch_size)

        if args.cuda:
            images = images.cuda()
            questions = questions.cuda()
            answers = answers.cuda()

        model.zero_grad()
        answers_hat = model(images, questions)
        loss = F.nll_loss(answers_hat, answers)
        # print("The loss is getting calculated here")
        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]
        pred = answers_hat.data.max(1)[1]
        correct = pred.eq(answers.data).cpu().sum()
        accuracy = correct * 100. / len(answers)
        train_accuracy += accuracy
    train_accuracy = train_accuracy / updates_per_epoch
    train_loss = train_loss / updates_per_epoch
    return train_loss, train_accuracy

# Define the validation step, currently doing it for 10 random batches
def val():
    model.eval()
    val_accuracy = 0.0
    for val_iteration in range(updates_per_epoch_val):
        val_images, val_questions, val_answers = val_dataset.next_batch(args.batch_size)

        # Convert the validation images into tensor Variables
        val_images = Variable(torch.from_numpy(val_images).permute(0, 3, 1, 2).float())

        # Process the questions and the answers
        val_questions = Variable(torch.LongTensor(val_questions))
        val_answers = Variable(torch.LongTensor(val_answers)).view(args.batch_size)

        if args.cuda:
            val_images = val_images.cuda()
            val_questions = val_questions.cuda()
            val_answers = val_answers.cuda()

        val_answers_hat = model(val_images, val_questions)
        val_pred = val_answers_hat.data.max(1)[1]
        val_correct = val_pred.eq(val_answers.data).cpu().sum()
        accuracy = val_correct * 100.0 / len(val_answers)
        val_accuracy += accuracy
    val_accuracy = val_accuracy / 10.0
    return val_accuracy


# Define the parameters to be used by the progress bar
number_examples = train_dataset._num_examples  # See this value and if this works.
updates_per_epoch = number_examples // cfg.TRAIN.BATCH_SIZE

number_examples_val = val_dataset._num_examples
updates_per_epoch_val = number_examples_val // cfg.TRAIN.BATCH_SIZE

# Create one-hot answers dictionary to be used in prepare answers
with open('./data/clver_rn/answer_to_ix.json', 'r') as answer_file:
    answer_to_ix = json.load(answer_file)

answer_to_one_hot = {}
one_hot_init_vector = [0] * len(answer_to_ix)

# Set up the training loop
for epoch in range(1, args.epochs + 1):
    epoch_start_time = time.time()
    widgets = ['epoch #%d|' % epoch, Percentage(), Bar(), ETA()]
    pbar = ProgressBar(maxval=updates_per_epoch, widgets=widgets)
    pbar.start()
    
    # Call the train and the test step for the dataset
    epoch_loss, epoch_accuracy = train(updates_per_epoch)
    log_line_train = '%s: %s; %s: %s; ' % ("Training Loss", epoch_loss, "Training Accuracy", epoch_accuracy)
    val_accuracy = val(updates_per_epoch_val)
    log_line_val = '%s: %s ' % ("Validation Accuracy", val_accuracy)
    epoch_end_time = time.time()
    time_taken = epoch_end_time - epoch_start_time
    log_time_line = '%s: %s' % ("Time taken for the current epoch", time_taken)
    print("Epoch %d | " % (epoch) + log_line_train + log_line_val + log_time_line)
    sys.stdout.flush()
