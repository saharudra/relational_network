"""
Code to access the CLEVR dataset. The images will be first downsampled to 
(128x128) then the images will be padded to make them of size (136x136)
then randomly cropped to (128x128) and randomly rotated between -0.05 rads
to +0.05 rads. 

Here the questions are parsed and the vocabulary dictionary is being made.
This is being stored in word_to_ix. This integer encoding of the vocab dictionary
is being stored as part of each of the image.

In the model section, the prepare_sequence function will take the batch input
of the questions and convert them into the required tensor that will be read
in by the model.

currently padding this with -1 so that the h5py framework can save it.
While reading it in, remove the -1 as that is not part of the vocab dictionary.
"""
import numpy as np
import h5py
import random, math, time
import json
import os
import string
import pdb
from os import listdir
from os.path import isfile, join
import cv2
from utils import *
from config import cfg

MAX_QUESTION_LENGTH = cfg.MAX_QUESTION_LENGTH
clever_dataset_location = '/media/exx/ssd/DATASETS/CLEVR_v1.0/'
images_location = clever_dataset_location + 'images/'
questions_location = clever_dataset_location + 'questions/'

# Method to downsample and pad the images.
def downsample_pad(img):
    # Nearest neighbor downsampling instead of bicubic downsampling
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_NEAREST)
    top = bottom = left = right = 4
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT)
    return img

# Method to crop and rotate the images.
def random_crop_rotate(img):
    left_offset = np.random.randint(0, 9)
    top_offset = np.random.randint(0, 9)
    new_img = img[left_offset:left_offset + 128, top_offset: top_offset+128, :]
    num_cols, num_rows = new_img.shape[0], new_img.shape[1]

    # Find the degree by which the image has to be rotated
    rad = random.uniform(-0.05, 0.05)
    degrees = math.degrees(rad)

    # Get the rotation matrix to rotate the image
    rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), degrees, 1)
    new_img = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
    return new_img

# Need to create the dictionary for the questions and encode them to integers here.
# Also need to perform one-hot encoding for answers here as h5py doesnot support
# unicode encoding for strings. The vocab and answer dictionaries are global dictionaries
# that will take care of all the words encountered across the data sets.
word_to_ix = {}
answer_to_ix = {}  # The size of this should be 28.

data_set = ['val', 'train', 'test']
for data_part in data_set:
    print("In data part %s" % (data_part))
    start_time = time.time()
    # Load all the images for the current data part
    mypath_images = images_location + data_part
    mypath_images_file = [f for f in listdir(mypath_images) if isfile(join(mypath_images, f))]

    # Load all the questions for the current data part
    mypath_questions_file = questions_location + 'CLEVR_' + data_part + '_questions.json'
    with open(mypath_questions_file) as qst_file:
        qst_data = json.load(qst_file)["questions"]

    images = []
    questions = []
    answers = []
    for img_file in mypath_images_file:
        img_filename = join(mypath_images, img_file)

        # Read images
        img = cv2.imread(img_filename)

        # Downsample and pad images
        img = downsample_pad(img)

        # Random crop and rotate images
        img = random_crop_rotate(img)

        for qst in qst_data:
            if qst['image_filename'] == img_file:
                curr_question = str(qst['question'])

                # Removing the punctuation from the string.
                curr_question = curr_question.translate(None, string.punctuation)

                curr_qst_vector = []
                curr_ans_vector = []  # The len of this vector will always be 1

                # Split the question and convert the words into lower case
                words_in_qst = [x.lower() for x in curr_question.split()]

                # Create the the vocabulary dictionary
                for word in words_in_qst:
                    if word not in word_to_ix:
                        word_to_ix[word] = len(word_to_ix) + 1
                        curr_qst_vector.append(word_to_ix[word])
                    else:
                        curr_qst_vector.append(word_to_ix[word])
                images.append(img)
                if len(curr_qst_vector) < MAX_QUESTION_LENGTH:
                    temp = [curr_qst_vector.append(0) for _ in range(MAX_QUESTION_LENGTH - len(curr_qst_vector))]
                else:
                    print(len(curr_qst_vector))
                    print(curr_question)
                    pdb.set_trace()
                questions.append(curr_qst_vector)
                if data_part is not 'test':
                    if qst['answer'] not in answer_to_ix:
                        answer_to_ix[qst['answer']] = len(answer_to_ix)
                        curr_ans_vector.append(answer_to_ix[qst['answer']])
                    else:
                        curr_ans_vector.append(answer_to_ix[qst['answer']])
                answers.append(curr_ans_vector)
                print(len(images), len(questions), len(answers))  # This count should reach 700,000 for training set

    # Dividing it in the directory structure as follows
    # |data|
    #      |clevr_rn
    #          |train
    #              |images.h5
    #              |questions.h5
    #              |answers.h5
    #          |val
    #              |images.h5
    #              |questions.h5
    #              |answers.h5
    #          |test
    #              |images.h5
    #              |questions.h5
    # Define the file paths and make the required structure
    data_part_path = './data/clver_rn/%s' % (data_part)
    images_filename = data_part_path + '/images.h5'
    questions_filename = data_part_path + '/questions.h5'
    if data_part is not 'test':
        answers_filename = data_part_path + '/answers.h5'
    mkdir_p(data_part_path)

    # Save data into the respective files
    with h5py.File(images_filename, 'w') as hf:
        hf.create_dataset('clevr_rn_%s_images' % (data_part), data=images)
    with h5py.File(questions_filename, 'w') as hf:
        hf.create_dataset('clevr_rn_%s_questions' % (data_part), data=questions)
    if data_part is not 'test':
        with h5py.File(answers_filename, 'w') as hf:
            hf.create_dataset('clevr_rn_%s_answers' % (data_part), data=answers)
    print(data_part)
    print(len(images), len(questions), len(answers))
    end_time = time.time()
    print("Current data part is: ", data_part)
    time_eplapsed = end_time - start_time
    print("Time elapsed: " + str(time_eplapsed))

# Save the vocabulary dictionary word_to_ix and answers dictionary answer_to_ix
save_path = './data/clver_rn/'
vocab_dict_filename = save_path + 'word_to_ix.json'
anser_dict_filename = save_path + 'answer_to_ix.json'


with open(vocab_dict_filename, 'w+') as f:
    json.dump(word_to_ix, f)

with open(anser_dict_filename, 'w+') as f:
    json.dump(answer_to_ix, f)

print("Printing the vocabulary size")  # There are 81 words in total
print(len(word_to_ix))
print("Printing the size of the final layer of the model")  # This should be 28
print(len(answer_to_ix))
