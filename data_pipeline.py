from __future__ import division
from __future__ import print_function

import numpy as np
import h5py
import os
from config import cfg

class Dataset(object):
    def __init__(self, images, questions, imsize, datadir, data_part=None, answers=None,
                 class_id=None, class_range=None):
        super(Dataset, self).__init__()
        self._images = images
        self._questions = questions
        self.data_part = data_part
        if self.data_part is not 'test':
            self._answers = answers
        self.datadir = datadir
        self._num_examples = len(images)
        self._epochs_completed = -1

        # shuffle on first run
        self._index_in_epoch = self._num_examples
        self._class_id = np.array(class_id)
        self._class_range = class_range
        self._imsize = imsize
        self._perm = None

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # Finished epochs
            self._epochs_completed += 1
            # shuffle the data
            self._perm = np.arange(self._num_examples)
            np.random.shuffle(self._perm)

            # start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        current_ids = self._perm[start: end]

        sampled_images = self._images[current_ids]
        sampled_questions = self._questions[current_ids]
        if self.data_part is not 'test':
            sampled_answers = self._answers[current_ids]

        # TODO: Try once without converting it in the (-1, 1) range
        sampled_images = sampled_images * (2.0 / 255.0) - 1.0

        # Make sure to not return answers for test data part as there are none.
        if self.data_part is not 'test':
            ret_list = [sampled_images, sampled_questions, sampled_answers]
        else:
            ret_list = [sampled_images, sampled_questions]

        return ret_list


class ClevrDataset(object):
    def __init__(self, train_data_dir=None, val_data_dir=None):  # No need for test data as of now
        super(ClevrDataset, self).__init__()
        self.image_shape = cfg.IMAGE_SHAPE
        if train_data_dir is not None:
            self.datadir = train_data_dir
            self.train_images_file_path = os.path.join(self.datadir + '/images.h5')
            self.train_questions_file_path = os.path.join(self.datadir + '/questions.h5')
            self.train_answers_file_path = os.path.join(self.datadir + '/answers.h5')
        if val_data_dir is not None:
            self.datadir = val_data_dir
            self.val_images_file_path = os.path.join(self.datadir + '/images.h5')
            self.val_questions_file_path = os.path.join(self.datadir + '/questions.h5')
            self.val_answers_file_path = os.path.join(self.datadir + '/answers.h5')
        # if test_data_dir is not None:
        #     self.datadir = test_data_dir
        #     self.test_images_file_path = os.path.join(self.datadir + '/images.h5')
        #     self.test_questions_file_path = os.path.join(self.datadir + '/questions.h5')

    # Return an object of the class Dataset
    def get_train_data(self):
        print("Getting training data")
        with h5py.File(self.train_images_file_path, 'r') as hf:
            images = hf['clevr_rn_train_images'][:]
            images = np.array(images, copy=False)
            print("CLEVR train images: ", images.shape)
        with h5py.File(self.train_questions_file_path, 'r') as hf:
            questions = hf['clevr_rn_train_questions'][:]
            questions = np.array(questions, copy=False)
            print("CLEVR train questions: ", questions.shape)
        with h5py.File(self.train_answers_file_path, 'r') as hf:
            answers = hf['clevr_rn_train_answers'][:]
            answers = np.array(answers, copy=False)
            print("CLEVR train answers: ", answers.shape)

        return Dataset(images=images, questions=questions, answers=answers, datadir=self.datadir, imsize=self.image_shape[0])

    # Return an object of the class Dataset
    def get_val_data(self):
        print("Getting validation data")
        with h5py.File(self.val_images_file_path, 'r') as hf:
            images = hf['clevr_rn_val_images'][:]
            images = np.array(images, copy=False)
            print("CLEVR val images: ", images.shape)
        with h5py.File(self.val_questions_file_path, 'r') as hf:
            questions = hf['clevr_rn_val_questions'][:]
            questions = np.array(questions, copy=False)
            print("CLEVR val questions: ", questions.shape)
        with h5py.File(self.val_answers_file_path, 'r') as hf:
            answers = hf['clevr_rn_val_answers'][:]
            answers = np.array(answers, copy=False)
            print("CLEVR val answers: ", answers.shape)

        return Dataset(images=images, questions=questions, answers=answers, datadir=self.datadir, imsize=self.image_shape[0])
