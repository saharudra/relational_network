from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.DATASET_NAME = 'CLEVR'

# E1 is the baseline experiment of RN for CLEVR dataset
__C.EXPERIMENT_NAME = 'E1'

# Training options default values
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 64
__C.TRAIN.EMBEDDING_DIM = 32  # Embedding dim used in the RN paper
__C.TRAIN.VOCAB_SIZE = 50  # This will be reset before the run based on the word_to_ix dictionary
__C.TRAIN.ANSWER_SIZE = 28  # The size of the final layer of f_phi
__C.TRAIN.HIDDEN_DIM = 128  # Output dim of the hidden state of lstm mentioned in the paper
__C.TRAIN.QUESTION_VECTOR_SIZE = 128
__C.TRAIN.IMG_DIM = 3
__C.TRAIN.USE_CUDA = True  # Whether to use the gpu or not, this will be set from the parse of the arguments.
__C.TRAIN.LEARNING_RATE = 2.5e-4
__C.TRAIN.RNN_TYPE = 'lstm'

# Common parameters
__C.MAX_QUESTION_LENGTH = 50
__C.GPU_ID = 0
__C.DATA_DIR = './data/clver_rn'
__C.IMAGE_SHAPE = [128, 128, 3]

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_edict(yaml_cfg):
    _merge_a_into_b(yaml_cfg, __C)
