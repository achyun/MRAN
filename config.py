from __future__ import division
from __future__ import print_function

import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.seed = 0
__C.gpu_id = 0
__C.multi_gpus = False

# Training options
__C.train = edict()
__C.train.batch_size = 64
__C.train.max_epochs = 50
__C.train.model_dim = 512
__C.train.vision_dim = 2048
__C.train.hand_dim = 2048
__C.train.k_max_frame_level = 16
__C.train.n_frames = 8
__C.train.constants = {'PAD': 0, 'UNK': 1, 'BOS': 2, 'EOS': 3, 'MASK': 4, 'VIS': 5}
__C.train.token_max_len = 30
__C.train.category_num = 20
__C.train.LN_eps = 0.00001
__C.train.dropout = 0.5
__C.train.attention_heads = 8
__C.train.attention_dropout = 0.0
__C.train.activation = 'gelu_new' # ['gelu', 'gelu_new', 'swish', 'relu']
__C.train.pretrained_path = None
# learning rate options
__C.train.weight_decay = 0.0005
__C.train.restore_lr = True
__C.train.learning_rate = 5e-4
__C.train.minimum_learning_rate = 5e-5
__C.train.decay = 0.95
__C.train.n_warmup_steps = 0

# test
__C.test = edict()
__C.test.beam_size = 5

# Dataset options
__C.dataset = edict()
__C.dataset.name = 'CFO'
__C.dataset.data_dir = 'data'
__C.dataset.appearance_feat = '{}/{}/{}_appearance_feat.hdf5'
__C.dataset.motion_feat = '{}/{}/{}_motion_feat.hdf5'
__C.dataset.hand_feat = '{}/{}/{}_hand_feat.hdf5'
# __C.dataset.audio_feat = '{}/{}_audio_feat.hdf5'
__C.dataset.corpus = '{}/{}/{}-corpus.pkl'
__C.dataset.refs = '{}/{}/{}-refs.pkl'
__C.dataset.save_dir = 'results'  # dir to save the results


# credit https://github.com/tohinz/pytorch-mac-network/blob/master/code/config.py
def merge_cfg(yaml_cfg, cfg):
    if type(yaml_cfg) is not edict:
        return

    for k, v in yaml_cfg.items():
        if not k in cfg:
            raise KeyError('{} is not a valid config key'.format(k))

        old_type = type(cfg[k])
        if old_type is not type(v):
            if isinstance(cfg[k], np.ndarray):
                v = np.array(v, dtype=cfg[k].dtype)
            elif isinstance(cfg[k], list):
                v = v.split(",")
                v = [int(_v) for _v in v]
            elif cfg[k] is None:
                if v == "None":
                    continue
                else:
                    v = v
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(cfg[k]),
                                                               type(v), k))
        # recursively merge dicts
        if type(v) is edict:
            try:
                merge_cfg(yaml_cfg[k], cfg[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            cfg[k] = v


def cfg_from_file(file_name):
    import yaml
    with open(file_name, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    merge_cfg(yaml_cfg, __C)

# import argparse
#
# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--seed', default=0)
#
#
#
#     args = parser.parse_args()
#     if args.dataset == 'CFO':
#         args.beta = [0.35, 0.9]
#         args.max_len = 30
#         args.with_category = True
#     return args
