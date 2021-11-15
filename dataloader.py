import pickle
import math
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils import padding


class MultimodalDataset(Dataset):
    def __init__(self, infoset, app_feat_id_to_index, app_feat_h5, motion_feat_id_to_index, motion_feat_h5,
                 hand_feat_id_to_index, hand_feat_h5):
        self.infoset = infoset
        self.app_feat_id_to_index = app_feat_id_to_index
        self.app_feat_h5 = app_feat_h5
        self.motion_feat_id_to_index = motion_feat_id_to_index
        self.motion_feat_h5 = motion_feat_h5
        self.hand_feat_id_to_index = hand_feat_id_to_index
        self.hand_feat_h5 = hand_feat_h5

    def __getitem__(self, ix):
        vid = self.infoset[ix]['vid']
        feat_vid = str(int(vid[2:]))
        app_index = self.app_feat_id_to_index[feat_vid]
        motion_index = self.motion_feat_id_to_index[feat_vid]
        hand_index = self.hand_feat_id_to_index[feat_vid]
        with h5py.File(self.app_feat_h5, 'r') as f_app:
            app_feat = f_app['resnet_features'][app_index]
        with h5py.File(self.motion_feat_h5, 'r') as f_motion:
            motion_feat = f_motion['resnext_features'][motion_index]
        with h5py.File(self.hand_feat_h5, 'r') as f_hand:
            hand_feat = f_hand['resnext_features'][hand_index]

        cap_id = self.infoset[ix]['cap_id']
        labels = self.infoset[ix]['labels']
        taggings = self.infoset[ix]['pos_tags']
        tokens = padding(labels, add_eos=True)
        labels = padding(labels, add_eos=True)
        taggings = padding(taggings, add_eos=True)

        return {
            'video_ids': vid,
            'app_feat': torch.FloatTensor(app_feat),
            'motion_feat': torch.FloatTensor(motion_feat),
            'hand_feat': torch.FloatTensor(hand_feat),
            'caption_ids': cap_id,
            'tokens': torch.LongTensor(tokens),
            'labels': torch.LongTensor(labels),
            'taggings': torch.LongTensor(taggings),
            'length_target': torch.FloatTensor(self.infoset[ix]['length_target']),
            'category': torch.LongTensor([self.infoset[ix]['category']])
        }

    def __len__(self):
        return len(self.infoset)


class MultimodalDataLoader(DataLoader):
    def __init__(self, cfg, mode):
        print('loading corpus from %s' % cfg.dataset.corpus)
        with open(cfg.dataset.corpus, 'rb') as corpus_file:
            self.corpus = pickle.load(corpus_file)
        infoset = list()
        for ix in self.corpus['info']['split'][mode]:
            vid = 'G_%05d' % ix
            category = self.corpus['info']['itoc'][ix]
            captions = self.corpus['captions'][vid]
            pos_tags = self.corpus['pos_tags'][vid]
            assert len(captions) == len(pos_tags)
            length_target = self.corpus['info']['length_info'][vid]
            length_target = length_target[:cfg.train.token_max_len]
            if len(length_target) < cfg.train.token_max_len:
                length_target += [0] * (cfg.train.token_max_len - len(length_target))
            length_target = np.array(length_target) / sum(length_target)
            if mode == 'train':
                cap_id_set = [i for i in range(len(captions))]
            else:
                cap_id_set = [0]
            for cap_id in cap_id_set:
                infoset.append({
                    'vid': vid,
                    'labels': captions[cap_id],
                    'pos_tags': pos_tags[cap_id],
                    'category': category,
                    'length_target': length_target,
                    'cap_id': cap_id,
                })

        print('loading reference file from %s' % cfg.dataset.refs)
        with open(cfg.dataset.refs, 'rb') as refs_file:
            self.refs = pickle.load(refs_file)
        print('loading appearance feat from %s'% cfg.dataset.appearance_feat)
        with h5py.File(cfg.dataset.appearance_feat, 'r') as app_file:
            app_video_ids = app_file['ids'][()]
        app_feat_id_to_index = {str(id): i for i, id in enumerate(app_video_ids)}
        print('loading motion feature from %s' % cfg.dataset.motion_feat)
        with h5py.File(cfg.dataset.motion_feat, 'r') as motion_features_file:
            motion_video_ids = motion_features_file['ids'][()]
        motion_feat_id_to_index = {str(id): i for i, id in enumerate(motion_video_ids)}
        print('loading sign language feature from %s' % cfg.dataset.hand_feat)
        with h5py.File(cfg.dataset.hand_feat, 'r') as hand_features_file:
            hand_video_ids = hand_features_file['ids'][()]
        hand_feat_id_to_index = {str(id): i for i, id in enumerate(hand_video_ids)}
        self.app_feat_h5 = cfg.dataset.appearance_feat
        self.motion_feat_h5 = cfg.dataset.motion_feat
        self.hand_feat_h5 = cfg.dataset.hand_feat
        self.batch_size = cfg.train.batch_size
        self.dataset = MultimodalDataset(infoset, app_feat_id_to_index, self.app_feat_h5,
                                         motion_feat_id_to_index, self.motion_feat_h5, hand_feat_id_to_index, self.hand_feat_h5)
        super().__init__(self.dataset, batch_size=self.batch_size, shuffle=True if mode == 'train' else False)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)
