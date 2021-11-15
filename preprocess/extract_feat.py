import argparse, os
import h5py
from scipy.misc import imresize
import skvideo.io
from PIL import Image
import json

import torch
from torch import nn
import torchvision
import random
import numpy as np

from models import resnext
from tqdm import tqdm


def load_file_paths(args):
    ''' Load a list of (path,image_id tuples).'''
    file_paths = list()
    with open(args.annotation_file, 'r', encoding='utf-8') as anno_file:
        instances = json.load(anno_file)
    video_ids = [instance['video_id'] for instance in instances['videos']]
    video_ids = set(video_ids)
    for vid in video_ids:
        if args.feature_type in ['appearance', 'motion']:
            file_paths.append((args.dataset_dir + 'video/{}.avi'.format(vid), vid))
        elif args.feature_type == 'hand':
            file_paths.append((args.dataset_dir + 'hand/{}hand.avi'.format(vid), vid))
        else:
            file_paths.append((args.dataset_dir + 'audio/{}.wav'.format(vid), vid))

    return file_paths

def build_resnet():
    if not hasattr(torchvision.models, args.model):
        raise ValueError('Invalid model "%s"' % args.model)
    if not 'resnet' in args.model:
        raise ValueError('Feature extraction only supports ResNets')
    cnn = getattr(torchvision.models, args.model)(pretrained=True)
    model = torch.nn.Sequential(*list(cnn.children())[:-1])
    model.cuda()
    model.eval()
    return model


def build_resnext():
    model = resnext.resnet101(num_classes=400, shortcut_type='B', cardinality=32,
                              sample_size=112, sample_duration=16,
                              last_fc=False)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)
    assert os.path.exists('preprocess/pretrained/resnext-101-kinetics.pth')
    model_data = torch.load('preprocess/pretrained/resnext-101-kinetics.pth', map_location='cpu')
    model.load_state_dict(model_data['state_dict'])
    model.eval()
    return model


def run_batch(cur_batch, model):
    """
    Args:
        cur_batch: treat a video as a batch of images
        model: ResNet model for feature extraction
    Returns:
        ResNet extracted feature.
    """
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)

    image_batch = np.concatenate(cur_batch, 0).astype(np.float32)
    image_batch = (image_batch / 255.0 - mean) / std
    image_batch = torch.FloatTensor(image_batch).cuda()
    with torch.no_grad():
        image_batch = torch.autograd.Variable(image_batch)

    feats = model(image_batch)
    feats = feats.data.cpu().clone().numpy()

    return feats


def extract_clips_with_consecutive_frames(path, num_clips, num_frames_per_clip):
    """
    Args:
        path: path of a video
        num_clips: expected numbers of splitted clips
        num_frames_per_clip: number of frames in a single clip, pretrained model only supports 16 frames
    Returns:
        A list of raw features of clips.
    """
    valid = True
    clips = list()
    try:
        video_data = skvideo.io.vread(path)
    except:
        print('file {} error'.format(path))
        valid = False
        if args.model == 'resnext101':
            return list(np.zeros(shape=(num_clips, 3, num_frames_per_clip, 112, 112))), valid
        else:
            return list(np.zeros(shape=(num_clips, num_frames_per_clip, 3, 224, 224))), valid
    total_frames = video_data.shape[0]
    img_size = (args.image_height, args.image_width)
    for i in np.linspace(0, total_frames, num_clips + 2, dtype=np.int32)[1:num_clips + 1]:
        clip_start = int(i) - int(num_frames_per_clip / 2)
        clip_end = int(i) + int(num_frames_per_clip / 2)
        if clip_start < 0:
            clip_start = 0
        if clip_end > total_frames:
            clip_end = total_frames - 1
        clip = video_data[clip_start:clip_end]
        if clip_start == 0:
            shortage = num_frames_per_clip - (clip_end - clip_start)
            added_frames = []
            for _ in range(shortage):
                added_frames.append(np.expand_dims(video_data[clip_start], axis=0))
            if len(added_frames) > 0:
                added_frames = np.concatenate(added_frames, axis=0)
                clip = np.concatenate((added_frames, clip), axis=0)
        if clip_end == (total_frames - 1):
            shortage = num_frames_per_clip - (clip_end - clip_start)
            added_frames = []
            for _ in range(shortage):
                added_frames.append(np.expand_dims(video_data[clip_end], axis=0))
            if len(added_frames) > 0:
                added_frames = np.concatenate(added_frames, axis=0)
                clip = np.concatenate((clip, added_frames), axis=0)
        new_clip = []
        for j in range(num_frames_per_clip):
            frame_data = clip[j]
            img = Image.fromarray(frame_data)
            img = imresize(img, img_size, interp='bicubic')
            img = img.transpose(2, 0, 1)[None]
            frame_data = np.array(img)
            new_clip.append(frame_data)
        new_clip = np.asarray(new_clip)  # (num_frames, channels, width, height)
        if args.model in ['resnext101']:
            new_clip = np.squeeze(new_clip)
            new_clip = np.transpose(new_clip, axes=(1, 0, 2, 3))
        clips.append(new_clip)
    return clips, valid


def extract_feature(model, files_path, num_clips, outfile):
    """
    Args:
        model: loaded pretrained model for feature extraction
        files_path: list of files path
        num_clips: expected numbers of splitted clips
        outfile: path of output file to be written
    Returns:
        h5 file containing visual features of splitted clips.
    """
    if not os.path.exists('data/{}'.format(args.dataset)):
        os.makedirs('data/{}'.format(args.dataset))

    dataset_size = len(files_path)

    with h5py.File(outfile, 'w') as fd:
        feat_dset = None
        file_ids_dset = None
        i0 = 0
        for i, (file_path, file_id) in enumerate(tqdm(files_path)):
            if args.feature_type == 'audio':
                try:
                    feat = model.forward(file_path)
                except:
                    feat = torch.zeros(8, 128)
                if feat_dset is None:
                    (C, D) = (8, 128)
                    feat_dset = fd.create_dataset('VGG_features', (dataset_size, C, D),
                                                  dtype=np.float32)
                    file_ids_dset = fd.create_dataset('ids', shape=(dataset_size,), dtype=int)

                i1 = i0 + 1
                if feat.size(0) != C:
                    if len(feat.shape) == 1:
                        feat = torch.unsqueeze(feat, dim=0)
                    sample_list = np.linspace(0, feat.size(0) - 1, C, dtype=int)
                    feat_dset[i0:i1] = feat.cpu().detach().numpy()[sample_list]
                else:
                    feat_dset[i0:i1] = feat.cpu().detach().numpy()
                file_ids_dset[i0:i1] = int(file_id[2:])
                i0 = i1

            else:
                clips, valid = extract_clips_with_consecutive_frames(file_path, num_clips=num_clips,
                                                                     num_frames_per_clip=16)
                if args.feature_type == 'appearance':
                    clip_feat = list()
                    if valid:
                        for clip_id, clip in enumerate(clips):
                            feats = run_batch(clip, model)  # (16, 2048)
                            feats = feats.squeeze()
                            clip_feat.append(feats)
                    else:
                        clip_feat = np.zeros(shape=(num_clips, 16, 2048))
                    clip_feat = np.asarray(clip_feat)  # (8, 16, 2048)
                    if feat_dset is None:
                        C, F, D = clip_feat.shape
                        feat_dset = fd.create_dataset('resnet_features', (dataset_size, C, F, D),
                                                      dtype=np.float32)
                        file_ids_dset = fd.create_dataset('ids', shape=(dataset_size,), dtype=np.int)
                elif args.feature_type in ['motion', 'hand']:
                    clip_torch = torch.FloatTensor(np.asarray(clips)).cuda()
                    if valid:
                        clip_feat = model(clip_torch)  # (8, 2048)
                        clip_feat = clip_feat.squeeze()
                        clip_feat = clip_feat.detach().cpu().numpy()
                    else:
                        clip_feat = np.zeros(shape=(num_clips, 2048))
                    if feat_dset is None:
                        C, D = clip_feat.shape
                        feat_dset = fd.create_dataset('resnext_features', (dataset_size, C, D),
                                                      dtype=np.float32)
                        file_ids_dset = fd.create_dataset('ids', shape=(dataset_size,), dtype=np.int)

                i1 = i0 + 1
                feat_dset[i0:i1] = clip_feat
                file_ids_dset[i0:i1] = int(file_id[2:])
                i0 = i1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='specify which gpu will be used')
    # dataset info
    parser.add_argument('--dataset', default='CFO', choices=['CFO'], type=str)
    # output
    parser.add_argument('--out', dest='outfile',
                        help='output filepath',
                        default="data/{}/{}_{}_feat.hdf5", type=str)
    # image sizes
    parser.add_argument('--num_clips', default=8, type=int)
    parser.add_argument('--image_height', default=224, type=int)
    parser.add_argument('--image_width', default=224, type=int)

    # network params
    parser.add_argument('--feature_type', default='appearance', choices=['appearance', 'motion', 'hand', 'audio'], type=str)
    parser.add_argument('--seed', default='666', type=int, help='random seed')
    args = parser.parse_args()
    if args.feature_type == 'appearance':
        args.model = 'resnet101'
    elif args.feature_type in ['motion', 'hand']:
        args.model = 'resnext101'
    elif args.feature_type == 'audio':
        args.model = 'VGGish'
    else:
        raise Exception('Feature type not supported!')
    # set gpu
    if args.model != 'resnext101':
        torch.cuda.set_device(args.gpu_id)  # use GPU
    torch.manual_seed(args.seed)    # set the random seed
    np.random.seed(args.seed)

    # annotation files
    if args.dataset == 'CFO':
        # args.annotation_file = './data/dataset/CFO/info.json'
        # args.dataset_dir = './data/dataset/CFO/'
        args.annotation_file = 'E:/数据集/G_17251-G_20670/info.json'
        args.dataset_dir = 'E:/数据集/G_17251-G_20670/'
        video_paths = load_file_paths(args)
        random.shuffle(video_paths)
        # load model
        if args.feature_type == 'appearance':
            model = build_resnet()
        elif args.feature_type in ['motion', 'hand']:
            model = build_resnext()
        elif args.feature_type == 'audio':
            model = torch.hub.load('harritaylor/torchvggish', 'vggish')
        extract_feature(model, video_paths, args.num_clips,
                    args.outfile.format(args.dataset, args.dataset, args.feature_type))

