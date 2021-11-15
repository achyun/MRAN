import itertools
from utils import resample
import torch
import torch.nn as nn
from torch.nn.modules.module import Module
import numpy as np


class HighWay(nn.Module):
    def __init__(self, hidden_size, with_gate=True):
        super(HighWay, self).__init__()
        self.with_gate = with_gate
        self.w1 = nn.Linear(hidden_size, hidden_size)
        if self.with_gate:
            self.w2 = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        #self._init_weights()

    def forward(self, x):
        y = self.tanh(self.w1(x))
        if self.with_gate:
            gate = torch.sigmoid(self.w2(x))
            return gate * x + (1 - gate) * y
        else:
            return x + y


class Feature_Aggregation_Layer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.5):
        super(Feature_Aggregation_Layer, self).__init__()
        self.feature_aggregation_model = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            HighWay(output_dim, with_gate=True),
            nn.Dropout(dropout)
        )
        self.BN = nn.BatchNorm1d(output_dim)

    def forward(self, input_feat):
        feat_FAL = self.feature_aggregation_model(input_feat)
        dim_num = len(feat_FAL.shape)
        if dim_num == 4:
            batch_size, clip_len, frame_len, _ = feat_FAL.shape
            output_feat = self.BN(feat_FAL.contiguous().view(batch_size * clip_len * frame_len, -1)).view(batch_size, clip_len, frame_len, -1)
        else:
            batch_size, seq_len, _ = feat_FAL.shape
            output_feat = self.BN(feat_FAL.contiguous().view(batch_size * seq_len, -1)).view(batch_size, seq_len, -1)
        return output_feat


class CRN(Module):
    def __init__(self, module_dim, num_objects, max_subset_size, gating=False, spl_resolution=1):
        super(CRN, self).__init__()
        self.module_dim = module_dim
        self.gating = gating

        self.k_objects_fusion = nn.ModuleList()
        if self.gating:
            self.gate_k_objects_fusion = nn.ModuleList()
        for i in range(min(num_objects, max_subset_size + 1), 1, -1):
            self.k_objects_fusion.append(nn.Linear(2 * module_dim, module_dim))
            if self.gating:
                self.gate_k_objects_fusion.append(nn.Linear(2 * module_dim, module_dim))
        self.spl_resolution = spl_resolution
        self.activation = nn.ELU()
        self.max_subset_size = max_subset_size

    def forward(self, object_list, cond_feat):
        """
        :param object_list: list of tensors or vectors
        :param cond_feat: conditioning feature
        :return: list of output objects
        """
        scales = [i for i in range(len(object_list), 1, -1)]  # [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]

        relations_scales = []
        subsample_scales = []
        for scale in scales:
            relations_scale = self.relationset(len(object_list), scale)
            relations_scales.append(relations_scale)
            subsample_scales.append(min(self.spl_resolution, len(relations_scale)))

        crn_feats = []
        if len(scales) > 1 and self.max_subset_size == len(object_list):
            start_scale = 1
        else:
            start_scale = 0
        for scaleID in range(start_scale, min(len(scales), self.max_subset_size)):
            idx_relations_randomsample = np.random.choice(len(relations_scales[scaleID]),
                                                          subsample_scales[scaleID], replace=False)
            mono_scale_features = 0
            for id_choice, idx in enumerate(idx_relations_randomsample):
                clipFeatList = [object_list[obj].unsqueeze(1) for obj in relations_scales[scaleID][idx]]
                # g_theta
                g_feat = torch.cat(clipFeatList, dim=1)
                g_feat = g_feat.mean(1)
                if len(g_feat.size()) == 2:
                    h_feat = torch.cat((g_feat, cond_feat), dim=-1)
                elif len(g_feat.size()) == 3:
                    cond_feat_repeat = cond_feat.repeat(1, g_feat.size(1), 1)
                    h_feat = torch.cat((g_feat, cond_feat_repeat), dim=-1)
                if self.gating:
                    h_feat = self.activation(self.k_objects_fusion[scaleID](h_feat)) * torch.sigmoid(
                        self.gate_k_objects_fusion[scaleID](h_feat))
                else:
                    h_feat = self.activation(self.k_objects_fusion[scaleID](h_feat))
                mono_scale_features += h_feat
            crn_feats.append(mono_scale_features / len(idx_relations_randomsample))
        return crn_feats

    def relationset(self, num_objects, num_object_relation):
        return list(itertools.combinations([i for i in range(num_objects)], num_object_relation))


class Relation_Extraction_Module(nn.Module):
    def __init__(self, k_max_frame_level, k_max_clip_level, spl_resolution, vision_dim, module_dim=512):
        super(Relation_Extraction_Module, self).__init__()

        self.clip_level_motion_cond = CRN(module_dim, k_max_frame_level, k_max_frame_level, gating=False, spl_resolution=spl_resolution)
        self.clip_level_hand_cond = CRN(module_dim, k_max_frame_level-2, k_max_frame_level-2, gating=True, spl_resolution=spl_resolution)
        self.video_level_motion_cond = CRN(module_dim, k_max_clip_level, k_max_clip_level, gating=False, spl_resolution=spl_resolution)
        self.video_level_hand_cond = CRN(module_dim, k_max_clip_level-2, k_max_clip_level-2, gating=True,spl_resolution=spl_resolution)

        self.motion_sequence_encoder = nn.LSTM(vision_dim, module_dim, batch_first=True, bidirectional=False)
        self.hand_sequence_encoder = nn.LSTM(vision_dim, module_dim, batch_first=True, bidirectional=False)
        self.video_level_motion_proj = nn.Linear(module_dim, module_dim)
        self.video_level_hand_proj = nn.Linear(module_dim, module_dim)

        self.module_dim = module_dim

    def forward(self, appearance_video_feat, motion_video_feat, hand_video_feat):
        """
        Args:
            appearance_video_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            motion_video_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            hand_video_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
        return:
            encoded video feature: [Tensor] (batch_size, N, module_dim)
        """
        batch_size = appearance_video_feat.size(0)
        clip_level_crn_outputs = []
        for i in range(appearance_video_feat.size(1)):
            clip_level_motion_proj = motion_video_feat[:, i, :]  # (bz, 512)
            clip_level_hand_proj = hand_video_feat[:, i, :]

            clip_level_appearance_proj = appearance_video_feat[:, i, :, :]  # (bz, 16, 512)
            # clip level CRNs
            clip_level_crn_motion = self.clip_level_motion_cond(torch.unbind(clip_level_appearance_proj, dim=1),
                                                                clip_level_motion_proj)
            clip_level_crn_hand = self.clip_level_hand_cond(clip_level_crn_motion, clip_level_hand_proj)

            clip_level_crn_output = torch.cat(      # (32, 12, 512)
                [frame_relation.unsqueeze(1) for frame_relation in clip_level_crn_hand],
                dim=1)
            clip_level_crn_output = clip_level_crn_output.view(batch_size, -1, self.module_dim)
            clip_level_crn_outputs.append(clip_level_crn_output)

        # Encode video level motion
        _, (video_level_motion, _) = self.motion_sequence_encoder(motion_video_feat)
        video_level_motion = video_level_motion.transpose(0, 1)
        video_level_motion_feat_proj = self.video_level_motion_proj(video_level_motion)
        # Encode video level hand
        _, (video_level_hand, _) = self.hand_sequence_encoder(hand_video_feat)
        video_level_hand = video_level_hand.transpose(0, 1)
        video_level_hand_feat_proj = self.video_level_hand_proj(video_level_hand)
        # video level CRNs
        video_level_crn_motion = self.video_level_motion_cond(clip_level_crn_outputs, video_level_motion_feat_proj)
        video_level_crn_hand = self.video_level_hand_cond(video_level_crn_motion, video_level_hand_feat_proj)

        video_level_crn_output = torch.cat([clip_relation.unsqueeze(1) for clip_relation in video_level_crn_hand],
                                           dim=1)
        video_level_crn_output = video_level_crn_output.view(batch_size, -1, self.module_dim)
        return video_level_crn_output # (32, 48, 512)


class Joint_Representation(nn.Module):
    def __init__(self, cfg):
        super(Joint_Representation, self).__init__()
        self.app_FAL = Feature_Aggregation_Layer(cfg.train.vision_dim, cfg.train.model_dim)
        self.motion_FAL = Feature_Aggregation_Layer(cfg.train.vision_dim, cfg.train.model_dim)
        self.hand_FAL = Feature_Aggregation_Layer(cfg.train.hand_dim, cfg.train.model_dim)
        # self.crn_app_FAL = Feature_Aggregation_Layer(cfg.train.hand_dim, cfg.train.model_dim)
        self.realiton_extraction_module = Relation_Extraction_Module(k_max_frame_level=cfg.train.k_max_frame_level,
                                                                     k_max_clip_level=cfg.train.n_frames,
                                                                     spl_resolution=1,
                                                                     vision_dim=cfg.train.model_dim,
                                                                     module_dim=cfg.train.model_dim)

        self.n_frames = cfg.train.n_frames
        self.seed = cfg.seed


    def forward(self, app_feat, motion_feat, hand_feat):
        app_rep = self.app_FAL(app_feat)
        motion_rep = self.motion_FAL(motion_feat)
        hand_rep = self.hand_FAL(hand_feat)
        # crn_app_FAL = self.crn_app_FAL(crn_app_feat)

        # resample the app_feat
        sample_dict = resample(clip_num=app_rep.size(1),
                               frame_num=app_rep.size(2),
                               sample_num=self.n_frames,
                               seed=self.seed)
        feat_sample_list = list()
        for clip, sample_frames in sample_dict.items():
            feat_sample_list.append(app_rep[:, clip, sample_frames, :])
        crn_app_rep = app_rep
        app_rep = torch.cat(feat_sample_list, dim=1)

        relations = self.realiton_extraction_module(crn_app_rep, motion_rep, hand_rep)
        joint_representation = torch.cat([motion_rep, app_rep, relations], dim=1)
        return joint_representation

