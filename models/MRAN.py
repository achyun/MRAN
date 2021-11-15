import torch.nn as nn
import torch

from .Encoder import Joint_Representation
from .Decoder import Decoder


class MRAN(nn.Module):
    def __init__(self, cfg):
        super(MRAN, self).__init__()
        self.encoder = Joint_Representation(cfg)
        self.decoder = Decoder(cfg)
        self.tag_proj = nn.Linear(cfg.train.model_dim, cfg.dataset.vocab_size, bias=False)

    def forward(self, app_feat, motion_feat, hand_feat, tag_tokens, category):
        tag_tokens = tag_tokens[:, :-1]
        joint_rep = self.encoder(app_feat, motion_feat, hand_feat)
        hidden_states, embs, *_ = self.decoder(joint_rep, tag_tokens, category)
        tag_logits = [self.tag_proj(item) for item in hidden_states]
        tgt_word_logprobs = [torch.log_softmax(item, dim=-1) for item in tag_logits]
        return {'tgt_word_logprobs': tgt_word_logprobs}
