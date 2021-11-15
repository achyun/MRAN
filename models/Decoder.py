import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import get_att_pad_mask


class TokenEmbeddings(nn.Module):
    def __init__(self, cfg):
        super(TokenEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(cfg.dataset.vocab_size, cfg.train.model_dim,
                                            padding_idx=cfg.train.constants['PAD'])
        self.position_embeddings = nn.Embedding(cfg.train.token_max_len, cfg.train.model_dim)
        self.category_embeddings = nn.Embedding(cfg.train.category_num, cfg.train.model_dim)
        self.layerNorm = nn.LayerNorm(cfg.train.model_dim, eps=cfg.train.LN_eps)
        self.dropout = nn.Dropout(cfg.train.dropout)

    def forward(self, tag_tokens, category):
        words_embeddings = self.word_embeddings(tag_tokens)
        tok_len = tag_tokens.size(1)
        position_tok = torch.arange(tok_len, dtype=torch.long, device=tag_tokens.device)
        position_tok = position_tok.unsqueeze(0).expand_as(tag_tokens)
        position_embeddings = self.position_embeddings(position_tok)
        category_embeddings = self.category_embeddings(category).repeat(1, words_embeddings.size(1), 1)
        embeddings = words_embeddings + position_embeddings + category_embeddings
        embeddings = self.layerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class Attention(nn.Module):
    def __init__(self, cfg):
        super(Attention, self).__init__()
        if cfg.train.model_dim % cfg.train.attention_heads != 0:
            raise ValueError("The model_dim should be divisible by attention_heads!")
        self.attention_heads = cfg.train.attention_heads
        self.attention_head_size = int(cfg.train.model_dim / cfg.train.attention_heads)
        self.all_head_size = self.attention_heads * self.attention_head_size

        self.query = nn.Linear(cfg.train.model_dim, self.all_head_size)
        self.key = nn.Linear(cfg.train.model_dim, self.all_head_size)
        self.value = nn.Linear(cfg.train.model_dim, self.all_head_size)

        self.dropout = nn.Dropout(cfg.train.attention_dropout)

    def forward(self, q, k, v, attention_mask, head_mask=None, output_attentions=False):
        d_k, d_v, n_head = self.attention_head_size, self.attention_head_size, self.attention_heads

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        q = self.query(q).view(sz_b, len_q, n_head, d_k)
        k = self.key(k).view(sz_b, len_k, n_head, d_k)
        v = self.value(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        if attention_mask is not None:
            attention_mask = attention_mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..

        attention_scores = torch.bmm(q, k.transpose(1, 2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask, -10e6)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        outputs = torch.bmm(attention_probs, v)

        outputs = outputs.view(n_head, sz_b, len_q, d_v)
        outputs = outputs.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        return (outputs, attention_probs.view(n_head, sz_b, len_q, len_k)) if output_attentions else (outputs,)


class Add_and_LN(nn.Module):
    def __init__(self, input_dim, module_dim, LN_eps, dropout=0.5, with_lN=True):
        super(Add_and_LN, self).__init__()
        self.linear = nn.Linear(input_dim, module_dim)
        self.LayerNorm = nn.LayerNorm(module_dim, eps=LN_eps) if with_lN else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor=None, dropout_last=False):
        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if input_tensor is not None:
            hidden_states = hidden_states + input_tensor
        if self.LayerNorm is not None:
            hidden_states = self.LayerNorm(hidden_states)
        if dropout_last:
            hidden_states = self.dropout(hidden_states)
        return hidden_states


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg, pos=False, with_residual=True, is_self=True):
        super(MultiHeadAttention, self).__init__()
        self.attention = Attention(cfg)
        self.output = Add_and_LN(cfg.train.model_dim, cfg.train.model_dim, cfg.train.LN_eps, cfg.train.dropout, with_lN=False)
        self.pos = pos
        self.with_residual = with_residual

    def forward(self, q, k, v, attention_mask, head_mask):
        att_output = self.attention(q, k, v, attention_mask, head_mask)
        output = self.output(att_output[0], q if self.with_residual else None)
        output = (output,) + att_output[1:]
        return output


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super(FeedForward, self).__init__()
        self.linear = nn.Linear(cfg.train.model_dim, cfg.train.vision_dim)
        if cfg.train.activation == 'gelu':
            self.activation = lambda x: x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        elif cfg.train.activation == 'gelu_new':
            self.activation = lambda x: 0.5 * x * (
                        1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        elif cfg.train.activation == 'swish':
            self.activation = lambda x: x * torch.sigmoid(x)
        else:
            self.activation = F.relu
        self.output = Add_and_LN(cfg.train.vision_dim, cfg.train.model_dim, cfg.train.LN_eps, cfg.train.dropout, with_lN=False)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.linear(hidden_states)
        hidden_states = self.activation(hidden_states)
        output = self.output(hidden_states, input_tensor, dropout_last=True)
        return output


class AttentionLayer(nn.Module):
    def __init__(self, cfg):
        super(AttentionLayer, self).__init__()
        self.intra_attention = MultiHeadAttention(cfg, with_residual=True)
        self.inter_attention = MultiHeadAttention(cfg, is_self=False)
        self.feed_forward = FeedForward(cfg)

    def forward(self, hidden_states, non_pad_mask=None, attention_mask=None,
                joint_rep=None, rep_tag_mask=None, head_mask=None):
        all_attentions = ()
        intra_att_outputs = self.intra_attention(hidden_states, hidden_states, hidden_states,
                                                 attention_mask, head_mask)
        intra_att_output = intra_att_outputs[0]
        all_attentions += intra_att_outputs[1:]

        if non_pad_mask is not None:
            intra_att_output = intra_att_output * non_pad_mask
        inter_att_outputs = self.inter_attention(intra_att_output, joint_rep, joint_rep,
                                                 rep_tag_mask, head_mask)
        inter_att_output = inter_att_outputs[0]
        all_attentions += intra_att_outputs[1:]
        if non_pad_mask is not None:
            inter_att_output = inter_att_output * non_pad_mask
        layer_output = self.feed_forward(inter_att_output, intra_att_output)
        if non_pad_mask is not None:
            layer_output *= non_pad_mask
        embs = layer_output.sum(1) / non_pad_mask.sum(1)
        outputs = (layer_output, embs,) + (all_attentions,)
        return outputs


class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.PAD = cfg.train.constants['PAD']
        self.embedding = TokenEmbeddings(cfg)
        self.attention_layer = AttentionLayer(cfg)

    def forward(self, joint_rep, tag_tokens, category):
        self_att_mask_pad = get_att_pad_mask(tag_tokens, tag_tokens, self.PAD)
        batch_size, seq_len = tag_tokens.size()
        self_att_sub_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=tag_tokens.device, dtype=torch.uint8), diagonal=1
        )
        self_att_sub_mask = self_att_sub_mask.unsqueeze(0).expand(batch_size, -1, -1)
        self_att_mask = (self_att_mask_pad + self_att_sub_mask).gt(0)
        non_pad_mask = tag_tokens.ne(self.PAD).type(torch.float).unsqueeze(-1)
        rep_size_ones = torch.ones(joint_rep.size(0), joint_rep.size(1)).to(joint_rep.device)
        rep_tag_mask = get_att_pad_mask(rep_size_ones, tag_tokens, self.PAD)
        hidden_states = self.embedding(tag_tokens, category)
        layer_outputs = self.attention_layer(
            hidden_states,
            non_pad_mask=non_pad_mask,
            attention_mask=self_att_mask,
            joint_rep=joint_rep,
            rep_tag_mask=rep_tag_mask,
        )
        res = [layer_outputs[0]]
        embs = layer_outputs[1]
        outputs = (res, embs,)
        return outputs
