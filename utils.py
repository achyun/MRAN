import numpy as np
import random
from config import cfg


def resample(clip_num, frame_num, sample_num, seed):
    np.random.seed(seed)
    if sample_num >= clip_num:
        sample_clips = list(range(clip_num))
        sample_frames = [sample_num // clip_num] * clip_num
        for i in range(sample_num % clip_num):
            sample_frames[random.randint(0, clip_num - 1)] += 1
    else:
        sample_frames = [1] * sample_num
        sample_clips = np.linspace(0, clip_num - 1, sample_num, dtype=int)
    sample_dict = dict()
    for i, clip in enumerate(sample_clips):
        tmp = np.linspace(0, frame_num - 1, sample_frames[i] + 1, dtype=int)
        idx = list()
        for j in range(sample_frames[i]):
            idx.append(int((tmp[j] + tmp[j + 1]) // 2))
        sample_dict[clip] = idx
    return sample_dict


def get_att_pad_mask(k, q, PAD):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = q.size(1)
    padding_mask = k.eq(PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


def padding(seq, add_eos=True):
    if seq is None:
        return None
    res = seq.copy()
    if len(res) > cfg.train.token_max_len:
        res = res[:cfg.train.token_max_len]
        if add_eos:
            res[-1] = cfg.train.constants['EOS']
    else:
        res += [cfg.train.constants['PAD']] * (cfg.train.token_max_len - len(res))
    return res


def enlarge(info, beam_size):
    bsz, *rest_shape = info.shape
    if len(rest_shape) == 2:
        info = info.unsqueeze(1).repeat(1, beam_size, 1, 1)
    elif len(rest_shape) == 1:
        info = info.unsqueeze(1).repeat(1, beam_size, 1)
    else:
        info = info.unsqueeze(1).repeat(1, beam_size)
    return info.view(bsz * beam_size, *rest_shape)


def auto_enlarge(info, beam_size):
    if isinstance(info, list):
        if isinstance(info[0], tuple):
            return [
                tuple([enlarge(_, beam_size) for _ in item])
                for item in info
            ]
        else:
            return [enlarge(item, beam_size) for item in info]
    else:
        if isinstance(info, tuple):
            return tuple([enlarge(item, beam_size) for item in info])
        else:
            return enlarge(info, beam_size)


def to_sentence(hyp, vocab, break_words=[cfg.train.constants['EOS'], cfg.train.constants['PAD']], skip_words=[]):
    sent = []
    for word_id in hyp:
        if word_id in skip_words:
            continue
        if word_id in break_words:
            break
        word = vocab[word_id]
        sent.append(word)
    return ' '.join(sent)


def cal_gt_n_gram(data, vocab, splits, n=1):
    gram_count = {}
    gt_sents = {}
    for i in splits['train']:
        k = 'video%d'% int(i)
        caps = data[k]
        for tmp in caps:
            cap = [vocab[wid] for wid in tmp[1:-1]]
            gt_sents[' '.join(cap)] = gt_sents.get(' '.join(cap), 0) + 1
            for j in range(len(cap)-n+1):
                key = ' '.join(cap[j:j+n])
                gram_count[key] = gram_count.get(key, 0) + 1
    return gram_count, gt_sents


def cal_n_gram(data, n=1):
    gram_count = {}
    sents = {}
    ave_length, count = 0, 0
    for k in data.keys():
        for i in range(len(data[k])):
            sents[data[k][i]['caption']] = sents.get(data[k][i]['caption'], 0) + 1
            cap = data[k][i]['caption'].split(' ')
            ave_length += len(cap)
            count += 1
            for j in range(len(cap)-n+1):
                key = ' '.join(cap[j:j+n])
                gram_count[key] = gram_count.get(key, 0) + 1
    return gram_count, sents, ave_length/count, count


def analyze_length_novel_unique(gt_data, data, vocab, splits, n=1, calculate_novel=True):
    novel_count = 0
    hy_res, hy_sents, ave_length, hy_count = cal_n_gram(data, n)
    if calculate_novel:
        gt_res, gt_sents = cal_gt_n_gram(gt_data, vocab, splits, n)
        for k1 in hy_sents.keys():
            if k1 not in gt_sents.keys():
                novel_count += 1

    novel = novel_count / hy_count
    unique = len(hy_sents.keys()) / hy_count
    vocabulary_usage = len(hy_res.keys())

    gram4, _, _, _ = cal_n_gram(data, n=4)
    return ave_length, novel, unique, vocabulary_usage, hy_res, len(gram4)


class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, learning_rate, minimum_learning_rate, epoch_decay_rate,
                 grad_clip=2, n_warmup_steps=0, summarywriter=None):
        self._optimizer = optimizer
        self.n_current_steps = 0
        self.lr = learning_rate
        self.mlr = minimum_learning_rate
        self.decay = epoch_decay_rate
        self.grad_clip = grad_clip
        self.n_warmup_steps = n_warmup_steps
        self.summarywriter = summarywriter

    def clip_gradient(self):
        for group in self._optimizer.param_groups:
            for param in group['params']:
                param.grad.data.clamp_(-self.grad_clip, self.grad_clip)

    def step(self):
        "Step with the inner optimizer"
        self.step_update_learning_rate()
        # self.clip_gradient()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def epoch_update_learning_rate(self):
        if self.n_current_steps > self.n_warmup_steps:
            self.lr = max(self.mlr, self.decay * self.lr)

    def step_update_learning_rate(self):
        ''' Learning rate scheduling per step '''
        self.n_current_steps += 1
        ratio = min(self.n_current_steps / (self.n_warmup_steps + 1.0), 1)
        learning_rate = self.lr * ratio

        if self.summarywriter is not None:
            self.summarywriter.add_scalar('learning_rate', learning_rate, global_step=self.n_current_steps)

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = learning_rate

    def get_lr(self):
        return self.lr

    def get_optimizer(self):
        return self._optimizer
