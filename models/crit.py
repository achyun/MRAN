from misc.logger import AverageMeter
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
from collections import defaultdict

from config import cfg


class CritBase(nn.Module):
    def __init__(self, weights=1.0, batch_mean=True):
        super(CritBase, self).__init__()
        self.keys = ('tgt_word_logprobs', 'tgt_word_labels')
        self.weights = weights
        self.batch_mean = batch_mean

    def _step(self, *inputs):
        raise NotImplementedError()

    def forward(self, kwargs):
        sources1, sources2, *others = [kwargs[key] for key in self.keys]

        if not isinstance(sources1, list):
            assert type(sources1) == torch.Tensor
            sources1 = [sources1]

        if not isinstance(sources2, list):
            assert type(sources2) == torch.Tensor
            sources2 = [sources2] * len(sources1)
        else:
            assert len(sources1) == len(sources2)

        if not isinstance(self.weights, list):
            self.weights = [self.weights] * len(sources1)

        assert len(sources1) == len(self.weights)

        loss = None
        dinominator = sources1[0].size(0) if self.batch_mean else 1.0

        for i, (weight, src1, src2) in enumerate(zip(self.weights, sources1, sources2)):
            if loss is None:
                loss = weight * self._step(i, src1, src2, *others) / dinominator
            else:
                loss = loss + weight * self._step(i, src1, src2, *others) / dinominator

        return loss, dinominator


class LanguageGeneration(CritBase):
    def __init__(self, cfg, weights=1.0, batch_mean=True):
        super().__init__(weights, batch_mean)
        self.loss_fn = nn.NLLLoss(reduce=False)
        self.ignore_index = cfg.train.constants['PAD']
        self.num_word_acc = 1
        self.visual_word_generation = False

    def _step(self, index_indicator, tgt_word_logprobs, tgt_word_labels, *others):
        """
            args:
                tgt_word_logprobs: [batch_size, seq_len, vocab_size]
                tgt_word_labels: [batch_size, seq_len]
        """
        assert not len(others)
        assert tgt_word_logprobs.size(1) == tgt_word_labels.size(1)

        # calculate the top-1 accuracy of the generated words
        self.calculate_word_acc(index_indicator, tgt_word_logprobs, tgt_word_labels)
        # calculate the perplexity of the generated words
        self.calculate_perplexity(index_indicator, tgt_word_logprobs, tgt_word_labels)

        tgt_word_logprobs = tgt_word_logprobs.contiguous().view(-1, tgt_word_logprobs.size(2))
        tgt_word_labels = tgt_word_labels.contiguous().view(-1)
        loss = self.loss_fn(tgt_word_logprobs, tgt_word_labels)

        if self.ignore_index is not None:
            mask = tgt_word_labels.ne(self.ignore_index).float()
            return torch.sum(loss * mask)
        else:
            return torch.sum(loss)

    def calculate_word_acc(self, index_indicator, preds, gts):
        ind = gts.ne(cfg.train.constants['PAD'])
        if index_indicator == 0 and self.visual_word_generation:
            ind = ind & gts.ne(cfg.train.constants['MASK'])

        predict_res = preds.max(-1)[1][ind]
        target_res = gts[ind]

        self.word_acc_recorder[index_indicator].update(
            (predict_res == target_res).sum().item(),
            predict_res.size(0),
            multiply=False
        )

    def calculate_perplexity(self, index_indicator, preds, gts):
        # for the methods with visual word generation
        # we only compute the perplexity of the caption genration process
        if index_indicator == 0 and self.visual_word_generation:
            return None

        assert len(preds.shape) == 3
        assert preds.shape[:-1] == gts.shape

        log_probs = preds.gather(2, gts.unsqueeze(2)).squeeze(2)
        mask = gts.ne(cfg.train.constants['PAD'])
        num_words = float(torch.sum(mask))

        per_word_cross_entropy = -torch.sum(log_probs * mask) / num_words
        self.perplexity_recorder.update(per_word_cross_entropy.item(), num_words)

    def get_fieldsnames(self):
        return ['Word Acc%d' % i for i in range(self.num_word_acc)] + ['Perplexity']

    def get_info(self):
        info = [meter.avg for meter in self.word_acc_recorder]
        info += [math.exp(self.perplexity_recorder.avg)]
        return self.get_fieldsnames(), info

    def reset_recorder(self):
        self.word_acc_recorder = [AverageMeter() for _ in range(self.num_word_acc)]
        self.perplexity_recorder = AverageMeter()


class Criterion(object):
    """
        Calculating losses or some metrics for all tasks

        Standard operations:
            1. before a epoch, Criterion.reset_loss_recorder()
            2. during a epoch, Criterion.get_loss(forward_results)
            3. after  a epoch, Criterion.get_loss_info()
    """

    def __init__(self, crit_objects, keys, names, scales, summarywriter=None):
        assert len(crit_objects) == len(keys)
        assert len(keys) == len(names)
        assert len(names) == len(scales)
        self.crit_objects = crit_objects
        self.num_loss = len(crit_objects)
        self.keys = keys
        self.names = names
        self.scales = scales
        self.summarywriter = summarywriter
        self.n_current_round = 0

    def reset_loss_recorder(self):
        self.loss_recorder = [AverageMeter() for _ in range(self.num_loss)]
        for crit_object in self.crit_objects:
            if getattr(crit_object, 'reset_recorder', None) is not None:
                crit_object.reset_recorder()

    def get_loss(self, results):
        """
            args:
                results: dict, contains the forward results of the model and some ground-truths
        """
        loss = []
        for i in range(self.num_loss):
            # calculate the i-th loss
            if isinstance(self.crit_objects[i], CritBase):
                i_loss, num_samples = self.crit_objects[i](results)
            else:
                # prepare the predictions and its corresponding ground-truths
                preds = results[self.keys[i][0]]
                gts = results[self.keys[i][1]]
                i_loss = self.crit_objects[i](preds, gts)
                num_samples = gts.size(0)

            # weighting the i-th loss
            loss.append(i_loss * self.scales[i])

            # update the statistics of the i-th loss
            self.loss_recorder[i].update(i_loss.item(), num_samples)

        # loss = loss1 * scale1 + loss2 * scale2 + ...
        loss = torch.stack(loss, dim=0).sum(0)
        return loss

    def get_loss_info(self):
        all_names = self.names
        all_info = [meter.avg for meter in self.loss_recorder]

        for crit_object in self.crit_objects:
            if getattr(crit_object, 'get_info', None) is not None:
                this_name, this_info = crit_object.get_info()
                all_names += this_name
                all_info += this_info

        if self.summarywriter is not None:
            self.n_current_round += 1
            for name, loss in zip(all_names, all_info):
                self.summarywriter.add_scalar(name, loss, global_step=self.n_current_round)

        # e.g., ['Cap Loss', 'Word Acc0', 'Perplexity'], [31.8, 0.385, 53.0]
        return all_names, all_info

    def get_fieldsnames(self):
        exclude_index_set = []
        fieldsnames = []
        for i, crit_object in enumerate(self.crit_objects):
            if isinstance(crit_object, LanguageGeneration):
                exclude_index_set.append(i)
            elif getattr(crit_object, 'get_fieldsnames', None) is not None:
                fieldsnames += crit_object.get_fieldsnames()

        fieldsnames += [n for i, n in enumerate(self.names) if i not in exclude_index_set]
        return fieldsnames