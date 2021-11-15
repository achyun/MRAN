import argparse
import json
import os
import csv
import sys
from collections import defaultdict
from termcolor import colored

import numpy as np
import torch
from tqdm import tqdm

from misc.Translator import Translator
from utils import to_sentence, analyze_length_novel_unique
from misc.cocoeval import suppress_stdout_stderr, COCOScorer
from dataloader import MultimodalDataLoader
import models.MRAN as MRAN


def evaluate(cfg, model, loader, device, vocab, save_path='', analyze=False, no_score=False):
    model.eval()
    gts = loader.refs
    predict_caps = defaultdict(list)
    translator = Translator(model=model, cfg=cfg)
    with torch.no_grad():
        for data in tqdm(loader, total=len(loader)):
            app_feat, motion_feat, hand_feat, category, labels, tokens = map(
                lambda x: x.to(device),
                [data['app_feat'], data['motion_feat'], data['hand_feat'], data['category'], data['labels'],
                 data['tokens']]
            )
            joint_rep = model.encoder(app_feat=app_feat, motion_feat=motion_feat, hand_feat=hand_feat)
            # start_index = 1
            # tgt_word_labels = labels[:, start_index:]
            all_hyp, all_scores = translator.translate_batch(joint_rep, category)

            if isinstance(all_hyp, torch.Tensor):
                if len(all_hyp.shape) == 2:
                    all_hyp = all_hyp.unsqueeze(1)
                all_hyp = all_hyp.tolist()
            if isinstance(all_scores, torch.Tensor):
                if len(all_scores.shape) == 2:
                    all_scores = all_scores.unsqueeze(1)
                all_scores = all_scores.tolist()
            video_ids = np.array(data['video_ids']).reshape(-1)

            for k, hyps in enumerate(all_hyp):
                video_id = video_ids[k]
                if not no_score:
                    assert len(hyps) == 1
                for j, hyp in enumerate(hyps):
                    sent = to_sentence(hyp, vocab)
                    predict_caps[video_id].append({'image_id': video_id, 'caption': sent})

    res = dict()
    if analyze:
        ave_length, novel, unique, usage, hy_res, gram4 = analyze_length_novel_unique(loader.corpus['captions'],
                                                                                      predict_caps, vocab,
                                                                                      splits=loader.corpus['info']['split'], n=1)
        res.update({'ave_length': ave_length, 'novel': novel, 'unique': unique, 'usage': usage, 'gram4': gram4})

    scorer = COCOScorer()
    if not no_score:
        with suppress_stdout_stderr():
            valid_score, detail_scores = scorer.score(gts, predict_caps, predict_caps.keys())

        res.update(valid_score)
        res['Sum'] = sum([res["Bleu_4"], res["METEOR"], res["ROUGE_L"], res["CIDEr"]])

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with open(os.path.join(save_path, 'results.json'), 'w') as prediction_results:
            json.dump({"predictions": predict_caps, "scores": valid_score}, prediction_results)
            prediction_results.close()
    return res


def main():
    parser = argparse.ArgumentParser(description='evaluate.py')
    parser.add_argument('--model_path', default='results/CFO/ckpt/best_model.pt', type=str)
    parser.add_argument('--save_path', default='results/predictions', type=str)
    parser.add_argument('--is_val', default=False, type=bool)

    opt = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('loading pretrained model from %s' % opt.model_path)
    ckpt = torch.load(opt.model_path, map_location='cpu')
    cfg = ckpt['model_kwargs']
    model = MRAN.MRAN(cfg).to(device)
    model.load_state_dict(ckpt['state_dict'])
    modes = ['validate', 'test'] if opt.is_val else ['test']
    for mode in modes:
        loader = MultimodalDataLoader(cfg, mode)
        vocab = loader.corpus['info']['itow']
        save_path = opt.save_path if mode == 'test' else ''
        res = evaluate(cfg, model, loader, device, vocab, save_path, analyze=False)

        headers = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4',
                    'METEOR', 'ROUGE_L', 'CIDEr', 'Sum']
        csvfile_path = os.path.join(cfg.dataset.save_dir, mode + '_record.csv')
        if not os.path.isfile(csvfile_path):
            with open(csvfile_path, 'w') as f:
                f_csv = csv.writer(f)
                f_csv.writerow(headers)
        res_row = [res[i] for i in headers if i in res.keys()]
        with open(csvfile_path, 'a') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(res_row)
        print('~~~~~~ {} Score: {:.4f} ~~~~~~~'.format(mode, res['Sum']))
        sys.stdout.write('\tBLEU(4): {BLEU}  METEOR: {METEOR}  ROUGE_L: {ROUGE_L}  CIDEr: {CIDEr}  Sum: {Sum}\n'.format(
            BLEU=colored("{:.2f}".format(res['Bleu_4'] * 100), "blue", attrs=['bold']),
            METEOR=colored("{:.2f}".format(res['METEOR'] * 100), "blue", attrs=['bold']),
            ROUGE_L=colored("{:.2f}".format(res['ROUGE_L'] * 100), "blue", attrs=['bold']),
            CIDEr=colored("{:.2f}".format(res['CIDEr'] * 100), "blue", attrs=['bold']),
            Sum=colored("{:.2f}".format(res['Sum']), "blue", attrs=['bold'])))
        sys.stdout.flush()


if __name__ == '__main__':
    main()


