import argparse
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_value_
import os
import random
import time
import numpy as np
import pickle
import logging
import csv

from termcolor import colored

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()

from utils import ScheduledOptim
from dataloader import MultimodalDataLoader
import models.MRAN as MRAN
import models.crit as Crit
from config import cfg, cfg_from_file
from evaluate import evaluate


def train(cfg):
    model_kwargs_tosave = cfg
    model = MRAN.MRAN(cfg)
    _optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), weight_decay=cfg.train.weight_decay)
    lr = cfg.train.learning_rate
    logging.info("model params: {}, number for trained: {}".format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))
    logging.info(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    start_epoch = 0
    if cfg.train.pretrained_path is not None:
        logging.info('loading pretrained model from %s' % cfg.train.pretrained_path)
        ckpt = torch.load(cfg.train.pretrained_path, map_location=lambda storage, loc: storage)
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['state_dict'])
        _optimizer.load_state_dict(ckpt['optimizer'])
        if cfg.train.restore_lr:
            lr = ckpt['lr']
    if torch.cuda.device_count() > 1 and cfg.multi_gpus:
        model = model.cuda()
        logging.info("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=None)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    optimizer = ScheduledOptim(
        optimizer=_optimizer,
        learning_rate=lr,
        minimum_learning_rate=cfg.train.minimum_learning_rate,
        epoch_decay_rate=cfg.train.decay,
        n_warmup_steps=cfg.train.n_warmup_steps
    )
    crit = Crit.Criterion(
        crit_objects=[Crit.LanguageGeneration(cfg)],
        keys=[('tgt_word_logprobs', 'tgt_word_labels')],
        names=['Cap Loss'],
        scales=[1.0]
    )
    train_loader = MultimodalDataLoader(cfg, 'train')
    val_loader = MultimodalDataLoader(cfg, 'validate')
    test_loader = MultimodalDataLoader(cfg, 'test')
    vocab = train_loader.corpus['info']['itow']

    headers = [
        'epoch', 'train_loss',
        'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4',
        'METEOR', 'ROUGE_L', 'CIDEr', 'Sum'] + crit.get_fieldsnames()
    csvfile_path = os.path.join(cfg.dataset.save_dir, 'train_record.csv')
    if not os.path.isfile(csvfile_path):
        with open(csvfile_path, 'w') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(headers)

    best_score = 0
    logging.info("start training........")
    for epoch in range(start_epoch, cfg.train.max_epochs):
        logging.info('>>>>>> epoch {epoch} lr={lr} <<<<<<'.format(
                    epoch=colored("{}".format(epoch), "green", attrs=["bold"]),
                    lr=colored("{}".format(optimizer.get_lr()), "green", attrs=["bold"])))
        model.train()
        crit.reset_loss_recorder()
        total_loss, avg_loss = 0.0, 0.0
        for i, data in enumerate(iter(train_loader)):
            progress = epoch + i / len(train_loader)
            optimizer.zero_grad()
            app_feat, motion_feat, hand_feat, category, labels, tokens = map(
                lambda x: x.to(device),
                [data['app_feat'], data['motion_feat'], data['hand_feat'], data['category'], data['labels'], data['tokens']]
            )
            results = model(
                app_feat=app_feat,
                motion_feat=motion_feat,
                hand_feat=hand_feat,
                tag_tokens=tokens,
                category=category,
            )
            start_index = 1
            results['tgt_word_labels'] = labels[:, start_index:]
            loss = crit.get_loss(results)
            loss.backward()
            total_loss += loss.detach()
            avg_loss = total_loss / (i + 1)
            clip_grad_value_(model.parameters(), 5)
            optimizer.step()
            sys.stdout.write(
                "\rProgress = {progress}  loss = {loss}  avg_loss = {avg_loss}".format(
                    progress=colored("{:.3f}".format(progress), "green", attrs=['bold']),
                    loss=colored("{:.4f}".format(loss.item()), "blue", attrs=['bold']),
                    avg_loss=colored("{:.4f}".format(avg_loss), "red", attrs=['bold'])
                ))
            sys.stdout.flush()
        sys.stdout.write("\n")
        name, loss_info = crit.get_loss_info()
        logging.info('\t'.join(['%10s: %05.3f' % (item[0], item[1]) for item in zip(name, loss_info)]))
        train_loss = loss_info[0]
        if (epoch + 1) % 5 == 0:
            optimizer.epoch_update_learning_rate()

        res = evaluate(cfg, model, val_loader, device, vocab, analyze=True)
        res['train_loss'] = train_loss
        res['epoch'] = epoch

        res_row = [res[i] for i in headers if i in res.keys()]
        with open(csvfile_path, 'a') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(res_row)

        ckpt_dir = os.path.join(cfg.dataset.save_dir, 'ckpt')
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        else:
            assert os.path.isdir(ckpt_dir)
        save_checkpoint(epoch, model, optimizer.get_optimizer(), optimizer.get_lr(), model_kwargs_tosave,
                        os.path.join(ckpt_dir, 'model.pt'))
        if best_score < res['Sum']:
            best_score = res['Sum']
            save_checkpoint(epoch, model, optimizer.get_optimizer(), optimizer.get_lr(), model_kwargs_tosave,
                            os.path.join(cfg.dataset.save_dir, 'ckpt', 'best_model.pt'))
            sys.stdout.write('\n >>>>>>A better checkpoint saves to %s <<<<<< \n' %
                             (os.path.join(cfg.dataset.save_dir, 'ckpt', 'best_model.pt')))
            sys.stdout.flush()

        logging.info('~~~~~~ Valid Score: %.4f ~~~~~~~' % res['Sum'])
        sys.stdout.write('\tBLEU(4): {BLEU}  METEOR: {METEOR}  ROUGE_L: {ROUGE_L}  CIDEr: {CIDEr}  Sum: {Sum}\n'.format(
            BLEU=colored("{:.2f}".format(res['Bleu_4']*100), "blue", attrs=['bold']),
            METEOR=colored("{:.2f}".format(res['METEOR']*100), "blue", attrs=['bold']),
            ROUGE_L=colored("{:.2f}".format(res['ROUGE_L']*100), "blue", attrs=['bold']),
            CIDEr=colored("{:.2f}".format(res['CIDEr']*100), "blue", attrs=['bold']),
            Sum=colored("{:.2f}".format(res['Sum']), "blue", attrs=['bold'])))
        sys.stdout.flush()

    test_result = evaluate(cfg, model, test_loader, device, vocab, analyze=False)
    logging.info('~~~~~~ test Score: %.4f ~~~~~~~' % test_result['Sum'])
    sys.stdout.write('\tBLEU(4): {BLEU}  METEOR: {METEOR}  ROUGE_L: {ROUGE_L}  CIDEr: {CIDEr}  Sum: {Sum}\n'.format(
        BLEU=colored("{:.2f}".format(test_result['Bleu_4'] * 100), "red", attrs=['bold']),
        METEOR=colored("{:.2f}".format(test_result['METEOR'] * 100), "red", attrs=['bold']),
        ROUGE_L=colored("{:.2f}".format(test_result['ROUGE_L'] * 100), "red", attrs=['bold']),
        CIDEr=colored("{:.2f}".format(test_result['CIDEr'] * 100), "red", attrs=['bold']),
        Sum=colored("{:.2f}".format(test_result['Sum']), "red", attrs=['bold'])))
    sys.stdout.flush()


def save_checkpoint(epoch, model, optimizer, lr, model_kwargs, filename):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr': lr,
        'model_kwargs': model_kwargs,
    }
    time.sleep(10)
    torch.save(state, filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='configs/CFO.yml', type=str)
    args = parser.parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if not cfg.multi_gpus:
        torch.cuda.set_device(cfg.gpu_id)
    cfg.dataset.save_dir = os.path.join(cfg.dataset.save_dir, cfg.dataset.name)
    if not os.path.exists(cfg.dataset.save_dir):
        os.makedirs(cfg.dataset.save_dir)
    log_file = os.path.join(cfg.dataset.save_dir, "log")
    if not os.path.exists(log_file):
        os.mkdir(log_file)

    # get feats' dir
    cfg.dataset.appearance_feat = cfg.dataset.appearance_feat.format(cfg.dataset.data_dir, cfg.dataset.name, cfg.dataset.name)
    cfg.dataset.motion_feat = cfg.dataset.motion_feat.format(cfg.dataset.data_dir, cfg.dataset.name, cfg.dataset.name)
    cfg.dataset.hand_feat = cfg.dataset.hand_feat.format(cfg.dataset.data_dir, cfg.dataset.name, cfg.dataset.name)
    # cfg.dataset.audio_feat = cfg.dataset.audio_feat.format(cfg.dataset.data_dir, cfg.dataset.name, cfg.dataset.name)
    cfg.dataset.corpus = cfg.dataset.corpus.format(cfg.dataset.data_dir, cfg.dataset.name, cfg.dataset.name)
    cfg.dataset.refs = cfg.dataset.refs.format(cfg.dataset.data_dir, cfg.dataset.name, cfg.dataset.name)
    cfg.dataset.vocab_size = len(pickle.load(open(cfg.dataset.corpus, 'rb'))['info']['itow'].keys())

    fileHandler = logging.FileHandler(os.path.join(log_file, 'stdout.log'), 'w+')
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    for k, v in vars(cfg).items():
        logging.info(k + ':' + str(v))

    # set random seed
    random.seed(cfg.seed)
    os.environ['PYTHONHASHSEED'] = str(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train(cfg)

if __name__ == '__main__':
    main()