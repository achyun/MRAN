import argparse
import json
import os
import pickle
import string
from collections import defaultdict

import nltk
from tqdm import tqdm


def get_length_info(captions):
    length_info = {}
    max_length = 50

    for vid, caps in captions.items():
        length_info[vid] = [0] * max_length
        for cap in caps:
            length = len(cap) - 2  # exclude <bos>, <eos>
            if length >= max_length:
                continue
            length_info[vid][length] += 1

    return length_info


def main(args):
    with open(args.annotation_file.format(args.dataset), 'r', encoding='utf-8') as anno_file:
        instances = json.load(anno_file)

    # data preprocess
    id_split = {'train': [], 'validate': [], 'test': []}
    itoc = dict()
    category = {'train': defaultdict(list), 'validate': defaultdict(list), 'test': defaultdict(list)}
    for instance in instances['videos']:
        id_split[instance['split']].append(int(instance['id']))
        itoc[instance['id']] = instance['category']
        category[instance['split']][int(instance['category'])].append(int(instance['id']))

    all_caps, train_caps, ref = defaultdict(list), defaultdict(list), defaultdict(list)
    for instance in tqdm(instances['sentences']):
        vid = instance['video_id']
        tokens = [word.lower() for word in instance['caption'].split() if word not in string.punctuation]
        all_caps[vid].append(tokens)
        start_index = 2 if args.dataset == 'CFO' else 5
        if int(vid[start_index:]) in id_split['train']:
            train_caps['vid'].append(tokens)
        ref[vid].append({
            'image_id': vid,
            'cap_id': len(ref[vid]),
            'caption': ' '.join(tokens)
        })

    # build_vocab
    counts = dict()
    for vid, caps in train_caps.items():
        for cap in caps:
            for w in cap:
                counts[w] = counts.get(w, 0) + 1

    bad_words = [w for w, n in counts.items() if n <= args.threshold]
    vocab = [(w, n) for w, n in counts.items() if n > args.threshold]
    vocab = sorted(vocab, key=lambda x: -x[1])
    vocab = [w for w, _ in vocab]
    bad_count = sum(counts[w] for w in bad_words)
    total_words = sum(counts.values())
    print('number of bad words: %d/%d = %.2f%%' %
          (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
    print('number of words in vocab would be %d' % (len(vocab),))
    print('number of UNKs: %d/%d = %.2f%%' %
          (bad_count, total_words, bad_count * 100.0 / total_words))
    print('50 top frequency words:', vocab[:50])

    itow = {0: '<pad>', 1: '<unk>', 2: '<bos>', 3: '<eos>', 4: '<mask>', 5: '<vis>'}
    ptoi = {w: i for i, w in itow.items()}
    for i, w in enumerate(vocab):
        itow[i + 6] = w
    wtoi = {w: i for i, w in itow.items()}
    tag_start_i = 6

    # mapping of nltk pos tags
    pos_tag_mapping = {}
    content = [
        [["``", "''", ",", "-LRB-", "-RRB-", ".", ":", "HYPH", "NFP"], "PUNCT"],
        [["$", "SYM"], "SYM"],
        [["VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "MD"], "VERB"],
        [["WDT", "WP$", "PRP$", "DT", "PDT"], "DET"],
        [["NN", "NNP", "NNPS", "NNS"], "NOUN"],
        [["WP", "EX", "PRP"], "PRON"],
        [["JJ", "JJR", "JJS", "AFX"], "ADJ"],
        [["ADD", "FW", "GW", "LS", "NIL", "XX"], "X"],
        [["SP", "_SP"], "SPACE"],
        [["RB", "RBR", "RBS", "WRB"], "ADV"],
        [["IN", "RP"], "ADP"],
        [["CC"], "CCONJ"],
        [["CD"], "NUM"],
        [["POS", "TO"], "PART"],
        [["UH"], "INTJ"]
    ]
    for item in content:
        ks, v = item
        for k in ks:
            pos_tag_mapping[k] = v

    captions, pos_tags = defaultdict(list), defaultdict(list)
    for vid, caps in tqdm(all_caps.items()):
        for cap in caps:
            tag_res = nltk.pos_tag(cap)
            caption_id, tagging_id = [ptoi['<bos>']], [ptoi['<bos>']]

            for w, t in zip(cap, tag_res):
                assert t[0] == w
                tag = pos_tag_mapping[t[1]]

                if w in wtoi.keys():
                    caption_id += [wtoi[w]]
                    if tag not in ptoi.keys():
                        ptoi[tag] = tag_start_i
                        tag_start_i += 1
                    tagging_id += [ptoi[tag]]
                else:
                    caption_id += [ptoi['<unk>']]
                    tagging_id += [ptoi['<unk>']]

            caption_id += [ptoi['<eos>']]
            tagging_id += [ptoi['<eos>']]

            captions[vid].append(caption_id)
            pos_tags[vid].append(tagging_id)

    itop = {i: t for t, i in ptoi.items()}
    length_info = get_length_info(captions)
    info = {
        'split': id_split,  # {'train': [0, 1, 2, ...], 'validate': [...], 'test': [...]}
        'split_category': category,
        'itoc': itoc,
        'itow': itow,  # id to word
        'itop': itop,  # id to POS tag
        'length_info': length_info,  # id to length info
    }

    with open(args.output_path.format(args.dataset, args.dataset, 'corpus'), 'wb') as f:
        pickle.dump({'info': info, 'captions': captions, 'pos_tags': pos_tags}, f)
    with open(args.output_path.format(args.dataset, args.dataset, 'refs'), 'wb') as f:
        pickle.dump(ref, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-H', '--threshold', default=2, type=int)
    parser.add_argument('--dataset', default='CFO', choices=['CFO', 'MSRVTT'], type=str)
    args = parser.parse_args()

    if not os.path.exists('./data/{}'.format(args.dataset)):
        os.makedirs('./data/{}'.format(args.dataset))
    args.annotation_file = './data/dataset/{}/info.json'
    args.output_path = './data/{}/{}-{}.pkl'
    main(args)
