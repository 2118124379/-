# coding=utf-8
import torch
import re
import os
import unicodedata
import pickle
import json

# depends on the word_vocab file
SOS_ID = 0
EOS_ID = 1
PAD_ID = 3


class Vocabulary:
    def __init__(self, name, save_dir):
        self.name = name
        # save_dir  = 'E:/毕设/oklen-Coarse-to-Fine-Review-Generation-master/Coarse-to-Fine-Review-Generation'
        with open(os.path.join(save_dir, 'topic.pkl'), 'rb') as fp:
            self.topic2idx = pickle.load(fp)
        with open(os.path.join(save_dir, 'topic_rev.pkl'), 'rb') as fp:
            self.idx2topic = pickle.load(fp)
        with open(os.path.join(save_dir, 'sketch2index.pkl'), 'rb') as fp:
            self.sketch2idx = pickle.load(fp)
        with open(os.path.join(save_dir, 'sketch_rev.pkl'), 'rb') as fp:
            self.idx2sketch = pickle.load(fp)
        self.n_topics = len(self.topic2idx)
        self.n_sketchs = len(self.sketch2idx)


def tokenize(path, corpus_name, vocab, save_dir):
    print("Reading {}".format(path))
    # combine attributes and reviews into pairs
    # save_dir = 'E:/毕设/oklen-Coarse-to-Fine-Review-Generation-master/Coarse-to-Fine-Review-Generation/'
    with open(os.path.join(save_dir, 'user.pkl'), 'rb') as fp:
        user_dict = pickle.load(fp)
    with open(os.path.join(save_dir, 'item.pkl'), 'rb') as fp:
        item_dict = pickle.load(fp)
    pairs = []
    with open(path, 'r') as f:
        for l in json.load(f):
            user = l['reviewerID']
            item = l['asin']
            rating = l['overall']

            user_id = user_dict[user]
            item_id = item_dict[item]
            rating = rating - 1

            topic = l['topic_tok']
            sents = l['sketchText'].split("||")
            sketch = [["<sos>"] + s.strip().split() + ["<eos>"] for s in sents]
            revs = l['text'].split("||")
            review = [["<sos>"] + r.strip().split() + ["<eos>"] for r in revs]
            aux = [user_id, item_id, rating]
            pair = [aux, topic, sketch, review]  # two list
            pairs.append(pair)
    return pairs


def tokenize_test(path, corpus_name, vocab, save_dir):
    print("Reading {}".format(path))
    # combine attributes and reviews into pairs
    # save_dir = 'E:/毕设/oklen-Coarse-to-Fine-Review-Generation-master/Coarse-to-Fine-Review-Generation/'
    with open(os.path.join(save_dir, 'user.pkl'), 'rb') as fp:
        user_dict = pickle.load(fp)
    with open(os.path.join(save_dir, 'item.pkl'), 'rb') as fp:
        item_dict = pickle.load(fp)
    pairs = []
    with open(path, 'r') as f:
        for l in json.load(f):
            user = l['reviewerID']
            item = l['asin']
            rating = l['overall']

            user_id = user_dict[user]
            item_id = item_dict[item]
            rating = rating - 1

            topic = l['topic_tok']
            topic_gen = l['topic_gen']
            sents = l['sketchText'].split("||")
            sketch = [["<sos>"] + s.strip().split() + ["<eos>"] for s in sents]
            revs = l['text'].split("||")
            review = [["<sos>"] + r.strip().split() + ["<eos>"] for r in revs]
            aux = [user_id, item_id, rating]
            pair = [aux, topic, sketch, review, topic_gen]  # two list
            pairs.append(pair)
    return pairs


# actually we do not use corpus_name
def prepareData(vocab, corpus_name, save_dir):
    vocab = None  # Not used
    # save_dir = 'E:/毕设/oklen-Coarse-to-Fine-Review-Generation-master/Coarse-to-Fine-Review-Generation/'
    train_pairs = tokenize(os.path.join(save_dir, 'train_tok.json'), corpus_name, vocab, save_dir)
    valid_pairs = tokenize(os.path.join(save_dir, 'valid_tok.json'), corpus_name, vocab, save_dir)
    test_pairs = tokenize_test(os.path.join(save_dir, 'test_tok.json'), corpus_name, vocab, save_dir)

    torch.save(train_pairs, os.path.join(save_dir, '{!s}.tar'.format('train_pairs')))
    torch.save(valid_pairs, os.path.join(save_dir, '{!s}.tar'.format('valid_pairs')))
    torch.save(test_pairs, os.path.join(save_dir, '{!s}.tar'.format('test_pairs')))
    return train_pairs, valid_pairs, test_pairs


def loadPrepareData(corpus_name, save_dir):
    try:
        print("Start loading training data ...")
        vocab = Vocabulary(corpus_name, save_dir)
        train_pairs = torch.load(os.path.join(save_dir, 'train_pairs.tar'))
        valid_pairs = torch.load(os.path.join(save_dir, 'valid_pairs.tar'))
        test_pairs = torch.load(os.path.join(save_dir, 'test_pairs.tar'))

    except FileNotFoundError:
        print("Saved data not found, start preparing training data ...")
        vocab = Vocabulary(corpus_name, save_dir)
        train_pairs, valid_pairs, test_pairs = prepareData(vocab, corpus_name, save_dir)
    return vocab, train_pairs, valid_pairs, test_pairs


