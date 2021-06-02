import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pack_padded_sequence
from masked_cross_entropy import *
import itertools
import random
import math
import sys
import os
from load import loadPrepareData
from load import SOS_ID, EOS_ID, PAD_ID, UNK_ID
import pickle
import logging
import numpy as np
logging.basicConfig(level=logging.INFO)
    
#############################################
# Prepare Training Data
#############################################
def indexesFromTopic(vocab, sentence):
    ids = []
    for words in sentence:
        idx = []
        for word in words:
            idx.append(vocab.topic2idx[word])
        ids.append(idx)
    return ids

def indexesFromSketch(vocab, sentence):
    ids = []
    for word in sentence:
        word = word
        ids.append(vocab.sketch2idx[word])
    return ids

def indexesFromReview(vocab, sentence):
    ids = []
    for word in sentence:
        word = word
        if word in vocab.word2idx:
            ids.append(vocab.word2idx[word])
        else:
            ids.append(UNK_ID)
    return ids

# batch_first: true -> false, i.e. shape: seq_len * batch
def Padding(l, fillvalue=PAD_ID):
    mx = 0
    for i in l:
        mx=max(mx,len(i))
    for i in range(len(l)):
        while len(l[i]) < mx:
            l[i].append(fillvalue)
    return l

def binaryMatrix(l, value=PAD_ID):
    m = []
    for i in range(len(l)):  # 
        m.append([])
        for j in range(len(l[i])):
            if l[i][j] >= 3 and l[i][j] <= 37:  # part-of-speech idx
                m[i].append(1)
            else:
                m[i].append(0) # mask = 1 if not padding
    return m

# return attribute index and input pack_padded_sequence
def inputVar(data, vocab, evaluation=False):
    
    attr = [[d[0], d[1], d[2]] for d in data]  # length: batch 
    attrVar = Variable(torch.LongTensor(attr), volatile=evaluation) # (batch_size, attribute_num), in our case it is 2

    return attrVar 

def ReviewVar(rb, pb, tb, vocab):

    topic = indexesFromTopic(vocab, tb)
    # topic = [indexesFromTopic(vocab, b) for b in tb]
    # manually padding
    review_input = []
    review_output = []
    sketch_output = []
    for i in range(len(rb)):
        inp = []
        out = []
        pat = []
        top = []
        for j in range(len(rb[i])):
            review = rb[i][j]
            sketch = pb[i][j]
            # top += [topic[i]] * (len(review) - 1)
            # inp += indexesFromReview(vocab, review[:-1])
            # out += indexesFromReview(vocab, review[1:])
            # pat += indexesFromSketch(vocab, sketch[1:])
            while len(topic[i]) < (len(review) -1):
                topic[i].append(vocab.topic2idx['<pad>'])
            review_input.append(indexesFromReview(vocab, review[:-1]))
            review_output.append(indexesFromReview(vocab, review[1:]))
            sketch_output.append(indexesFromSketch(vocab, sketch[1:]))
    for i in range(len(review_input)):
        for j in review_input[i]:
            if j==PAD_ID:
                print('pas')
                exit(0)
        if len(review_input[i]) != len(review_output[i]):
            print(len(review_input[i]),len(review_output[i]))
            exit(0)
    topicList = Padding(topic)
    inpadList = Padding(review_input)
    outpadList = Padding(review_output)
    sketchList = Padding(sketch_output)
    mask = binaryMatrix(sketchList)
    # print(review_input)
    for i in range(len(inpadList)):
        for j in range(len(inpadList[i])):
            if int(inpadList[i][j]) == PAD_ID and int(outpadList[i][j]) != PAD_ID:
                print("222222")
                exit(0)
    topicVar = Variable(torch.LongTensor(topicList))
    inpadVar = Variable(torch.LongTensor(inpadList))
    outpadVar = Variable(torch.LongTensor(outpadList))
    sketchVar = Variable(torch.LongTensor(sketchList))
    mask = Variable(torch.BoolTensor(mask))

    return topicVar, sketchVar, inpadVar, outpadVar, mask

# pair_batch is a list of (input, output) with length batch_size
# sort list of (input, output) pairs by output length, reverse input
# return input, lengths for pack_padded_sequence, output_variable, mask
def batch2TrainData(vocab, pair_batch, evaluation=False):
    # pair_batch.sort(key=lambda x: len(x[1]), reverse=True) # sort on topic length
    input_batch, topic_batch, sketch_batch, review_batch = [], [], [], []
    for i in range(len(pair_batch)):
        input_batch.append(pair_batch[i][0])
        # print(pair_batch[i][0])
        # exit(0)
        topic_batch.append(pair_batch[i][1])
        sketch_batch.append(pair_batch[i][2])
        review_batch.append(pair_batch[i][3])
    attr_input = inputVar(input_batch, vocab, evaluation=evaluation)
    topic_input, sketch_output, review_input, review_output, mask= ReviewVar(review_batch, sketch_batch, topic_batch, vocab)
    return attr_input, topic_input, sketch_output, review_input, review_output,mask

