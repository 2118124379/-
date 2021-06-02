import torch
from torch.autograd import Variable
import random
from model import *
from util import *
import sys
import os
from fun import *
from masked_cross_entropy import *
import itertools
import random
import math
from load import SOS_ID, EOS_ID, PAD_ID
from model import ReviewAttnDecoderRNN, TopicAttnDecoderRNN, SketchAttnDecoderRNN, AttributeEncoder
import pickle
import logging
import  mymodel
logging.basicConfig(level=logging.INFO)
USE_CUDA = torch.cuda.is_available()

MAX_TOPIC = 10
MIN_TOPIC = 2
pos = ['NN', 'JJ', 'JJS', 'VBP', 'NNS', 'MD', 'VB', 'IN', 'VBN', 'VBG', 'VBD', 'WDT', 'RB', 'VBZ', 'WP', 'PRP$', 'DT',
         'PRP', 'CD', 'JJR', 'RP', 'WRB', 'CC', 'RBR', 'FW', 'NNP', 'PDT', 'UH', 'WP$', 'RBS', 'TO', 'SYM', 'LS', 'NNPS', 'EX']


class Hypothesis(object):
    def __init__(self, tokens, log_probs, state):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state

    def extend(self, token, log_prob, state):
        return Hypothesis(tokens = self.tokens + [token],
                        log_probs = self.log_probs + [log_prob],
                        state = state)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property 
    def log_prob(self):
        return sum(self.log_probs)

    @property 
    def avg_log_prob(self):
        return self.log_prob / len(self.tokens)
 
def sort_hyps(hyps):
    return sorted(hyps, key=lambda h: h.log_prob, reverse=True)

def greedy_decoder(model, src_input, temp_input, start_symbol,max_length):
    """
    For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
    target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
    Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target input
    """

    dec_input = torch.zeros(1, max_length).type_as(src_input.data)
    topic_size = torch.tensor([len(i) for i in src_input])
    sketch_size = torch.tensor([len(i) for i in temp_input])
    review_size = torch.tensor([len(i) for i in dec_input])
    if USE_CUDA:
        src_input = src_input.cuda()
        temp_input = temp_input.cuda()
        dec_input = dec_input.cuda()
        topic_size = topic_size.cuda()
        sketch_size = sketch_size.cuda()
        review_size = review_size.cuda()
    src_outputs, src_self_attns = model.src_encoder(src_input,topic_size)
    temp_outputs, temp_self_attns = model.temp_encoder(temp_input,sketch_size)
    next_symbol = start_symbol
    for i in range(0, max_length):
        dec_input[0][i] = next_symbol
        dec_outputs, _, _, _ = model.decoder(dec_input, review_size, src_outputs, temp_outputs)
        projected = model.linear(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[i]
        next_symbol = next_word.item()
    return dec_input

def beam_decode(review_decoder, topics, sketchs, vocab):
    decoded_reviews = []
    for ti in range(len(topics)):
        nti = vocab.topic2idx[topics[ti]]
        topic = Variable(torch.LongTensor([[nti]]))
        topic = topic.cuda() if USE_CUDA else topic

        sketch_tokens=[vocab.sketch2idx[t] for t in sketchs[ti]]
        sketch_out = [[t] for t in sketch_tokens[1:]]

        # sketch_rnn = Variable(torch.LongTensor(sketch_out))
        sketch_rnn = Variable(torch.LongTensor([sketch_tokens[1:]]))
        sketch_rnn = sketch_rnn.cuda() if USE_CUDA else sketch_rnn

        seq_len = len(sketch_out)
        review_input = greedy_decoder(review_decoder, topic, sketch_rnn, EOS_ID, seq_len)
        # print(review_input.size())
        review_input = review_input.cuda() if USE_CUDA else review_input
        review_size = torch.tensor([len(i) for i in review_input])
        review_size = review_size.cuda() if USE_CUDA else review_size
        topic_size = torch.tensor([len(i) for i in topic])
        topic_size = topic_size.cuda() if USE_CUDA else topic_size
        sketch_size = torch.tensor([len(i) for i in sketch_rnn])
        sketch_size = sketch_size.cuda() if USE_CUDA else sketch_size
        predict, _, _, _, _, _ = review_decoder(topic, topic_size, sketch_rnn, sketch_size, review_input, review_size)
        predict = predict.data.max(2, keepdim=True)[1]
        review = [vocab.idx2word[n.item()] for n in predict.squeeze()]
        decoded_reviews.extend(review)

    return decoded_reviews


def evaluate(review_decoder, vocab, pair):
    attribute = pair[0] # (user_id, item_id, rating)
    attr_input = Variable(torch.LongTensor([attribute]), volatile=True)
    attr_input = attr_input.cuda() if USE_CUDA else attr_input

    # attribute encoder
    # topic_encoder_out, topic_encoder_hidden = topic_encoder(attr_input)
    # sketch_encoder_out, sketch_encoder_hidden = sketch_encoder(attr_input)
    # # review_encoder_out, review_encoder_hidden = review_encoder(attr_input)
    #
    # # topic embedding
    # topic_decoder_hidden = topic_encoder_hidden[:topic_decoder.n_layers]
    #
    # # sketch decoder
    # sketch_decoder_hidden = sketch_encoder_hidden[:sketch_decoder.n_layers]
    
    # review decoder
    # review_decoder_hidden = review_encoder_hidden[:review_decoder.n_layers]
    
    return beam_decode(review_decoder, pair[4], pair[5], vocab)


def evaluateRandomly(review_decoder, vocab, user_rdict, item_rdict, pairs, n_pairs, save_dir):
    path = os.path.join(save_dir, 'decode')
    if not os.path.exists(path):
        os.makedirs(path)
    f1 = open(path + "/decoded.txt", 'w')
    ids_predict = []
    feature_batch = []
    feature_test = []
    token_test = []
    for i in range(n_pairs):

        pair = pairs[i]

        user = pair[0][0]
        item = pair[0][1]
        rating = pair[0][2]
        attribute = '\t'.join([user_rdict[user], item_rdict[item], str(rating + 1)])
        feature_test.append([vocab.topic2idx[t] for t in pair[1]])
        feature_batch.append([vocab.topic2idx[t] for t in pair[4]])
        topic = " ".join(pair[1])
        sketchs = []
        for sketch in pair[2]:
            sketchs.append(" ".join(sketch[1:-1]))
        sketchs = " || ".join(sketchs)
        sketch_gen = []
        for sketch in pair[5]:
            sketch_gen.append(" ".join(sketch[1:-1]))
        sketch_gen = " || ".join(sketch_gen)
        reviews = []
        token = []
        for review in pair[3]:
            reviews.append(" ".join(review[1:-1]))
            token.extend(review[1:])
        token_test.append(token)
        reviews = " || ".join(reviews)
        print("=============================================================")
        print('Attribute > ', attribute)
        print('Topic > ', topic)
        print('sketch > ', sketchs)
        print('Review > ', reviews)

        f1.write(
            'Attribute: ' + attribute + '\n' + 'Topic: ' + topic + '\n' + 'sketch: ' + sketchs + '\n' + 'Review: ' + reviews + '\n')
        output_reviews = evaluate(review_decoder, vocab, pair)
        ids_predict.append([vocab.word2idx[t] for t in output_reviews])
        review_words = []
        for wd in output_reviews:
            if wd == "<eos>":
                review_words.append("||")
            elif "_" in wd:
                review_words.extend(wd.split("_"))
            else:
                review_words.append(wd)
        review_sentence = ' '.join(review_words[:-1])
        print('Generation topic < ', " ".join(pair[4]))
        print('Generation sketch < ', sketch_gen)
        print('Generation review < ', review_sentence)
        f1.write('Generation topic: ' + " ".join(pair[4]) + "\n")
        f1.write('Generation sketch: ' + sketch_gen + "\n")
        f1.write('Generation review: ' + review_sentence + "\n")
    f1.close()

    PUS, NUS = unique_sentence_percent(ids_predict)
    print(get_now_time() + 'USN on test set: {}'.format(NUS))
    print(get_now_time() + 'USR on test set: {}'.format(PUS))

    # DIV really takes time
    DIV = feature_diversity(feature_batch)
    print(get_now_time() + 'DIV on test set: {}'.format(DIV))
    FCR = feature_coverage_ratio(feature_batch, vocab.topic2idx)
    print(get_now_time() + 'FCR on test set: {}'.format(FCR))
    FMR = feature_matching_ratio(feature_batch, feature_test)
    print(get_now_time() + 'FMR on test set: {}'.format(FMR))

    token_predict = [ids2tokens(vocab.idx2word, ids) for ids in ids_predict]
    BLEU_1 = bleu_score(token_test, token_predict, n_gram=1, smooth=False)
    print(get_now_time() + 'BLEU-1 on test set: {}'.format(BLEU_1))
    BLEU_4 = bleu_score(token_test, token_predict, n_gram=4, smooth=False)
    print(get_now_time() + 'BLEU-4 on test set: {}'.format(BLEU_4))

    text_predict = [' '.join(tokens) for tokens in token_predict]
    text_test = [' '.join(tokens) for tokens in token_test]
    ROUGE = rouge_score(text_test, text_predict)  # a dictionary
    print(get_now_time() + 'ROUGE on test set:')
    for (k, v) in ROUGE.items():
        print('{}: {}'.format(k, v))

def runTest(corpus, n_layers, hidden_size, embed_size, attr_size, attr_num, overall, rv_modelFile, sk_modelFile, tp_modelFile, beam_size, max_length, min_length, save_dir):

    vocab, train_pairs, valid_pairs, test_pairs = loadPrepareData(corpus, save_dir)
    
    print('Building encoder and decoder ...')

    # topic encoder
    with open(os.path.join(save_dir, 'user_rev.pkl'), 'rb') as fp:
        user_rdict = pickle.load(fp)
    with open(os.path.join(save_dir, 'item_rev.pkl'), 'rb') as fp:
        item_rdict = pickle.load(fp)
        
    num_user = len(user_rdict)
    num_item = len(item_rdict)
    num_over = overall 
    
    attr_embeddings = []
    attr_embeddings.append(nn.Embedding(num_user, attr_size))
    attr_embeddings.append(nn.Embedding(num_item, attr_size))
    attr_embeddings.append(nn.Embedding(num_over, attr_size))

    if USE_CUDA:
        for attr_embedding in attr_embeddings:
            attr_embedding = attr_embedding.cuda()

    topic_encoder = AttributeEncoder(attr_size, attr_num, hidden_size, attr_embeddings, n_layers)

    # topic decoder
    topic_embedding = nn.Embedding(vocab.n_topics, embed_size)

    if USE_CUDA:
        topic_embedding = topic_embedding.cuda()

    topic_decoder = TopicAttnDecoderRNN(topic_embedding, embed_size, hidden_size, attr_size, vocab.n_topics, n_layers)

    # checkpoint = torch.load(tp_modelFile)
    # topic_encoder.load_state_dict(checkpoint['encoder'])
    # topic_decoder.load_state_dict(checkpoint['topic_decoder'])

    # use cuda
    if USE_CUDA:
        topic_encoder = topic_encoder.cuda()
        topic_decoder = topic_decoder.cuda()

    # train mode set to false, effect only on dropout, batchNorm
    topic_encoder.train(False)
    topic_decoder.train(False)

    # sketch encoder
    attr_embeddings = []
    attr_embeddings.append(nn.Embedding(num_user, attr_size))
    attr_embeddings.append(nn.Embedding(num_item, attr_size))
    attr_embeddings.append(nn.Embedding(num_over, attr_size))

    if USE_CUDA:
        for attr_embedding in attr_embeddings:
            attr_embedding = attr_embedding.cuda()

    sketch_encoder = AttributeEncoder(attr_size, attr_num, hidden_size, attr_embeddings, n_layers)

    # sketch decoder
    topic_embedding = nn.Embedding(vocab.n_topics, embed_size)
    sketch_embedding = nn.Embedding(vocab.n_sketchs, embed_size)

    if USE_CUDA:
        topic_embedding = topic_embedding.cuda()
        sketch_embedding = sketch_embedding.cuda()

    sketch_decoder = SketchAttnDecoderRNN(topic_embedding, sketch_embedding, embed_size, hidden_size, attr_size, vocab.n_sketchs, n_layers)

    checkpoint = torch.load(sk_modelFile)
    sketch_encoder.load_state_dict(checkpoint['encoder'])
    sketch_decoder.load_state_dict(checkpoint['sketch_decoder'])

    # use cuda
    if USE_CUDA:
        sketch_encoder = sketch_encoder.cuda()
        sketch_decoder = sketch_decoder.cuda()

    # train mode set to false, effect only on dropout, batchNorm
    sketch_encoder.train(False)
    sketch_decoder.train(False)
    #
    # # review encoder
    # attr_embeddings = []
    # attr_embeddings.append(nn.Embedding(num_user, attr_size))
    # attr_embeddings.append(nn.Embedding(num_item, attr_size))
    # attr_embeddings.append(nn.Embedding(num_over, attr_size))
    # if USE_CUDA:
    #     for attr_embedding in attr_embeddings:
    #         attr_embedding = attr_embedding.cuda()
    #
    # review_encoder = AttributeEncoder(attr_size, attr_num, hidden_size, attr_embeddings, n_layers)
    #
    # # birnn encoder
    # sketch_embedding = nn.Embedding(vocab.n_sketchs, embed_size)
    # if USE_CUDA:
    #     sketch_embedding = sketch_embedding.cuda()
    #
    # birnn_encoder = EncoderRNN(embed_size, hidden_size, sketch_embedding, n_layers)
    #
    # review decoder
    topic_embedding = nn.Embedding(vocab.n_topics, embed_size)
    sketch_embedding = nn.Embedding(vocab.n_sketchs, embed_size)
    word_embedding = nn.Embedding(vocab.n_words, embed_size)

    # with open(os.path.join(save_dir, 'aspect_ids.pkl'), 'rb') as fp:
    #     ids = pickle.load(fp)
    #
    # aspect_ids = nn.Embedding(vocab.n_topics-3, 100)
    # aspect_ids.weight.data.copy_(torch.from_numpy(np.array(ids)))

    if USE_CUDA:
        topic_embedding = topic_embedding.cuda()
        sketch_embedding = sketch_embedding.cuda()
        word_embedding = word_embedding.cuda()
    #
    # review_decoder = ReviewAttnDecoderRNN(topic_embedding, sketch_embedding, word_embedding, embed_size, hidden_size, attr_size, vocab.n_words, aspect_ids, n_layers)
    review_decoder = mymodel.Transformer(vocab.n_topics, 500, vocab.n_sketchs, 500, vocab.n_words, 500)
    checkpoint = torch.load(rv_modelFile)
    # review_encoder.load_state_dict(checkpoint['encoder'])
    # birnn_encoder.load_state_dict(checkpoint['birnn_encoder'])
    review_decoder.load_state_dict(checkpoint['review_decoder'])

    # use cuda
    if USE_CUDA:
        # review_encoder = review_encoder.cuda()
        # birnn_encoder = birnn_encoder.cuda()
        review_decoder = review_decoder.cuda()

    # train mode set to false, effect only on dropout, batchNorm
    # review_encoder.train(False)
    # birnn_encoder.train(False)
    review_decoder.train(False)

    evaluateRandomly(review_decoder,vocab, user_rdict, item_rdict, test_pairs, len(test_pairs), save_dir)
   
