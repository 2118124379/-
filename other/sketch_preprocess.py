import pickle
import nltk
import stanza


def sort_and_clip(dic, top_k):
    return dict(list({k: v for k, v in sorted(dic.items(), key=lambda item: item[1],reverse=True)}.items())[:top_k])


def sentence_process(sentence):
    sentence = sentence.split(' ')
    ret_sentence = []
    for word in sentence:
        if word != '':
            ret_sentence.append(word)
    return ret_sentence

def main():
    data = pickle.load(open('reviews.pickle','rb'))
    uni_grams = dict()
    two_grams = dict()
    three_grams = dict()
    feature_cnt = dict()
    sketch_word2index = dict()
    feature_sketch = [dict()]*54
    feature_to_index = dict()
    sketch_word_cnt = 0
    feature_index = 0

    sketch2index = dict()
    index2sketch = dict()
    index2topic = dict()
    word2index = dict()
    review_rev = dict()

    for example in data:

        for feature_now in example['fine_aspect']:
            if feature_to_index.get(feature_now) is None:
                feature_to_index[feature_now] = []
            feature_to_index[feature_now].append(feature_index)


            if feature_cnt.get(feature_now) is None:
                feature_cnt[feature_now] = 1
            else:
                feature_cnt[feature_now] += 1
            if sketch_word2index.get(feature_now) is None:
                sketch_word2index[feature_now] = sketch_word_cnt
                sketch_word_cnt += 1
        feature_index += 1
        sentence = sentence_process(example['text'])
        # for word in sentence:
            # if sketch_word2index.get(word) is None:
        two_g = nltk.ngrams(sentence,2)
        three_g = nltk.ngrams(sentence,3)
        for word in sentence:
            if uni_grams.get(word) is None:
                uni_grams[word] = 1
            else:
                uni_grams[word] += 1
        for word in two_g:
            if two_grams.get(word) is None:
                two_grams[word] = 1
            else:
                two_grams[word] += 1
        for word in three_g:
            if three_grams.get(word) is None:
                three_grams[word] = 1
            else:
                three_grams[word] += 1

    count_feature_cnt = len(feature_cnt)
    uni_grams = sort_and_clip(uni_grams, 50)
    two_grams = sort_and_clip(two_grams, 200)
    three_grams = sort_and_clip(three_grams, 200)
    feature_used = min(count_feature_cnt,50) # how many feature to use
    feature_cnt = sort_and_clip(feature_cnt, feature_used)  # Only use top 5 features

    real_data = []
    feature_tops = dict()
    feature_tops['<sos>'] = 0
    feature_tops['<eos>'] = 1
    feature_tops['<unk>'] = 2
    feature_tops['<pad>'] = 3

    index2topic[0] = '<sos>'
    index2topic[1] = '<eos>'
    index2topic[2] = '<unk>'
    index2topic[3] = '<pad>'

    sketch2index['<sos>'] = 0
    sketch2index['<eos>'] = 1
    sketch2index['<unk>'] = 2
    sketch2index['<pad>'] = 3

    index2sketch[0] = '<sos>'
    index2sketch[1] = '<eos>'
    index2sketch[2] = '<unk>'
    index2sketch[3] = '<pad>'

    word2index['<sos>'] = 0
    word2index['<eos>'] = 1
    word2index['<unk>'] = 2
    word2index['<pad>'] = 3

    review_rev[0] = '<sos>'
    review_rev[1] = '<eos>'
    review_rev[2] = '<unk>'
    review_rev[3] = '<pad>'
    for rank, item in enumerate(feature_cnt.items()):
        feature_name = item[0]
        feature_rank = rank + 4

        feature_tops[feature_name] = feature_rank
        index2topic[feature_rank] = feature_name
        for index in feature_to_index[feature_name]:
            example = data[index]
            sentence = example['text']
            words = sentence_process(sentence)
            for word in words:
                if feature_sketch[feature_rank].get(word) is None:
                    feature_sketch[feature_rank][word] = 1
                else:
                    feature_sketch[feature_rank][word] += 1


    # print('uni_grams', uni_grams)
    # print('two_grams', two_grams)
    # print('three_grams', three_grams)
    # print('feature_cnt', feature_cnt)
    # print('feature count:', count_feature_cnt)
    # exit(0)

    for i in range(4,feature_used+4):
        feature_sketch[i] = sort_and_clip(feature_sketch[i], 50)  # Only use top 50 word

    nlp = stanza.Pipeline(lang='en',processors='tokenize,pos',tokenize_pretokenized=True,tokenize_no_ssplit=True,
                          tokenize_batch_size=1024,
                          pos_batch_size=2000000)
    print("feature sketch:{}".format(len(feature_sketch[0])))

    sketch_word_index = 4
    review_word_index = 4

    ccount = 0
    for example in data:
        ccount += 1
        if ccount % 10000==0:
            print('ccount=',ccount)
        fine_aspect = []
        for feature_now in example['fine_aspect']:
            if feature_cnt.get(feature_now) is not None:
                fine_aspect.append(feature_now)
        if len(fine_aspect)<2:
            continue
        example['fine_aspect'] = fine_aspect
        sentence = example['text']
        words = sentence_process(sentence)
        mask = [0] * len(words)
        two_g = [*nltk.ngrams(words, 2)]
        three_g = [*nltk.ngrams(words, 3)]
        doc = nlp(sentence).to_dict()
        doc = doc[0]
        # print(doc)
        # print(words)
        for word in words:
            if word2index.get(word) is None:
                word2index[word] = review_word_index
                review_rev[review_word_index] = word
                review_word_index += 1
        for feature_now in example['fine_aspect']:
            feature_rank = feature_tops[feature_now]

            for i in range(len(words)):
                if feature_sketch[feature_rank].get(words[i]) is not None or uni_grams.get(words[i]) is not None:
                    mask[i] = 1
                if i < len(two_g) and two_grams.get(two_g[i]) is not None:
                    mask[i] = mask[i+1] = 1
                if i < len(three_g) and three_grams.get(three_g[i]) is not None:
                    mask[i] = mask[i+1] = mask[i+2] = 1
        sketch = []

        for i in range(len(words)):
            if mask[i] == 1:
                sketch.append(words[i])
            else:
                sketch.append(doc[i]['xpos'])
            if sketch2index.get(sketch[-1]) is None:
                sketch2index[sketch[-1]] = sketch_word_index
                index2sketch[sketch_word_index] = sketch[-1]
                sketch_word_index += 1

        example['sketch'] = ' '.join(sketch)


    pickle.dump(data,open('sketch_reviews.pickle','wb'))
    pickle.dump(sketch2index, open('./sketch2index.pkl','wb'))
    pickle.dump(feature_tops, open('./topic2index.pkl', 'wb'))
    pickle.dump(index2sketch, open('./sketch_rev.pkl', 'wb'))
    pickle.dump(index2topic, open('./topic_rev.pkl', 'wb'))
    pickle.dump(word2index, open('./review.pkl','wb'))
    pickle.dump(review_rev, open('./review_rev.pkl', 'wb'))
    print('sketch vob size:{}'.format(len(sketch2index)))


main()


