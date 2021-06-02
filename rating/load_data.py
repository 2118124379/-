from sklearn.feature_extraction.text import CountVectorizer
from utils import *
import json


def load_data(user_dict,item_dict):
    # collect all users id and items id





    # # convert id to array index
    # user_list = list(user_set)
    # item_list = list(item_set)
    # user2index = {x: i for i, x in enumerate(user_list)}
    # item2index = {x: i for i, x in enumerate(item_list)}

    max_rating = 5
    min_rating = 1
    def format_data(data_type):
        tuple_list = []
        with open(data_type + '_tok.json', 'r') as f:
            for l in json.load(f):
                user = l['reviewerID']
                item = l['asin']
                rating = int(l['overall'])

                user = user_dict[user]
                item = item_dict[item]
                # if max_rating < rating:
                #     max_rating = rating
                # if min_rating > rating:
                #     min_rating = rating
        # fea_set = set()
        # for idx in indexes:
        #     rev = reviews[idx]
        #     u = user2index[rev['user']]
        #     i = item2index[rev['item']]
        #     r = rev['rating']
                tuple_list.append([user, item, rating])
        return tuple_list

    train_tuple_list = format_data('train')
    validation_tuple_list = format_data('valid')
    test_tuple_list = format_data('test')
    user2items_test = {}
    for x in test_tuple_list:
        u = x[0]
        i = x[1]
        if u in user2items_test:
            user2items_test[u].add(i)
        else:
            user2items_test[u] = {i}



    return train_tuple_list, validation_tuple_list, test_tuple_list, 5, 1, user2items_test


# def get_word2index(doc_list, max_word_num):
#     def split_words_by_space(text):
#         return text.split(' ')
#
#     vectorizer = CountVectorizer(max_features=max_word_num, analyzer=split_words_by_space)
#     vectorizer.fit(doc_list)
#     word_list = vectorizer.get_feature_names()
#     word_list.extend(['<UNK>', '<GO>', '<EOS>', '<PAD>'])
#     word2index = {w: i for i, w in enumerate(word_list)}
#
#     return word2index, word_list
