import pickle
import json
import random


def main():
    data = pickle.load(open('./sketch_reviews.pickle','rb'))
    new_data = []
    user_count = 0
    user2index = {}
    for example in data:
        if example.get('sketch') is None:
            continue
        if example['user'] not in user2index.keys():
            user2index[example['user']] = user_count
            user_count += 1
            new_data.append([{
                'reviewerID': example['user'],
                'asin': example['item'],
                'overall': example['rating'],
                'topic_tok': example['fine_aspect'],
                'sketchText': example['sketch'],
                'text': example['text']
            }])
        else :
            new_data[user2index[example['user']]].append({
                'reviewerID': example['user'],
                'asin': example['item'],
                'overall': example['rating'],
                'topic_tok': example['fine_aspect'],
                'sketchText': example['sketch'],
                'text': example['text']
            })
    dev = []
    test = []
    train = []
    c = 0
    for i in range(user_count):
        random.shuffle(new_data[i])
        usz = len(new_data[i])
        if usz<5:
            c+=1
            continue
        if usz>10:
            dev.extend(new_data[i][:usz // 10])
            test.extend(new_data[i][usz // 10:usz // 10 * 2])
            train.extend(new_data[i][usz // 10 * 2:])
        else:
            dev.extend(new_data[i][:1])
            test.extend(new_data[i][1:2])
            train.extend(new_data[i][2:])
    user_dict = {}
    rev_user_dict = {}
    item_dict = {}
    rev_item_dict = {}
    uuser_count = 0
    item_count = 0

    for i in range(user_count):
        if len(new_data[i])<5:
            continue
        for example in new_data[i]:
            if user_dict.get(example['reviewerID']) is None:
                user_dict[example['reviewerID']] = uuser_count
                rev_user_dict[uuser_count] = example['reviewerID']
                uuser_count += 1
            if item_dict.get(example['asin']) is None:
                item_dict[example['asin']] = item_count
                rev_item_dict[item_count] = example['asin']
                item_count += 1
    print(uuser_count,item_count)
    pickle.dump(user_dict, open('./user.pkl', 'wb'))
    pickle.dump(item_dict, open('./item.pkl', 'wb'))
    pickle.dump(rev_user_dict, open('./user_rev.pkl', 'wb'))
    pickle.dump(rev_item_dict, open('./item_rev.pkl', 'wb'))


    # random.shuffle(new_data)
    print(c)
    print(len(dev),len(test),len(train))
    json.dump(dev,open('./valid_tok.json','w'))
    json.dump(train, open('./train_tok.json','w'))
    json.dump(test, open('./test_tok.json','w'))

main()