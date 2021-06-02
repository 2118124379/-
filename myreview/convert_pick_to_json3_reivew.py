import pickle
import json
import random


def main():
    data = pickle.load(open('./sketch_reviews.pickle','rb'))
    new_data = []
    for example in data:
        if example.get('sketch') is None:
            continue
        new_data.append({
            'reviewerID':example['user'],
            'asin':example['item'],
            'overall':example['rating'],
            'topic_tok':example['template'][0],
            'sketchText':example['sketch'],
            'reviewText':example['template'][2],
        })
    # random.shuffle(new_data)
    usz = len(new_data)
    dev = new_data[:usz//10]
    test = new_data[usz//10:usz//10*2]
    train = new_data[usz//10*2:]
    json.dump(dev,open('./valid_tok.json','w'))
    json.dump(train, open('./train_tok.json','w'))
    json.dump(test, open('./test_tok.json','w'))

main()