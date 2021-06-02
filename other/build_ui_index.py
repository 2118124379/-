import pickle
data = pickle.load(open('./sketch_reviews.pickle','rb'))
user_dict = {}
rev_user_dict = {}
item_dict = {}
rev_item_dict = {}
user_count = 0
item_count = 0

for example in data:
    if example.get('sketch') is None: continue
    if user_dict.get(example['user']) is None:
        user_dict[example['user']] = user_count
        rev_user_dict[user_count] = example['user']
        user_count+=1
    if item_dict.get(example['item']) is None:
        item_dict[example['item']] = item_count
        rev_item_dict[item_count] = example['item']
        item_count += 1

pickle.dump(user_dict,open('./topic/data/electronic/user.pkl', 'wb'))
pickle.dump(item_dict, open('./topic/data/electronic/item.pkl', 'wb'))
pickle.dump(rev_user_dict,open('./topic/data/electronic/user_rev.pkl','wb'))
pickle.dump(rev_item_dict, open('./topic/data/electronic/item_rev.pkl', 'wb'))