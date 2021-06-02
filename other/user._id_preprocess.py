import json
import pickle

filename = 'user.json'
data = json.load(open(filename,'r'))
user_dict = {}
index = 0
for user in data:
    user_dict[user['user']] = index
pickle.dump(user_dict, open('user.pickle','wb'))

filename = 'item.json'
data = json.load(open(filename,'r'))
item_dict = {}
index = 0
for item in data:
    user_dict[item['name']] = index
pickle.dump(user_dict, open('item.pickle','wb'))

