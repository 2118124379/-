import pickle
data = pickle.load(open('./sketch/sketch_predict.pkl','rb'))
rev = pickle.load(open('topic_rev.pkl','rb'))
print(rev)
# pickle.dump(data,open('reviews.pickle','wb'))