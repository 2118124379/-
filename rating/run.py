from model import NETE_r
from load_data import load_data
from utils import *
import argparse
import sys


parser = argparse.ArgumentParser()
parser.add_argument('-gd', '--gpu_device', type=str, help='device(s) on GPU, default=0', default='0')
parser.add_argument('-dp', '--data_path', type=str, help='path for loading pickle data', default=None)
parser.add_argument('-dr', '--data_ratio', type=str, help='ratio of train:validation:test', default='8:1:1')
parser.add_argument('-id', '--index_dir', type=str, help='create new indexes if the directory is empty, otherwise load indexes', default=None)

parser.add_argument('-rn', '--rating_layer_num', type=int, help='rating prediction layer number, default=4', default=4)
parser.add_argument('-ld', '--latent_dim', type=int, help='latent dimension of users and items, default=200', default=200)
parser.add_argument('-wd', '--word_dim', type=int, help='dimension of word embeddings, default=200', default=200)
parser.add_argument('-rd', '--rnn_dim', type=int, help='dimension of RNN hidden states, default=256', default=256)
parser.add_argument('-sm', '--seq_max_len', type=int, help='seq max len of a text, default=15', default=15)
parser.add_argument('-wn', '--max_word_num', type=int, help='number of words in vocabulary, default=20000', default=20000)
parser.add_argument('-dk', '--dropout_keep', type=float, help='dropout ratio in RNN, default=0.8', default=0.8)

parser.add_argument('-en', '--max_epoch_num', type=int, help='max epoch number, default=100', default=100)
parser.add_argument('-bs', '--batch_size', type=int, help='batch size, default=128', default=128)
parser.add_argument('-lr', '--learning_rate', type=float, help='learning rate, default=0.0001', default=0.0001)
parser.add_argument('-rr', '--reg_rate', type=float, help='regularization rate, default=0.0001', default=0.0001)

parser.add_argument('-pf', '--use_predicted_feature', type=int, help='use predicted features from PMI when testing, 0 means no, otherwise yes', default=0)
parser.add_argument('-pp', '--prediction_path', type=str, help='the path for saving predictions', default=None)
parser.add_argument('-tk', '--top_k', type=int, help='select top k to evaluate, default=5', default=5)
args = parser.parse_args()


print('-----------------------------ARGUMENTS-----------------------------')
for arg in vars(args):
    value = getattr(args, arg)
    if value is None:
        value = ''
    print('{:30} {}'.format(arg, value))
print('-----------------------------ARGUMENTS-----------------------------')


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_device
# if args.data_path is None:
#     sys.exit(get_now_time() + 'provide data_path for loading data')
# if args.index_dir is None:
#     sys.exit(get_now_time() + 'provide index_dir for saving and loading indexes')
# if args.prediction_path is None:
#     sys.exit(get_now_time() + 'provide prediction_path for saving predicted text')
# if not os.path.exists(args.index_dir) or len(os.listdir(args.index_dir)) == 0:
#     split_data(args.data_path, args.index_dir, args.data_ratio)

with open(os.path.join('./user.pkl'), 'rb') as fp:
    user2index = pickle.load(fp)
with open(os.path.join('./item.pkl'), 'rb') as fp:
    item2index = pickle.load(fp)
train_tuple_list, validation_tuple_list, test_tuple_list,max_rating,min_rating, user2items_test = load_data(user2index, item2index)
mean_r = (max_rating + min_rating) / 2


model_r = NETE_r(train_tuple_list, len(user2index), len(item2index), args.rating_layer_num, args.latent_dim, args.learning_rate,
                 args.batch_size, args.reg_rate)
# first train rating prediction module
previous_loss = 1e10
rating_validation, rating_test = None, None
for en in range(1, args.max_epoch_num + 1):
    print(get_now_time() + 'iteration {}'.format(en))

    train_loss = model_r.train_one_epoch()
    print(get_now_time() + 'loss on train set: {}'.format(train_loss))
    validation_loss = model_r.validate(validation_tuple_list)
    print(get_now_time() + 'loss on validation set: {}'.format(validation_loss))

    # early stop setting
    if validation_loss > previous_loss:
        print(get_now_time() + 'early stopped')
        break
    previous_loss = validation_loss

    rating_validation = model_r.get_prediction(validation_tuple_list)
    rating_test = model_r.get_prediction(test_tuple_list)

# evaluating
predicted_rating = []
for (x, r_p) in zip(test_tuple_list, rating_test):
    predicted_rating.append((x[2], r_p))
test_rmse = root_mean_square_error(predicted_rating, max_rating, min_rating)
print(get_now_time() + 'RMSE on test set: {}'.format(test_rmse))
test_mae = mean_absolute_error(predicted_rating, max_rating, min_rating)
print(get_now_time() + 'MAE on test set: {}'.format(test_mae))

# user2items_top = model_r.get_prediction_ranking(args.top_k, list(user2items_test.keys()), len(item2index))
# ndcg = evaluate_ndcg(user2items_test, user2items_top)
# print(get_now_time() + 'NDCG on test set: {}'.format(ndcg))
# precision, recall, f1 = evaluate_precision_recall_f1(user2items_test, user2items_top)
# print(get_now_time() + 'Precision on test set: {}'.format(precision))
# print(get_now_time() + 'HR on test set: {}'.format(recall))
# print(get_now_time() + 'F1 on test set: {}'.format(f1))

for i in range(len(test_tuple_list)):
    print("-"*30)
    print(test_tuple_list[i],rating_test[i])
    print("-" * 30)
# replace the ground-truth sentiments with predicted ratings
new_validation_list = []
new_test_list = []
for (x, r_p) in zip(validation_tuple_list, rating_validation):
    x[-1] = r_p
    new_validation_list.append(x)
for (x, r_p) in zip(test_tuple_list, rating_test):
    x[-1] = r_p
    new_test_list.append(x)
validation_tuple_list = new_validation_list
test_tuple_list = new_test_list

