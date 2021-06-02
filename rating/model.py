class NETE_r:
    def __init__(self, train_tuple_list, user_num, item_num, rating_layer_num=4, latent_dim=200, learning_rate=0.0001,
                 batch_size=128, reg_rate=0.0001):

        self.train_tuple_list = train_tuple_list
        self.batch_size = batch_size

        graph = tf.Graph()
        with graph.as_default():
            # input
            self.user_id = tf.placeholder(dtype=tf.int32, shape=[None], name='user_id')  # (batch_size,)
            self.item_id = tf.placeholder(dtype=tf.int32, shape=[None], name='item_id')
            self.rating = tf.placeholder(dtype=tf.float32, shape=[None], name='rating')

            # embeddings
            user_embeddings = tf.get_variable('user_embeddings', shape=[user_num, latent_dim], initializer=tf.random_uniform_initializer(-0.1, 0.1))
            item_embeddings = tf.get_variable('item_embeddings', shape=[item_num, latent_dim], initializer=tf.random_uniform_initializer(-0.1, 0.1))

            # rating prediction
            user_feature = tf.nn.embedding_lookup(user_embeddings, self.user_id)  # (batch_size, latent_dim)
            item_feature = tf.nn.embedding_lookup(item_embeddings, self.item_id)
            hidden = tf.concat(values=[user_feature, item_feature], axis=1)  # (batch_size, latent_dim * 2)
            for k in range(rating_layer_num):
                hidden = tf.layers.dense(inputs=hidden, units=latent_dim * 2, activation=tf.nn.sigmoid, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                         bias_initializer=tf.constant_initializer(0.0), name='layer-{}'.format(k))  # (batch_size, latent_dim * 2)
            prediction = tf.layers.dense(inputs=hidden, units=1, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                         bias_initializer=tf.constant_initializer(0.0), name='prediction')  # (batch_size, 1)
            self.predicted_rating = tf.reshape(prediction, shape=[-1])  # (batch_size,)
            rating_loss = tf.losses.mean_squared_error(self.rating, self.predicted_rating)

            regularization_cost = tf.reduce_sum([tf.nn.l2_loss(v) for v in tf.trainable_variables()])

            # optimization
            self.total_loss = rating_loss + reg_rate * regularization_cost
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.total_loss)

            init = tf.global_variables_initializer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=graph, config=config)
        self.sess.run(init)

    def train_one_epoch(self):
        sample_num = len(self.train_tuple_list)
        index_list = list(range(sample_num))
        random.shuffle(index_list)

        total_loss = 0

        step_num = int(math.ceil(sample_num / self.batch_size))
        for step in range(step_num):
            start = step * self.batch_size
            offset = min(start + self.batch_size, sample_num)

            user = []
            item = []
            rating = []
            for idx in index_list[start:offset]:
                x = self.train_tuple_list[idx]
                user.append(x[0])
                item.append(x[1])
                rating.append(x[2])
            user = np.asarray(user, dtype=np.int32)
            item = np.asarray(item, dtype=np.int32)
            rating = np.asarray(rating, dtype=np.float32)

            feed_dict = {self.user_id: user,
                         self.item_id: item,
                         self.rating: rating}
            _, loss = self.sess.run([self.optimizer, self.total_loss], feed_dict=feed_dict)
            total_loss += loss * (offset - start)

        return total_loss / sample_num

    def validate(self, tuple_list):
        sample_num = len(tuple_list)

        total_loss = 0

        step_num = int(math.ceil(sample_num / self.batch_size))
        for step in range(step_num):
            start = step * self.batch_size
            offset = min(start + self.batch_size, sample_num)

            user = []
            item = []
            rating = []
            for x in tuple_list[start:offset]:
                user.append(x[0])
                item.append(x[1])
                rating.append(x[2])
            user = np.asarray(user, dtype=np.int32)
            item = np.asarray(item, dtype=np.int32)
            rating = np.asarray(rating, dtype=np.float32)

            feed_dict = {self.user_id: user,
                         self.item_id: item,
                         self.rating: rating}
            loss = self.sess.run(self.total_loss, feed_dict=feed_dict)
            total_loss += loss * (offset - start)

        return total_loss / sample_num

    def get_prediction(self, tuple_list):
        sample_num = len(tuple_list)
        rating_prediction = []

        step_num = int(math.ceil(sample_num / self.batch_size))
        for step in range(step_num):
            start = step * self.batch_size
            offset = min(start + self.batch_size, sample_num)

            user = []
            item = []
            for x in tuple_list[start:offset]:
                user.append(x[0])
                item.append(x[1])
            user = np.asarray(user, dtype=np.int32)
            item = np.asarray(item, dtype=np.int32)

            feed_dict = {self.user_id: user,
                         self.item_id: item}
            rating_p = self.sess.run(self.predicted_rating, feed_dict=feed_dict)
            rating_prediction.extend(rating_p)

        return np.asarray(rating_prediction, dtype=np.float32)

    def get_prediction_ranking(self, top_k, users_test, item_num):
        user2items_train = {}
        for x in self.train_tuple_list:
            u = x[0]
            i = x[1]
            if u in user2items_train:
                user2items_train[u].add(i)
            else:
                user2items_train[u] = {i}

        user2items_top = {}
        for u in users_test:
            items = set(list(range(item_num))) - user2items_train[u]
            tuple_list = [[u, i] for i in items]
            predicted = self.get_prediction(tuple_list)
            item2rating = {}
            for i, p in zip(items, predicted):
                rating = p
                if rating == 0:
                    rating = random.random()
                item2rating[i] = rating
            top_list = heapq.nlargest(top_k, item2rating, key=item2rating.get)
            user2items_top[u] = top_list

        return user2items_top