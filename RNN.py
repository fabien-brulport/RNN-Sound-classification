from doctest import _OutputRedirectingPdb

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import pickle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
import time


class RNN:
    def __init__(self):
        self.learning_rate = 0.0005
        self.display_step = 100
        self.hm_epochs = 10
        self.n_classes = None
        self.batch_size = 128
        self.chunk_size = 20
        self.n_chunks = 41
        self.rnn_sizes = [128, 64]
        self.load = False
        self.export_dir = './networks/'
        self.model_name = 'multiple'

    def train(self):
        with open('./input/mfcc.p', 'rb') as fp:
            t = pickle.load(fp)
            tr_features, tr_labels = t
        # Encode string label to int
        enc = LabelEncoder()
        tr_labels = enc.fit_transform(tr_labels)
        # Encode int label to one hot [0, 0, 1, 0, 0]
        enc = OneHotEncoder()
        tr_labels = enc.fit_transform(tr_labels.reshape(-1, 1)).toarray()
        self.n_classes = tr_labels.shape[1]

        # Shuffle the train set
        idx = np.random.permutation(len(tr_labels))
        tr_labels = tr_labels[idx]
        tr_features = tr_features[idx]

        # Split in train and test set
        sep = int(0.9 * len(tr_labels))
        X = tr_features[:sep]
        Y = tr_labels[:sep]
        X_test = tr_features[sep:]
        y_test = tr_labels[sep:]

        # RNN
        tf.reset_default_graph()

        x = tf.placeholder("float", [None, self.n_chunks, self.chunk_size])
        y = tf.placeholder("float", [None, self.n_classes])
        keep_prob = tf.placeholder("float", name='keep_prob')

        prediction = self.build_rnn(x, keep_prob)

        # Define loss and optimizer
        loss_f = -tf.reduce_sum(y * tf.log(prediction + 1e-10))
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(loss_f)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Initializing the variables
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            saver = tf.train.Saver()
            if self.load:
                saver.restore(session, self.export_dir + self.model_name)
            else:
                session.run(init)

            for epoch in range(self.hm_epochs):
                print('\nEpoch {} ------------------------'.format(epoch))
                t0 = time.time()
                for steps in range(Y.shape[0] // self.batch_size):
                    offset = (steps * (epoch + 1) * self.batch_size) % (
                        Y.shape[0] - self.batch_size)
                    batch_x = X[offset:(offset + self.batch_size), :, :]
                    batch_y = Y[offset:(offset + self.batch_size), :]
                    _, c = session.run([optimizer, loss_f],
                                       feed_dict={x: batch_x, y: batch_y,
                                                  keep_prob: 0.5})

                    if steps % self.display_step == 0:
                        # Calculate batch accuracy
                        acc = session.run(accuracy,
                                          feed_dict={x: batch_x, y: batch_y,
                                                     keep_prob: 1})
                        # Calculate batch loss
                        loss = session.run(loss_f,
                                           feed_dict={x: batch_x, y: batch_y,
                                                      keep_prob: 1})
                        print("Iter " + str(steps) + " / " + str(
                            Y.shape[
                                0] // self.batch_size) + ", Minibatch Loss= " +
                              "{:.6f}".format(loss) + ", Training Accuracy= " +
                              "{:.5f}".format(acc))

                print('Test accuracy: ',
                      round(session.run(accuracy,
                                        feed_dict={x: X_test, y: y_test,
                                                   keep_prob: 1}),
                            3))
                print('Epoch time: {}'.format(time.time() - t0))

            saver.save(session, self.export_dir + self.model_name)

    def build_rnn(self, x, keep_prob):
        layer = {
            'weight': tf.Variable(
                tf.random_normal([self.rnn_sizes[-1], self.n_classes])),
            'bias': tf.Variable(tf.random_normal([self.n_classes]))}
        lstm_cells = [rnn_cell.LSTMCell(rnn_size) for rnn_size in
                      self.rnn_sizes]
        drop_cells = [
            tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob) for
            lstm in lstm_cells]

        lstm = rnn_cell.MultiRNNCell(drop_cells)
        output, state = tf.nn.dynamic_rnn(lstm, x, dtype=tf.float32)
        output = tf.transpose(output, [1, 0, 2])
        output = tf.gather(output, int(output.get_shape()[0]) - 1)

        return tf.nn.softmax(
            tf.tensordot(output, layer['weight'], [[1], [0]]) + layer[
                'bias'])


if __name__ == '__main__':
    rnn = RNN()
    rnn.train()
