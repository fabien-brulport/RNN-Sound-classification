import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import pickle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
import time
import pandas as pd
import random
import os
import glob
import librosa
import matplotlib.pyplot as plt


class RNN:
    def __init__(self):
        self.learning_rate = 0.0001
        self.display_step = 5
        self.hm_epochs = 4000
        self.n_classes = None
        self.batch_size = 128
        self.chunk_size = 2
        # self.n_chunks = 41
        self.rnn_sizes = [128, 128]
        self.max_length = 512*41
        self.load = False
        self.export_dir = './networks/'
        self.import_dir = './input/'
        self.model_name = 'sound'
        self.label_file = 'train.csv'
        self.init_encoders()

    def init_encoders(self):
        df = pd.read_csv(self.import_dir + self.label_file)
        df = df.set_index('fname')

        # Encode string label to int
        self.enc = LabelEncoder()
        self.enc.fit(df.label)
        self.n_classes = len(self.enc.classes_)

        # Encode int label to one hot [0, 0, 1, 0, 0]
        self.enc_hot = OneHotEncoder()
        self.enc_hot.fit(self.enc.transform(df.label).reshape(-1, 1))

    def get_bounds(self, sound):
        start = 0
        length = len(sound)
        while start + self.max_length / 2 < length:
            yield start, min(start + self.max_length, length)
            start += self.max_length

    @staticmethod
    def normalize(sound):
        return (sound - np.mean(sound)) / np.std(sound)

    def batch_generator(self):
        df = pd.read_csv(self.import_dir + self.label_file)
        df = df.set_index('fname')
        rep = 4
        batch = np.zeros(
            (self.batch_size * rep, self.max_length, self.chunk_size))
        labels = []
        i = 0
        while True:
            for fn in glob.glob(
                    os.path.join(self.import_dir + 'audio_train/', '*.wav')):

                label = df.loc[fn[-12:]].label
                sound_clip, s = librosa.load(fn)
                sound_clip = self.normalize(sound_clip)
                for start, end in self.get_bounds(sound_clip):
                    batch[i, :end - start, 0] = sound_clip[start:end]
                    batch[i, :end - start, 1] = np.abs(np.fft.fft(
                        sound_clip[start:end], norm='ortho'))
                    labels.append(label)
                    i += 1
                    if i == self.batch_size * rep:
                        idx = np.random.permutation(self.batch_size * rep)
                        l = self.enc_hot.transform(
                            self.enc.transform(labels).reshape(-1, 1)).toarray()
                        for _ in range(rep):
                            s = _ * self.batch_size
                            e = s + self.batch_size
                            yield batch[idx[s:e]], l[idx[s:e]]
                        i = 0
                        batch = np.zeros(
                            (self.batch_size * rep, self.max_length,
                             self.chunk_size))
                        labels = []

    def train(self):

        # RNN
        tf.reset_default_graph()

        x = tf.placeholder("float", [None, self.max_length, self.chunk_size])
        y = tf.placeholder("float", [None, self.n_classes])
        keep_prob = tf.placeholder("float", name='keep_prob')

        prediction = self.build_rnn(x, keep_prob)

        # Define loss and optimizer
        loss_f = -tf.reduce_sum(y * tf.log(prediction + 1e-10))
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate=self.learning_rate).minimize(loss_f)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Initializing the variables
        init = tf.global_variables_initializer()
        generator = self.batch_generator()
        with tf.Session() as session:
            saver = tf.train.Saver()
            if self.load:
                saver.restore(session, self.export_dir + self.model_name)
            else:
                session.run(init)

            t0 = time.time()
            for epoch in range(self.hm_epochs):

                batch_x, batch_y = next(generator)
                _, c = session.run([optimizer, loss_f],
                                   feed_dict={x: batch_x, y: batch_y,
                                              keep_prob: 1})

                if epoch % self.display_step == 0:
                    # Calculate batch accuracy
                    acc = session.run(accuracy,
                                      feed_dict={x: batch_x, y: batch_y,
                                                 keep_prob: 1})
                    # Calculate batch loss
                    loss = session.run(loss_f,
                                       feed_dict={x: batch_x, y: batch_y,
                                                  keep_prob: 1})

                    pred = session.run(prediction,
                                       feed_dict={x: batch_x, y: batch_y,
                                                  keep_prob: 1})
                    print("Iter " + str(epoch) + " / " + str(
                        self.hm_epochs) + ", Minibatch Loss= " +
                          "{:.6f}".format(loss) + ", Training Accuracy= " +
                          "{:.5f}".format(acc))

                    print(session.run(tf.argmax(prediction, 1),
                                      feed_dict={x: batch_x, y: batch_y,
                                                 keep_prob: 1}))
                    print(session.run(tf.argmax(y, 1),
                                      feed_dict={x: batch_x, y: batch_y,
                                                 keep_prob: 1}))

                    # print('Test accuracy: ',
                    #       round(session.run(accuracy,
                    #                         feed_dict={x: X_test, y: y_test,
                    #                                    keep_prob: 1}),
                    #             3))
                    print('{} epochs time: {}'.format(self.display_step,
                                                      time.time() - t0))
                    t0 = time.time()

            saver.save(session, self.export_dir + self.model_name)

    def build_rnn(self, x, keep_prob):
        layer = {
            'weight': tf.Variable(
                tf.truncated_normal([self.rnn_sizes[-1], self.n_classes],
                                    stddev=0.01)),
            'bias': tf.Variable(tf.constant(0.1, shape=[self.n_classes]))}
        lstm_cells = [rnn_cell.LSTMCell(rnn_size) for rnn_size in
                      self.rnn_sizes]
        drop_cells = [
            tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob) for
            lstm in lstm_cells]

        lstm = rnn_cell.MultiRNNCell(drop_cells)
        output, state = tf.nn.dynamic_rnn(lstm, x, dtype=tf.float32,
                                          sequence_length=self.length(x))
        last = self.last_relevant(output, self.length(x))

        return tf.nn.softmax(
            tf.tensordot(last, layer['weight'], [[1], [0]]) + layer[
                'bias'])

    def predict(self):
        with open('./input/mfcc_test.p', 'rb') as fp:
            ts_features = pickle.load(fp)

        with open('./input/mfcc_labels.p', 'rb') as fp:
            tr_labels = pickle.load(fp)

        with open('./input/mfcc_test_name.p', 'rb') as fp:
            fname = pickle.load(fp)

        # Encode string label to int
        enc = LabelEncoder()
        tr_labels = enc.fit_transform(tr_labels)
        self.n_classes = len(enc.classes_)

        # RNN
        tf.reset_default_graph()

        x = tf.placeholder("float", [None, self.n_chunks, self.chunk_size])
        y = tf.placeholder("float", [None, self.n_classes])
        keep_prob = tf.placeholder("float", name='keep_prob')

        prediction = self.build_rnn(x, keep_prob)
        pred = tf.argmax(prediction, 1)

        # Initializing the variables
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            saver = tf.train.Saver()
            saver.restore(session, self.export_dir + self.model_name)

            results = {'label': [], 'fname': []}
            for i, sound in enumerate(ts_features):
                if len(sound) == 0:
                    sound = np.zeros((1, self.n_chunks, self.chunk_size))
                predictions = session.run(prediction,
                                          feed_dict={x: np.array(sound),
                                                     keep_prob: 1})

                n_classe = 0
                threshold = 1
                while n_classe < 3:
                    top_labels = np.argsort(predictions, axis=1)[:,
                                 -threshold:].reshape(-1)
                    labels, counts = np.unique(top_labels, return_counts=True)
                    n_classe = len(labels)
                    threshold += 1

                top3_labels = labels[np.argsort(counts)[-3:]]
                top3_labels = " ".join(
                    [enc.inverse_transform(el) for el in top3_labels])
                results['label'].append(top3_labels)
                results['fname'].append(fname[i])

                print('Label for {}: {}'.format(i, top3_labels))

        df = pd.DataFrame(results)
        print(df.head())
        df.to_csv("output/{}.csv".format(self.model_name), index=False)

    @staticmethod
    def length(sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length

    @staticmethod
    def last_relevant(output, length):
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1]
        out_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant


if __name__ == '__main__':
    rnn = RNN()
    rnn.train()
    # gen = rnn.batch_generator()
    # batch, _ = next(gen)
    # print(batch.shape)
    # for i in batch:
    #     plt.plot(i[:,0])
    #     plt.plot(i[:,1])
    #     plt.show()
