import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import pickle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, scale
import numpy as np
import time
import pandas as pd
import os
import glob
import librosa


class RNN:
    def __init__(self):
        # Audio parameters
        self.audio_length = 2
        self.sampling_rate = 44100
        step = 512  # numbers of point librosa takes to create 20 features
        self.time_steps = self.audio_length * self.sampling_rate // step

        # RNN parameters
        self.learning_rate = 0.001
        self.display_step = 50
        self.test_step = 200
        self.nb_epochs = 4000
        self.batch_size = 64
        self.n_features = 20
        self.rnn_sizes = [128, 128]
        self.model_name = 'mfcc_drop_scaled'
        self.load_model_name = 'mfcc_drop_verified'
        self.load = True

        self.export_dir = './networks/'
        self.import_dir = './input/'
        self.label_file = 'train.csv'
        self._init_encoders()

    def _init_encoders(self):
        df = pd.read_csv(self.import_dir + self.label_file)

        # Encode string label to int
        self.enc = LabelEncoder()
        self.enc.fit(df.label)
        self.n_classes = len(self.enc.classes_)

        # Encode int label to one hot [0, 0, 1, 0, 0]
        self.enc_hot = OneHotEncoder()
        self.enc_hot.fit(self.enc.transform(df.label).reshape(-1, 1))

    def extract_mfcc(self, train=True):
        """
        Split the audio files in segments of 2 seconds, and compute the mfcc
        features for all of the
        :param train: True for train set, False for test set
        """
        if train:
            df = pd.read_csv(self.import_dir + self.label_file)
            df = df.set_index('fname')
            labels = []
            verifieds = []
            path = 'audio_train/'
        else:
            path = 'audio_test/'

        mfccs = []
        fnames = []
        count = 0

        for fn in glob.glob(
                os.path.join(self.import_dir + path, '*.wav')):
            fname = fn.split('/')[-1]
            if train:
                label = df.loc[fname].label
                verified = df.loc[fname].manually_verified
            sound_clip, s = librosa.load(fn, sr=self.sampling_rate)

            try:
                mfcc = librosa.feature.mfcc(y=sound_clip, sr=s,
                                            n_mfcc=self.n_features).T
            except ValueError:
                # Some files of the test set cause problems
                mfcc = np.ones((10, self.n_features))
            time_steps, _ = mfcc.shape

            mfcc = scale(mfcc, axis=0)

            pad = self.time_steps - time_steps % self.time_steps

            # pad with zeros to have the same time_steps
            if pad < self.time_steps // 3 or time_steps // self.time_steps == 0:
                mfcc = np.pad(mfcc, ((0, pad), (0, 0)), mode='constant',
                              constant_values=(0, 0))
                mfcc = mfcc.reshape(time_steps // self.time_steps + 1,
                                    self.time_steps, self.n_features)

            # remove the last part if it is too short
            else:
                mfcc = mfcc[:time_steps // self.time_steps * self.time_steps, :]
                mfcc = mfcc.reshape(time_steps // self.time_steps,
                                    self.time_steps,
                                    self.n_features)

            for i in range(mfcc.shape[0]):
                mfccs.append(mfcc[i, :, :])
                fnames.append(fname)
                if train:
                    labels.append(label)
                    verifieds.append(verified)

            count += 1
            if count % 100 == 0:
                print("file {}".format(count))

        if train:
            df_mfcc = pd.DataFrame({
                'fname': fnames,
                'label': labels,
                'verified': verifieds,
            })
            df_mfcc.to_csv(self.import_dir + 'mfcc_train.csv')
            with open('./input/mfcc_train.p', 'wb') as fp:
                pickle.dump(np.array(mfccs), fp)
        else:
            df_mfcc = pd.DataFrame({
                'fname': fnames,
            })
            df_mfcc.to_csv(self.import_dir + 'mfcc_test.csv')
            with open('./input/mfcc_test.p', 'wb') as fp:
                pickle.dump(np.array(mfccs), fp)

    def train(self, verified=False):
        """
        Train on the mfcc features
        """

        with open('./input/mfcc_train.p', 'rb') as fp:
            X = pickle.load(fp)

        df_mfcc = pd.read_csv(self.import_dir + 'mfcc_train.csv')

        y = self.enc_hot.transform(
            self.enc.transform(df_mfcc.label).reshape(-1, 1)).toarray()

        # take only the manually verified label
        if verified:
            df_mfcc = df_mfcc[df_mfcc.verified == 1]
            idx_verif = df_mfcc.index
            X = X[idx_verif]
            y = y[idx_verif]

        np.random.seed(0)
        idx = np.random.permutation(len(y))
        X = X[idx, :, :]
        y = y[idx]

        sep = int(len(y) * 0.9)
        X_train = X[:sep, :, :]
        X_test = X[sep:, :, :]

        y_train = y[:sep]
        y_test = y[sep:]

        # RNN
        tf.reset_default_graph()

        x = tf.placeholder("float", [None, self.time_steps, self.n_features])
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
        with tf.Session() as session:
            saver = tf.train.Saver()
            if self.load:
                saver.restore(session, self.export_dir + self.load_model_name)
            else:
                session.run(init)

            t0 = time.time()
            for epoch in range(self.nb_epochs):
                start = epoch * self.batch_size % (
                    len(y_train) - self.batch_size)
                batch_x = X_train[start:start + self.batch_size, :, :]
                batch_y = y_train[start:start + self.batch_size]

                _, c = session.run([optimizer, loss_f],
                                   feed_dict={x: batch_x, y: batch_y,
                                              keep_prob: 0.7})

                if epoch % self.display_step == 0:
                    # Calculate batch accuracy
                    acc = session.run(accuracy,
                                      feed_dict={x: batch_x, y: batch_y,
                                                 keep_prob: 1})
                    # Calculate batch loss
                    loss = session.run(loss_f,
                                       feed_dict={x: batch_x, y: batch_y,
                                                  keep_prob: 1})

                    print("Iter " + str(epoch) + " / " + str(
                        self.nb_epochs) + ", Minibatch Loss= " +
                          "{:.6f}".format(loss) + ", Training Accuracy= " +
                          "{:.5f}".format(acc))

                    print('{} epochs time: {}'.format(self.display_step,
                                                      time.time() - t0))
                    t0 = time.time()
                if epoch % self.test_step == 0:
                    print('Test accuracy: ',
                          round(session.run(accuracy,
                                            feed_dict={x: X_test, y: y_test,
                                                       keep_prob: 1}), 3))

            saver.save(session, self.export_dir + self.model_name)

    def build_rnn(self, x, keep_prob):
        """
        Build the network
        """
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

    @staticmethod
    def length(sequence):
        """
        From https://danijar.com/variable-sequence-lengths-in-tensorflow/
        """
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length

    @staticmethod
    def last_relevant(output, length):
        """
        Return the last relevant output of the LSTM cell, by removing the
        trailing zeros (!! Raise an error it the array is full of zeros)
        From https://danijar.com/variable-sequence-lengths-in-tensorflow/
        """
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1]
        out_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant

    def predict(self):
        with open('./input/mfcc_test.p', 'rb') as fp:
            X = pickle.load(fp)

        df_mfcc = pd.read_csv(self.import_dir + 'mfcc_test.csv')

        # RNN
        tf.reset_default_graph()

        # variables
        x = tf.placeholder("float", [None, self.time_steps, self.n_features])
        keep_prob = tf.placeholder("float", name='keep_prob')

        prediction = self.build_rnn(x, keep_prob)

        with tf.Session() as session:
            saver = tf.train.Saver()
            saver.restore(session, self.export_dir + self.model_name)
            unique = pd.unique(df_mfcc.fname)
            results = {'label': [], 'fname': []}
            for i in range(len(pd.unique(df_mfcc.fname))):
                idxs = df_mfcc.fname[
                    df_mfcc.fname == unique[i]].index.tolist()

                batch = X[idxs, :, :]
                if batch.sum() == 0:
                    batch = np.ones_like(batch)

                predictions = session.run(prediction,
                                          feed_dict={x: np.array(batch),
                                                     keep_prob: 1})
                predictions = predictions.mean(axis=0)
                top3_labels = self.top_3(predictions, return_string=True)
                results['label'].append(top3_labels)
                results['fname'].append(unique[i])

                print('Label for {}: {}'.format(i, top3_labels))

        df = pd.DataFrame(results)
        print(df.head())
        df.to_csv("output/{}.csv".format(self.model_name), index=False)

    def top_3(self, predictions, return_string=True):
        top_labels = np.argsort(predictions)
        top_labels = top_labels[::-1]
        top3_labels = top_labels[:3]

        if return_string:
            top3_labels = " ".join(
                [self.enc.inverse_transform(el) for el in top3_labels])
        return top3_labels


if __name__ == '__main__':
    rnn = RNN()
    rnn.extract_mfcc(train=True)
    rnn.train(verified=False)
    rnn.extract_mfcc(train=False)
    rnn.predict()
