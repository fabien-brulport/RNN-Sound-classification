import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import pickle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
import time

with open('./input/mfcc.p', 'rb') as fp:
    t = pickle.load(fp)
    tr_features, tr_labels = t
# Encode string label to int
enc = LabelEncoder()
tr_labels = enc.fit_transform(tr_labels)
# Encode int label to one hot [0, 0, 1, 0, 0]
enc = OneHotEncoder()
tr_labels = enc.fit_transform(tr_labels.reshape(-1, 1)).toarray()

idx = np.random.permutation(len(tr_labels))
tr_labels = tr_labels[idx]
tr_features = tr_features[idx]
sep = int(0.9 * len(tr_labels))
X = tr_features[:sep]
Y = tr_labels[:sep]
X_test = tr_features[sep:]
y_test = tr_labels[sep:]

# RNN
tf.reset_default_graph()

learning_rate = 0.0005
display_step = 100
hm_epochs = 100
n_classes = tr_labels.shape[1]
batch_size = 128
chunk_size = 20
n_chunks = 41
rnn_size = 128

x = tf.placeholder("float", [None, n_chunks, chunk_size])
y = tf.placeholder("float", [None, n_classes])


def RNN1(x):
    layer = {'weight': tf.Variable(tf.random_normal([rnn_size, n_classes])),
             'bias': tf.Variable(tf.random_normal([n_classes]))}

    lstm_cell = rnn_cell.LSTMCell(rnn_size)
    output, state = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
    output = tf.transpose(output, [1, 0, 2])
    output = output[-1, :, :]
    return tf.nn.softmax(
        tf.tensordot(output, layer['weight'], [[1], [0]]) + layer['bias'])


def RNN2(x):
    layer = {'weight': tf.Variable(tf.random_normal([rnn_size, n_classes])),
             'bias': tf.Variable(tf.random_normal([n_classes]))}

    lstm_cell = rnn_cell.LSTMCell(rnn_size)
    output, state = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
    output = tf.transpose(output, [1, 0, 2])
    output = tf.gather(output, int(output.get_shape()[0]) - 1)

    return tf.nn.softmax(
        tf.tensordot(output, layer['weight'], [[1], [0]]) + layer['bias'])


prediction = RNN1(x)

# Define loss and optimizer
loss_f = -tf.reduce_sum(y * tf.log(prediction))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_f)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
with tf.Session() as session:
    saver = tf.train.Saver()
    session.run(init)
    nbstep = hm_epochs * Y.shape[0] // batch_size

    for epoch in range(hm_epochs):
        print('\nEpoch {} ------------------------'.format(epoch))
        t0 = time.time()
        for steps in range(Y.shape[0] // batch_size):
            offset = (steps * (epoch + 1) * batch_size) % (
                Y.shape[0] - batch_size)
            batch_x = X[offset:(offset + batch_size), :, :]
            batch_y = Y[offset:(offset + batch_size), :]
            _, c = session.run([optimizer, loss_f],
                               feed_dict={x: batch_x, y: batch_y})

            if steps % display_step == 0:
                # Calculate batch accuracy
                acc = session.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                # Calculate batch loss
                loss = session.run(loss_f, feed_dict={x: batch_x, y: batch_y})
                print("Iter " + str(steps) + " / " + str(
                    Y.shape[0] // batch_size) + ", Minibatch Loss= " +
                      "{:.6f}".format(loss) + ", Training Accuracy= " +
                      "{:.5f}".format(acc))

        print('Test accuracy: ',
              round(session.run(accuracy, feed_dict={x: X_test, y: y_test}),
                    3))
        print('Epoch time: {}'.format(time.time() - t0))

    saver.save(session, './networks/my-test-model')
