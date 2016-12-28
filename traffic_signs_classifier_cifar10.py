#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 13:26:05 2016

@author: tz
"""

from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

import tensorflow as tf
import numpy as np
import time


batch_size = 64
graph = tf.Graph()

with graph.as_default():
    # Input data.
    train_dataset = tf.placeholder(tf.float32, shape=(None,32,32,3))
    train_labels = tf.placeholder(tf.int32, shape=(None))
    
    # Variables.
    w0_cl1 = tf.Variable(tf.truncated_normal([1,1,3,3], stddev=np.sqrt(2./3)))
    b0_cl1 = tf.Variable(tf.zeros([3]))
    w1_cl5 = tf.Variable(tf.truncated_normal([3,3,3,6], stddev=np.sqrt(2./27)))
    b1_cl5 = tf.Variable(tf.zeros([6]))
    w2_cl5 = tf.Variable(tf.truncated_normal([3,3,6,64], stddev=np.sqrt(2./54)))
    b2_cl5 = tf.Variable(tf.zeros([64]))
    w3_cl5 = tf.Variable(tf.truncated_normal([3,3,64,256], stddev=np.sqrt(2./576)))
    b3_cl5 = tf.Variable(tf.zeros([256]))

    w4_fc1 = tf.Variable(tf.truncated_normal([2*2*256,128], stddev=np.sqrt(2./512)))
    b4_fc1 = tf.Variable(tf.zeros([128]))
    w5_fc2 = tf.Variable(tf.truncated_normal([128,10], stddev=np.sqrt(2./128)))
    b5_fc2 = tf.Variable(tf.zeros([10]))
    
    # Model.
    def model(data, dropout = 1.0):
        conv0 = tf.nn.conv2d(data, w0_cl1, [1,1,1,1], 'SAME')
        hidden0 = tf.nn.relu(tf.nn.bias_add(conv0, b0_cl1))
        
        conv1 = tf.nn.conv2d(hidden0, w1_cl5, [1,1,1,1], 'VALID')
        hidden1 = tf.nn.relu(tf.nn.bias_add(conv1, b1_cl5))
        
        pool1 = tf.nn.max_pool(hidden1, [1,2,2,1], [1,2,2,1], 'VALID')
        
        conv2 = tf.nn.conv2d(pool1, w2_cl5, [1,1,1,1], 'VALID')
        hidden2 = tf.nn.relu(tf.nn.bias_add(conv2, b2_cl5))
        
        pool2 = tf.nn.max_pool(hidden2, [1,2,2,1], [1,2,2,1], 'VALID')
        
        conv3 = tf.nn.conv2d(pool2, w3_cl5, [1,1,1,1], 'VALID')
        hidden3 = tf.nn.relu(tf.nn.bias_add(conv3, b3_cl5))        
        
        pool3 = tf.nn.max_pool(hidden3, [1,2,2,1], [1,2,2,1], 'VALID')
        
        #s = pool3.get_shape().as_list()
        #fc1 = tf.reshape(pool3, [s[0],s[1]*s[2]*s[3]])
        
        fc1 = tf.contrib.layers.flatten(pool3)
        fc1 = tf.nn.relu(tf.matmul(fc1, w4_fc1) + b4_fc1)
        fc1 = tf.nn.dropout(fc1, dropout)

        fc2 = tf.matmul(fc1, w5_fc2) + b5_fc2
        return fc2
        
    # Training computation.
    logits = model(train_dataset, 0.5)
    
    # Loss.
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, train_labels))

    # Optimizer.
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    
    # Predictions.
    train_prediction = tf.nn.softmax(logits)

def accuracy(predictions, labels):
    return (np.sum(np.equal(np.argmax(predictions, 1),labels.reshape(-1))))/predictions.shape[0]



with tf.Session(graph = graph) as sess:
    training_epochs = 20
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(training_epochs):
        total_batch = np.int(X_train.shape[0]/batch_size)
        t0 = time.time()
        for i in range(total_batch):
            offset = i * batch_size
            batch_x = X_train[offset:(offset+batch_size), ]/255.0
            batch_y = y_train[offset:(offset+batch_size)].reshape(batch_size)
            feed_dict = {train_dataset: batch_x, train_labels: batch_y}
            _, l, predictions = sess.run([optimizer, loss, train_prediction],feed_dict=feed_dict)

        acc = 0.0
        for k in range(0, X_test.shape[0], batch_size):
            p = sess.run(train_prediction, feed_dict={train_dataset: X_test[k:(k+batch_size),]/255.0})
            acc += accuracy(p, y_test[k:(k+batch_size)]) * p.shape[0]
        print("Accuracy for {} epoch: {:.3%}".format(epoch, acc/X_test.shape[0]))
        print("Time for {} epoch: {}".format(epoch, time.time()-t0))
        
        
"""
Accuracy for 0 epoch: 35.670%
Time for 0 epoch: 30.26082491874695
Accuracy for 1 epoch: 43.800%
Time for 1 epoch: 30.46102499961853
Accuracy for 2 epoch: 51.790%
Time for 2 epoch: 30.418233156204224
Accuracy for 3 epoch: 56.070%
Time for 3 epoch: 30.37011408805847
Accuracy for 4 epoch: 57.910%
Time for 4 epoch: 30.37199592590332
Accuracy for 5 epoch: 60.430%
Time for 5 epoch: 30.566981077194214
Accuracy for 6 epoch: 60.920%
Time for 6 epoch: 30.814717054367065
Accuracy for 7 epoch: 61.700%
Time for 7 epoch: 29.927985906600952
Accuracy for 8 epoch: 62.120%
Time for 8 epoch: 30.04191493988037
Accuracy for 9 epoch: 63.310%
Time for 9 epoch: 29.949213981628418
Accuracy for 10 epoch: 63.040%
Time for 10 epoch: 30.191231966018677
Accuracy for 11 epoch: 63.450%
Time for 11 epoch: 32.2960569858551
Accuracy for 12 epoch: 63.730%
Time for 12 epoch: 33.39171504974365
Accuracy for 13 epoch: 62.870%
Time for 13 epoch: 31.74323296546936
Accuracy for 14 epoch: 63.480%
Time for 14 epoch: 32.0549840927124
Accuracy for 15 epoch: 63.740%
Time for 15 epoch: 33.09568810462952
Accuracy for 16 epoch: 64.120%
Time for 16 epoch: 31.596776962280273
Accuracy for 17 epoch: 63.360%
Time for 17 epoch: 33.36310696601868
Accuracy for 18 epoch: 63.730%
Time for 18 epoch: 45.5346360206604
Accuracy for 19 epoch: 62.730%
Time for 19 epoch: 36.5439510345459
"""