# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 13:58:46 2019

@author: Sooram Kang

"""
import tensorflow as tf
import os
import sys

config = {
    "n_batch": 256,
    "n_classes": 10,
    "n_epoch": 1000,
    "img_W": 30,
    "img_H": 30,
    "img_D": 30,
    "img_C": 1,
    "lr": 1e-4,
    "n_filters": 128,
    "dim_dense": 1024,
    'padding': 'same',
    "keep_prob_ratio": 0.8,
    "fine_tune": False
}

class Model():
    def __init__(self, inputs, labels, config, keep_prob):
        self.config = config
        self.inputs = inputs
        self.labels = labels
        self.keep_prob = keep_prob
    
        # conv => 30*30*30
        conv1 = tf.layers.conv3d(inputs=self.inputs, filters=16, kernel_size=[3,3,3], padding='same', activation=tf.nn.relu)
        # conv => 30*30*30
        conv2 = tf.layers.conv3d(inputs=conv1, filters=32, kernel_size=[3,3,3], padding='same', activation=tf.nn.relu)
        # pool => 15*15*15
        pool3 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2, 2, 2], strides=2)
        pool3 = tf.nn.dropout(pool3, self.keep_prob)

        # conv => 15*15*15
        conv4 = tf.layers.conv3d(inputs=pool3, filters=64, kernel_size=[3,3,3], padding='same', activation=tf.nn.relu)
        # conv => 15*15*15
        conv5 = tf.layers.conv3d(inputs=conv4, filters=128, kernel_size=[3,3,3], padding='same', activation=tf.nn.relu)
        # pool => 7*7*7
        pool6 = tf.layers.max_pooling3d(inputs=conv5, pool_size=[2, 2, 2], strides=2)
        pool6 = tf.nn.dropout(pool6, self.keep_prob)

        cnn3d_bn = tf.layers.batch_normalization(inputs=pool6, training=True)

        last_size = int(config['img_W'] / 4)
        flattening = tf.reshape(cnn3d_bn, [-1, last_size*last_size*last_size*config['n_filters']])
        dense = tf.layers.dense(inputs=flattening, units=config['dim_dense'], activation=tf.nn.relu)

        output = tf.layers.dense(inputs=dense, units=config['n_classes'])
        self.output = output
        self.proba = tf.nn.softmax(output)
        
        correct = tf.equal(tf.argmax(self.proba, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=self.labels))
        self.loss = loss
        
        optimizer = tf.train.AdamOptimizer(config['lr']).minimize(loss)
        self.train_op = optimizer

        self.saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=50)

           
    def save(self, sess, logdir, step):
        model_name = 'model.ckpt'
        checkpoint_path = os.path.join(logdir, model_name)
        print('Storing checkpoint to {} ...'.format(logdir), end="")
        sys.stdout.flush()

        if not os.path.exists(logdir):
            os.makedirs(logdir)

        self.saver.save(sess, checkpoint_path, global_step=step)
        print(' Done.')

    def load(self, sess, logdir):
        print("Trying to restore saved checkpoints from {} ...".format(logdir),
              end="")

        ckpt = tf.train.get_checkpoint_state(logdir)
        if ckpt:
            print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
            global_step = int(ckpt.model_checkpoint_path
                              .split('/')[-1]
                              .split('-')[-1])
            print("  Global step was: {}".format(global_step))
            print("  Restoring...", end="")
            self.saver.restore(sess, ckpt.model_checkpoint_path)
            print(" Done.")
            return global_step, sess
        else:
            print(" No checkpoint found.")
            return None, sess