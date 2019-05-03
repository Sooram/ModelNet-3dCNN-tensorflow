# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:06:25 2019

@author: Sooram Kang

Reference: 
    https://github.com/guoguo12/modelnet-cnn3d_bn
    http://aguo.us/writings/classify-modelnet.html

"""
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from model import config, Model

data = np.load('modelnet10.npz')
X, Y = shuffle(data['X_train'], data['y_train'])
X_test, Y_test = shuffle(data['X_test'], data['y_test'])

X.shape   # (3991, 30, 30, 30)
X_test.shape

#%%
tf.reset_default_graph()

inputs = tf.placeholder(tf.float32, shape=[None, config['img_D'], config['img_H'], config['img_W'], config['img_C']])
labels = tf.placeholder(tf.int64, [None, config["n_classes"]], name='labels')
keep_prob = tf.placeholder(tf.float32)

model = Model(inputs, labels, config, keep_prob)

""" open session """
c = tf.ConfigProto()
c.gpu_options.visible_device_list = "0"

sess = tf.Session(config=c)
sess.run(tf.global_variables_initializer())

#model.load(sess, model_dir)

#%%
""" train """
start_epoch = 0
num_iters = int(len(X)/config['n_batch']) + 1
log_dir = './train_log/'
saving_cycle = 1


for epoch in range(start_epoch, config["n_epoch"]):      ################ check!!!!
    for step in range(num_iters):
        batch_images = X[step*config['n_batch']: (step+1)*config['n_batch']] 
        batch_images = batch_images.reshape(-1, config['img_D'], config['img_H'], config['img_W'], config['img_C'])
        
        batch_labels = Y[step*config['n_batch']: (step+1)*config['n_batch']] 
        one_hot_labels = np.eye(config['n_classes'])[batch_labels]
        
        feed_dict = {inputs: batch_images, labels: one_hot_labels, keep_prob: 0.8}

        _, train_loss = sess.run([model.train_op, model.loss], feed_dict=feed_dict)
        print("Epoch: %d, Step: %d, Training Loss: %f" % (epoch, step, train_loss))

        with open(log_dir + "training_loss.txt", "a+") as file:
            file.write("Epoch: %d\t Step: %d\t Training Loss: %f\n" % (epoch, step, train_loss))
            
    if(epoch % saving_cycle == 0):   
        model.save(sess, log_dir, epoch)
        
#%%
""" test """
log_dir = './train_log/'
model.load(sess, log_dir)

batch_images = X_test
batch_images = batch_images.reshape(-1, config['img_W'], config['img_W'], config['img_W'], 1)

batch_labels = Y_test
one_hot_labels = np.eye(config['n_classes'])[batch_labels]
one_hot_labels = one_hot_labels.reshape(-1, config['n_classes'])

feed_dict = {inputs: batch_images, labels: one_hot_labels, keep_prob: 1.0}
output, softmax, performance = sess.run([model.output, model.proba, model.accuracy], feed_dict=feed_dict)
print(performance)        
        
        
        
     