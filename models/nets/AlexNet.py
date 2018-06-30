# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Project: Region-DH
# Module: models.nets.AlexNet
# Copyright (c) 2018
# Written by: Franck FOTSO
# Licensed under MIT License
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope

from finetuning import Network

""" AlexNet: classification with fine-tuning"""

class AlexNet(Network):
    
    def __init__(self):
        Network.__init__(self)
        self._scope = 'alexnet'
    
    """ input => conv5 """
    def _image_to_head(self, is_training, reuse=None):
        with tf.variable_scope(self._scope, self._scope, reuse=reuse):
            
            net = slim.conv2d(self._image, 64, [11, 11], 4, padding='VALID', scope='conv1')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
            net = slim.conv2d(net, 192, [5, 5], scope='conv2')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
            net = slim.conv2d(net, 384, [3, 3], scope='conv3')
            net = slim.conv2d(net, 384, [3, 3], scope='conv4')
            net = slim.conv2d(net, 256, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')
    
        self._act_summaries.append(net)
        self._layers['head'] = net
        
        return net
        
    """ conv5 => fc7 """
    def _head_to_tail(self, pool5, is_training, reuse=None):
        with tf.variable_scope(self._scope, self._scope, reuse=reuse):
            pool5_flat = slim.flatten(pool5, scope='flatten')
            fc6 = slim.fully_connected(pool5_flat, 4096, scope='fc6')
            if is_training:
                fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True, scope='dropout6')
            fc7 = slim.fully_connected(fc6, 4096, scope='fc7')
            if is_training:
                fc7 = slim.dropout(fc7, keep_prob=0.5, is_training=True, scope='dropout7')
    
        return fc7
    
    
    def get_variables_to_restore(self, variables, var_keep_dic):
        variables_to_restore = []
    
        for v in variables:
            # exclude the conv weights that are fc weights in vgg16
            if v.name == (self._scope + '/fc6/weights:0') or \
                v.name == (self._scope + '/fc7/weights:0'):
                self._variables_to_fix[v.name] = v
                continue
            # exclude the first conv layer to swap RGB to BGR
            if v.name == (self._scope + '/conv1/conv1_1/weights:0'):
                self._variables_to_fix[v.name] = v
                continue
            if v.name.split(':')[0] in var_keep_dic:
                print('Variables restored: %s' % v.name)
                variables_to_restore.append(v)
    
        return variables_to_restore
    
    
    def fix_variables(self, sess, pretrained_model):
        print('Fix VGG16 layers..')
        with tf.variable_scope('Fix_VGG16') as scope:
            with tf.device("/cpu:0"):
                # fix the vgg16 issue from conv weights to fc weights
                # fix RGB to BGR
                fc6_conv = tf.get_variable("fc6_conv", [7, 7, 512, 4096], trainable=False)
                fc7_conv = tf.get_variable("fc7_conv", [1, 1, 4096, 4096], trainable=False)
                conv1_rgb = tf.get_variable("conv1_rgb", [3, 3, 3, 64], trainable=False)
                restorer_fc = tf.train.Saver({self._scope + "/fc6/weights": fc6_conv, 
                                              self._scope + "/fc7/weights": fc7_conv,
                                              self._scope + "/conv1/conv1_1/weights": conv1_rgb})
                restorer_fc.restore(sess, pretrained_model)
        
                sess.run(tf.assign(self._variables_to_fix[self._scope + '/fc6/weights:0'], tf.reshape(fc6_conv, 
                                    self._variables_to_fix[self._scope + '/fc6/weights:0'].get_shape())))
                sess.run(tf.assign(self._variables_to_fix[self._scope + '/fc7/weights:0'], tf.reshape(fc7_conv, 
                                    self._variables_to_fix[self._scope + '/fc7/weights:0'].get_shape())))
                sess.run(tf.assign(self._variables_to_fix[self._scope + '/conv1/conv1_1/weights:0'], 
                                    tf.reverse(conv1_rgb, [2])))
    