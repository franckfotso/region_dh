# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Project: Region-DH
# Module: models.nets.VGG16
# Copyright (c) 2018
# Written by: Franck FOTSO
# Licensed under MIT License
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope

""" VGG16: backbone network"""

class VGG16(object):
    
    def __init__(self):
        self._scope = 'vgg_16'
        
    """ input => conv5 """
    def _image_to_head(self, is_training, 
                       trainables=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'], 
                       last_pool=True, reuse=None):
        # net.layers[-1]: input layer
        
        with tf.variable_scope(self._scope, self._scope, reuse=reuse):
            net = slim.repeat(self._images, 2, slim.conv2d, 64, [3, 3], 
                              trainable=('conv1' in trainables), scope='conv1')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], 
                              trainable=('conv2' in trainables), scope='conv2')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], 
                              trainable=('conv3' in trainables), scope='conv3')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], 
                              trainable=('conv4' in trainables), scope='conv4')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], 
                              trainable=('conv5' in trainables), scope='conv5')            
            if last_pool:
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool5')
                #net = tf.reduce_max(net, [1, 2], keep_dims=True, name='pool5')
        
        self._act_summaries.append(net)
        self._layers['head'] = net
        
        return net
    
    """ conv5 => fc7 """
    def _head_to_tail(self, net, is_training, reuse=None, global_pool=None):
        # net.layers[-1]: last conv layer
        
        with tf.variable_scope(self._scope, self._scope, reuse=reuse):            
            net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
            #print("fc6.shape: ", net.shape)
            if is_training:
                net = slim.dropout(net, keep_prob=0.5, is_training=is_training, scope='dropout6')
            
            net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
            #print("fc7.shape: ", net.shape)
            if is_training:
                net = slim.dropout(net, keep_prob=0.5, is_training=is_training, scope='dropout7')
                
            if global_pool != None:
                if global_pool == "MEAN":
                    net = tf.reduce_mean(net, [1, 2], keepdims=True, name='global_pool')
                elif global_pool == "MAX":
                    net = tf.reduce_max(net, [1, 2], keepdims=True, name='global_pool')
                else:
                    raise NotImplementedError    
        return net
    
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