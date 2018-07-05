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

from nets.finetuning import Network

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
        pass
    
    
    def fix_variables(self, sess, pretrained_model):
        pass
    