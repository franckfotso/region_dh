# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Project: Region-DH
# Module: models.nets.dlbhc
# Copyright (c) 2018
# Written by: Franck FOTSO
# Based on: tf-faster-rcnn 
#    (https://github.com/endernewton/tf-faster-rcnn)
# Licensed under MIT License
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np

import tensorflow as tf

import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope

from nets.VGG16 import VGG16

class VGG16_DLBHC(VGG16):
    
    def __init__(self, cfg, num_bits):
        VGG16.__init__(self)
        self._num_bits = num_bits
        self._predictions = {}
        self._targets = {}
        self._losses = {}
        self._accuracies = {}
        self._layers = {}
        self._act_summaries = []
        self._score_summaries = {}
        self._train_summaries = []
        self._event_summaries = {}
        self._variables_to_fix = {}
        self.cfg = cfg
        
    def create_architecture(self, mode, num_classes, tag=None):
        
        training = mode == 'TRAIN'
        testing = mode == 'TEST'
        
        if training:
            # mode: TRAIN
            self._images = tf.placeholder(tf.float32, shape=[self.cfg.TRAIN_BATCH_CFC_NUM_IMG, None, None, 3])
            self._labels = tf.placeholder(tf.int32, shape=[self.cfg.TRAIN_BATCH_CFC_NUM_IMG, 1])
        else:
            # mode: TEST
            self._images = tf.placeholder(tf.float32, shape=[self.cfg.TEST_BATCH_CFC_NUM_IMG, None, None, 3])            
        
        self._tag = tag    
        self._num_classes = num_classes
        self._mode = mode        
    
        assert tag != None
    
        # handle most of the regularizers here
        weights_regularizer = tf.contrib.layers.l2_regularizer(self.cfg.TRAIN_DEFAULT_WEIGHT_DECAY)
        biases_regularizer = tf.no_regularizer
    
        # list as many types of layers as possible, even if they are not used now
        with arg_scope([slim.conv2d, slim.conv2d_in_plane, \
                        slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected], 
                        weights_regularizer=weights_regularizer,
                        biases_regularizer=biases_regularizer, 
                        biases_initializer=tf.constant_initializer(0.0)): 
            cls_prob, cls_pred = self._build_network(training)
    
        layers_to_output = {}
    
        for var in tf.trainable_variables():
            self._train_summaries.append(var)
    
        if testing:
            pass
        else:
            self._add_losses()
            layers_to_output.update(self._losses)
            
            val_summaries = []
            with tf.device("/cpu:0"):                
                #val_summaries.append(self._add_gt_image_summary())
                
                for key, var in self._event_summaries.items():
                    val_summaries.append(tf.summary.scalar(key, var))
                for key, var in self._score_summaries.items():
                    self._add_score_summary(key, var)
                for var in self._act_summaries:
                    self._add_act_summary(var)
                for var in self._train_summaries:
                    self._add_train_summary(var)
            
            self._summary_op = tf.summary.merge_all()
            self._summary_op_val = tf.summary.merge(val_summaries)
    
        layers_to_output.update(self._predictions)
    
        return layers_to_output
    
    
    def _build_network(self, is_training=True):        
        pool5 = self._image_to_head(is_training)    
        fc7 = self._head_to_tail(pool5, is_training)
        
        if not is_training:
            self._predictions["fc7"] = tf.reshape(fc7, [fc7.shape[0],-1]) # keep it as deep features
        
        with tf.variable_scope(self._scope, self._scope):
            # image hashing
            fc_hash = self._image_hashing(fc7, is_training)
            
            # image classification
            cls_prob, cls_pred = self._image_classification(fc_hash, is_training)
    
        self._score_summaries.update(self._predictions)
    
        return cls_prob, cls_pred
    
    def _image_hashing(self, net, is_training):
        # net.layers[-1]: last fc layer (fc7)
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.005)
        
        # fc layer: random initializer & sigmoid activation
        fc_hash = slim.conv2d(net, self._num_bits, [1, 1],
                          weights_initializer=initializer,
                          trainable=is_training,
                          activation_fn=tf.nn.sigmoid, scope='fc_hash')
        
        self._predictions["fc_hash"] = tf.reshape(fc_hash, [fc_hash.shape[0],-1])
        
        return fc_hash
    
    def _image_classification(self, net, is_training):
        # net.layers[-1]: fc hash
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        
        cls_score = slim.conv2d(net, self._num_classes, [1, 1],
                          weights_initializer=initializer,
                          trainable=is_training,
                          activation_fn=None, scope='cls_score')
        cls_score = tf.reshape(cls_score, [cls_score.shape[0],-1])
        
        cls_prob = tf.nn.softmax(cls_score, name="cls_prob")
        
        cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")
        cls_pred = tf.reshape(cls_pred, [-1])
            
        self._predictions["cls_score"] = cls_score
        self._predictions["cls_pred"] = cls_pred
        self._predictions["cls_prob"] = cls_prob
    
        return cls_prob, cls_pred
    
    
    def _add_losses(self):
        with tf.variable_scope('LOSS_' + self._tag) as scope:
            
            # class loss
            cls_score = self._predictions["cls_score"]                  
            
            labels = self._labels
            labels = tf.reshape(labels, [-1])
            
            loss_cls = tf.losses.sparse_softmax_cross_entropy(logits=cls_score, labels=labels)
            self._losses['loss_cls'] = loss_cls
            self._losses['total_loss'] = tf.losses.get_total_loss()
                       
            self._event_summaries.update(self._losses)
            
            # Evaluation metrics
            cls_pred = tf.to_int32(self._predictions["cls_pred"])
            acc_cls = tf.contrib.metrics.accuracy(predictions=cls_pred, labels=labels, name="acc_cls")
            self._accuracies['acc_cls'] = acc_cls
            
            self._event_summaries.update(self._accuracies)
    
        return loss_cls, acc_cls
    
    
    def _add_act_summary(self, tensor):
        tf.summary.histogram('ACT/' + tensor.op.name + '/activations', tensor)
        tf.summary.scalar('ACT/' + tensor.op.name + '/zero_fraction',
                          tf.nn.zero_fraction(tensor))
        

    def _add_score_summary(self, key, tensor):
        tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)
        
    
    def _add_train_summary(self, var):
        tf.summary.histogram('TRAIN/' + var.op.name, var)
        
    def get_summary(self, sess, blobs):
        feed_dict = {self._images: blobs['data'], self._labels: blobs['labels']}
        summary = sess.run(self._summary_op_val, feed_dict=feed_dict)
    
        return summary
    
    # only useful during testing mode
    def test_image(self, sess, im_blob):
        feed_dict = {self._images: im_blob}

        cls_score, cls_prob, cls_pred, fc_hash, fc7 = sess.run([self._predictions["cls_score"],
                                                          self._predictions['cls_prob'],
                                                          self._predictions['cls_pred'],
                                                          self._predictions["fc_hash"],
                                                          self._predictions["fc7"]],                                                 
                                                          feed_dict=feed_dict)
        return cls_score, cls_prob, cls_pred, fc_hash, fc7
    
    
    def train_step(self, sess, blobs, train_op):
        feed_dict = {self._images: blobs['data'], self._labels: blobs['labels']}
        total_loss, loss_cls, acc_cls, _ = sess.run([self._losses["total_loss"],
                                                     self._losses['loss_cls'],
                                                     self._accuracies['acc_cls'],
                                                     train_op], feed_dict=feed_dict)        
        return total_loss, loss_cls, acc_cls
            
        
    def train_step_with_summary(self, sess, blobs, train_op):
        feed_dict = {self._images: blobs['data'], self._labels: blobs['labels']}
        total_loss, loss_cls, acc_cls, summary, _ = sess.run([self._losses["total_loss"],
                                                              self._losses['loss_cls'],
                                                              self._accuracies['acc_cls'],
                                                              self._summary_op, train_op], feed_dict=feed_dict)        
        return total_loss, loss_cls, acc_cls, summary 
    
    
    