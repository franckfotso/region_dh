# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Project: Region-DH
# Module: models.nets.finetuning
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

""" VGG16_FT: finetuning with VGG16 """

class VGG16_FT(VGG16):
    
    def __init__(self, cfg):
        VGG16.__init__(self)
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
        self._images = tf.placeholder(tf.float32, shape=[self.cfg.TRAIN_BATCH_CFC_NUM_IMG, None, None, 3])        
        self._labels = tf.placeholder(tf.int32, shape=[self.cfg.TRAIN_BATCH_CFC_NUM_IMG, 1])
        
        #self._labels = tf.placeholder(tf.int32, shape=[self.cfg.TRAIN_BATCH_CFC_NUM_IMG, num_classes])
        self._tag = tag
    
        self._num_classes = num_classes
        self._mode = mode
    
        training = mode == 'TRAIN'
        testing = mode == 'TEST'
    
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
        # set initializers: random
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
    
        pool5 = self._image_to_head(is_training)    
        fc7 = self._head_to_tail(pool5, is_training)
        with tf.variable_scope(self._scope, self._scope):
            # image classification
            cls_prob, cls_pred = self._image_classification(fc7, is_training, initializer)
    
        self._score_summaries.update(self._predictions)
    
        return cls_prob, cls_pred
    
    
    def _image_classification(self, net, is_training, initializer):
        """
        cls_score = slim.fully_connected(net, self._num_classes, 
                                           weights_initializer=initializer,
                                           trainable=is_training,
                                           activation_fn=None, scope='cls_score')
        #"""
        
        cls_score = slim.conv2d(net, self._num_classes, [1, 1],
                          weights_initializer=initializer,
                          trainable=is_training,
                          activation_fn=None, scope='cls_score')
        cls_score = tf.reshape(cls_score, [cls_score.shape[0],-1]) 
        
        cls_prob = tf.nn.softmax(cls_score, name="cls_prob")
        
        cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")
        cls_pred = tf.reshape(cls_pred, [-1])
        
        #print("_image_classification > cls_score.shape: ", cls_score.shape)
        #print("_image_classification > cls_pred.shape: ", cls_pred.shape)
    
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
            
            #print("cls_score.shape: ", cls_score.shape)
            #print("labels.shape: ", labels.shape)
            
            # single-label
            """
            _loss_cls = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=labels)
            loss_cls = tf.reduce_mean(_loss_cls)
            
            regularization_loss = tf.add_n(tf.losses.get_regularization_losses(), 'regu')
            self._losses['loss_cls'] = loss_cls + regularization_loss
            #"""
            
            #"""
            loss_cls = tf.losses.sparse_softmax_cross_entropy(logits=cls_score, labels=labels)
            self._losses['loss_cls'] = loss_cls
            self._losses['total_loss'] = tf.losses.get_total_loss()
            #"""
            
            # multi-label
            #loss_cls = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits=cls_score, multi_class_labels=labels))
                       
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
    def test_image(self, sess, im_blob, label_blob):
        feed_dict = {self._images: im_blob, self._labels: label_blob}

        cls_score, cls_prob, cls_pred = sess.run([self._predictions["cls_score"],
                                                  self._predictions['cls_prob'],
                                                  self._predictions['cls_pred']],
                                                  feed_dict=feed_dict)
        return cls_score, cls_prob, cls_pred
    
    
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
    
    
    