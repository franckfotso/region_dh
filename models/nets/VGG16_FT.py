# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Project: Region-DH
# Module: models.nets.VGG16_FT
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

from faster_rcnn import layers

""" VGG16_FT: finetuning with VGG16 """

class VGG16_FT(VGG16):
    
    def __init__(self, cfg, multilabel=False):
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
        self._loss_summaries = {}
        self._acc_summaries = {}
        self._variables_to_fix = {}
        self.multilabel = multilabel
        self.cfg = cfg
        
    def create_architecture(self, mode, num_classes, tag=None):
        
        training = mode == 'TRAIN'
        testing = mode == 'TEST'
        
        if training:
            # mode: TRAIN    
            
            if self.multilabel:
                self._images = tf.placeholder(tf.float32, shape=[self.cfg.TRAIN_BATCH_DET_IMS_PER_BATCH, None, None, 3])
                self._labels = tf.placeholder(tf.int32, shape=[self.cfg.TRAIN_BATCH_DET_IMS_PER_BATCH, num_classes-1])
                self._pos_weights = tf.placeholder(tf.float32, 
                                                   shape=[self.cfg.TRAIN_BATCH_DET_IMS_PER_BATCH, num_classes-1])
            else:
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
                for key, var in self._loss_summaries.items():
                    # add loss to summary
                    val_summaries.append(self._add_loss_summary(key, var))
                for key, var in self._acc_summaries.items():
                    # add acc to summary
                    val_summaries.append(self._add_acc_summary(key, var))
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
        pool5 = self._image_to_head(is_training, last_pool=True)
        print("pool5.shape: ", pool5.shape)
        fc7 = self._head_to_tail(pool5, is_training, global_pool="MEAN")
        print("fc7.shape: ", fc7.shape)
        
        if not is_training:
            self._predictions["fc7"] = tf.reshape(fc7, [fc7.shape[0],-1])
                
        with tf.variable_scope(self._scope, self._scope):            
            # image classification
            cls_prob, cls_pred = self._image_classification(fc7, is_training, 
                                                            self.cfg.TEST_DEFAULT_CFC_THRESH)
        self._score_summaries.update(self._predictions)
    
        return cls_prob, cls_pred
    
    
    def _image_classification(self, net, is_training, threshold=0.3):
        # net.layers[-1]: fc hash
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        num_cls = self._num_classes if not self.multilabel else self._num_classes-1
        
        cls_score = slim.fully_connected(net, self._num_classes-1, 
                              weights_initializer=initializer,
                              trainable=is_training,
                              activation_fn=None,
                              scope='cls_score')        
        print("cls_score.shape: ", cls_score.shape)
        
        if self.multilabel:
            cls_prob = tf.nn.sigmoid(cls_score, name="cls_prob")
            cls_prob = tf.reshape(cls_prob, [cls_prob.shape[0], num_cls])
            cls_pred = tf.to_int32((cls_prob >= threshold))
            cls_pred = tf.reshape(cls_pred, [cls_prob.shape[0], num_cls])
        else:
            # single-label
            cls_prob = tf.nn.softmax(cls_score, name="cls_prob")
            cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")
            cls_pred = tf.reshape(cls_pred, [-1])
            
        self._predictions["cls_score"] = cls_score
        self._predictions["cls_prob"] = cls_prob
        self._predictions["cls_pred"] = cls_pred        
    
        return cls_prob, cls_pred
    
    def _add_losses(self):
        with tf.variable_scope('LOSS_' + self._tag) as scope:
            
            """ classification loss"""
            cls_score = self._predictions["cls_score"]            
            labels = self._labels
            
            cls_score = tf.reshape(cls_score, [cls_score.shape[0], -1])
            labels = tf.reshape(labels, [labels.shape[0], -1])
            print("cls_score.shape: ", cls_score.shape)
            print("labels.shape: ", labels.shape)
                        
            if self.multilabel:
                #labels = tf.reshape(labels, [-1, self._num_classes-1])
                pos_weights = tf.reshape(self._pos_weights, [self._pos_weights.shape[0], -1])                
                #print("pos_weights.shape: ", pos_weights.shape)
                
                pos_weight = pos_weights[:,0]
                pos_weight = tf.reshape(pos_weight, [pos_weight.shape[0], -1])
                print("pos_weight.shape: ", pos_weight.shape)
                
                loss_cls = tf.nn.weighted_cross_entropy_with_logits(logits=cls_score,
                                                                   targets=tf.to_float(labels),
                                                                   pos_weight=pos_weight) # try 1, 2, 3
                                                                   #pos_weight=pos_weights) # try 4, 5, 6
                loss_cls = tf.reduce_mean(loss_cls)
                tf.losses.add_loss(loss_cls)
            else:
                labels = tf.reshape(labels, [-1])
                loss_cls = tf.losses.sparse_softmax_cross_entropy(logits=cls_score, 
                                                                 labels=labels, 
                                                                 weights=self.alpha)
            self._losses['loss_cls'] = loss_cls                 
            
            """ overall loss: loss_cls + regu """            
            self._losses['total_loss'] = tf.losses.get_total_loss()
                       
            self._event_summaries.update(self._losses)
            self._loss_summaries.update(self._losses)
            
            # Evaluation metrics
            cls_pred =self._predictions["cls_pred"]
            
            if self.multilabel:
                print("labels.shape: ", labels.shape)
                print("cls_pred.shape: ", cls_pred.shape)
                acc_cls, precision, recall = layers._precision_recall_score(labels, cls_pred, "prec_rec_score")
            else:
                acc_cls = tf.contrib.metrics.accuracy(predictions=cls_pred, labels=labels, name="acc_cls")
            
            self._accuracies['acc_cls'] = acc_cls            
            self._event_summaries.update(self._accuracies)
            self._acc_summaries.update(self._accuracies)
    
    def _add_loss_summary(self, key, var):
        return tf.summary.scalar('LOSS/' + key, var)        
        
    def _add_acc_summary(self, key, var):
        return tf.summary.scalar('ACC/' + key, var)
    
    def _add_act_summary(self, tensor):
        tf.summary.histogram('ACT/' + tensor.op.name + '/activations', tensor)
        tf.summary.scalar('ACT/' + tensor.op.name + '/zero_fraction',
                          tf.nn.zero_fraction(tensor))
        

    def _add_score_summary(self, key, tensor):
        tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)
        
    
    def _add_train_summary(self, var):
        tf.summary.histogram('TRAIN/' + var.op.name, var)
        
    def get_summary(self, sess, blobs):
        feed_dict = {self._images: blobs['data'], 
                     self._labels: blobs['labels']}
        if self.multilabel:
            feed_dict[self._pos_weights] = blobs['pos_weights']
            
        summary = sess.run(self._summary_op_val, feed_dict=feed_dict)
    
        return summary
    
    # only useful during testing mode
    def test_image(self, sess, blobs):
        feed_dict = {self._images: blobs["data"]}

        cls_score, cls_prob, cls_pred, fc7 = sess.run([self._predictions["cls_score"],
                                                  self._predictions['cls_prob'],
                                                  self._predictions['cls_pred'],
                                                  self._predictions['fc7']],
                                                  feed_dict=feed_dict)
        return cls_score, cls_prob, cls_pred, fc7, fc7
    
    
    def train_step(self, sess, blobs, train_op):
        feed_dict = {self._images: blobs['data'], 
                     self._labels: blobs['labels']}
        if self.multilabel:
            feed_dict[self._pos_weights] = blobs['pos_weights']
            
        total_loss, loss_cls, acc_cls, _ = sess.run([self._losses["total_loss"],
                                                     self._losses['loss_cls'],
                                                     self._accuracies['acc_cls'],
                                                     train_op], feed_dict=feed_dict)
        
        return {"losses": {"total_loss": total_loss, "loss_cls": loss_cls},
                "accuracies": {"acc_cls": acc_cls}}
            
        
    def train_step_with_summary(self, sess, blobs, train_op):
        feed_dict = {self._images: blobs['data'], 
                     self._labels: blobs['labels']}
        if self.multilabel:
            feed_dict[self._pos_weights] = blobs['pos_weights']
            
        total_loss, loss_cls, acc_cls, summary, _ = sess.run([self._losses["total_loss"],
                                                              self._losses['loss_cls'],
                                                              self._accuracies['acc_cls'],
                                                              self._summary_op, train_op], feed_dict=feed_dict)
    
        return {"losses": {"total_loss": total_loss, "loss_cls": loss_cls},
                "accuracies": {"acc_cls": acc_cls},
                "summary": summary}
    