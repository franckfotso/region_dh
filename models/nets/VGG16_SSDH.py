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

class VGG16_SSDH(VGG16):
    
    def __init__(self, cfg, num_bits, multilabel=False):
        VGG16.__init__(self)
        self._num_bits = num_bits
        self._predictions = {}
        self._targets = {}
        self._losses = {}
        self._gradients = {}
        self._accuracies = {}
        self._layers = {}
        self._act_summaries = []
        self._score_summaries = {}
        self._train_summaries = []
        self._event_summaries = {}
        self._variables_to_fix = {}
        self.multilabel = multilabel
        self.alpha = cfg.TRAIN_BATCH_CFC_ALPHA
        self.beta = cfg.TRAIN_BATCH_CFC_BETA
        self.gamma = cfg.TRAIN_BATCH_CFC_GAMMA
        self.cfg = cfg
        
    def create_architecture(self, mode, num_classes, tag=None):
        
        training = mode == 'TRAIN'
        testing = mode == 'TEST'
        
        if training:
            # mode: TRAIN
            self._images = tf.placeholder(tf.float32, shape=[self.cfg.TRAIN_BATCH_CFC_NUM_IMG, None, None, 3])
            
            if self.multilabel:
                self._labels = tf.placeholder(tf.int32, shape=[self.cfg.TRAIN_BATCH_CFC_NUM_IMG, num_classes])
            else:
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
                for key, var in self._event_summaries.items():
                    #print("key: {}, var: {}".format(key, var))
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
            self._predictions["fc7"] = tf.reshape(fc7, [fc7.shape[0],-1])
        
        with tf.variable_scope(self._scope, self._scope):
            # image encoding
            fc_emb = self._image_encoding(fc7, is_training)
            
            # image classification
            cls_prob, cls_pred = self._image_classification(fc_emb, is_training)
    
        self._score_summaries.update(self._predictions)
    
        return cls_prob, cls_pred
    
    def _image_encoding(self, net, is_training):
        # net.layers[-1]: last fc layer (fc7)
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.005)
        
        # fc layer: random initializer & sigmoid activation
        fc_emb = slim.conv2d(net, self._num_bits, [1, 1],
                          weights_initializer=initializer,
                          trainable=is_training,
                          activation_fn=tf.nn.sigmoid, scope='fc_emb')
        
        if is_training:
            self._predictions["fc_emb"] = fc_emb
        else:
            self._predictions["fc_emb"] = tf.reshape(fc_emb, [fc_emb.shape[0],-1])
        
        return fc_emb
    
    def _image_classification(self, net, is_training, threshold=0.5):
        # net.layers[-1]: fc hash
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        
        cls_score = slim.conv2d(net, self._num_classes, [1, 1],
                          weights_initializer=initializer,
                          trainable=is_training,
                          activation_fn=None, scope='cls_score')
        cls_score = tf.reshape(cls_score, [cls_score.shape[0],-1])
        
        if self.multilabel:
            cls_prob = tf.nn.sigmoid(cls_score, name="cls_prob")
            cls_pred = tf.round(cls_prob)
            cls_pred = tf.reshape(cls_pred, [-1, self._num_classes])
        else:
            # single-label
            cls_prob = tf.nn.softmax(cls_score, name="cls_prob")
            cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")
            cls_pred = tf.reshape(cls_pred, [-1])
            
        self._predictions["cls_score"] = cls_score
        self._predictions["cls_prob"] = cls_prob
        self._predictions["cls_pred"] = cls_pred        
    
        return cls_prob, cls_pred
    
    def k1_euclidean_loss(self, norm_order=2, weights=1.0):
        """ forcing binary: alternative to the quantization loss """
        
        fc_emb = self._predictions["fc_emb"]
        fc_emb = tf.reshape(fc_emb, [fc_emb.shape[0],-1])
        
        #num_bits = fc_emb.shape[-1]
        #print("fc_emb.shape: ", fc_emb.shape)
        
        _cal = tf.subtract(fc_emb, 0.5)        
        _cal = tf.multiply(_cal, _cal)
        #_cal = tf.sqrt(_cal)
        _cal = tf.reduce_mean(_cal, 1)
        loss = tf.reduce_mean(_cal, 0)*(-1)
        
        return tf.scalar_mul(weights, loss)
    
    
    def k1_euclidean_grad(self, norm_order=2, weights=1.0):
        """ forcing binary: gradients """
        
        fc_emb = self._predictions["fc_emb"]
        fc_emb = tf.reshape(fc_emb, [fc_emb.shape[0],-1])
        
        _cal = tf.subtract(fc_emb, 0.5) 
        _cal = tf.multiply(tf.sign(_cal), _cal)
        _cal = tf.reduce_mean(_cal, 1) 
        _cal = tf.reduce_mean(_cal, 0)
        grad = (_cal/norm_order)*(-1)
        
        return tf.scalar_mul(weights, grad)
        
    
    def k2_euclidean_loss(self, norm_order=1, weights=1.0):
        """ 50% fire for each bit: avoid preference for hidden values to be 0 or 1 """
        
        fc_emb = self._predictions["fc_emb"]
        fc_emb = tf.reshape(fc_emb, [fc_emb.shape[0],-1])
       
        fc_avg = tf.reduce_mean(fc_emb, 1)
        
        _cal = tf.subtract(fc_avg, 0.5)      
        _cal = tf.multiply(_cal, _cal)
        #_cal = tf.sqrt(_cal)
        loss = tf.reduce_mean(_cal, 0)
        
        return tf.scalar_mul(weights, loss)
    
    def k2_euclidean_grad(self, norm_order=2, weights=1.0):
        """ 50% fire for each bit: gradients """
        
        fc_emb = self._predictions["fc_emb"]
        fc_emb = tf.reshape(fc_emb, [fc_emb.shape[0],-1])
        
        fc_avg = tf.reduce_mean(fc_emb, 1)
        
        _cal = tf.subtract(fc_avg, 0.5) 
        _cal = tf.multiply(tf.sign(_cal), _cal)
        _cal = tf.reduce_mean(_cal, 0)
        grad = _cal/norm_order
        
        return tf.scalar_mul(weights, grad)
    
    
    def _add_losses(self):
        with tf.variable_scope('LOSS_' + self._tag) as scope:
            
            """ classification loss: E1 """
            cls_score = self._predictions["cls_score"]          
            labels = self._labels
                        
            if self.multilabel:
                labels = tf.reshape(labels, [-1, self._num_classes])
                loss_E1 = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits=cls_score, 
                                                                         multi_class_labels=labels,
                                                                         weights=self.alpha))
            else:
                labels = tf.reshape(labels, [-1])
                loss_E1 = tf.losses.sparse_softmax_cross_entropy(logits=cls_score, 
                                                                 labels=labels, 
                                                                 weights=self.alpha)
            self._losses['loss_E1'] = loss_E1
            
            """ forcing binary loss: E2 """
            loss_E2 = self.k1_euclidean_loss(weights=self.beta)
            tf.losses.add_loss(loss_E2)
            self._losses['loss_E2'] = loss_E2
            
            """ 50% fire for each bit loss: E3 """
            loss_E3 = self.k2_euclidean_loss(weights=self.gamma)
            tf.losses.add_loss(loss_E3)
            self._losses['loss_E3'] = loss_E3           
            
            """ overall loss: E1 + E2 + E3 + regu """
            # apha, beta, & gamma are hyperparameters balancing the total loss
            #regu_loss = tf.add_n(tf.losses.get_regularization_losses(), 'regu')
            #self._losses['total_loss'] = loss_E1 + loss_E2 + loss_E3 + regu_loss
            
            self._losses['total_loss'] = tf.losses.get_total_loss()
                       
            self._event_summaries.update(self._losses)
            
            # Evaluation metrics
            cls_pred = tf.to_int32(self._predictions["cls_pred"])
            acc_cls = tf.contrib.metrics.accuracy(predictions=cls_pred, labels=labels, name="acc_cls")
            
            #print("cls_pred.shape: ", cls_pred.shape)
            #print("labels.shape: ", labels.shape)
            """
            acc_cls = tf.equal(cls_pred, labels)
            acc_cls = tf.reduce_mean(tf.cast(acc_cls, tf.float32))            
            """
            self._accuracies['acc_cls'] = acc_cls
            
            self._event_summaries.update(self._accuracies)
    
    
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

        cls_score, cls_prob, cls_pred, fc_emb, fc7 = sess.run([self._predictions["cls_score"],
                                                          self._predictions['cls_prob'],
                                                          self._predictions['cls_pred'],
                                                          self._predictions["fc_emb"],
                                                          self._predictions["fc7"]],
                                                          feed_dict=feed_dict)
        return cls_score, cls_prob, cls_pred, fc_emb, fc7
    
    
    def train_step(self, sess, blobs, train_op):
        feed_dict = {self._images: blobs['data'], self._labels: blobs['labels']}
        total_loss, loss_E1, loss_E2, loss_E3, acc_cls, _ = sess.run([self._losses["total_loss"],
                                                                         self._losses['loss_E1'],
                                                                         self._losses['loss_E2'],
                                                                         self._losses['loss_E3'],
                                                                         self._accuracies['acc_cls'],
                                                                         train_op], feed_dict=feed_dict)
    
        return {"losses": {"total_loss": total_loss, "loss_E1": loss_E1, "loss_E2": loss_E2, "loss_E3": loss_E3},
                "accuracies": {"acc_cls": acc_cls}}
            
            
        
    def train_step_with_summary(self, sess, blobs, train_op):
        feed_dict = {self._images: blobs['data'], self._labels: blobs['labels']}
        total_loss, loss_E1, loss_E2, loss_E3, acc_cls, summary, _ = sess.run([self._losses["total_loss"],
                                                                              self._losses['loss_E1'],
                                                                              self._losses['loss_E2'],
                                                                              self._losses['loss_E3'],
                                                                              self._accuracies['acc_cls'],
                                                                              self._summary_op, train_op], feed_dict=feed_dict)
        
        return {"losses": {"total_loss": total_loss, "loss_E1": loss_E1, "loss_E2": loss_E2, "loss_E3": loss_E3},
                "accuracies": {"acc_cls": acc_cls},
                "summary": summary }
    
    
    