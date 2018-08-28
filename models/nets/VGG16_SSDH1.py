# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Project: Region-DH
# Module: models.nets.VGG16_SSDH
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

""" SSDH implementation: single fc_emb, multiple binary classifiers"""

class VGG16_SSDH1(VGG16):
    
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
        self._loss_summaries = {}
        self._acc_summaries = {}
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
        fc7 = self._head_to_tail(pool5, is_training, global_pool="MEAN")
        
        with tf.variable_scope(self._scope, self._scope):
            # compute embeddings
            fc_emb = self._image_encoding(fc7, is_training, scope="fc_emb")        
            
            if self.multilabel:
                # multi-label dataset
                cls_score, cls_prob, cls_pred = self._multi_image_classification(fc_emb, is_training)
            else:
                # single-label dataset
                cls_score, cls_prob, cls_pred = self._single_image_classification(fc_emb, is_training)
                
            print("cls_score.shape: ", cls_score.shape)
            print("cls_prob.shape: ", cls_prob.shape)
            print("cls_pred.shape: ", cls_pred.shape)
            
            self._predictions["cls_score"] = cls_score
            self._predictions["cls_prob"] = cls_prob
            self._predictions["cls_pred"] = cls_pred  
            
        if not is_training:
            self._predictions["fc7"] = tf.reshape(fc7, [fc7.shape[0],-1])
            self._predictions["fc_emb"] = tf.reshape(fc_emb, [fc_emb.shape[0],-1])
        else:
            self._predictions["fc_emb"] = fc_emb
            
        self._score_summaries.update(self._predictions)
        
        return cls_prob, cls_pred
    
    
    def _image_encoding(self, net, is_training, scope):
        # net.layers[-1]: last fc layer (fc7)
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.005)
        
        # fc layer: random initializer & sigmoid activation
        fc_emb = slim.conv2d(net, self._num_bits, [1, 1],
                          weights_initializer=initializer,
                          trainable=is_training,
                          activation_fn=tf.nn.sigmoid, scope=scope)
                    
        return fc_emb
    
    
    def _single_image_classification(self, net, is_training):
        # net.layers[-1]: fc hash
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        num_cls = self._num_classes
        
        cls_score = slim.fully_connected(net, num_cls, 
                              weights_initializer=initializer,
                              trainable=is_training,
                              activation_fn=None,
                              scope='cls_score')        
        print("cls_score.shape: ", cls_score.shape)
        
        cls_prob = tf.nn.softmax(cls_score, name="cls_prob")
        cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")
        cls_pred = tf.reshape(cls_pred, [-1]) 
        
        return cls_score, cls_prob, cls_pred
    
    
    def _multi_image_classification(self, net, is_training):
        # using binary softmax classifiers        
        cls_score, cls_prob, cls_pred = ([], [], [])
        
        for i in range(self._num_classes-1):
            initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)

            _cls_score = slim.fully_connected(net, 2, 
                          weights_initializer=initializer,
                          trainable=is_training,
                          activation_fn=None,
                          scope='cls_score'+str(i))

            _cls_prob = tf.nn.softmax(_cls_score, name="cls_prob"+str(i))
            _cls_prob = tf.reshape(_cls_prob, [_cls_prob.shape[0], -1])
            _cls_pred = tf.argmax(_cls_prob, axis=1, name="cls_pred"+str(i))

            _cls_score = tf.reshape(_cls_score, [_cls_score.shape[0], 1, -1])
            _cls_prob = tf.reshape(_cls_prob, [_cls_prob.shape[0], 1, -1])
            _cls_pred = tf.reshape(_cls_pred, [_cls_pred.shape[0], 1, -1])                             
            cls_score.append(_cls_score)
            cls_prob.append(_cls_prob)
            cls_pred.append(_cls_pred)                                     

        cls_score = tf.concat(cls_score, axis=1, name="cls_score")
        cls_prob = tf.concat(cls_prob, axis=1, name="cls_prob")
        cls_pred = tf.concat(cls_pred, axis=1, name="cls_pred")
        
        return cls_score, cls_prob, cls_pred
    
    def k1_euclidean_loss(self, norm_order=2, weights=1.0):
        """ forcing binary: alternative to the quantization loss """
        
        fc_emb = self._predictions["fc_emb"]
        fc_emb = tf.reshape(fc_emb, [fc_emb.shape[0],-1])
                
        _cal = tf.subtract(fc_emb, 0.5)        
        _cal = tf.multiply(_cal, _cal)
        _cal = tf.reduce_mean(_cal, 1)
        _cal = tf.reduce_mean(_cal, 0)*(-1)
        loss = tf.scalar_mul(weights, _cal)
        
        tf.losses.add_loss(loss)        
        self._losses['loss_E2'] = loss
        
        return loss
        
    
    def k2_euclidean_loss(self, norm_order=1, weights=1.0):
        """ 50% fire for each bit: avoid preference for hidden values to be 0 or 1 """
        
        fc_emb = self._predictions["fc_emb"]
        fc_emb = tf.reshape(fc_emb, [fc_emb.shape[0],-1])
       
        fc_avg = tf.reduce_mean(fc_emb, 1)  
        _cal = tf.subtract(fc_avg, 0.5)   
        _cal = tf.multiply(_cal, _cal)
        _cal = tf.reduce_mean(_cal, 0)
        loss = tf.scalar_mul(weights, _cal)
        
        tf.losses.add_loss(loss)        
        self._losses['loss_E3'] = loss
        
        return loss    
    
    def margin_loss(self, logits, labels, pos_weights):
        prob = tf.nn.sigmoid(logits)
        cal = tf.subtract(labels, prob)
        cal = tf.multiply(cal, cal)
        cal = tf.multiply(cal, 0.5)
        cal = tf.multiply(cal, pos_weights)
        cal = tf.reduce_mean(cal, 0)        
        loss = tf.reduce_mean(cal)
        
        return loss
    
    
    def _add_losses(self):
        with tf.variable_scope('LOSS_' + self._tag) as scope:
            
            """ classification loss: E1 """
            cls_score = self._predictions["cls_score"]            
            labels = self._labels
            
            labels = tf.reshape(labels, [labels.shape[0], -1])
            print("cls_score.shape: ", cls_score.shape)
            print("labels.shape: ", labels.shape)
                        
            if self.multilabel:
                pos_weights = tf.reshape(self._pos_weights, [self._pos_weights.shape[0], -1])                
                #print("pos_weights.shape: ", pos_weights.shape)               
                
                loss_E1 = 0
                for i in range(self._num_classes-1):
                    pos_weight = pos_weights[:, i]
                    pos_weight = tf.reshape(pos_weight, [pos_weight.shape[0], -1])
                
                    _cls_score = cls_score[:, i, :]
                    _labels = labels[:, i]
                    
                    _labels = tf.reshape(_labels, [-1])
                    _loss_E1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=_cls_score, 
                                                                                            labels=_labels,
                                                                                            name="loss_E1."+str(i)))
                    #_loss_E1 = _loss_E1*pos_weight[0] # weighting loss
                    tf.losses.add_loss(_loss_E1)
                    loss_E1 += _loss_E1                       
            else:
                labels = tf.reshape(labels, [-1])
                loss_E1 = tf.losses.sparse_softmax_cross_entropy(logits=cls_score, 
                                                                 labels=labels, 
                                                                 weights=self.alpha)
            self._losses['loss_E1'] = loss_E1
            
            """ forcing binary loss: E2 """
            loss_E2 = self.k1_euclidean_loss(weights=self.beta) # loss added inside
            
            """ 50% fire for each bit loss: E3 """
            loss_E3 = self.k2_euclidean_loss(weights=self.gamma) # loss added inside        
                              
            """ overall loss: E1 + E2 + E3 + regu """            
            self._losses['total_loss'] = tf.losses.get_total_loss()
                       
            self._event_summaries.update(self._losses)
            self._loss_summaries.update(self._losses)
            
            # Evaluation metrics
            cls_pred = self._predictions["cls_pred"]
            
            if self.multilabel:
                print("labels.shape: ", labels.shape)
                print("cls_pred.shape: ", cls_pred.shape)
                acc_cls, _, _ = layers._precision_recall_score(labels, cls_pred, "acc_cls")
            else:
                acc_cls = tf.contrib.metrics.accuracy(predictions=cls_preds, labels=labels, name="acc_cls")
            
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

        cls_score, cls_prob, cls_pred, fc_emb, fc7 = sess.run([self._predictions["cls_score"],
                                                          self._predictions['cls_prob'],
                                                          self._predictions['cls_pred'],
                                                          self._predictions["fc_emb"],
                                                          self._predictions["fc7"]],
                                                          feed_dict=feed_dict)
        return cls_score, cls_prob, cls_pred, fc_emb, fc7
    
    
    def train_step(self, sess, blobs, train_op):
        feed_dict = {self._images: blobs['data'], 
                     self._labels: blobs['labels']}
        if self.multilabel:
            feed_dict[self._pos_weights] = blobs['pos_weights']
            
        total_loss, loss_E1, loss_E2, \
        loss_E3, acc_cls, _ = sess.run([self._losses["total_loss"],
                                         self._losses['loss_E1'],
                                         self._losses['loss_E2'],
                                         self._losses['loss_E3'],
                                         self._accuracies['acc_cls'],
                                         train_op], feed_dict=feed_dict)
    
        return {"losses": {"total_loss": total_loss, "loss_E1": loss_E1, "loss_E2": loss_E2, "loss_E3": loss_E3},
                "accuracies": {"acc_cls": acc_cls}}
            
            
        
    def train_step_with_summary(self, sess, blobs, train_op):
        feed_dict = {self._images: blobs['data'], 
                     self._labels: blobs['labels']}
        if self.multilabel:
            feed_dict[self._pos_weights] = blobs['pos_weights']
            
        total_loss, loss_E1, loss_E2, \
        loss_E3, acc_cls, summary, _ = sess.run([self._losses["total_loss"],
                                                  self._losses['loss_E1'],
                                                  self._losses['loss_E2'],
                                                  self._losses['loss_E3'],
                                                  self._accuracies['acc_cls'],
                                                  self._summary_op, train_op], feed_dict=feed_dict)
        
        return {"losses": {"total_loss": total_loss, "loss_E1": loss_E1, "loss_E2": loss_E2, "loss_E3": loss_E3},
                "accuracies": {"acc_cls": acc_cls},
                "summary": summary } 
    