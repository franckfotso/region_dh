# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Project: Region-DH
# Module: models.nets.VGG16_RegionDH
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

class VGG16_RegionDH(VGG16):
    
    def __init__(self, cfg, num_bits, multilabel=True):
        VGG16.__init__(self)        
        self._predictions = {}
        self._anchor_targets = {}
        self._proposal_targets = {}
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
        
        self._feat_stride = [16, ]
        self._num_bits = num_bits
        self.alphas = cfg.TRAIN_BATCH_DET_ALPHAS
        self.betas = cfg.TRAIN_BATCH_DET_BETAS
        self.gammas_H1 = cfg.TRAIN_BATCH_DET_GAMMAS_H1        
        self.gammas_H2 = cfg.TRAIN_BATCH_DET_GAMMAS_H2
        self.cfg = cfg
        
    def create_architecture(self, mode, num_classes, tag=None, 
                           anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):        
        training = mode == 'TRAIN'
        testing = mode == 'TEST'
        
        if training:
            # mode: TRAIN
            self._images = tf.placeholder(tf.float32, shape=[1, None, None, 3])
            self._im_info = tf.placeholder(tf.float32, shape=[3])
            self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
            self._labels = tf.placeholder(tf.int32, shape=[1, num_classes-1])
            self._pos_weights = tf.placeholder(tf.float32, shape=[1, num_classes-1])
            
        else:
            # mode: TEST
            self._images = tf.placeholder(tf.float32, shape=[1, None, None, 3])  
            self._im_info = tf.placeholder(tf.float32, shape=[3])
        
        self._tag = tag    
        self._num_classes = num_classes
        self._mode = mode     
        
        self._anchor_scales = anchor_scales
        self._num_scales = len(anchor_scales)
        self._anchor_ratios = anchor_ratios
        self._num_ratios = len(anchor_ratios)
        self._num_anchors = self._num_scales * self._num_ratios
    
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
            
            rois = self._build_network(training)
    
        layers_to_output = {"rois": rois}
    
        for var in tf.trainable_variables():
            self._train_summaries.append(var)
    
        if testing:
            stds = np.tile(np.array(self.cfg.TRAIN_BATCH_DET_BBOX_NORMALIZE_STDS), (self._num_classes))
            means = np.tile(np.array(self.cfg.TRAIN_BATCH_DET_BBOX_NORMALIZE_MEANS), (self._num_classes))
            self._predictions["bbox_xy_pred"] *= stds
            self._predictions["bbox_xy_pred"] += means
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
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)        
    
        net_conv = self._image_to_head(is_training, 
                                       trainables=['conv_3', 'conv_4', 'conv_5'],
                                       last_pool=False)
        
        with tf.variable_scope(self._scope, self._scope):
            # build the anchors for the image
            layers._anchor_component(self)
            
            # region proposal network: RPN
            rois = layers._region_proposal(self, net_conv, is_training, initializer)
            num_rois = rois.shape[0]
            
            # region of interest pooling: ROIPooling
            pool5 = layers._crop_pool_layer(self, net_conv, rois, "pool5")
            
        # bbox_feat: N x b
        bbox_feat = self._head_to_tail(pool5, is_training) # fc7
        print("bbox_feat.shape: ", bbox_feat.shape)
        
        with tf.variable_scope(self._scope, self._scope):
            # rois encoding
            embs_H1 = self._image_encoding(bbox_feat, is_training, scope="embs_H1")
            print("embs_H1.shape: ", embs_H1.shape)
            
            # bbox_xy_pred: N x 4c
            bbox_xy_pred = self._region_regression(bbox_feat, is_training)
            
            # bbox_score: N x c, bbox_prob: N x c, bbox_cls_pred: N x 1
            bbox_score, bbox_prob, bbox_cls_pred = self._region_classification(embs_H1, is_training)
                        
            # cross-proposal fusion => instance-aware feature (iaf): c x b    
            # we exclude the bg class [1:] to emphasize the fg interest
            #ia_feat = tf.tensordot(tf.transpose(bbox_prob)[1:], bbox_feat, axes=1, name="ia_feat")
            ia_feat = tf.tensordot(tf.transpose(bbox_prob)[1:], embs_H1, axes=1, name="ia_feat")
            print("ia_feat.shape: ", ia_feat.shape)
            ia_feat = tf.reshape(ia_feat, [1, -1])
            print("ia_feat.shape: ", ia_feat.shape)
            
            # image encoding
            embs_H2 = self._image_encoding(ia_feat, is_training, scope="embs_H2")
            ia_feat = tf.reshape(ia_feat, [ia_feat.shape[0], -1])
            print("embs_H2.shape: ", embs_H2.shape)
            
            # im_score: 1 x c, im_prob: 1 x c, im_pred: 1 x 1
            im_score, im_prob, im_pred = self._image_classification(embs_H2, is_training) 
            
            self._predictions["bbox_xy_pred"] = bbox_xy_pred
            self._predictions["bbox_score"] = bbox_score
            self._predictions["bbox_prob"] = bbox_prob
            self._predictions["bbox_cls_pred"] = bbox_cls_pred
            self._predictions["embs_H1"] = embs_H1
            self._predictions["embs_H2"] = embs_H2
            self._predictions["im_score"] = im_score
            self._predictions["im_prob"] = im_prob
            self._predictions["im_pred"] = im_pred
            
            if not is_training:
                self._predictions["bbox_feat"] = bbox_feat
                self._predictions["ia_feat"] = ia_feat
        
        self._score_summaries.update(self._predictions)
    
        return rois
    
    
    def _head_to_tail(self, net, is_training, reuse=None):
        # net.layers[-1]: last conv layer
        
        with tf.variable_scope(self._scope, self._scope, reuse=reuse):
            
            net = slim.flatten(net, scope='flatten')
            net = slim.fully_connected(net, 4096, scope='fc6')
            if is_training:
                net = slim.dropout(net, keep_prob=0.5, is_training=True, 
                                    scope='dropout6')
            net = slim.fully_connected(net, 4096, scope='fc7')
            if is_training:
                net = slim.dropout(net, keep_prob=0.5, is_training=True, 
                                    scope='dropout7')    
        return net
    
    
    def get_variables_to_restore(self, variables, var_keep_dic):
        variables_to_restore = []
    
        for v in variables:
            """
            # exclude the conv weights that are fc weights in vgg16
            if v.name == (self._scope + '/fc6/weights:0') or \
                v.name == (self._scope + '/fc7/weights:0'):
                self._variables_to_fix[v.name] = v
                continue
            # exclude the first conv layer to swap RGB to BGR
            if v.name == (self._scope + '/conv1/conv1_1/weights:0'):
                self._variables_to_fix[v.name] = v
                continue
            """
            if v.name.split(':')[0] in var_keep_dic:
                print('Variables restored: %s' % v.name)
                variables_to_restore.append(v)
    
        return variables_to_restore 
    
    
    def fix_variables(self, sess, pretrained_model):
        print('Fix VGG16 layers...') # skip if pretrained is ckpt        
    
    
    def _image_encoding(self, net, is_training, scope="fc_emb"):
        # net.layers[-1]: last fc layer (fc7)
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.005)
        
        # fc layer: random initializer & sigmoid activation        
        fc_emb = slim.fully_connected(net, self._num_bits, 
                                      weights_initializer=initializer,
                                      trainable=is_training,
                                      activation_fn=tf.nn.sigmoid,
                                      scope=scope)        
        return fc_emb
    
      
    def _region_regression(self, net, is_training):
        # net.layers[-1]: last fc layer (fc7)
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.001)
        
        bbox_xy_pred = slim.fully_connected(net, self._num_classes * 4, 
                                     weights_initializer=initializer,
                                     trainable=is_training,
                                     activation_fn=None, scope='bbox_xy_pred')
        return bbox_xy_pred
    
        
    def _region_classification(self, net, is_training):
        # net.layers[-1]: fc_emb
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        
        bbox_score = slim.fully_connected(net, self._num_classes, 
                              weights_initializer=initializer,
                              trainable=is_training,
                              activation_fn=None,
                              scope='bbox_score')
                
        bbox_prob = tf.nn.softmax(bbox_score, name="bbox_prob")
        bbox_cls_pred = tf.argmax(bbox_score, axis=1, name="bbox_cls_pred")
        bbox_cls_pred = tf.reshape(bbox_cls_pred, [-1])
        
        return bbox_score, bbox_prob, bbox_cls_pred
    
    
    def _image_classification(self, net, is_training):
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
    
       
    def k1_euclidean_loss(self, scope="fc_emb",norm_order=2, weights=1.0):
        """ forcing binary: alternative to the quantization loss """
        
        fc_emb = self._predictions[scope]
        fc_emb = tf.reshape(fc_emb, [fc_emb.shape[0],-1])
                
        _cal = tf.subtract(fc_emb, 0.5)        
        _cal = tf.multiply(_cal, _cal)
        _cal = tf.reduce_mean(_cal, 1)
        _cal = tf.reduce_mean(_cal, 0)*(-1)
        loss = tf.scalar_mul(weights, _cal)
        tf.losses.add_loss(loss)
                
        return loss
        
    
    def k2_euclidean_loss(self, scope="fc_emb",norm_order=1, weights=1.0):
        """ 50% fire for each bit: avoid preference for hidden values to be 0 or 1 """
        
        fc_emb = self._predictions[scope]
        fc_emb = tf.reshape(fc_emb, [fc_emb.shape[0],-1])
       
        fc_avg = tf.reduce_mean(fc_emb, 1)  
        _cal = tf.subtract(fc_avg, 0.5)   
        _cal = tf.multiply(_cal, _cal)
        _cal = tf.reduce_mean(_cal, 0)
        loss = tf.scalar_mul(weights, _cal)
        tf.losses.add_loss(loss)
                
        return loss    
    
    
    def _add_losses(self, sigma_rpn=3.0):
        with tf.variable_scope('LOSS_' + self._tag) as scope:
            # RPN, bbox regression loss: L_reg1
            rpn_bbox_pred = self._predictions['rpn_bbox_pred']
            rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
            rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
            rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']
            # = rpn_loss_box
            L_reg1 = self.alphas[0]*layers._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                              rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])
            tf.losses.add_loss(L_reg1)
            self._losses['L_reg1'] = L_reg1
            
            # RPN, classification loss: L_cls1
            rpn_cls_score = tf.reshape(self._predictions['rpn_cls_score_reshape'], [-1, 2])
            rpn_label = tf.reshape(self._anchor_targets['rpn_labels'], [-1])
            rpn_select = tf.where(tf.not_equal(rpn_label, -1))
            rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
            rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
            # = rpn_cross_entropy
            L_cls1 = self.betas[0]*tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, 
                                                                                                 labels=rpn_label))
            tf.losses.add_loss(L_cls1)
            self._losses['L_cls1'] = L_cls1
            
            # RCNN, bbox regression loss: L_reg2          
            bbox_xy_pred = self._predictions["bbox_xy_pred"]
            bbox_xy_pred = tf.reshape(bbox_xy_pred, [bbox_xy_pred.shape[0], -1])
            bbox_targets = self._proposal_targets['bbox_targets']
            bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
            bbox_outside_weights = self._proposal_targets['bbox_outside_weights']
            # = loss_box
            L_reg2 = self.alphas[1]*layers._smooth_l1_loss(bbox_xy_pred, bbox_targets, 
                                                          bbox_inside_weights, 
                                                          bbox_outside_weights)
            tf.losses.add_loss(L_reg2)
            self._losses['L_reg2'] = L_reg2            

            # RCNN, classification loss: L_cls2
            bbox_score = self._predictions["bbox_score"]
            bbox_label = tf.reshape(self._proposal_targets["labels"], [-1])
            # = cross_entropy
            L_cls2 = self.betas[1]*tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=bbox_score, 
                                                                                                 labels=bbox_label))
            tf.losses.add_loss(L_cls2)
            self._losses['L_cls2'] = L_cls2
            
            # Hashing, instance-aware loss: L_H1 (E1 & E2)
            # == forcing binary loss: E1
            L_H1_E1 = self.k1_euclidean_loss(scope="embs_H1", weights=self.gammas_H1[0])            
            # == 50% fire for each bit loss: E2
            L_H1_E2 = self.k2_euclidean_loss(scope="embs_H1", weights=self.gammas_H1[1])            
            #self._losses['L_H1_E1'] = L_H1_E1
            #self._losses['L_H1_E2'] = L_H1_E2
            self._losses['L_H1'] = L_H1_E1 + L_H1_E2
            
            # Hashing, image loss: L_H2 (E1 & E2)
            # == forcing binary loss: E1
            L_H2_E1 = self.k1_euclidean_loss(scope="embs_H2", weights=self.gammas_H2[0])            
            # == 50% fire for each bit loss: E2
            L_H2_E2 = self.k2_euclidean_loss(scope="embs_H2", weights=self.gammas_H2[1])
            #self._losses['L_H2_E1'] = L_H2_E1
            #self._losses['L_H2_E2'] = L_H2_E2
            self._losses['L_H2'] = L_H2_E1 + L_H2_E2
            
            # RCNN, classification loss: L_cls3
            im_score = self._predictions["im_score"]
            im_label = self._labels          
            im_label = tf.reshape(im_label, [-1, self._num_classes-1])
            
            # weighting loss
            pos_weights = tf.reshape(self._pos_weights, [self._pos_weights.shape[0], -1])                
            #print("pos_weights.shape: ", pos_weights.shape)
            
            L_cls3 = 0
            for i in range(self._num_classes-1):
                pos_weight = pos_weights[:, i]
                pos_weight = tf.reshape(pos_weight, [pos_weight.shape[0], -1])

                _cls_score = im_score[:, i, :]
                _labels = im_label[:, i]

                _labels = tf.reshape(_labels, [-1])
                _L_cls3 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=_cls_score, 
                                                                                        labels=_labels,
                                                                                        name="L_cls3."+str(i)))
                #_loss_E1 = _loss_E1*pos_weight[0] # weighting loss
                tf.losses.add_loss(_L_cls3)
                L_cls3 += _L_cls3  
            
            self._losses['L_cls3'] = L_cls3                     
            
            # Overall loss: L_reg(1,2) + L_cls(1,2,3) + L_H(1,2) + regu
            self._losses['total_loss'] = tf.losses.get_total_loss()
                       
            self._event_summaries.update(self._losses)
            self._loss_summaries.update(self._losses)
            
            self._losses['L_H1_E1'] = L_H1_E1
            self._losses['L_H1_E2'] = L_H1_E2
            self._losses['L_H2_E1'] = L_H2_E1
            self._losses['L_H2_E2'] = L_H2_E2
            
            # Evaluation metrics
            im_pred = tf.to_int32(self._predictions["im_pred"])
            bbox_cls_pred = tf.to_int32(self._predictions["bbox_cls_pred"])
            
            im_acc_cls, _, _ = layers._precision_recall_score(im_label, im_pred, "prec_rec_score")
            bbox_acc_cls = tf.contrib.metrics.accuracy(predictions=bbox_cls_pred, labels=bbox_label, name="bbox_acc_cls")
            
            self._accuracies['im_acc_cls'] = im_acc_cls
            self._accuracies['bbox_acc_cls'] = bbox_acc_cls            
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
                     self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes'],
                     self._labels: blobs['labels'],
                     self._pos_weights: blobs['pos_weights']}
        
        summary = sess.run(self._summary_op_val, feed_dict=feed_dict)
    
        return summary
    
    
    # only useful during testing mode
    def test_image(self, sess, blobs):
        feed_dict = {self._images: blobs['data'], self._im_info: blobs['im_info']}
        
        bbox_score, bbox_prob, bbox_pred, \
        im_score, im_prob, im_pred, \
        embs_H1, embs_H2, bbox_feat, ia_feat = sess.run([self._predictions["bbox_score"],
                                                         self._predictions['bbox_prob'],
                                                         self._predictions['bbox_xy_pred'],
                                                         self._predictions['im_score'],
                                                         self._predictions['im_prob'],
                                                         self._predictions['im_pred'],
                                                         self._predictions["embs_H1"],
                                                         self._predictions["embs_H2"],
                                                         self._predictions["bbox_feat"],
                                                         self._predictions["ia_feat"]],
                                                          feed_dict=feed_dict)
        
        #return bbox_score, bbox_prob, bbox_pred, im_score, im_prob, im_pred, embs_H1, embs_H2, bbox_feat, ia_feat
        
        return im_score, im_prob, im_pred, embs_H2, ia_feat
    
    
    def train_step(self, sess, blobs, train_op):
        feed_dict = {self._images: blobs['data'],
                     self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes'],
                     self._labels: blobs['labels'], 
                     self._pos_weights: blobs['pos_weights']}
        
        total_loss, L_reg1, L_reg2, L_cls1, L_cls2, \
        L_cls3, L_H1_E1, L_H1_E2, L_H2_E1, L_H2_E2, \
        im_acc_cls, bbox_acc_cls, _ = sess.run([self._losses["total_loss"],
                                                    self._losses['L_reg1'],
                                                    self._losses['L_reg2'],
                                                    self._losses['L_cls1'],
                                                    self._losses['L_cls2'],
                                                    self._losses['L_cls3'],
                                                    self._losses['L_H1_E1'],
                                                    self._losses['L_H1_E2'],
                                                    self._losses['L_H2_E1'],
                                                    self._losses['L_H2_E2'],
                                                    self._accuracies['im_acc_cls'],
                                                    self._accuracies['bbox_acc_cls'],
                                                    train_op], feed_dict=feed_dict)    
        return {"losses": {"total_loss": total_loss, 
                           "L_reg1": L_reg1, 
                           "L_reg2": L_reg2, 
                           "L_cls1": L_cls1,
                           "L_cls2": L_cls2,
                           "L_cls3": L_cls3,
                           "L_H1_E1": L_H1_E1,
                           "L_H1_E2": L_H1_E2,
                           "L_H2_E1": L_H2_E1,
                           "L_H2_E2": L_H2_E2
                          },
                "accuracies": {"im_acc_cls": im_acc_cls,
                               "bbox_acc_cls": bbox_acc_cls,
                              }
               }          
            
        
    def train_step_with_summary(self, sess, blobs, train_op):
        feed_dict = {self._images: blobs['data'],
                     self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes'],
                     self._labels: blobs['labels'],
                     self._pos_weights: blobs['pos_weights']}
        
        total_loss, L_reg1, L_reg2, L_cls1, L_cls2, \
        L_cls3, L_H1_E1, L_H1_E2, L_H2_E1, L_H2_E2, \
        im_acc_cls, bbox_acc_cls, summary, _ = sess.run([self._losses["total_loss"],
                                                    self._losses['L_reg1'],
                                                    self._losses['L_reg2'],
                                                    self._losses['L_cls1'],
                                                    self._losses['L_cls2'],
                                                    self._losses['L_cls3'],
                                                    self._losses['L_H1_E1'],
                                                    self._losses['L_H1_E2'],
                                                    self._losses['L_H2_E1'],
                                                    self._losses['L_H2_E2'],
                                                    self._accuracies['im_acc_cls'],
                                                    self._accuracies['bbox_acc_cls'],
                                                    self._summary_op, train_op], feed_dict=feed_dict)
        
        return {"losses": {"total_loss": total_loss, 
                           "L_reg1": L_reg1, 
                           "L_reg2": L_reg2, 
                           "L_cls1": L_cls1,
                           "L_cls2": L_cls2,
                           "L_cls3": L_cls3,
                           "L_H1_E1": L_H1_E1,
                           "L_H1_E2": L_H1_E2,
                           "L_H2_E1": L_H2_E1,
                           "L_H2_E2": L_H2_E2
                          },
                "accuracies": {"im_acc_cls": im_acc_cls,
                               "bbox_acc_cls": bbox_acc_cls,
                              },
                "summary": summary
               }
    