# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Project: Region-DH
# Module: libs.region_dh.faster_rcnn
# Copyright (c) 2018
# Written by: Franck FOTSO
# Based on: tf-faster-rcnn 
#    (https://github.com/endernewton/tf-faster-rcnn)
# Licensed under MIT License
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim

from layer_utils.snippets import generate_anchors_pre, generate_anchors_pre_tf
from layer_utils.proposal_layer import proposal_layer, proposal_layer_tf
#from layer_utils.proposal_top_layer import proposal_top_layer, proposal_top_layer_tf
from layer_utils.anchor_target_layer import anchor_target_layer
from layer_utils.proposal_target_layer import proposal_target_layer

from sklearn.metrics import precision_score, recall_score


def _reshape_layer(bottom, num_dim, name):
    input_shape = tf.shape(bottom)
    with tf.variable_scope(name) as scope:
        # change the channel to the caffe format
        to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
        # then force it to have channel 2
        reshaped = tf.reshape(to_caffe,
                            tf.concat(axis=0, values=[[1, num_dim, -1], [input_shape[2]]]))
        # then swap the channel back
        to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
        return to_tf
    

def _softmax_layer(bottom, name):
    if name.startswith('rpn_cls_prob_reshape'):
        input_shape = tf.shape(bottom)
        bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
        reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
        return tf.reshape(reshaped_score, input_shape)
    return tf.nn.softmax(bottom, name=name)    

"""
def proposal_top_layer(net, rpn_cls_prob, rpn_bbox_pred, name):
    with tf.variable_scope(name) as scope:
        if net.cfg.TRAIN_DEFAULT_USE_E2E_TF:
            rois, rpn_scores = proposal_top_layer_tf(
              rpn_cls_prob,
              rpn_bbox_pred,
              net._im_info,
              net._feat_stride,
              net._anchors,
              net._num_anchors
            )
        else:
            raise NotImplemented

    return rois, rpn_scores
"""

def _proposal_layer(net, rpn_cls_prob, rpn_bbox_pred, name):
    with tf.variable_scope(name) as scope:
        if net.cfg.TRAIN_DEFAULT_USE_E2E_TF:
            rois, rpn_scores = proposal_layer_tf(
              rpn_cls_prob,
              rpn_bbox_pred,
              net._im_info,
              net._mode,
              net._feat_stride,
              net._anchors,
              net._num_anchors, 
              net.cfg
            )
        else:
            raise NotImplemented
            
        #print("rois.shape: ", rois.shape)
        #print("rpn_scores.shape: ", rpn_scores.shape)

        rois.set_shape([None, 5])
        rpn_scores.set_shape([None, 1])

    return rois, rpn_scores


# Only use it if you have roi_pooling op written in tf.image
def _roi_pool_layer(bootom, rois, name):
    with tf.variable_scope(name) as scope:
        return tf.image.roi_pooling(bootom, rois,
                                  pooled_height=cfg.TRAIN_BATCH_DET_POOLING_SIZE,
                                  pooled_width=cfg.TRAIN_BATCH_DET_POOLING_SIZE,
                                  spatial_scale=1. / 16.)[0]

def _crop_pool_layer(net, bottom, rois, name):
    with tf.variable_scope(name) as scope:
        
        batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
        # Get the normalized coordinates of bounding boxes
        bottom_shape = tf.shape(bottom)
        height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(net._feat_stride[0])
        width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(net._feat_stride[0])                
        x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
        y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
        x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
        y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
        
        # Won't be back-propagated to rois anyway, but to save time
        bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
        pre_pool_size = net.cfg.TRAIN_BATCH_DET_POOLING_SIZE * 2
        crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), 
                                         [pre_pool_size, pre_pool_size], name="crops")

    return slim.max_pool2d(crops, [2, 2], padding='SAME')


def _dropout_layer(bottom, name, ratio=0.5):
    return tf.nn.dropout(bottom, ratio, name=name)

def precision_recall_score(y_true, y_pred):
    f1_score, precision, recall = (0, 0, 0)
    
    for y1, y2 in zip(y_true, y_pred):
        precision += precision_score(y1, y2, average='weighted')
        recall += recall_score(y1, y2, average='weighted')
        
    precision = precision/y_true.shape[0]
    recall = recall/y_true.shape[0]
    
    if precision <=0 or recall <= 0:
        f1_score = 0.0
    else:
        f1_score = 2*precision*recall/(precision+recall)
    
    return np.float32(f1_score), np.float32(precision), np.float32(recall)

def _precision_recall_score(y_true, y_pred, name):
    with tf.variable_scope(name) as scope:
        f1_score, precision, recall = tf.py_func(precision_recall_score, 
                                       [y_true, y_pred], 
                                       [tf.float32, tf.float32, tf.float32], 
                                                 name="precision_recall_score")
    return f1_score, precision, recall


def print_shape(var):
    print("_print_shape, var.shape: ", var.shape)

def _print_shape(var, name):
    with tf.variable_scope(name) as scope:
        tf.py_func(print_shape, [var], [tf.float32], name="_print_shape")

def _anchor_target_layer(net, rpn_cls_score, name):
    with tf.variable_scope(name) as scope:
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, \
        rpn_bbox_outside_weights = tf.py_func(anchor_target_layer,
        [rpn_cls_score, net._gt_boxes, net._im_info, net._feat_stride, net._anchors, net._num_anchors],
        [tf.float32, tf.float32, tf.float32, tf.float32], name="anchor_target")

        rpn_labels.set_shape([1, 1, None, None])
        rpn_bbox_targets.set_shape([1, None, None, net._num_anchors * 4])
        rpn_bbox_inside_weights.set_shape([1, None, None, net._num_anchors * 4])
        rpn_bbox_outside_weights.set_shape([1, None, None, net._num_anchors * 4])

        rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
        net._anchor_targets['rpn_labels'] = rpn_labels
        net._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
        net._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
        net._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

        net._score_summaries.update(net._anchor_targets)

    return rpn_labels


def _proposal_target_layer(net, rois, roi_scores, name):
    with tf.variable_scope(name) as scope:
        rois, roi_scores, labels, bbox_targets, \
        bbox_inside_weights, bbox_outside_weights = tf.py_func(proposal_target_layer,
        [rois, roi_scores, net._gt_boxes, net._num_classes],
        [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32], name="proposal_target")

        rois.set_shape([net.cfg.TRAIN_BATCH_DET_BATCH_SIZE, 5])
        roi_scores.set_shape([net.cfg.TRAIN_BATCH_DET_BATCH_SIZE])
        labels.set_shape([net.cfg.TRAIN_BATCH_DET_BATCH_SIZE, 1])
        bbox_targets.set_shape([net.cfg.TRAIN_BATCH_DET_BATCH_SIZE, net._num_classes * 4])
        bbox_inside_weights.set_shape([net.cfg.TRAIN_BATCH_DET_BATCH_SIZE, net._num_classes * 4])
        bbox_outside_weights.set_shape([net.cfg.TRAIN_BATCH_DET_BATCH_SIZE, net._num_classes * 4])

        net._proposal_targets['rois'] = rois
        net._proposal_targets['labels'] = tf.to_int32(labels, name="to_int32")
        net._proposal_targets['bbox_targets'] = bbox_targets
        net._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
        net._proposal_targets['bbox_outside_weights'] = bbox_outside_weights

        net._score_summaries.update(net._proposal_targets)

    return rois, roi_scores


def _anchor_component(net):
    with tf.variable_scope('ANCHOR_' + net._tag) as scope:
        # just to get the shape right
        height = tf.to_int32(tf.ceil(net._im_info[0] / np.float32(net._feat_stride[0])))
        width = tf.to_int32(tf.ceil(net._im_info[1] / np.float32(net._feat_stride[0])))  
        
        if net.cfg.TRAIN_DEFAULT_USE_E2E_TF:
            anchors, anchor_length = generate_anchors_pre_tf(
              height,
              width,
              net._feat_stride,
              net._anchor_scales,
              net._anchor_ratios
            )
        else:
            raise NotImplemented
            
        anchors.set_shape([None, 4])
        anchor_length.set_shape([])
        
        net._anchors = anchors
        net._anchor_length = anchor_length


def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = tf.abs(in_box_diff)
    smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
    in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = tf.reduce_mean(tf.reduce_sum(
      out_loss_box,
      axis=dim
    ))
    return loss_box


def _region_proposal(net, net_conv, is_training, initializer):
    rpn = slim.conv2d(net_conv, net.cfg.TRAIN_BATCH_DET_RPN_CHANNELS, [3, 3], 
                      trainable=is_training, weights_initializer=initializer,
                        scope="rpn_conv/3x3")
    net._act_summaries.append(rpn)
    rpn_cls_score = slim.conv2d(rpn, net._num_anchors * 2, [1, 1], trainable=is_training,
                                weights_initializer=initializer,
                                padding='VALID', activation_fn=None, scope='rpn_cls_score')
    # change it so that the score has 2 as its channel size
    rpn_cls_score_reshape = _reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
    rpn_cls_prob_reshape = _softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
    rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score_reshape, [-1, 2]), axis=1, name="rpn_cls_pred")
    rpn_cls_prob = _reshape_layer(rpn_cls_prob_reshape, net._num_anchors * 2, "rpn_cls_prob")
    rpn_bbox_pred = slim.conv2d(rpn, net._num_anchors * 4, [1, 1], trainable=is_training,
                                weights_initializer=initializer,
                                padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
    if is_training:
        rois, roi_scores = _proposal_layer(net, rpn_cls_prob, rpn_bbox_pred, "rois")
        rpn_labels = _anchor_target_layer(net, rpn_cls_score, "anchor")
        
        # Try to have a deterministic order for the computing graph, for reproducibility
        with tf.control_dependencies([rpn_labels]):
            rois, _ = _proposal_target_layer(net, rois, roi_scores, "rpn_rois")
        
    else:
        # TEST MODE: NMS
        rois, _ = _proposal_layer(net, rpn_cls_prob, rpn_bbox_pred, "rois")

    net._predictions["rpn_cls_score"] = rpn_cls_score
    net._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
    net._predictions["rpn_cls_prob"] = rpn_cls_prob
    net._predictions["rpn_cls_pred"] = rpn_cls_pred
    net._predictions["rpn_bbox_pred"] = rpn_bbox_pred
    net._predictions["rois"] = rois

    return rois

"""
def region_classification(net, fc7, is_training, initializer, initializer_bbox):
    cls_score = slim.fully_connected(fc7, net._num_classes, 
                                       weights_initializer=initializer,
                                       trainable=is_training,
                                       activation_fn=None, scope='cls_score')
    cls_prob = softmax_layer(cls_score, "cls_prob")
    cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")
    bbox_pred = slim.fully_connected(fc7, net._num_classes * 4, 
                                     weights_initializer=initializer_bbox,
                                     trainable=is_training,
                                     activation_fn=None, scope='bbox_pred')
    
    net._predictions["cls_score"] = cls_score
    net._predictions["cls_pred"] = cls_pred
    net._predictions["cls_prob"] = cls_prob
    net._predictions["bbox_pred"] = bbox_pred
    
    return cls_prob, bbox_pred
"""