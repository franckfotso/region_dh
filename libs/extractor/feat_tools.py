# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Project: Region-DH
# Module: libs.extractor.DeepFeaturesExtractor
# Copyright (c) 2018
# Written by: Franck FOTSO
# Licensed under MIT License
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os
import numpy as np
import numpy.random as npr
import tensorflow as tf
from nets.VGG16_FT import VGG16_FT
from nets.VGG16_DLBHC import VGG16_DLBHC
from nets.VGG16_SSDH1 import VGG16_SSDH1
from nets.VGG16_SSDH2 import VGG16_SSDH2
from nets.VGG16_RegionDH import VGG16_RegionDH
from datasets.Image import Image
from datasets.IMGenerator import IMGenerator
from datasets.ROIGenerator import ROIGenerator 


def tf_init_feat(weights, net_name, num_cls, num_bits, techno, multilabel, cfg):
    
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True
    # init session
    sess = tf.Session(config=tfconfig)
        
    net = None    
    if net_name == "VGG16":
        if techno == "DLBHC":
            net = VGG16_DLBHC(cfg, num_bits, multilabel)
            net.create_architecture('TEST', num_cls, tag='default')
            
        elif techno == "FT":
            net = VGG16_FT(cfg, multilabel)
            net.create_architecture('TEST', num_cls, tag='default')
            
        elif techno == "SSDH1":
            net = VGG16_SSDH1(cfg, num_bits, multilabel)
            net.create_architecture('TEST', num_cls, tag='default')
            
        elif techno == "SSDH2":
            net = VGG16_SSDH2(cfg, num_bits, multilabel)
            net.create_architecture('TEST', num_cls, tag='default')
            
        elif techno == "Region-DH":
            net = VGG16_RegionDH(cfg, num_bits)
            net.create_architecture('TEST', num_cls, tag='default',
                                  anchor_scales=cfg.TRAIN_BATCH_DET_ANCHOR_SCALES,
                                  anchor_ratios=cfg.TRAIN_BATCH_DET_ANCHOR_RATIOS)                   
        else:
            raise NotImplementedError       
        
    elif net_name == "RESNET50":
        raise NotImplemented
    
    assert net != None, "[ERROR] wrong network provided, found: {}".format(net_name)
    
    print(('Loading model from {:s}').format(weights))
    saver = tf.train.Saver()
    saver.restore(sess, weights)
    print('Loaded.')
    
    return sess, net


def _batch_foward(sess, net, images, data_gen, cfg):
    random_scale_inds = npr.randint(0, high=len(cfg.TEST_DEFAULT_SCALES),size=len(images))
    
    im_blob, im_scales = data_gen.built_image_blob(images, random_scale_inds)    
    im_info = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)
    #print("im_info: ", im_info)
    
    blobs = {"data": im_blob, "im_info": im_info}
        
    return net.test_image(sess, blobs)


def tf_batch_feat(images, sess, net, techno, cfg):
        
    if net.multilabel:
        data_gen = ROIGenerator(images, None, None, cfg)        
    else:
        data_gen = IMGenerator(images, None, cfg)
    
    _, _, _, fc_hash, fc7 = _batch_foward(sess, net, images, data_gen, cfg)
    
    if techno in ["DLBHC"]:
        # apply a binary thresholding: 0.5
        binary_codes = np.where(fc_hash >= 0.5, 1, 0)
        
    elif techno in ["SSDH1", "SSDH2", "Region-DH"]:
        binary_codes = (np.sign(fc_hash - 0.5) + 1)/2
        
    else:
        raise NotImplementedError
        
    return binary_codes, fc7