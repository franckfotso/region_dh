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
from nets.VGG16_DLBHC import VGG16_DLBHC
from datasets.Image import Image
from datasets.IMGenerator import IMGenerator


def tf_init_feat(weights, net_name, num_cls, num_bits, techno, cfg):
    
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True
    # init session
    sess = tf.Session(config=tfconfig)
        
    net = None    
    if net_name == "VGG16":
        if techno == "DLBHC":
            net = VGG16_DLBHC(cfg, num_bits)
        else:
            raise NotImplemented
        net.create_architecture('TEST', num_cls, tag='default')
        
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
    #label_blob = data_gen.built_label_blob(images)
        
    return net.test_image(sess, im_blob)


def tf_batch_feat(im_pns, sess, net, techno, cfg):
    
    images = []
    
    for im_pn in im_pns:
        im_fn = im_pn[im_pn.rfind(os.path.sep) + 1:]
        img = Image(im_fn, im_pn)
        images.append(img)
        
    data_gen = IMGenerator(images, None, cfg)
    
    _, _, _, fc_hash, fc7 = _batch_foward(sess, net, images, data_gen, cfg)
    
    if techno == "DLBHC":
        # apply a binary thresholding: 0.5
        binary_codes = np.where(fc_hash >= 0.5, 1, 0)
    else:
        # thresholding already applied into the net
        binary_codes = fc_hash
        
    return binary_codes, fc7