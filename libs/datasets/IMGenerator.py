# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Project: Region-DH
# Module: libs.datasets.IMGenerator
# Copyright (c) 2018
# Written by: Franck FOTSO
# Based on: py-faster-rcnn 
#    (https://github.com/rbgirshick/py-faster-rcnn)
# Licensed under MIT License
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import scipy.io as sio
import numpy as np
import numpy.random as npr

import cv2

#from utils.regression import *
#from utils.transformation import *

class IMGenerator(object):
        
    def __init__(self, images, dataset, cfg):
        self._images = images
        self._cfg = cfg
        self.dataset = dataset
        
        self._cur_idx, self._perm_ids = self.shuffe_images()

      
    def shuffe_images(self):
        """Randomly permute the training images"""        
        
        self.perm_ids = np.random.permutation(np.arange(len(self.images)))        
        self.cur_idx = 0
        
        return self.cur_idx, self.perm_ids
    
    
    def get_next_minibatch_ids(self):
        
        if self.cur_idx + self.cfg.TRAIN_BATCH_CFC_NUM_IMG >= len(self.images):
            self.shuffe_images()
            
        batch_ids = self.perm_ids[self.cur_idx:self.cur_idx+self.cfg.TRAIN_BATCH_CFC_NUM_IMG]
        self.cur_idx += self.cfg.TRAIN_BATCH_CFC_NUM_IMG
        
        return batch_ids
    
    
    def get_next_minibatch(self):
        
        batch_ids = self.get_next_minibatch_ids()
        minibatch_imgs = [self.images[i] for i in batch_ids]
        
        return self.get_minibatch(minibatch_imgs)
        
        
    def get_minibatch(self, images):
        """Given an image obj, construct a minibatch sampled from it."""
        
        num_imgs = len(images)
        random_scale_inds = npr.randint(0, high=len(self.cfg.TRAIN_DEFAULT_SCALES),
                                    size=num_imgs)
        
        assert self.cfg.TRAIN_BATCH_CFC_NUM_IMG % num_imgs == 0, \
            "[ERROR] wrong size for the minibatch, found: {}".format(len(images))
        
        """ build blobs for images """
        
        # resize & build data blob: according caffe format
        im_blob, im_scales = self.built_image_blob(images, random_scale_inds)
        label_blob = self.built_label_blob(images)
        
        blobs = {'data': im_blob, "labels": label_blob }
        
        assert self.cfg.TRAIN_BATCH_CFC_NUM_IMG % len(im_scales) == 0, "Wrong batch size"
        assert self.cfg.TRAIN_BATCH_CFC_NUM_IMG % len(images) == 0, "Wrong batch size"
        
        return blobs
    
    
    def built_label_blob(self, images):
        num_images = len(images)
        
        if self.dataset.name in ["voc_2007","voc_2012","nus_wide"]:
            # multi-label data        
            blob = np.zeros((num_images, self.dataset.num_cls), dtype=np.int32)
            for im_i in range(num_images):
                image = images[im_i]
                gt_classes = image.gt_rois["gt_classes"]
                classes = np.unique(gt_classes)
                blob[im_i, classes] = 1
                
        elif self.dataset.name in ["cifar10", "cifar100"]:
            # single-label data
            blob = np.zeros((num_images, 1), dtype=np.int32)
            for im_i in range(num_images):
                image = images[im_i]
                blob[im_i] = image.label
        else:
            raise NotImplemented
            
        return blob
            
        
    def built_image_blob(self, images, scale_inds):
        num_imgs = len(images)
        
        for im_i in range(num_imgs):
            im_RAW = cv2.imread(images[im_i].pathname) 
               
            built_im_RAWs = []
            built_im_scales = []
            
            target_size = self.cfg.TRAIN_DEFAULT_SCALES[scale_inds[im_i]]
            PIXEL_MEANS = np.array([[self.cfg.MAIN_DEFAULT_PIXEL_MEANS]])
            im_RAW, im_scale = self.prep_im_for_blob(im_RAW, 
                                        PIXEL_MEANS, target_size,
                                        self.cfg.TRAIN_DEFAULT_MAX_SIZE)
            built_im_RAWs.append(im_RAW)
            built_im_scales.append(im_scale)
        
        # Create a blob to hold the input images
        blob = self.images_to_blob(built_im_RAWs)

        return blob, built_im_scales
    
        
    def prep_im_for_blob(self, im_RAW, pixel_means, target_size, max_size):
        
        im_RAW = im_RAW.astype(np.float32, copy=False)
        im_RAW -= pixel_means
        im_shape = im_RAW.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        
        im_scale = float(target_size) / float(im_size_min)        
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
            
        im_RAW = cv2.resize(im_RAW, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        
        return im_RAW, im_scale
       
    def images_to_blob(self, im_RAWs):
        max_shape = np.array([im_RAW.shape for im_RAW in im_RAWs]).max(axis=0)
        num_images = len(im_RAWs)
        blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
        
        for i in range(num_images):
            im_RAW = im_RAWs[i]
            blob[i, 0:im_RAW.shape[0], 0:im_RAW.shape[1], :] = im_RAW
    
        # Move channels (axis 3) to axis 1
        # Axis order will become: (batch elem, channel, height, width)
        channel_swap = (0, 3, 1, 2)
        blob = blob.transpose(channel_swap)
        
        return blob


    def get_perm_ids(self):
        return self._perm_ids


    def get_images(self):
        return self._images


    def set_perm_ids(self, value):
        self._perm_ids = value


    def set_images(self, value):
        self._images = value


    def del_perm_ids(self):
        del self._perm_ids


    def del_images(self):
        del self._images


    def get_cur_idx(self):
        return self._cur_idx


    def set_cur_idx(self, value):
        self._cur_idx = value


    def del_cur_idx(self):
        del self._cur_idx


    def get_cfg(self):
        return self._cfg


    def set_cfg(self, value):
        self._cfg = value


    def del_cfg(self):
        del self._cfg
    
    
    cfg = property(get_cfg, set_cfg, del_cfg, "cfg's docstring")
    cur_idx = property(get_cur_idx, set_cur_idx, del_cur_idx, "cur_idx's docstring")
    perm_ids = property(get_perm_ids, set_perm_ids, del_perm_ids, "perm_ids's docstring")
    images = property(get_images, set_images, del_images, "images's docstring")
    
