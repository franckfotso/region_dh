# Project: segm_cfm
# Module: libs.datasets.ROIGenerator
# Copyright (c) 2017
# Written by: Romuald FOTSO
# Based on: py-faster-rcnn 
#    (https://github.com/rbgirshick/py-faster-rcnn)
# Licensed under MIT License

import scipy.io as sio
import numpy as np
import numpy.random as npr
import cv2

class ROIGenerator(object):
        
    def __init__(self, images, dataset, pos_weights, cfg):
        self.images = images     
        self.dataset = dataset
        self.cfg = cfg
        self.pos_weights = pos_weights
        
        self.cur_idx, self.perm_ids = self.shuffe_images()

      
    def shuffe_images(self):
        """Randomly permute the training images"""        
        
        if self.cfg.TRAIN_BATCH_DET_ASPECT_GROUPING:
            widths = np.array([im.rois["gt"]['im_info']['width'] for im in self.images])            
            heights = np.array([im.rois["gt"]['im_info']['height'] for im in self.images])
            
            horz = (widths >= heights)
            vert = np.logical_not(horz)
            horz_inds = np.where(horz)[0]
            vert_inds = np.where(vert)[0]
            inds = np.hstack((
                np.random.permutation(horz_inds),
                np.random.permutation(vert_inds)))
            inds = np.reshape(inds, (-1, 2))
            row_perm = np.random.permutation(np.arange(inds.shape[0]))
            self.perm_ids = np.reshape(inds[row_perm, :], (-1,))
        else:
            self.perm_ids = np.random.permutation(np.arange(len(self.images)))    
        
        self.cur_idx = 0
        return self.cur_idx, self.perm_ids
    
    
    def get_next_minibatch_ids(self):
        
        if self.cur_idx + self.cfg.TRAIN_BATCH_DET_IMS_PER_BATCH >= len(self.images):
            self.shuffe_images()
            
        batch_ids = self.perm_ids[self.cur_idx:self.cur_idx+self.cfg.TRAIN_BATCH_DET_IMS_PER_BATCH]
        self.cur_idx += self.cfg.TRAIN_BATCH_DET_IMS_PER_BATCH
        
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
        
        assert num_imgs == self.cfg.TRAIN_BATCH_DET_IMS_PER_BATCH, \
        "[ERROR] Single batch only, found: {}".format(num_imgs)
        
        """ build blobs for images """
        
        # resize & build data blob: according caffe format
        im_blob, im_scales = self.built_image_blob(images, random_scale_inds)
        label_blob = self.built_label_blob(images)
        
        if self.cfg.TRAIN_BATCH_DET_USE_ALL_GT:
            gt_inds = np.where(images[0].rois["gt"]['gt_classes'] != 0)[0]
            
        gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
        gt_boxes[:, 0:4] = images[0].rois["gt"]['boxes'][gt_inds, :] * im_scales[0]
        gt_boxes[:, 4] = images[0].rois["gt"]['gt_classes'][gt_inds]
        
        im_info = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)
        
        # positive weights over the batch only
        batch_weights = np.ones((label_blob.shape[0], label_blob.shape[1]))       
           
        for cls_id in range(label_blob.shape[1]):
            pos_ids = np.where(label_blob[:, cls_id] == 1)[-1]
            neg_ids = np.where(label_blob[:, cls_id] == 0)[-1]
            pos_ratio = len(pos_ids)/label_blob.shape[0]
            batch_weights[pos_ids, cls_id] = pos_ratio
            batch_weights[neg_ids, cls_id] = 1 - pos_ratio
        
        pos_weights = np.ones((label_blob.shape[0], label_blob.shape[1])) # try 0, weights
        #pos_weights = batch_weights # try 1
        #pos_weights = 1-pos_weights # try 2
        
        # positive weights over all the samples # try 3
        """
        if self.pos_weights != None:
            pos_weights = self.pos_weights
        else:
            pos_weights = np.ones((label_blob.shape[1]))
        """        
        #print("pos_weights: ", pos_weights)
        #print("pos_weight: ", pos_weights[:,0])
                   
        blobs = {"data": im_blob,
                 "im_info": im_info,
                 "gt_boxes": gt_boxes, 
                 "labels": label_blob,
                 "pos_weights": pos_weights}
                
        return blobs
    
    def built_label_blob(self, images):
        num_images = len(images)
        
        blob = np.zeros((num_images, self.dataset.num_cls-1), dtype=np.int32)
        for im_i in range(num_images):
            image = images[im_i]
            blob[im_i] = image.rois["gt"]["labels"]
            
        return blob
    
    def built_image_blob(self, images, scale_inds):
        num_imgs = len(images)
        #print("built_image_blob > num_imgs: ", num_imgs)
        
        built_im_RAWs = []
        built_im_scales = []
        
        for im_i in range(num_imgs):
            im_RAW = cv2.imread(images[im_i].pathname)
            
            #print("built_image_blob > im_RAW.shape: ", im_RAW.shape)            
            if images[im_i].rois["gt"]['flipped']:
                im_RAW = im_RAW[:, ::-1, :]
            
            target_size = self.cfg.TRAIN_DEFAULT_SCALES[scale_inds[im_i]]
            PIXEL_MEANS = np.array([[self.cfg.MAIN_DEFAULT_PIXEL_MEANS]])
            im_RAW, im_scale = self.prep_im_for_blob(im_RAW, 
                                        PIXEL_MEANS, target_size,
                                        self.cfg.TRAIN_DEFAULT_MAX_SIZE)
            #print("built_image_blob > im_scale: ", im_scale)
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
        
        # caffe - only
        # Move channels (axis 3) to axis 1
        # Axis order will become: (batch elem, channel, height, width)
        #channel_swap = (0, 3, 1, 2)
        #blob = blob.transpose(channel_swap)
        
        return blob
