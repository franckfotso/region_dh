# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Project: Region-DH
# Module: libs.datasets.CIFAR10
# Copyright (c) 2018
# Written by: Franck FOTSO
# Licensed under MIT License
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os, pickle, copy
import numpy as np
import os.path as osp
from datasets.Image import Image
from datasets.Dataset import Dataset

class CIFAR10(Dataset):
    
    def __init__(self,
                 name, 
                 path_dir,
                 classes=None,
                 cls_to_id=None,
                 sets=None,
                 cfg=None):
        self.cfg = cfg
        
        super(CIFAR10, self).__init__(name, 
                 path_dir,
                 classes,
                 cls_to_id,
                 sets,
                 cfg)
        
        self.load_classes(cfg.CIFAR10_DATASET_FILE_CLS)
        
    
    def built_im_path(self, im_nm, im_DIR):
        im_fn = None
        im_pn = None
        
        for ext in self.cfg.PASCAL_DATASET_DEFAULT_EXT:
            im_pn = osp.join(im_DIR, im_nm+"."+ext)
            if osp.exists(im_pn):
                im_fn = im_nm+"."+ext
                break
            
        assert im_fn != None, \
        "[ERROR] unable to load image {} in {}".format(im_nm, im_DIR)
        
        return im_fn, im_pn
    
    
    def load_images(self, im_names=None):
        train_images, val_images = ([],[])
        train_labels, val_labels = ([],[])
        
        cache_images_file = osp.join(self.cfg.MAIN_DIR_ROOT, "cache",self.name+"_images.pkl")
        if osp.exists(cache_images_file):
            with open(cache_images_file,'rb') as fp:
                (train_images, val_images), (train_labels, val_labels) = pickle.load(fp)
                print ('[INFO] images_obj loaded from {}'.format(cache_images_file))
                fp.close()
            return (train_images, val_images), (train_labels, val_labels)
        
        train_dir = osp.join(self.cfg.MAIN_DIR_ROOT, "data", self.name, 
                              self.cfg.CIFAR10_DATASET_DIR_TRAIN)
        
        val_dir = osp.join(self.cfg.MAIN_DIR_ROOT, "data", self.name, 
                              self.cfg.CIFAR10_DATASET_DIR_VAL)
        
        sub_dirs = os.listdir(train_dir)
        for label in sub_dirs:
            label_dir = osp.join(train_dir, label)
            im_fns = os.listdir(label_dir)
            
            for im_fn in im_fns:
                im_pn = osp.join(label_dir, im_fn)
                img = Image(im_fn, im_pn, label=label)
                train_images.append(img)
                train_labels.append(label)     
        
        sub_dirs = os.listdir(val_dir)
        for label in sub_dirs:
            label_dir = osp.join(val_dir, label)
            im_fns = os.listdir(label_dir)
            
            for im_fn in im_fns:
                im_pn = osp.join(label_dir, im_fn)
                img = Image(im_fn, im_pn, label=label)
                val_images.append(img)
                val_labels.append(label)
            
            
        if not osp.exists(cache_images_file):
            with open(cache_images_file,'wb') as fp:
                to_dump = [(train_images, val_images), (train_labels, val_labels)]
                pickle.dump(to_dump, fp)
                print ('[INFO] images_obj saved to {}'.format(cache_images_file))
                fp.close()
            
        return (train_images, val_images), (train_labels, val_labels)
    
    
    def load_sets(self):
        sets = {
            "train":    {"im_fns": [], "images": [], "num_items":0},
            "trainval": {"im_fns": [], "images": [], "num_items":0},
            "val":      {"im_fns": [], "images": [], "num_items":0},
            "test":     {"im_fns": [], "images": [], "num_items":0}
        }      
        train_dir = osp.join(self.cfg.MAIN_DIR_ROOT, "data", self.name, 
                              self.cfg.CIFAR10_DATASET_DIR_TRAIN)
        
        val_dir = osp.join(self.cfg.MAIN_DIR_ROOT, "data", self.name, 
                              self.cfg.CIFAR10_DATASET_DIR_VAL)
         
        
        if osp.exists(train_dir):
            train_labels = os.listdir(train_dir)
            for label in train_labels:
                label_dir = osp.join(train_dir, label)
                im_fns = os.listdir(label_dir)
                sets["train"]["im_fns"] = im_fns
                sets["train"]["num_items"] += len(im_fns)
            del im_fns
        else:
            print ("[WARN] unable to access dir {}".format(train_dir))
            
        if osp.exists(val_dir):
            val_labels = os.listdir(val_dir)
            for label in val_labels:
                label_dir = osp.join(val_dir, label)
                im_fns = os.listdir(label_dir)
                sets["val"]["im_fns"] = im_fns
                sets["val"]["num_items"] += len(im_fns)
            del im_fns
        else:
            print ("[WARN] unable to access dir {}".format(val_dir))  
        
        self.sets = sets
    
    def append_flipped_images(self, images, src, num_proc=1):
        pass
    
    def built_output_dir(self, root_dir, phase, net):
        output_dir = osp.join(root_dir, self.cfg.MAIN_DIR_OUTPUTS, 
                              self.name, phase+'_'+net)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        return output_dir
    
    
    