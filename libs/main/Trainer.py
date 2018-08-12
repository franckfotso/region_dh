# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Project: Region-DH
# Module: libs.main.Trainer
# Copyright (c) 2018
# Written by: Franck FOTSO
# Licensed under MIT License
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from main.BasicWorker import BasicWorker
from main.SolverWrapper import SolverWrapper
from datasets.IMGenerator import IMGenerator
from datasets.ROIGenerator import ROIGenerator

import tensorflow as tf

class Trainer(BasicWorker):
    
    def __init__(self,
                 dataset,
                 model,
                 cfg):
        
        super(Trainer, self).__init__(dataset,
                 model,
                 cfg)
           
    def run(self, train_images, val_images, tb_dir,
            output_dir, techno, max_epochs=20, pos_weights=None):
        """ Train the network """
            
        if self.dataset.name in ["cifar10", "cifar10_m"]:            
            train_gen = IMGenerator(train_images, self.dataset, self.cfg)
            val_gen = IMGenerator(val_images, self.dataset, self.cfg)
                                    
        elif self.dataset.name in ["voc_2007", "voc_2012"]:
            self.model['net'].multilabel = True
            train_gen = ROIGenerator(train_images, self.dataset, pos_weights, self.cfg)
            val_gen = ROIGenerator(val_images, self.dataset, None, self.cfg)
            
        else:
            print("[ERROR] wrong dataset name, found: ", self.dataset.name)
            raise NotImplementedError
        
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True
        
        _batch_size = 1
        if self.cfg.MAIN_DEFAULT_TASK == "CFC":
            _batch_size = self.cfg.TRAIN_BATCH_CFC_NUM_IMG
        elif self.cfg.MAIN_DEFAULT_TASK == "DET":
            _batch_size = self.cfg.TRAIN_BATCH_DET_IMS_PER_BATCH
        else:
            raise NotImplementedError
        
        max_iters = int((max_epochs*len(train_images))/_batch_size)+1
        
        with tf.Session(config=tfconfig) as sess:
            sw = SolverWrapper(self.model['net'], self.model['weights'],techno, self.dataset, 
                   train_gen, val_gen, tb_dir, output_dir, self.cfg)
            print('Solving...')
            sw.train_model(sess, max_iters)
            print('done solving')