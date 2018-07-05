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
            output_dir, techno, max_iters=40000):
        """ Train a CFM network """
        
        assert techno in self.cfg.MAIN_DEFAULT_TECHNOS, \
            '[ERROR] unknown task name provided: {}'.format(self.task)
            
        if techno == 'SSDH':            
            train_gen = IMGenerator(train_images, self.dataset, self.cfg)
            val_gen = IMGenerator(val_images, self.dataset, self.cfg)
                                    
        elif techno == 'RegionDH':
            raise NotImplemented
        
        elif techno == 'ISDH':
            raise NotImplemented
        
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True
        
        with tf.Session(config=tfconfig) as sess:
            sw = SolverWrapper(self.model['net'], self.model['weights'],techno, self.dataset, 
                   train_gen, val_gen, tb_dir, output_dir, self.cfg)
            print('Solving...')
            sw.train_model(sess, max_iters)
            print('done solving')    
    