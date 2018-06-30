# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Project: Region-DH
# Module: libs.main.Trainer
# Copyright (c) 2018
# Written by: Franck FOTSO
# Licensed under MIT License
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from main.BasicWorker import BasicWorker
from main.SolverWrapper import SolverWrapper
from datasets.SSDHGenerator import SSDHGenerator


class Trainer(BasicWorker):
    
    def __init__(self,
                 dataset,
                 model,
                 cfg):
        
        super(Trainer, self).__init__(dataset,
                 model,
                 cfg)
        
    def run(self, all_images, all_masks, 
            cache_im_dir, output_dir, task, max_iters=40000):
        """ Train a CFM network """
        
        assert task in self.cfg.MAIN_DEFAULT_TASKS, \
            '[ERROR] unknown task name provided: {}'.format(self.task)
            
        if task == 'SSDH':            
            data_gen = SSDHGenerator(images=all_images, masks=all_masks,
                                    bbox_means=None, bbox_stds=None,
                                    num_cls=self.dataset.num_cls, cfm_t=None,
                                    in_gt_dir=None, in_pr_dir=None,
                                    cache_im_dir=cache_im_dir, cfg=self.cfg)
                                    
        elif self.task == 'RegionDH':
            raise NotImplemented
        
        elif self.task == 'ISDH':
            raise NotImplemented            
        
        sw = SolverWrapper(self.model['solver'], 
                           self.model['pretrained_model'], 
                           data_gen, output_dir, self.cfg)

        print 'Solving...'
        model_paths = sw.train_model(max_iters)
        print 'done solving'
        return model_paths
    
    