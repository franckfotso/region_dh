# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Project: Region-DH
# Module: libs.extractor.DeepExtractor
# Copyright (c) 2018
# Written by: Franck FOTSO
# Licensed under MIT License
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from extractor.feat_tools import *

class DeepExtractor:

    def __init__(self, techno, arch, num_cls, num_bits, weights, cfg):
        self.techno = techno
        self.num_bits = num_bits
        self.weights = weights
        self.cfg = cfg
        
        sess, net = tf_init_feat(weights, arch, num_cls, num_bits, techno, cfg) 
        self.sess = sess
        self.net = net

    def extract(self, im_pns):        
        binary_codes = None
        if self.net != None:
            binary_codes, deep_features = tf_batch_feat(im_pns,
                                                       self.sess,
                                                       self.net,
                                                       self.techno,
                                                       self.cfg)
        else:
            print ("[ERROR] DeepExtractor > extract: Error, net is not defined")
            
        return binary_codes, deep_features