# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Project: Region-DH
# Module: libs.extractor.DeepExtractor
# Copyright (c) 2018
# Written by: Franck FOTSO
# Licensed under MIT License
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class DeepExtractor:

    def __init__(self, techno, num_bits, df_len, weights, cfg):
        self.techno = techno
        self.num_bits = num_bits
        self.df_len = df_len
        self.weights = weights
        self.cfg = cfg
        
        sess, net = tf_init_feat(weights, num_bits)
        self.sess = sess
        self.net = net


    def extract(self, im_pns):
        print '----------------------------------------------------------'
        print ('{}-bits binary codes & {}-vector deep features extraction'.format(self.num_bits, self.df_len))
        print '----------------------------------------------------------'
        binary_codes = None
        if self.net != None:
            binary_codes, deep_features = tf_batch_feat(im_pns,
                                                       self.sess,
                                                       self.net,
                                                       self.techno,
                                                       self.cfg)
        else:
            print "[ERROR] DeepExtractor > extract: Error, net is not defined"
            
        return binary_codes, deep_features