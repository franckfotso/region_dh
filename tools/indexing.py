# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Project: Region-DH
# Module: tools.indexing
# Copyright (c) 2018
# Written by: Franck FOTSO
# Licensed under MIT License
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Goal: extract and index hash codes and deep features

import _init_paths
import argparse, os
import numpy as np
from indexer.DeepIndexer import DeepIndexer
from extractor.DeepExtractor import DeepExtractor
from Config import Config
import scipy.io as sio
from PIL import Image


def parse_args():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", dest="net", required=True, 
                    help="backbone network", type=str)
    ap.add_argument("--weights", required=True,
                    help="path to model file")
    ap.add_argument('--techno', dest='techno',
                    help='implemented techno for hashing', required=True,
                    default='DLHBC', type=str)
    ap.add_argument("--num_cls", dest="num_cls", required=True, 
                        help="number of classes for the trained model", type=int)
    ap.add_argument("--num_bits", dest="num_bits", default=48,
                        help="number of bits for the hashing layer", type=int)
    ap.add_argument("--im_dir", required=True,
                    help="input for target images")
    ap.add_argument("--deep_db", required=True,
                    help="path for the deepfeatures db")
    args = vars(ap.parse_args())
    
    return args


if __name__ == '__main__':
    args = parse_args()
    
    print('[INFO] Called with args:')
    print(args)
    
    # setup & load configs
    _C = Config(config_pn="config/config.ini")
    cfg = _C.cfg
    
    # TODO: handle multi-size batch for the forward
    cfg.TEST_BATCH_CFC_NUM_IMG = cfg.INDEXING_BATCH_SIZE
    bacth_size = cfg.INDEXING_BATCH_SIZE

    # hash codes & deep features extractor
    deep_extr = DeepExtractor(args["techno"], args["net"], args["num_cls"], 
                              args["num_bits"], args["weights"], cfg)

    im_pns = []
    im_dir = args['im_dir']
    categs = os.listdir(im_dir)
    for i, categ in enumerate(categs):
        categ_dir = os.path.join(im_dir, categ)
        im_fns = os.listdir(categ_dir)
        
        if cfg.INDEXING_CHECK_DS:
            print ('[{}/{}] Loading & checking category: {}'.format(i + 1, len(categs), categ))
        else:
            print ('[{}/{}] Loading category: {}'.format(i + 1, len(categs), categ))

        for im_fn in im_fns:
            im_pn = os.path.join(categ_dir, im_fn)
            if not os.path.isdir(im_pn):
                try:
                    if cfg.INDEXING_CHECK_DS == 1:
                        im = Image.open(im_pn)
                        del im
                    im_pns.append(im_pn)
                except:
                    os.remove(im_pn)
                    print ('[ERROR] indexing > IOError: cannot identify image file')
                    print ('[ERROR] Wrong file deleted: {}'.format(im_pn))     
    
    im_pns = np.array(im_pns)
    
    assert len(im_pns) % bacth_size == 0, \
     "[ERROR] BATCH_SIZE must be divible with total images. Set INDEXING_BATCH_SIZE = 1 to continue"
    
    # initialize the deep feature indexer
    di = DeepIndexer(args["deep_db"], 
                     estNumImages=cfg.INDEXING_NUM_IM_EST,
                     maxBufferSize=cfg.INDEXING_MAX_BUF_SIZE, verbose=True)
    
    print ('----------------------------------------------------------')
    print ('{}-bits binary codes & deep features extraction'.format(args["num_bits"]))
    print ('----------------------------------------------------------')
    
    for i in range(0,len(im_pns), bacth_size):
        _im_pns = im_pns[i:i+bacth_size]
        
        # extract binary codes & deep features
        binary_codes, deep_features = deep_extr.extract(_im_pns)        
        
        """
            indexing data extracted in hdfs file
        """ 
        for j, (binary_code, feature_vector) in enumerate(zip(binary_codes, deep_features)):            
            im_pn = _im_pns[j]
            im_fn = im_pn[im_pn.rfind(os.path.sep) + 1:]

            # adding fields
            di.add(im_fn, feature_vector, binary_code)
        
        # check to see if progress should be displayed
        if i > 0 and bacth_size > 1 and i % bacth_size == 0:
            di._debug("saved {} images".format(i), msgType="[PROGRESS]")
        elif i > 0 and bacth_size == 1 and i % 10 == 0:
            di._debug("saved {} images".format(i), msgType="[PROGRESS]")

    # finish the indexing process
    di._debug("{} images sucessfully indexed".format(len(im_fns)), msgType="[INFO]")
    di.finish()

        