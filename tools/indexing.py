# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Project: Region-DH
# Module: tools.indexing
# Copyright (c) 2018
# Written by: Franck FOTSO
# Licensed under MIT License
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Goal: extract and index hash codes and deep features

import _init_paths
import argparse, os, pprint
import os.path as osp
import numpy as np
import scipy.io as sio
from PIL import Image

from Config import Config
from datasets.Pascal import Pascal
from datasets.CIFAR10 import CIFAR10
from indexer.DeepIndexer import DeepIndexer
from extractor.DeepExtractor import DeepExtractor

def parse_args():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", dest="dataset", required=True, 
                    help="dataset name to use", type=str)
    ap.add_argument('--gt_set', dest='gt_set',
                    help='gt set use to list data', required=True,
                    default='train', type=str)
    ap.add_argument("--net", dest="net", required=True, 
                    help="backbone network", type=str)
    ap.add_argument("--weights", required=True,
                    help="path to model file")
    ap.add_argument('--techno', dest='techno',
                    help='implemented techno for hashing', required=True,
                    default='DLHBC', type=str)
    ap.add_argument("--num_bits", dest="num_bits", default=48,
                        help="number of bits for the hashing layer", type=int)
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
    
    print('Using config:')
    pprint.pprint(cfg)
    
    print ('[INFO] loading dataset {} for training...'.format(args["dataset"]))
    dataset_DIR = osp.join(cfg.MAIN_DIR_ROOT, "data", args["dataset"])
    gt_set = args['gt_set']
    techno = args['techno']
    
    dataset = None
    ds_pascal = ["voc_2007", "voc_2012"]
    ds_coco = []
    
    if args["dataset"] in  ds_pascal:
        dataset = Pascal(name=args["dataset"], path_dir=dataset_DIR, cfg=cfg)
    
    elif args["dataset"] in ["cifar10"]:
        dataset = CIFAR10(name=args["dataset"], path_dir=dataset_DIR, cfg=cfg)
        
    assert dataset != None, \
    "[ERROR] unable to build {} dataset. Available: {}".format(args["dataset"], ds_pascal)
        
    dataset.load_sets()
    print ('[INFO] dataset.name: {}'.format(dataset.name))
    print ('[INFO] dataset.num_cls: {}'.format(dataset.num_cls))
    print ('[INFO] dataset.train: {}'.format(dataset.sets["train"]["num_items"]))
    print ('[INFO] dataset.trainval: {}'.format(dataset.sets["trainval"]["num_items"]))
    print ('[INFO] dataset.test: {}'.format(dataset.sets["test"]["num_items"]))
    print ('[INFO] dataset.val: {}'.format(dataset.sets["val"]["num_items"]))
    
    if gt_set == "train_val":
        if dataset.name in ds_pascal:
            images, _ = dataset.load_gt_rois(gt_set="trainval")
            
        elif dataset.name in ["cifar10"]:
            (images, _), _ = dataset.load_images()           
           
    elif gt_set == "test":
        if dataset.name in ds_pascal:
            images, _ = dataset.load_gt_rois(gt_set="test")
            
        elif dataset.name in ["cifar10"]:
            (_, images), _ = dataset.load_images() 
        
    print ('[INFO] {}, images.num: {}'.format(gt_set, len(images))) 
    
    # TODO: handle multi-size batch for the forward
    cfg.TEST_BATCH_CFC_NUM_IMG = cfg.INDEXING_BATCH_SIZE
    bacth_size = cfg.INDEXING_BATCH_SIZE

    # hash codes & deep features extractor
    deep_extr = DeepExtractor(args["techno"], args["net"], dataset.num_cls, 
                              args["num_bits"], args["weights"], cfg)

    im_pns = [image.pathname for image in images]
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
    di._debug("{} images sucessfully indexed".format(len(im_pns)), msgType="[INFO]")
    di.finish()

        