# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Project: Region-DH
# Module: tools.train
# Copyright (c) 2018
# Written by: Franck FOTSO
# Licensed under MIT License
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# usage: python tools/train_net.py --gpu_id 0 --dataset voc_2007 \
#        --gt_set train --net AlexNet \
#        --iters 40000 --cache_im_dir cache/voc2007_train
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from __future__ import print_function
import _init_paths
import argparse, pprint
import os.path as osp
import numpy as np
import tensorflow as tf

from Config import Config
from datasets.Pascal import Pascal
from main.Trainer import Trainer

def parse_args():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu_id", dest="gpu_id", default=0, 
                    help="id of GPU device", type=int)
    ap.add_argument("--dataset", dest="dataset", required=True, 
                    help="dataset name to use", type=str)
    ap.add_argument('--gt_set', dest='gt_set',
                    help='gt set use to list data', required=True,
                    default='train', type=str)
    ap.add_argument('--techno', dest='techno',
                    help='implemented techno to execute', required=True,
                    default='SSDH', type=str)
    ap.add_argument("--net", dest="net", required=True, 
                    help="backbone network", type=str)
    ap.add_argument("--iters", dest="iters", default=50000,
                    help="number of iterations for training", type=int)
    ap.add_argument('--cache_im_dir', dest='cache_im_dir', required=True,
                    help='directory for images prepared', type=str)
    ap.add_argument("--rand", dest="rand", default=False,
                    help="randomize (do not use a fixed seed)", type=bool)
    ap.add_argument("--verbose", dest="verbose", default=False,
                    help="verbosity level", type=bool)
    args = vars(ap.parse_args())
    
    return args

if __name__ == '__main__':
    args = parse_args()
    
    print('[INFO] Called with args:')
    print(args)
    
    # setup & load configs
    _C = Config(config_pn="config/config.ini")
    cfg = _C.cfg
    
    cfg.MAIN_DEFAULT_GPU_ID = args['gpu_id']
    
    print('Using config:')
    pprint.pprint(cfg)
    
    # setup tensorflow & tensorboard
    
    
    print ('[INFO] loading dataset {} for training...'.format(args["dataset"]))
    dataset_DIR = osp.join(cfg.MAIN_DIR_ROOT, "data", args["dataset"])
    gt_set = args['gt_set']
    techno = args['techno']
    
    dataset = None
    ds_pascal = ["voc_2007", "voc_2012"]
    ds_coco = []
    
    if args["dataset"] in  ds_pascal:
        dataset = Pascal(name=args["dataset"], path_dir=dataset_DIR, cfg=cfg)
        
    assert dataset != None, \
    "[ERROR] unable to build {} dataset. Available: {}".format(args["dataset"], ds_pascal)
        
    dataset.load_sets()
    print ('[INFO] dataset.name: {}'.format(dataset.name))
    print ('[INFO] dataset.num_cls: {}'.format(dataset.num_cls))
    print ('[INFO] dataset.train: {}'.format(dataset.sets["train"]["num_items"]))
    print ('[INFO] dataset.trainval: {}'.format(dataset.sets["trainval"]["num_items"]))
    print ('[INFO] dataset.test: {}'.format(dataset.sets["test"]["num_items"]))
    print ('[INFO] dataset.val: {}'.format(dataset.sets["val"]["num_items"]))
    
    images = dataset.load_gt_rois(gt_set=gt_set)
    print ('[INFO] images.num: {}'.format(len(images)))
    
    # tensorboard directory
    tb_DIR = osp.join(cfg.MAIN_DIR_LOGS, args["dataset"])
    
    # launch train process
    root_dir = cfg.MAIN_DIR_ROOT
    output_dir = dataset.built_output_dir(root_dir, 'train')
    cache_im_dir = osp.join(root_dir, args['cache_im_dir'])
    print ('[INFO] Start the training process on {}'.format(args["dataset"]))
    
    
    