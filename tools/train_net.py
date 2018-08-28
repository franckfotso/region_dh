# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Project: Region-DH
# Module: tools.train
# Copyright (c) 2018
# Written by: Franck FOTSO
# Licensed under MIT License
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# usage: python tools/train_net.py --gpu_id 0 --dataset voc_2007 \
#        --gt_set train_val --net VGG16 --techno SSDH \
#        --iters 40000 --cache_im_dir cache/voc2007_train
#
# usage: python tools/train_net.py --gpu_id 0 --dataset cifar10 \
#        --gt_set train_val --net VGG16 --techno SSDH \
#        --iters 80000 --cache_im_dir cache/voc2007_train
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from __future__ import print_function
import _init_paths
import argparse, pprint, os
import os.path as osp
import numpy as np
import tensorflow as tf

from Config import Config
from datasets.Pascal import Pascal
from datasets.CIFAR10 import CIFAR10 
from main.Trainer import Trainer
from nets.VGG16_FT import VGG16_FT
from nets.VGG16_DLBHC import VGG16_DLBHC
from nets.VGG16_SSDH1 import VGG16_SSDH1
from nets.VGG16_SSDH2 import VGG16_SSDH2
from nets.VGG16_RegionDH import VGG16_RegionDH

def parse_args():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
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
    ap.add_argument("--weights", dest="weights", default="",
                    help="path to checkpoint model file")
    ap.add_argument("--epochs", dest="epochs", required=True, default=20,
                    help="number of epochs for the training", type=int)
    ap.add_argument("--num_bits", dest="num_bits", default=48,
                    help="number of bits for the hashing layer", type=int)
    ap.add_argument("--rand", dest="rand", default=False,
                    help="randomize (do not use a fixed seed)", type=bool)
    ap.add_argument("--verbose", dest="verbose", default=False,
                    help="verbosity level", type=bool)
    ap.add_argument("--config", dest="config", required=True, type=str,
                    help="config file for the techno")
    args = vars(ap.parse_args())
    
    return args

if __name__ == '__main__':
    args = parse_args()
    
    print('[INFO] Called with args:')
    print(args)
    
    # setup & load configs
    _C = Config(config_pn=args['config'])
    cfg = _C.cfg
    
    cfg.TRAIN_DEFAULT_SNAPSHOT_PREFIX = args['net']+"_"+args['techno']
    
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
    
    elif args["dataset"] in ["cifar10", "cifar10_m"]:
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
    
    if gt_set in ["train_val", "train_test", "trainval_test"]:
        if dataset.name in ds_pascal:
            train_set, val_set = gt_set.split("_")
            train_images, pos_weights = dataset.load_gt_rois(gt_set=train_set)
            val_images, _ = dataset.load_gt_rois(gt_set=val_set)
            
        elif dataset.name in ["cifar10", "cifar10_m"]:
            (train_images, val_images), _ = dataset.load_images()
            
        print ('[INFO] train_images.num: {}'.format(len(train_images)))
        print ('[INFO] val_images.num: {}'.format(len(val_images)))        
    else:
        train_images = dataset.load_gt_rois(gt_set=gt_set)
        print ('[INFO] train_images.num: {}'.format(len(train_images)))
    
    # tensorboard directory
    tb_dir = osp.join(cfg.MAIN_DIR_LOGS, args["dataset"]+"_"+args['net'])
    if not osp.exists(tb_dir):
        os.makedirs(tb_dir)
    
    net = None
    weights = args["weights"] # pretrained weights or checkpoint
    
    if args['net'] == "VGG16":
        if techno == "FT":
            net = VGG16_FT(cfg)
            
        elif techno == "DLBHC":
            net = VGG16_DLBHC(cfg, args["num_bits"])
            
        elif techno == "SSDH1":
            net = VGG16_SSDH1(cfg, args["num_bits"])
            
        elif techno == "SSDH2":
            net = VGG16_SSDH2(cfg, args["num_bits"])
            
        elif techno == "Region-DH":
            # slow down display
            cfg.TRAIN_DEFAULT_DISPLAY = 20 
            net = VGG16_RegionDH(cfg, args["num_bits"])
            
        if args["weights"] == "":
            weights = "models/pretrained/imagenet_weights/vgg_16.ckpt"
        
    assert net != None, \
    "[ERROR] invalid network provided. Found: {}".format(args["net"])
    
    model = {'net': net,'weights': weights}
    trainer = Trainer(dataset=dataset, model=model, cfg=cfg)
    
    # launch train process
    root_dir = cfg.MAIN_DIR_ROOT
    output_dir = dataset.built_output_dir(root_dir, 'train', args["net"])
    
    print ('[INFO] Start the training process on {}'.format(args["dataset"]))
    max_epochs = args['epochs']    
    trainer.run(train_images, val_images, tb_dir, output_dir, 
                techno, max_epochs=max_epochs, pos_weights=pos_weights)    
    
    