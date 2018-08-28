# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Project: Region-DH
# Module: libs.Config
# Copyright (c) 2018
# Written by: Franck FOTSO
# Licensed under MIT License
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import configparser as cp
import importlib
from easydict import EasyDict as edict

class Config(object):
    
    def __init__(self, config_pn, extras=None):
        self._parser = cp.RawConfigParser()
        self._parser.read(config_pn)
        
        self._extras = extras
        self._cfg = edict()
        self._cfg.FILE = config_pn
        
        self.load_config()

    def get_parser(self):
        return self._parser


    def set_parser(self, value):
        self._parser = value


    def del_parser(self):
        del self._parser
        

    def get_cfg(self):
        return self._cfg


    def set_cfg(self, value):
        self._cfg = value


    def del_cfg(self):
        del self._cfg


    def get_extras(self):
        return self._extras


    def set_extras(self, value):
        self._extras = value


    def del_extras(self):
        del self._extras

    parser = property(get_parser, set_parser, del_parser, "parser's docstring")
    extras = property(get_extras, set_extras, del_extras, "extras's docstring")
    cfg = property(get_cfg, set_cfg, del_cfg, "cfg's docstring")
    
    ''' 
        load_config: load config parameters from config.ini
    '''        
    def load_config(self):
        try:
            ''' 
                MAIN.DEFAULT
            '''
            self.cfg.MAIN_DEFAULT_GPU_ID = self.load_param("MAIN.DEFAULT", "GPU_ID", "int")
            self.cfg.MAIN_DEFAULT_RNG_SEED = self.load_param("MAIN.DEFAULT", "RNG_SEED", "int")
            self.cfg.MAIN_DEFAULT_TASK = self.load_param("MAIN.DEFAULT", "TASK")
            self.cfg.MAIN_DEFAULT_USE_GPU_NMS = self.load_param("MAIN.DEFAULT", "USE_GPU_NMS", "bool")
            self.cfg.MAIN_DEFAULT_PIXEL_MEANS = self.load_param("MAIN.DEFAULT", "PIXEL_MEANS", "list", "float")
            self.cfg.MAIN_DEFAULT_EPS = self.load_param("MAIN.DEFAULT", "EPS", "float")
            self.cfg.MAIN_DEFAULT_BINARIZE_THRESH = self.load_param("MAIN.DEFAULT", "BINARIZE_THRESH","float")
            self.cfg.MAIN_DEFAULT_TECHNOS = self.load_param("MAIN.DEFAULT", "TECHNOS","list")
            self.cfg.MAIN_DEFAULT_DATASETS = self.load_param("MAIN.DEFAULT", "DATASETS","list")
            self.cfg.MAIN_DEFAULT_EXT = self.load_param("MAIN.DEFAULT", "EXT", "list")
            self.cfg.MAIN_DEFAULT_CORES = self.load_param("MAIN.DEFAULT", "CORES", "int")
            
            ''' 
                MAIN.DIR
            '''
            self.cfg.MAIN_DIR_ROOT = self.load_param("MAIN.DIR", "ROOT")
            self.cfg.MAIN_DIR_CACHE = self.load_param("MAIN.DIR", "CACHE")
            self.cfg.MAIN_DIR_LOGS = self.load_param("MAIN.DIR", "LOGS")
            self.cfg.MAIN_DIR_OUTPUTS = self.load_param("MAIN.DIR", "OUTPUTS")
           
            
            ''' 
                PASCAL.DATASET.DIR
            '''
            self.cfg.PASCAL_DATASET_DIR_MAIN_SET = self.load_param("PASCAL.DATASET.DIR", "MAIN_SET")            
            self.cfg.PASCAL_DATASET_DIR_IMAGE = self.load_param("PASCAL.DATASET.DIR", "IMAGE")
            self.cfg.PASCAL_DATASET_DIR_ANNOTATION = self.load_param("PASCAL.DATASET.DIR", "ANNOTATION")

            ''' 
                PASCAL.DATASET.FILE
            '''
            self.cfg.PASCAL_DATASET_FILE_VAL = self.load_param("PASCAL.DATASET.FILE", "VAL")
            self.cfg.PASCAL_DATASET_FILE_TEST = self.load_param("PASCAL.DATASET.FILE", "TEST")
            self.cfg.PASCAL_DATASET_FILE_TRAIN = self.load_param("PASCAL.DATASET.FILE", "TRAIN")
            self.cfg.PASCAL_DATASET_FILE_TRAINVAL = self.load_param("PASCAL.DATASET.FILE", "TRAINVAL")           
            self.cfg.PASCAL_DATASET_FILE_CLS = self.load_param("PASCAL.DATASET.FILE", "CLS")
            
            ''' 
                CIFAR10.DATASET.DIR
            '''
            self.cfg.CIFAR10_DATASET_DIR_TRAIN = self.load_param("CIFAR10.DATASET.DIR", "TRAIN")
            self.cfg.CIFAR10_DATASET_DIR_VAL = self.load_param("CIFAR10.DATASET.DIR", "VAL")
            
            ''' 
                CIFAR10.DATASET.FILE
            '''
            self.cfg.CIFAR10_DATASET_FILE_TRAIN = self.load_param("CIFAR10.DATASET.FILE", "TRAIN")
            self.cfg.CIFAR10_DATASET_FILE_VAL = self.load_param("CIFAR10.DATASET.FILE", "VAL")
            self.cfg.CIFAR10_DATASET_FILE_CLS = self.load_param("CIFAR10.DATASET.FILE", "CLS")
            
            
            ''' 
                TRAIN.DEFAULT
            '''
            self.cfg.TRAIN_DEFAULT_LEARNING_RATE = self.load_param("TRAIN.DEFAULT", "LEARNING_RATE","float")
            self.cfg.TRAIN_DEFAULT_MOMENTUM = self.load_param("TRAIN.DEFAULT", "MOMENTUM","float")
            self.cfg.TRAIN_DEFAULT_WEIGHT_DECAY = self.load_param("TRAIN.DEFAULT", "WEIGHT_DECAY","float")
            self.cfg.TRAIN_DEFAULT_GAMMA = self.load_param("TRAIN.DEFAULT", "GAMMA","float")
            self.cfg.TRAIN_DEFAULT_STEPSIZE_RATE = self.load_param("TRAIN.DEFAULT", "STEPSIZE_RATE", "int")
            self.cfg.TRAIN_DEFAULT_DOUBLE_BIAS = self.load_param("TRAIN.DEFAULT", "DOUBLE_BIAS","bool")
            self.cfg.TRAIN_DEFAULT_SUMMARY_INTERVAL = self.load_param("TRAIN.DEFAULT", "SUMMARY_INTERVAL","int")
            self.cfg.TRAIN_DEFAULT_DISPLAY = self.load_param("TRAIN.DEFAULT", "DISPLAY","int")
            self.cfg.TRAIN_DEFAULT_SNAPSHOT_KEPT = self.load_param("TRAIN.DEFAULT", "SNAPSHOT_KEPT","int")
            self.cfg.TRAIN_DEFAULT_SNAPSHOT_PREFIX = self.load_param("TRAIN.DEFAULT", "SNAPSHOT_PREFIX")
            self.cfg.TRAIN_DEFAULT_USE_E2E_TF = self.load_param("TRAIN.DEFAULT", "USE_E2E_TF", "bool")
            
            self.cfg.TRAIN_DEFAULT_DEBUG = self.load_param("TRAIN.DEFAULT", "DEBUG", "bool")
            self.cfg.TRAIN_DEFAULT_USE_FLIPPED = self.load_param("TRAIN.DEFAULT", "USE_FLIPPED", "bool")            
            self.cfg.TRAIN_DEFAULT_SNAPSHOT_EPOCHS = self.load_param("TRAIN.DEFAULT", "SNAPSHOT_EPOCHS", "int")                     
            self.cfg.TRAIN_DEFAULT_SCALES = self.load_param("TRAIN.DEFAULT", "SCALES","list","int")
            self.cfg.TRAIN_DEFAULT_MAX_SIZE = self.load_param("TRAIN.DEFAULT", "MAX_SIZE","int")

            ''' 
                TRAIN.BATCH.CFC
            '''
            self.cfg.TRAIN_BATCH_CFC_NUM_IMG = self.load_param("TRAIN.BATCH.CFC", "NUM_IMG", "int")
            self.cfg.TRAIN_BATCH_CFC_ALPHA = self.load_param("TRAIN.BATCH.CFC", "ALPHA", "float")
            self.cfg.TRAIN_BATCH_CFC_BETA = self.load_param("TRAIN.BATCH.CFC", "BETA", "float")
            self.cfg.TRAIN_BATCH_CFC_GAMMA = self.load_param("TRAIN.BATCH.CFC", "GAMMA", "float")
            
            ''' 
                TRAIN.BATCH.DET
            '''
            self.cfg.TRAIN_BATCH_DET_ASPECT_GROUPING = self.load_param("TRAIN.BATCH.DET", "ASPECT_GROUPING", "bool")
            self.cfg.TRAIN_BATCH_DET_PROPOSAL_METHOD = self.load_param("TRAIN.BATCH.DET", "PROPOSAL_METHOD", "str")
            self.cfg.TRAIN_BATCH_DET_HAS_RPN = self.load_param("TRAIN.BATCH.DET", "HAS_RPN", "bool")           
            self.cfg.TRAIN_BATCH_DET_RPN_POSITIVE_OVERLAP = self.load_param("TRAIN.BATCH.DET", "RPN_POSITIVE_OVERLAP", "float")
            self.cfg.TRAIN_BATCH_DET_RPN_NEGATIVE_OVERLAP = self.load_param("TRAIN.BATCH.DET", "RPN_NEGATIVE_OVERLAP", "float")
            self.cfg.TRAIN_BATCH_DET_RPN_FG_FRACTION = self.load_param("TRAIN.BATCH.DET", "RPN_FG_FRACTION", "float")
            self.cfg.TRAIN_BATCH_DET_RPN_BATCHSIZE = self.load_param("TRAIN.BATCH.DET", "RPN_BATCHSIZE", "int")
            self.cfg.TRAIN_BATCH_DET_RPN_NMS_THRESH = self.load_param("TRAIN.BATCH.DET", "RPN_NMS_THRESH", "float")
            self.cfg.TRAIN_BATCH_DET_RPN_PRE_NMS_TOP_N = self.load_param("TRAIN.BATCH.DET", "RPN_PRE_NMS_TOP_N", "int")
            self.cfg.TRAIN_BATCH_DET_RPN_POST_NMS_TOP_N = self.load_param("TRAIN.BATCH.DET", "RPN_POST_NMS_TOP_N", "int")
            self.cfg.TRAIN_BATCH_DET_RPN_POSITIVE_WEIGHT = self.load_param("TRAIN.BATCH.DET", "RPN_POSITIVE_WEIGHT", "float")
            self.cfg.TRAIN_BATCH_DET_USE_ALL_GT = self.load_param("TRAIN.BATCH.DET", "USE_ALL_GT", "bool")
            self.cfg.TRAIN_BATCH_DET_POOLING_SIZE = self.load_param("TRAIN.BATCH.DET", "POOLING_SIZE", "int")
            self.cfg.TRAIN_BATCH_DET_ANCHOR_SCALES = self.load_param("TRAIN.BATCH.DET", "ANCHOR_SCALES", "list", "int")
            self.cfg.TRAIN_BATCH_DET_ANCHOR_RATIOS = self.load_param("TRAIN.BATCH.DET", "ANCHOR_RATIOS", "list", "float")
            self.cfg.TRAIN_BATCH_DET_RPN_CHANNELS = self.load_param("TRAIN.BATCH.DET", "RPN_CHANNELS", "int")
            
            self.cfg.TRAIN_BATCH_DET_IMS_PER_BATCH = self.load_param("TRAIN.BATCH.DET", "IMS_PER_BATCH", "int")
            self.cfg.TRAIN_BATCH_DET_BATCH_SIZE = self.load_param("TRAIN.BATCH.DET", "BATCH_SIZE", "int")
            self.cfg.TRAIN_BATCH_DET_FG_FRACTION = self.load_param("TRAIN.BATCH.DET", "FG_FRACTION", "float")
            self.cfg.TRAIN_BATCH_DET_FG_THRESH = self.load_param("TRAIN.BATCH.DET", "FG_THRESH", "float")
            self.cfg.TRAIN_BATCH_DET_BG_THRESH_HI = self.load_param("TRAIN.BATCH.DET", "BG_THRESH_HI", "float")
            self.cfg.TRAIN_BATCH_DET_BG_THRESH_LO = self.load_param("TRAIN.BATCH.DET", "BG_THRESH_LO", "float")
            
            self.cfg.TRAIN_BATCH_DET_BBOX_REG = self.load_param("TRAIN.BATCH.DET", "BBOX_REG","bool")
            self.cfg.TRAIN_BATCH_DET_BBOX_THRESH = self.load_param("TRAIN.BATCH.DET", "BBOX_THRESH","float")            
            self.cfg.TRAIN_BATCH_DET_BBOX_NORMALIZE_TARGETS \
                = self.load_param("TRAIN.BATCH.DET", "BBOX_NORMALIZE_TARGETS","bool")
            self.cfg.TRAIN_BATCH_DET_BBOX_NORMALIZE_TARGETS_PRECOMPUTED \
                = self.load_param("TRAIN.BATCH.DET", "BBOX_NORMALIZE_TARGETS_PRECOMPUTED","bool")
            self.cfg.TRAIN_BATCH_DET_BBOX_INSIDE_WEIGHTS \
                = self.load_param("TRAIN.BATCH.DET", "BBOX_INSIDE_WEIGHTS","list","float")           
            self.cfg.TRAIN_BATCH_DET_BBOX_NORMALIZE_MEANS \
                = self.load_param("TRAIN.BATCH.DET", "BBOX_NORMALIZE_MEANS","list","float")
            self.cfg.TRAIN_BATCH_DET_BBOX_NORMALIZE_STDS \
                = self.load_param("TRAIN.BATCH.DET", "BBOX_NORMALIZE_STDS","list","float")
            
            self.cfg.TRAIN_BATCH_DET_ALPHAS = self.load_param("TRAIN.BATCH.DET", "ALPHAS", "list","float")
            self.cfg.TRAIN_BATCH_DET_BETAS = self.load_param("TRAIN.BATCH.DET", "BETAS", "list","float")
            self.cfg.TRAIN_BATCH_DET_GAMMAS_H1 = self.load_param("TRAIN.BATCH.DET", "GAMMAS_H1", "list","float")
            self.cfg.TRAIN_BATCH_DET_GAMMAS_H2 = self.load_param("TRAIN.BATCH.DET", "GAMMAS_H2", "list","float")
            self.cfg.TRAIN_BATCH_DET_BOTTOM_RATE = self.load_param("TRAIN.BATCH.DET", "BOTTOM_RATE","float")
     
            ''' 
                TRAIN.LAYER 
            '''
            
            ''' 
                TEST.DEFAULT
            '''
            self.cfg.TEST_DEFAULT_DEBUG = self.load_param("TEST.DEFAULT", "DEBUG", "bool")
            self.cfg.TEST_DEFAULT_USE_FLIPPED = self.load_param("TEST.DEFAULT", "USE_FLIPPED", "bool")                 
            self.cfg.TEST_DEFAULT_SCALES = self.load_param("TEST.DEFAULT", "SCALES","list","int")
            self.cfg.TEST_DEFAULT_MAX_SIZE = self.load_param("TEST.DEFAULT", "MAX_SIZE", "int")
            
            self.cfg.TEST_DEFAULT_HAS_RPN = self.load_param("TEST.DEFAULT", "HAS_RPN", "bool")
            self.cfg.TEST_DEFAULT_PROPOSAL_METHOD = self.load_param("TEST.DEFAULT", "PROPOSAL_METHOD")
            self.cfg.TEST_DEFAULT_NMS = self.load_param("TEST.DEFAULT", "NMS", "float")
            self.cfg.TEST_DEFAULT_RPN_NMS_THRESH = self.load_param("TEST.DEFAULT", "RPN_NMS_THRESH", "float")
            self.cfg.TEST_DEFAULT_RPN_PRE_NMS_TOP_N = self.load_param("TEST.DEFAULT", "RPN_PRE_NMS_TOP_N", "int")
            self.cfg.TEST_DEFAULT_RPN_POST_NMS_TOP_N = self.load_param("TEST.DEFAULT", "RPN_POST_NMS_TOP_N", "int")
            self.cfg.TEST_DEFAULT_CFC_THRESH = self.load_param("TEST.DEFAULT", "CFC_THRESH", "float")
            
            ''' 
                TEST.BATCH.CFC
            '''
            self.cfg.TEST_BATCH_CFC_NUM_IMG = self.load_param("TEST.BATCH.CFC", "NUM_IMG", "int")            
           
            ''' 
                EVAL.DEFAULT
            '''
            self.cfg.EVAL_DEFAULT_METRIC = self.load_param("EVAL.DEFAULT", "METRIC")
            
            ''' 
                INDEXING
            '''
            self.cfg.INDEXING_NUM_IM_EST = self.load_param("INDEXING", "NUM_IM_EST", "int")
            self.cfg.INDEXING_MAX_BUF_SIZE = self.load_param("INDEXING", "MAX_BUF_SIZE", "int")
            self.cfg.INDEXING_CHECK_DS = self.load_param("INDEXING", "CHECK_DS", "bool")
            self.cfg.INDEXING_BATCH_SIZE = self.load_param("INDEXING", "BATCH_SIZE", "int")
            
            ''' 
                SEARCH.DEFAULT
            '''
            self.cfg.SEARCH_DEFAULT_MAX_CAND = self.load_param("SEARCH.DEFAULT", "MAX_CAND", "int")
            self.cfg.SEARCH_DEFAULT_TOP_K = self.load_param("SEARCH.DEFAULT", "TOP_K", "int")
            
            ''' 
                SEARCH.RENDERING
            '''
            self.cfg.SEARCH_RENDERING_ITEM_SIZE = self.load_param("SEARCH.RENDERING", "ITEM_SIZE", "list", "int")
            self.cfg.SEARCH_RENDERING_ROWS = self.load_param("SEARCH.RENDERING", "ROWS", "int")
            self.cfg.SEARCH_RENDERING_HEIGHT = self.load_param("SEARCH.RENDERING", "HEIGHT", "int")
            
            
        except Exception as e:
            print ("[Error] loading config: {}".format(str(e)))
    
    ''' 
        load_param: load casted parameters
    '''
    def load_param(self, bloc, param, proto="str", sub_proto="str"):
        
        if proto == "list":
            cls = None
            try:
                module = importlib.import_module('builtins')
                cls = getattr(module, sub_proto)
                
            except AttributeError:
                module, sub_proto = sub_proto.rsplit(".", 1)
                module = importlib.import_module(module)
                cls = getattr(module, sub_proto)    
                
            except Exception as e:
                print ("[Error] load_param: {}".format(str(e)))
                
            assert cls != None, "[Error] unable to load parameters: unknown type"
            
            vals = self.parser.get(bloc, param).split(",")
            vals = [ cls(v) for v in vals]
            return vals
        
        else:        
            cls = None       
            try:
                if proto == 'bool':
                    val = self.parser.get(bloc, param)
                    return val.lower() in ['true','yes','1']
                    
                module = importlib.import_module('builtins')
                cls = getattr(module, proto)
                
            except AttributeError:
                module, proto = proto.rsplit(".", 1)
                module = importlib.import_module(module)
                cls = getattr(module, proto)    
                
            except Exception as e:
                print ("[Error]load_param: {}".format(str(e)))
                
            assert cls != None, "[Error] unable to load parameters: unknown type"
            
            return cls(self.parser.get(bloc, param))
            
            
        
            