# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Project: Region-DH
# Module: tools.search
# Copyright (c) 2018
# Written by: Franck FOTSO
# Licensed under MIT License
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Goal: similarity search based on binary codes & deep features indexed

import _init_paths
import argparse, cv2, os, imutils
import numpy as np
from retriever.dist_tools import *
from retriever.DeepSearcher import DeepSearcher
from retriever.Rendering import Rendering
from indexer.DeepIndexer import DeepIndexer
from extractor.DeepExtractor import DeepExtractor
from Config import Config
import scipy.io as sio
from PIL import Image

def parse_args():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", dest="query", required=True, 
                    help="pathname of the query image ", type=str)
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
    ap.add_argument("--num_rlts", dest="num_rlts", default=49, 
                        help="top-k results", type=int)
    ap.add_argument("--view", dest="view", default=0, 
                        help="view results", type=int)
    ap.add_argument("--verbose", dest="verbose", default=0, 
                        help="display prints", type=int)
    args = vars(ap.parse_args())
    
    return args


if __name__ == '__main__':
    args = parse_args()
    
    print('[INFO] Called with args:')
    print(args)
    
    # setup & load configs
    _C = Config(config_pn="config/config.ini")
    cfg = _C.cfg
    
    cfg.TEST_BATCH_CFC_NUM_IMG = 1
    bacth_size = 1

    # hash codes & deep features extractor
    deep_extr = DeepExtractor(args["techno"], args["net"], args["num_cls"], 
                              args["num_bits"], args["weights"], cfg)

    im_pns = {}
    im_dir = args['im_dir']
    categs = os.listdir(im_dir)
    for i, categ in enumerate(categs):
        categ_dir = os.path.join(im_dir, categ)
        im_fns = os.listdir(categ_dir)
        
        for im_fn in im_fns:
            im_pns[im_fn] = os.path.join(categ_dir, im_fn)
            
            
    # extract binary codes & deep features
    binary_codes, deep_features = deep_extr.extract([args["query"]])
    qry_binCode = binary_codes[0]
    qry_featVec = deep_features[0]
    
    dSearcher = DeepSearcher(args["deep_db"], distanceMetric=chi2_distance)
    # compute similarities
    search_rlt = dSearcher.search(qry_binCode, qry_featVec, 
                                  numResults=args["num_rlts"], # numResults=20
                                  maxCandidates=cfg.SEARCH_DEFAULT_MAX_CAND) # maxCandidates=100
    print("[INFO] search took: {:.2f}s".format(search_rlt.search_time))
    
    # initialize the results montage
    #rendering = Rendering((240, 320), 5, 20)
    im_item_dim = (50, 50)
    rendering = Rendering(im_item_dim, 7, args["num_rlts"])

    # load the query image and process it
    queryImage = cv2.imread(args["query"])    
    if args["view"] == 1:
        cv2.imshow("Query", imutils.resize(queryImage, width=im_item_dim[0])) #320
        
    queryRelevant = [] # required gth on ranking
    
    # loop over the individual results
    for (i, (score, resultID, resultIdx)) in enumerate(search_rlt.results):
        # load the result image and display it
        #"""
        try:
            if args["verbose"] == 1:
                print("[RESULT] {result_num}. {result} - {score:.4f}".format(result_num=i + 1,
                                                                         result=resultID, 
                                                                         score=score))
            # resultID => im_fn
            result = cv2.imread("{}".format(im_pns[resultID])) 
            rendering.addResult(result, text="#{}".format(i + 1),
                              highlight=resultID in queryRelevant)
        except:
            print('Error: exception found on print')
        #"""

    # show the output image of results
    if args["view"] == 1:
        cv2.imshow("Results", imutils.resize(rendering.view, height=cfg.SEARCH_RENDERING_HEIGHT))
    
    # save results
    this_dir = os.path.dirname(__file__)
    out_rlts_DIR = os.path.join(this_dir,'..','outputs/qry_rlts')
    if not os.path.exists(out_rlts_DIR):
        os.mkdir(out_rlts_DIR)

    qry_pn = args["query"]
    qry_fn = qry_pn[qry_pn.rfind("/") + 1:]
    qry_nm, ext = qry_fn.split(".")
    
    out_qry_file = os.path.join(out_rlts_DIR, qry_nm+"_qry."+ext)
    out_rlts_file = os.path.join(out_rlts_DIR, qry_nm+"_rlts."+ext)
    cv2.imwrite(out_qry_file, imutils.resize(queryImage, height=im_item_dim[1]))
    cv2.imwrite(out_rlts_file, imutils.resize(rendering.view, height=cfg.SEARCH_RENDERING_HEIGHT))

    if args["view"] == 1:
        cv2.waitKey(0)
        
    dSearcher.finish()
    
    