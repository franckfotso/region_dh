# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Project: Region-DH
# Module: tools.eval_rank
# Copyright (c) 2018
# Written by: Franck FOTSO
# Based on: caffe-cvprw15
#    (https://github.com/kevinlin311tw/caffe-cvprw15)
# Licensed under MIT License
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Goal: evaluate performance of the retrieval system

import _init_paths
import h5py, csv, argparse, os, time
import numpy as np
from sklearn.metrics import hamming_loss
from sklearn.metrics.pairwise import pairwise_distances
from multiprocessing import Process, Queue

from Config import Config
from datasets.CIFAR10 import CIFAR10
from datasets.Pascal import Pascal
from retriever.dist_tools import *
from utils.timer import Timer

def parse_args():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()  
    ap.add_argument("--dataset", dest="dataset", required=True, 
                    help="dataset name to use", type=str)
    ap.add_argument('--gt_set', dest='gt_set',
                    help='gt set use to list data', required=True,
                    default='train', type=str)
    ap.add_argument("--train_db", dest='train_db', required=True,
                    help="path for the train deepfeatures db")
    ap.add_argument("--val_db", dest='val_db', required=True,
                help="path for the val deepfeatures db")
    ap.add_argument("--encoding", dest='encoding', default="bin_code",
                help="encoding of features to evaluate")
    ap.add_argument("--num_rlts", dest="num_rlts", default=500, 
                help="top-k results", type=int)
    ap.add_argument("--config", dest="config", required=True, type=str,
                    help="config file for the techno")
    args = vars(ap.parse_args())
    
    return args


def comp_precision(val_labels, val_codes, train_labels, train_codes, top_k, cfg, 
                   encoding="bin_code", dataset=None, multilabel=False):
    """ compute the Mean Average Precision (MAP) for the top-k retrievals """    
    num_qrys = len(val_codes)
    
    # index of the k positions
    m = np.array([ x+1 for x in range(top_k)])
    AP = np.zeros((num_qrys))
    w_AP = np.zeros((num_qrys))
    ACG = np.zeros((num_qrys))
    num_TP = np.zeros((len(m)))
    
    processes = []
    queues  = []
    num_proc = cfg.MAIN_DEFAULT_CORES
    num_items = len(val_codes)

    l_start = 0   
    if num_items <= num_proc:
        num_proc = num_items
    l_offset = int(np.ceil(num_items / float(num_proc)))
    
    # setup & start paralell processing
    for proc_id in range(num_proc):
        l_end = min(l_start + l_offset, num_items)
        q = Queue()
        p = Process(target=_comp_precision, 
                    args=(l_start, l_end, val_labels, val_codes, 
                          train_labels, train_codes, top_k, encoding, q,
                          dataset, multilabel))
        p.start()
        processes.append(p)
        queues.append(q)
        l_start += l_offset

    # collect results
    for proc_id in range(num_proc):
        _num_TP, _ids, _AP, _w_AP, _ACG = queues[proc_id].get()
        
        num_TP  += _num_TP
        for _i, _id in enumerate(_ids):
            AP[_id] = _AP[_i]
            w_AP[_id] = _w_AP[_i]
            ACG[_id] = _ACG[_i]
    
    MAP = np.mean(AP) # MAP
    w_MAP = np.mean(w_AP) # w_MAP
    ACG = np.mean(ACG) # ACG@k
    print ("[INFO] MAP: {:.6}".format(MAP))
    print ("[INFO] w_MAP: {:.6}".format(w_MAP))
    print ("[INFO] ACG: {:.6}".format(ACG))
    
    # precision at k results
    prec_k = num_TP/(m*num_qrys)
    
    return prec_k, MAP, w_MAP, ACG


def _comp_precision(l_start, l_end, val_labels, val_codes, 
                    train_labels, train_codes, 
                    top_k, encoding, queue, dataset=None, multilabel=False):
    
    # index of the k positions
    m = np.array([ x+1 for x in range(top_k)])
    _num_TP = np.zeros((len(m)))
    _AP = []
    _w_AP = []
    _ACG = []
    _ids = []

    for i in range(l_start, l_end):
        qry_label = val_labels[i]
        qry_code = val_codes[i]

        print ("[INFO] Processing query {}/{}".format(i+1, len(val_codes)))

        qry_rlts = []
        top_k = min(top_k, len(train_codes))

        if encoding == "bin_code":
            #print ("[INFO] == using hamming distance ")
            qry_codes = [qry_code]
            rlt_dists = pairwise_distances(np.array(qry_codes), np.array(train_codes), 'hamming')[0]            
            # get indexes sorted in min order
            s_indexes = rlt_dists.argsort()
            
            # sort HAMMING distance in ascending order
            _qry_rlts = [(rlt_dists[k], k) for k in s_indexes] # => (dist, idx)
            _qry_rlts = sorted(_qry_rlts) 
            qry_rlts = [(dist, train_labels[k], k) for dist, k in _qry_rlts]
            qry_rlts = qry_rlts[:top_k]
            #raise NotImplementedError

        else:
            #print ("[INFO] == using euclidean")
            rlt_dists = {}
            for id_code, train_code in enumerate(train_codes):
                # compute distance between the two feature vector
                d = euclidean_dist(qry_code, train_code) # euclidean   
                rlt_dists[id_code] = d
                #print("d: ", d)
                    
            # sort all results such that small distance values are in the top
            _qry_rlts = [(v, k) for (k, v) in rlt_dists.items()]
            _qry_rlts = sorted(_qry_rlts)
            qry_rlts = [(dist, train_labels[k], k) for dist, k in _qry_rlts]
            qry_rlts = qry_rlts[:top_k]

        # similarity (0, 1) observed for each j-th position
        r = np.zeros((len(qry_rlts))) # r(j)
        ACG = np.zeros((len(qry_rlts)))
        N_pos = 0 # number of relevant images over all results

        for j, (dist, label, idx) in enumerate(qry_rlts):
            if multilabel:
                for id_cls in range(len(label)):
                    if qry_label[id_cls] == 1 and label[id_cls] == qry_label[id_cls]:
                        r[j] = 1
                        ACG[j] += 1
                N_pos += r[j]
            else:
                if qry_label == label:
                    r[j] = 1 # number of shared label. 1 for single-label   
                    N_pos += 1

        # number of relevant images within the top j images
        P_at_j = np.cumsum(r)/m
        ACG_at_j = np.cumsum(ACG)/m

        # AP for each query
        if N_pos == 0:
            AP = 0.0
            w_AP = 0.0
        else:
            AP = np.sum(P_at_j*r)/N_pos
            w_AP = np.sum(ACG_at_j*r)/N_pos
            
        _AP.append(AP)
        _w_AP.append(w_AP)
        _ACG.append(ACG_at_j[-1])
        _ids.append(i)
        print ("[INFO] Processing query {}/{}, AP: {:.6}, w_AP: {:.6}, ACG: {:.6}".\
               format(i+1, len(val_codes), AP, w_AP, ACG_at_j[-1]))

        _num_TP = _num_TP + np.cumsum(r)

    queue.put([_num_TP, _ids, _AP, _w_AP, _ACG])
        
        
if __name__ == '__main__':
    args = parse_args()
    
    print('[INFO] Called with args:')
    print(args)
    
    # setup & load configs
    _C = Config(config_pn=args['config'])
    cfg = _C.cfg 
    
    dataset = None
    ds_pascal = ["voc_2007", "voc_2012"]
    ds_coco = []
    
    dataset_DIR = os.path.join(cfg.MAIN_DIR_ROOT, "data", args["dataset"])
    
    if args["dataset"] in  ds_pascal:
        dataset = Pascal(name=args["dataset"], path_dir=dataset_DIR, cfg=cfg)
    
    elif args["dataset"] == "cifar10":
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
        
    multilabel = False
    if dataset.name in ds_pascal:
        # TODO: get also labels
        train_set, val_set = args["gt_set"].split("_")
        train_images, _ = dataset.load_gt_rois(gt_set=train_set)
        train_images = [image for image in train_images if not image.rois["gt"]['flipped']]
        train_labels = [image.rois["gt"]["labels"] for image in train_images]
        
        val_images, _ = dataset.load_gt_rois(gt_set=val_set)
        val_labels = [image.rois["gt"]["labels"] for image in val_images]
        multilabel = True
        
    elif dataset.name == "cifar10":
        (train_images, val_images), (train_labels, val_labels) = dataset.load_images()

    print ('[INFO] train_images.num: {}'.format(len(train_images)))
    print ('[INFO] val_images.num: {}'.format(len(val_images)))
    
    # get all binary codes / deep features extracted    
    train_db = h5py.File(args["train_db"], mode="r")
    val_db = h5py.File(args["val_db"], mode="r")
    
    assert args["encoding"] in ["bin_code", "deep_feat"], \
    "[ERROR] wrong encoding provided, found: {}, available: {}".format(args["encoding"],
                                                                      ["bin_code", "deep_feat"])
    
    if args["encoding"] == "bin_code":
        train_codes = train_db["binarycode"]
        val_codes = val_db["binarycode"]
    else:
        train_codes = train_db["deepfeatures"]
        val_codes = val_db["deepfeatures"]
        
    train_codes = np.array(train_codes)
    val_codes = np.array(val_codes)
    #print("len(val_codes): ", len(val_codes))
    #print("len(val_labels): ", len(val_labels))
    #print("len(train_codes): ", len(train_codes))
    #print("len(train_labels): ", len(train_labels))
    #raise NotImplementedError
    
    train_db.close()
    val_db.close()
    
    now = time.time()
    #prec_k, MAP  = comp_precision(val_labels[:100], val_codes[:100], 
    prec_k, MAP, w_MAP, ACG  = comp_precision(val_labels, val_codes, 
                                  train_labels, train_codes, 
                                  args["num_rlts"], cfg, args["encoding"],
                                  dataset=dataset, multilabel=multilabel)
    print("comp_precision, speed: {:.3f}s".format(time.time() - now))
    
    # save results
    this_dir = os.path.dirname(__file__)
    out_rlts_DIR = os.path.join(this_dir,'..','outputs/eval_rlts')
    if not os.path.exists(out_rlts_DIR):
        os.mkdir(out_rlts_DIR)
        
    trace_fn = args["dataset"]+"_"+args["encoding"]+"_top_"+str(args["num_rlts"])+"_trace.csv"
    trace_pn = os.path.join(out_rlts_DIR, trace_fn)
    eval_fn = args["dataset"]+"_"+args["encoding"]+"_top_"+str(args["num_rlts"])+"_eval.csv"
    eval_pn = os.path.join(out_rlts_DIR, eval_fn)
    
    with open(trace_pn, 'w') as csvfile:
        fieldnames = ['position', 'precision']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, prec in enumerate(prec_k):       
            writer.writerow({'position': i+1, 'precision': prec})     
            
    with open(eval_pn, 'w') as csvfile:
        fieldnames = ['queries', 'targets', 'encoding', 'mAP', 'w_mAP', 'ACG']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
              
        writer.writerow({'queries': len(val_labels),
                         'targets': len(train_labels), 
                         'encoding': args["encoding"], 
                         'mAP': MAP, "w_mAP": w_MAP, "ACG": ACG})
    
  
        
    