# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Project: Region-DH
# Module: libs.datasets.BasicInput
# Copyright (c) 2018
# Written by: Franck FOTSO
# Based on: py-faster-rcnn 
#    [https://github.com/rbgirshick/py-faster-rcnn]
# Licensed under MIT License
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


import os, pickle, copy
import numpy as np
import os.path as osp
import scipy.io as sio
import xml.etree.ElementTree as ET
from multiprocessing import Process, Queue

from datasets.Dataset import Dataset
from datasets.Image import Image
from utils.timer import Timer

class Pascal(Dataset):
    
    def __init__(self,
                 name, 
                 path_dir,
                 classes=None,
                 cls_to_id=None,
                 sets=None,
                 year=2007,
                 metric=None, 
                 cfg=None):
        self._year = year
        self._metric = metric
        self._cfg = cfg
        
        super(Pascal, self).__init__(name, 
                 path_dir,
                 classes,
                 cls_to_id,
                 sets,
                 cfg)
        
        self.load_classes(cfg.PASCAL_DATASET_FILE_CLS)
        
    def get_cfg(self):
        return self._cfg


    def set_cfg(self, value):
        self._cfg = value


    def del_cfg(self):
        del self._cfg


    def get_year(self):
        return self._year


    def get_metric(self):
        return self._metric


    def set_year(self, value):
        self._year = value


    def set_metric(self, value):
        self._metric = value


    def del_year(self):
        del self._year


    def del_metric(self):
        del self._metric
        
    def built_im_path(self, im_nm, im_DIR):
        im_fn = None
        im_pn = None
        
        for ext in self.cfg.PASCAL_DATASET_DEFAULT_EXT:
            im_pn = osp.join(im_DIR, im_nm+"."+ext)
            if osp.exists(im_pn):
                im_fn = im_nm+"."+ext
                break
            
        assert im_fn != None, \
        "[ERROR] unable to load image {} in {}".format(im_nm, im_DIR)
        
        return im_fn, im_pn 
        
    def load_images(self, im_names):
        images = []
        
        cache_images_file = osp.join(self.cfg.MAIN_DIR_ROOT, "cache","pascal_images.pkl")
        if osp.exists(cache_images_file):
            with open(cache_images_file,'rb') as fp:
                images = pickle.load(fp)
                print ('[INFO] images_obj loaded from {}'.format(cache_images_file))
                fp.close()
            return images
        
        for im_nm in im_names:
            im_DIR = osp.join(self.cfg.MAIN_DIR_ROOT, "data", self.name, 
                              self.cfg.PASCAL_DATASET_DIR_IMAGE)
            anno_DIR = osp.join(self.cfg.MAIN_DIR_ROOT, "data", self.name, 
                              self.cfg.PASCAL_DATASET_DIR_ANNOTATION)
           
            im_fn, im_pn = self.built_im_path(im_nm, im_DIR)
            
            anno_pn = osp.join(anno_DIR, im_nm+".xml")
            assert osp.exists(anno_pn), \
                   "[ERROR] unable to load annotation {}".format(im_nm+".xml")
            
            rois = self.readXmlAnno(anno_pn)            
            img = Image(im_fn, im_pn, gt_rois=rois)           
            images.append(img)
            
        # prepare images for training
        images = self.prepare_images(images)
            
        if not osp.exists(cache_images_file):
            with open(cache_images_file,'wb') as fp:
                pickle.dump(images, fp)
                print ('[INFO] images_obj saved to {}'.format(cache_images_file))
                fp.close()
            
        return images
    
        
    def load_gt_rois(self, gt_set, num_proc=1):
        images = []        
        im_names = self.sets[gt_set]['im_names']
        
        imgs_DIR = osp.join(self.cfg.MAIN_DIR_ROOT, "data", self.name, 
                                  self.cfg.PASCAL_DATASET_DIR_IMAGE)
        anno_DIR = osp.join(self.cfg.MAIN_DIR_ROOT, "data", self.name, 
                                  self.cfg.PASCAL_DATASET_DIR_ANNOTATION)
        
        cache_images_fn = '{}_gt_{}_images.pkl'.format(self.name, gt_set)        
        
        cache_images_pn = osp.join(self.cfg.MAIN_DIR_ROOT,
                                       "cache",cache_images_fn)
        
        print ('[INFO] loading gt rois for {}...'.format(self.name))        
        if osp.exists(cache_images_pn):                
            with open(cache_images_pn,'rb') as fp:
                images = pickle.load(fp)
                print ('[INFO] images with gt loaded from {}'.format(cache_images_pn))
                fp.close()
                
            return images
        
        # sub-method for a multiprocessing
        def _load_gt_rois(proc_id, l_start, l_end, im_names, queue, cfg):
            _images = []
            timer = Timer()
            
            timer.tic()            
            for im_i in range(l_start, l_end):
                im_nm = im_names[im_i]  
                
                """ load rois data """
                img_fn, img_pn = self.built_im_path(im_nm, imgs_DIR)
                anno_pn = osp.join(anno_DIR, im_nm+".xml")
                
                image = Image(filename=img_fn,pathname=img_pn)                
                rois = self.readXmlAnno(anno_pn)
                
                image.gt_rois = rois
                _images.append(image)
            
            timer.toc()    
            #n_imgs = len(xrange(l_start, l_end))
            print ('[INFO] >> PROC.ID [{}]:  {}-{}/{} images processed in {:.3f}'.\
            format(proc_id, l_start, l_end, len(im_names), timer.average_time))
                
            #return on queue
            queue.put(_images)           
        
        processes = []
        queues  = []
        num_imgs = len(im_names)
        l_start = 0   
        if num_imgs <= num_proc:
            num_proc = num_imgs
        
        l_offset = int(np.ceil(num_imgs / float(num_proc)))
        
        for proc_id in range(num_proc):
            l_end = min(l_start + l_offset, num_imgs)
            q = Queue()
            p = Process(target=_load_gt_rois, 
                        args=(proc_id, l_start, l_end, im_names, q, self.cfg))        
            p.start()
            processes.append(p)
            queues.append(q)
            l_start += l_offset
            
        for proc_id in range(num_proc):            
            _images = queues[proc_id].get()
            images.extend(_images)
            processes[proc_id].join()
            
        print ('gt > bef. flipped, len(images): {}'.format(len(images)))
        
        """ append horizontal flipped  images """
        print ('[INFO] append horizontal flipped  gt images: {}'.format(self.name))
        
        if self.cfg.TRAIN_DEFAULT_USE_FLIPPED and gt_set != 'test':
            images = self.append_flipped_images(images=images, src='gt', num_proc=num_proc)
            
        print ('gt > aft. flipped, len(images): {}'.format(len(images)))
                
        if not osp.exists(cache_images_pn):
            with open(cache_images_pn,'wb') as fp:
                pickle.dump(images, fp)
                print ('[INFO] images with gt saved to {}'.format(cache_images_pn))
                fp.close()
        
        return images
   
    def load_sets(self):
        sets = {
            "train":    {"im_names": [], "images": [], "num_items":0},
            "trainval": {"im_names": [], "images": [], "num_items":0},
            "val":      {"im_names": [], "images": [], "num_items":0},
            "test":     {"im_names": [], "images": [], "num_items":0}
        }
        task = self.cfg.MAIN_DEFAULT_TASK
                
        if task == 'CFC' or task == 'DET':
            DATASET_DIR = self.cfg.PASCAL_DATASET_DIR_MAIN_SET
        elif task == 'SEGM':
            DATASET_DIR = self.cfg.PASCAL_DATASET_DIR_SEGM_SET
        else:
            raise('[ERROR] unknown task')
        
        if self.name == "bsd_voc2012":
            DATASET_DIR = ''
        
        train_file = osp.join(self.cfg.MAIN_DIR_ROOT, "data", self.name, 
                              DATASET_DIR, self.cfg.PASCAL_DATASET_FILE_TRAIN)
        
        trainval_file = osp.join(self.cfg.MAIN_DIR_ROOT, "data", self.name, 
                              DATASET_DIR, self.cfg.PASCAL_DATASET_FILE_TRAINVAL)
        
        test_file = osp.join(self.cfg.MAIN_DIR_ROOT, "data", self.name, 
                              DATASET_DIR, self.cfg.PASCAL_DATASET_FILE_TEST)
        
        val_file = osp.join(self.cfg.MAIN_DIR_ROOT, "data", self.name, 
                              DATASET_DIR, self.cfg.PASCAL_DATASET_FILE_VAL)        
        
        im_names = []
        if osp.exists(train_file):
            with open(train_file) as in_f:
                for im_nm in in_f:
                    im_nm = im_nm.split('\n')[0].split('\r')[0]
                    im_names.append(im_nm)
                in_f.close()
            sets["train"]["im_names"] = im_names
            sets["train"]["num_items"] = len(im_names)            
        else:
            print ("[WARN] unable to load file {}".format(train_file))
        del im_names
        
        im_names = []
        if osp.exists(trainval_file):
            with open(trainval_file) as in_f:
                for im_nm in in_f:
                    im_nm = im_nm.split('\n')[0].split('\r')[0]
                    im_names.append(im_nm)
                in_f.close()
            sets["trainval"]["im_names"] = im_names
            sets["trainval"]["num_items"] = len(im_names)            
        else:
            print ("[WARN] unable to load file {}".format(trainval_file))
        del im_names
        
        im_names = []
        if osp.exists(test_file):
            with open(test_file) as in_f:
                for im_nm in in_f:
                    im_nm = im_nm.split('\n')[0].split('\r')[0]
                    im_names.append(im_nm)
                in_f.close()
            sets["test"]["im_names"] = im_names
            sets["test"]["num_items"] = len(im_names)
        else:
            print ("[WARN] unable to load file {}".format(test_file))
        del im_names
        
        im_names = []
        if osp.exists(val_file):
            with open(val_file) as in_f:
                for im_nm in in_f:
                    im_nm = im_nm.split('\n')[0].split('\r')[0]
                    im_names.append(im_nm)
                in_f.close()
            sets["val"]["im_names"] = im_names
            sets["val"]["num_items"] = len(im_names)
        else:
            print ("[WARN] unable to load file {}".format(val_file))
        del im_names
        
        self.sets = sets
        
    def readXmlAnno(self, anno_pn):
        tree = ET.parse(anno_pn)
        root = tree.getroot()
            
        objs = root.findall('object')
        num_objs = len(objs)
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, len(self.cls_to_id)), dtype=np.float32)
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        is_segm = int(root.find('segmented').text)
        size = root.find('size')
        im_info = {"width": int(size.find('width').text),
                   "height": int(size.find('height').text),
                   "depth": int(size.find('depth').text)}
                    
        for id_obj, obj in enumerate(objs):
            xmin = int(float(obj.find('bndbox').find('xmin').text)) - 1
            ymin = int(float(obj.find('bndbox').find('ymin').text)) - 1
            xmax = int(float(obj.find('bndbox').find('xmax').text)) - 1
            ymax = int(float(obj.find('bndbox').find('ymax').text)) - 1
            id_cls = self.cls_to_id[obj.find('name').text.strip()]
            
            boxes[id_obj,:] = [xmin, ymin, xmax, ymax]
            gt_classes[id_obj] = id_cls
            overlaps[id_obj, id_cls] = 1.0
            seg_areas[id_obj] = (xmax - xmin + 1) * (ymax - ymin + 1)
        
        #overlaps = csr_matrix(overlaps)
        
        return {"boxes": boxes,
                "gt_classes": gt_classes,
                "gt_overlaps": overlaps,
                "seg_areas": seg_areas,
                "is_segm": is_segm,
                "im_info": im_info,
                "flipped": False}
        
    def filter_images(self, images):
        num_imgs = len(images)
        
        images_OK = []
        
        for im_i in range(num_imgs):
            gt_overlaps = images[im_i].gt_rois['max_overlaps']
            pr_overlaps = images[im_i].pr_rois['max_overlaps']            
            all_overlaps = gt_overlaps.extend(pr_overlaps)
            
            # find boxes with sufficient overlap
            fg_inds = np.where(all_overlaps >= self.cfg.TRAIN_DEFAULT_FG_THRESH)[0]
            # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
            bg_inds = np.where((all_overlaps < self.cfg.TRAIN_DEFAULT_BG_THRESH_HI) &
                               (all_overlaps >= self.cfg.TRAIN_DEFAULT_BG_THRESH_LO))[0]
            # image is only valid if such boxes exist
            valid = len(fg_inds) > 0 or len(bg_inds) > 0
            
            if valid:
                images_OK.append(images[im_i])
        
        num_after = len(images_OK)       
        print ('[INFO] Filtered {} images: {} -> {}'.format(num_imgs - num_after,
                                                    num_imgs, num_after))
        return images_OK
            
        
    def append_flipped_images(self, images, src, num_proc=1):
        num_imgs = len(images)
        widths = []
        
        assert src in ['gt','pr'], \
        '[ERROR] unknown rois src provided'
        
        if src == 'gt':
            widths = [image.gt_rois['im_info']['width'] for image in images]
        elif src == 'pr':
            widths = [image.pr_rois['im_info']['width'] for image in images]
        
        # sub-method for multiprocessing
        def _append_flipped_images(proc_id, l_start, l_end, queue, cfg):
            _images_flip = []
            timer = Timer()
            
            timer.tic() 
            for im_i in range(l_start, l_end):
                if src == 'gt':
                    boxes = images[im_i].gt_rois['boxes'].copy()
                elif src == 'pr':
                    boxes = images[im_i].pr_rois['boxes'].copy()
                    
                oldx1 = boxes[:, 0].copy()
                oldx2 = boxes[:, 2].copy()
                boxes[:, 0] = widths[im_i] - oldx2 - 1
                boxes[:, 2] = widths[im_i] - oldx1 - 1            
                assert (boxes[:, 2] >= boxes[:, 0]).all()
                assert (boxes[:, 0] >= 0).all()
                assert (boxes[:, 2] >= 0).all()
                            
                image = copy.deepcopy(images[im_i])
                if src == 'gt':
                    image.gt_rois['boxes'] = boxes
                    image.gt_rois['flipped'] = True
                elif src == 'pr':
                    image.pr_rois['boxes'] = boxes
                    image.pr_rois['flipped'] = True
                    
                _images_flip.append(image)
                
            timer.toc()    
            #n_imgs = len(xrange(l_start, l_end))
            print ('[INFO] >> PROC.ID [{}]:  {}-{}/{} images flipped in {:.3f}'.\
            format(proc_id, l_start, l_end, len(images), timer.average_time))
            
            #return on queue
            queue.put(_images_flip)
            
        processes = []
        queues = []
        l_start = 0
        if num_imgs <= num_proc:
            num_proc = num_imgs
            
        l_offset = int(np.ceil(num_imgs / float(num_proc)))
        
        for proc_id in range(num_proc):
            l_end = min(l_start + l_offset, num_imgs)
            q = Queue()
            p = Process(target=_append_flipped_images, 
                        args=(proc_id, l_start, l_end, q, self.cfg))        
            p.start()
            processes.append(p)
            queues.append(q)
            l_start += l_offset
        
        for proc_id in range(num_proc):
            images.extend(queues[proc_id].get())
            processes[proc_id].join()
                        
        return images

    def built_output_dir(self, root_dir, phase):
        output_dir = osp.join(root_dir, self.cfg.MAIN_DIR_OUTPUTS, self.name, 
                              phase+'_'+self.cfg.TRAIN_DEFAULT_ROI_METHOD)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        return output_dir

    year = property(get_year, set_year, del_year, "year's docstring")
    metric = property(get_metric, set_metric, del_metric, "metric's docstring")
    cfg = property(get_cfg, set_cfg, del_cfg, "cfg's docstring")
    
    
        
    
