# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Project: Region-DH
# Module: libs.datasets.Dataset
# Copyright (c) 2018
# Written by: Franck FOTSO
# Licensed under MIT License
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


import os.path as osp

class Dataset(object):
    
    def __init__(self, 
                 name, 
                 path_dir,
                 classes,
                 cls_to_id=None,
                 sets=None,
                 cfg=None
                 ):
        self._name = name
        self._path_dir = path_dir
        self._cls_to_id = cls_to_id
        self._classes = classes
        self._sets = sets
        
        self._classes = []
        self._rgb_to_cls = {}
        self._num_cls = 0       

    def get_num_cls(self):
        return self._num_cls


    def set_num_cls(self, value):
        self._num_cls = value


    def del_num_cls(self):
        del self._num_cls


    def get_name(self):
        return self._name


    def get_path_dir(self):
        return self._path_dir


    def get_classes(self):
        return self._classes


    def get_cls_to_id(self):
        return self._cls_to_id


    def get_sets(self):
        return self._sets


    def set_name(self, value):
        self._name = value


    def set_path_dir(self, value):
        self._path_dir = value


    def set_classes(self, value):
        self._classes = value


    def set_cls_to_id(self, value):
        self._cls_to_id = value


    def set_sets(self, value):
        self._sets = value


    def del_name(self):
        del self._name


    def del_path_dir(self):
        del self._path_dir


    def del_classes(self):
        del self._classes


    def del_cls_to_id(self):
        del self._cls_to_id


    def del_sets(self):
        del self._sets

        
    def load_classes(self, cls_file):
        self.classes = []
        cls_file = osp.join(self.cfg.MAIN_DIR_ROOT, "config", cls_file)
        
        with open(cls_file, 'r') as f:
            for line in f:
                cls = line.split('\n')[0].split('\r')[0]
                self.classes.append(cls)
            f.close()
            
        self.num_cls = len(self.classes)
        self.cls_to_id = dict(zip(self.classes, range(self.num_cls)))
        
        #print 'self.classes: {}'.format(self.classes)
        #print 'self.cls_to_id.keys(): {}'.format(self.cls_to_id.keys())
        
    def load_images(self):
        pass
    
    def load_segments(self):
        pass

    name = property(get_name, set_name, del_name, "name's docstring")
    path_dir = property(get_path_dir, set_path_dir, del_path_dir, "path_dir's docstring")
    classes = property(get_classes, set_classes, del_classes, "classes's docstring")
    cls_to_id = property(get_cls_to_id, set_cls_to_id, del_cls_to_id, "cls_to_id's docstring")
    sets = property(get_sets, set_sets, del_sets, "sets's docstring")
    num_cls = property(get_num_cls, set_num_cls, del_num_cls, "num_cls's docstring")
    

    
        
        