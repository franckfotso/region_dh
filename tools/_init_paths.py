import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        #print ("path: ", path)

this_dir = osp.dirname(__file__)

# Add models to PYTHONPATH
add_path(osp.join(this_dir, '..', 'models'))

# Add libs to PYTHONPATH
add_path(osp.join(this_dir, '..', 'libs'))

# Add faster_rcnn to PYTHONPATH
add_path(osp.join(this_dir, '..', 'libs', "faster_rcnn"))
