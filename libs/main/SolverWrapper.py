# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Project: Region-DH
# Module: libs.main.SolverWrapper
# Copyright (c) 2018
# Written by: Franck FOTSO
# Based on: tf-faster-rcnn 
#    (https://github.com/endernewton/tf-faster-rcnn)
# Licensed under MIT License
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os.path as osp
import numpy as np
import tensorflow as tf
import time, glob, os
from utils.timer import Timer

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

try:
  import cPickle as pickle
except ImportError:
  import pickle

class SolverWrapper(object):
    """A simple wrapper for the training process
    """

    def __init__(self, net, weights, techno, dataset,
                 train_gen, val_gen, tb_dir, output_dir, cfg):
        """Initialize the SolverWrapper."""
        self.net = net
        self.weights = weights
        self.techno = techno
        self.dataset = dataset
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.tb_dir = tb_dir
        self.output_dir = output_dir
        self.cfg = cfg
        
        """
        if cfg.MAIN_DEFAULT_TASK == "DET" and cfg.TRAIN_DEFAULT_BBOX_REG:
            cache_dir = osp.join(cfg.MAIN_DIR_ROOT,cfg.MAIN_DIR_CACHE)
            means_file = osp.join(cache_dir,'{}_bbox_means.npy'\
                                   .format(cfg.TRAIN_DEFAULT_SEGM_METHOD))
            stds_file = osp.join(cache_dir, '{}_bbox_stds.npy'\
                                   .format(cfg.TRAIN_DEFAULT_SEGM_METHOD))
            #print 'means_file: {}'.format(means_file)
            #print 'stds_file: {}'.format(stds_file)
            
            if os.path.exists(means_file) and os.path.exists(stds_file):
                self.bbox_means = np.load(means_file)
                self.bbox_stds = np.load(stds_file)
            else:
                print ('[INFO] SolverWrapper: compute bbox means & stds over the train set...')
                self.bbox_means, self.bbox_stds = compute_bbox_means_stds(train_gen.cache_im_dir, 
                                                                          train_gen.num_cls, cfg)                                                                
            self.data_gen.bbox_means = self.bbox_means
            self.data_gen.bbox_stds = self.bbox_stds
        """ 

    def train_model(self, sess, max_iters):
        """train a model, with snapshots and summaries"""
        
        # Construct the computation graph
        lr, train_op = self.construct_graph(sess)
    
        # Find previous snapshots if there is any to restore from
        lsf, nfiles, sfiles = self.find_previous()
    
        # Initialise the variables or restore them from the last snapshot
        if lsf == 0:
            rate, last_snapshot_iter, stepsizes, np_paths, ss_paths = self.initialize(sess)
        else:
            rate, last_snapshot_iter, stepsizes, np_paths, ss_paths = self.restore(sess, 
                                                                                str(sfiles[-1]), 
                                                                                str(nfiles[-1]))
        timer = Timer()
        iter = last_snapshot_iter + 1
        last_summary_time = time.time()
        # Make sure the lists are not empty
        stepsizes.append(max_iters)
        stepsizes.reverse()
        next_stepsize = stepsizes.pop()
        while iter < max_iters + 1:
            # Learning rate
            if iter == next_stepsize + 1:
                # Add snapshot here before reducing the learning rate
                self.snapshot(sess, iter)
                rate *= self.cfg.TRAIN_DEFAULT_GAMMA
                sess.run(tf.assign(lr, rate))
                next_stepsize = stepsizes.pop()
            
            assert self.techno in self.cfg.MAIN_DEFAULT_TECHNOS, \
                "[ERROR] unknown techno found {}, expected: {}".format(self.techno, self.cfg.MAIN_DEFAULT_TECHNOS)
            
            if self.techno == "SSDH":
                self.train_model_with_ssdh(timer, last_summary_time, sess, train_op, max_iters, lr)
                
            elif self.techno == 'RegionDH':
                raise NotImplemented
            
            elif self.techno == 'ISDH':
                raise NotImplemented 
    
            # Snapshotting
            if iter % self.cfg.TRAIN_DEFAULT_SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                ss_path, np_path = self.snapshot(sess, iter)
                np_paths.append(np_path)
                ss_paths.append(ss_path)
    
            # Remove the old snapshots if there are too many
            if len(np_paths) > self.cfg.TRAIN_DEFAULT_SNAPSHOT_KEPT:
                self.remove_snapshot(np_paths, ss_paths)
    
            iter += 1
    
        if last_snapshot_iter != iter - 1:
            self.snapshot(sess, iter - 1)
    
        self.writer.close()
        self.valwriter.close()
        
    def train_model_with_ssdh(self, timer, last_summary_time, sess, train_op, max_iters, lr):
        timer.tic()
        # Get training data, one batch at a time
        blobs = self.train_gen.get_next_minibatch()

        now = time.time()
        if iter == 1 or now - last_summary_time > self.cfg.TRAIN_DEFAULT_SUMMARY_INTERVAL:
            # Compute the graph with summary
            loss_cls, summary = self.net.train_step_with_summary(sess, blobs, train_op)
            self.writer.add_summary(summary, float(iter))
            
            # Also check the summary on the validation set
            blobs_val = self.val_gen.get_next_minibatch()
            
            summary_val = self.net.get_summary(sess, blobs_val)
            self.valwriter.add_summary(summary_val, float(iter))
            last_summary_time = now
        else:
            # Compute the graph without summary
            loss_cls = self.net.train_step(sess, blobs, train_op)
        timer.toc()

        # Display training information
        if iter % (self.cfg.TRAIN_DEFAULT_DISPLAY) == 0:
            print('iter: %d / %d, loss cls: %.6f\n >>> lr: %f' % (iter, max_iters, loss_cls, lr.eval()))
            print('speed: {:.3f}s / iter'.format(timer.average_time))
            
    def find_previous(self):
        sfiles = os.path.join(self.output_dir, self.cfg.TRAIN_DEFAULT_SNAPSHOT_PREFIX + '_iter_*.ckpt.meta')
        sfiles = glob.glob(sfiles)
        sfiles.sort(key=os.path.getmtime)
        # Get the snapshot name in TensorFlow
        redfiles = []
        for stepsize in self.cfg.TRAIN_DEFAULT_STEPSIZE:
            redfiles.append(os.path.join(self.output_dir, 
                          self.cfg.TRAIN_DEFAULT_SNAPSHOT_PREFIX + '_iter_{:d}.ckpt.meta'.format(stepsize+1)))
        sfiles = [ss.replace('.meta', '') for ss in sfiles if ss not in redfiles]
    
        nfiles = os.path.join(self.output_dir, self.cfg.TRAIN_DEFAULT_SNAPSHOT_PREFIX + '_iter_*.pkl')
        nfiles = glob.glob(nfiles)
        nfiles.sort(key=os.path.getmtime)
        redfiles = [redfile.replace('.ckpt.meta', '.pkl') for redfile in redfiles]
        nfiles = [nn for nn in nfiles if nn not in redfiles]
    
        lsf = len(sfiles)
        assert len(nfiles) == lsf
    
        return lsf, nfiles, sfiles
        
    
    def initialize(self, sess):
        # Initial file lists are empty
        np_paths = []
        ss_paths = []
        # Fresh train directly from ImageNet weights
        print('Loading initial model weights from {:s}'.format(self.weights))
        variables = tf.global_variables()
        # Initialize all variables first
        sess.run(tf.variables_initializer(variables, name='init'))
        var_keep_dic = self.get_variables_in_checkpoint_file(self.weights)
        # Get the variables to restore, ignoring the variables to fix
        variables_to_restore = self.net.get_variables_to_restore(variables, var_keep_dic)
    
        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(sess, self.weights)
        print('Loaded.')
        # Need to fix the variables before loading, so that the RGB weights are changed to BGR
        # For VGG16 it also changes the convolutional weights fc6 and fc7 to
        # fully connected weights
        self.net.fix_variables(sess, self.weights)
        print('Fixed.')
        last_snapshot_iter = 0
        rate = self.cfg.TRAIN_DEFAULT_LEARNING_RATE
        stepsizes = list(self.cfg.TRAIN_DEFAULT_STEPSIZE)
    
        return rate, last_snapshot_iter, stepsizes, np_paths, ss_paths
    

    def restore(self, sess, sfile, nfile):
        # Get the most recent snapshot and restore
        np_paths = [nfile]
        ss_paths = [sfile]
        # Restore model from snapshots
        last_snapshot_iter = self.from_snapshot(sess, sfile, nfile)
        # Set the learning rate
        rate = self.cfg.TRAIN_DEFAULT_LEARNING_RATE
        stepsizes = []
        for stepsize in self.cfg.TRAIN_DEFAULT_STEPSIZE:
            if last_snapshot_iter > stepsize:
                rate *= self.cfg.TRAIN_DEFAULT_GAMMA
            else:
                stepsizes.append(stepsize)
    
        return rate, last_snapshot_iter, stepsizes, np_paths, ss_paths
    
    
    def remove_snapshot(self, np_paths, ss_paths):
        to_remove = len(np_paths) - self.cfg.TRAIN_DEFAULT_SNAPSHOT_KEPT
        for c in range(to_remove):
            nfile = np_paths[0]
            os.remove(str(nfile))
            np_paths.remove(nfile)
    
        to_remove = len(ss_paths) - self.cfg.TRAIN_DEFAULT_SNAPSHOT_KEPT
        for c in range(to_remove):
            sfile = ss_paths[0]
            # To make the code compatible to earlier versions of Tensorflow,
            # where the naming tradition for checkpoints are different
            if os.path.exists(str(sfile)):
                os.remove(str(sfile))
            else:
                os.remove(str(sfile + '.data-00000-of-00001'))
                os.remove(str(sfile + '.index'))
            sfile_meta = sfile + '.meta'
            os.remove(str(sfile_meta))
            ss_paths.remove(sfile)
    
        
    def construct_graph(self, sess):
        with sess.graph.as_default():
            # Set the random seed for tensorflow
            tf.set_random_seed( self.cfg.MAIN_DEFAULT_RNG_SEED)
            # Build the main computation graph
            layers = self.net.create_architecture('TRAIN', self.dataset.num_cls, tag='default')
            # Define the loss
            loss = layers['loss_cls']
            # Set learning rate and momentum
            lr = tf.Variable(self.cfg.TRAIN_DEFAULT_LEARNING_RATE, trainable=False)
            self.optimizer = tf.train.MomentumOptimizer(lr, self.cfg.TRAIN_DEFAULT_MOMENTUM)
            
            # Compute the gradients with regard to the loss
            gvs = self.optimizer.compute_gradients(loss)
            # Double the gradient of the bias if set
            if self.cfg.TRAIN_DEFAULT_DOUBLE_BIAS:
                final_gvs = []
                with tf.variable_scope('Gradient_Mult') as scope:
                    for grad, var in gvs:
                        scale = 1.
                        if self.cfg.TRAIN_DEFAULT_DOUBLE_BIAS and '/biases:' in var.name:
                            scale *= 2.
                        if not np.allclose(scale, 1.0):
                            grad = tf.multiply(grad, scale)
                        final_gvs.append((grad, var))
                train_op = self.optimizer.apply_gradients(final_gvs)
            else:
                train_op = self.optimizer.apply_gradients(gvs)
            
            # We will handle the snapshots ourselves
            self.saver = tf.train.Saver(max_to_keep=100000)
            # Write the train and validation information to tensorboard
            
            train_tb_dir = osp.join(self.tb_dir, "train")
            if not osp.exists(train_tb_dir):
                os.makedirs(train_tb_dir)
                
            val_tb_dir = osp.join(self.tb_dir, "val")
            if not osp.exists(val_tb_dir):
                os.makedirs(val_tb_dir)
            
            self.writer = tf.summary.FileWriter(train_tb_dir, sess.graph)
            self.valwriter = tf.summary.FileWriter(val_tb_dir)
    
        return lr, train_op
    
    def snapshot(self, sess, iter):
        net = self.net
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Store the model snapshot
        filename = self.cfg.TRAIN_DEFAULT_SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.ckpt'
        filename = os.path.join(self.output_dir, filename)
        self.saver.save(sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))
        
        # Also store some meta information, random state, etc.
        nfilename = self.cfg.TRAIN_DEFAULT_SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.pkl'
        nfilename = os.path.join(self.output_dir, nfilename)
        # current state of numpy random
        st0 = np.random.get_state()
        # current position in the database
        cur = self.train_gen.cur_idx
        # current shuffled indexes of the database
        perm = self.train_gen.perm_ids
        # current position in the validation database
        cur_val = self.val_gen.cur_idx
        # current shuffled indexes of the validation database
        perm_val = self.val_gen.perm_ids
        
        # Dump the meta info
        with open(nfilename, 'wb') as fid:
            pickle.dump(st0, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(cur, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(perm, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(cur_val, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(perm_val, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(iter, fid, pickle.HIGHEST_PROTOCOL)
            
        return filename, nfilename
    
    def from_snapshot(self, sess, sfile, nfile):
        print('Restoring model snapshots from {:s}'.format(sfile))
        self.saver.restore(sess, sfile)
        print('Restored.')
        # Needs to restore the other hyper-parameters/states for training, (TODO xinlei) I have
        # tried my best to find the random states so that it can be recovered exactly
        # However the Tensorflow state is currently not available
        with open(nfile, 'rb') as fid:
            st0 = pickle.load(fid)
            cur = pickle.load(fid)
            perm = pickle.load(fid)
            cur_val = pickle.load(fid)
            perm_val = pickle.load(fid)
            last_snapshot_iter = pickle.load(fid)
            
            np.random.set_state(st0)
            self.data_layer._cur = cur
            self.data_layer._perm = perm
            self.data_layer_val._cur = cur_val
            self.data_layer_val._perm = perm_val
    
        return last_snapshot_iter

    def get_variables_in_checkpoint_file(self, file_name):
        try:
            reader = pywrap_tensorflow.NewCheckpointReader(file_name)
            var_to_shape_map = reader.get_variable_to_shape_map()
            return var_to_shape_map 
        except Exception as e:  # pylint: disable=broad-except
            print(str(e))
            if "corrupted compressed block contents" in str(e):
                print("It's likely that your checkpoint file has been compressed "
                  "with SNAPPY.")