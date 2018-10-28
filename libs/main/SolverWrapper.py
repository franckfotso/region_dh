# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Project: CBIR-360
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
        self.stepsizes = []

    def train_model(self, sess, max_iters):
        """train a model, with snapshots and summaries"""
        
        # define stepsizes for learning rate decrease
        stepsize_rate = self.cfg.TRAIN_DEFAULT_STEPSIZE_RATE
        
        for v in range(1, stepsize_rate):
            _stepsize = int(max_iters*(v/stepsize_rate))+1
            
            if _stepsize < max_iters:
                self.stepsizes.append(_stepsize)
        #print("stepsizes: ", sorted(self.stepsizes))
        
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
        
        # RFM
        _batch_size = 1
        if self.cfg.MAIN_DEFAULT_TASK == "CFC":
            _batch_size = self.cfg.TRAIN_BATCH_CFC_NUM_IMG
        elif self.cfg.MAIN_DEFAULT_TASK == "DET":
            _batch_size = self.cfg.TRAIN_BATCH_DET_IMS_PER_BATCH
        else:
            raise NotImplementedError
            
        max_epochs = max_iters*_batch_size
        max_epochs = max_epochs/len(self.train_gen.images)
        
        iters_per_epoch = int(len(self.train_gen.images)/_batch_size)+1
        snapshot_iters = iters_per_epoch*self.cfg.TRAIN_DEFAULT_SNAPSHOT_EPOCHS
               
        assert snapshot_iters < max_iters, \
        "snapshot_iters must be very small than max_iters, got: [{},{}]".format(snapshot_iters, max_iters)
        
        while iter < max_iters + 1:
            epoch = iter*_batch_size
            epoch = epoch/len(self.train_gen.images)
            
            # Learning rate
            if iter == next_stepsize + 1:
                # Add snapshot here before reducing the learning rate
                """
                try:
                except:
                """
                blobs_val = self.val_gen.get_next_minibatch()
                outputs = self.net.train_step(sess, blobs_val, train_op)
                
                if self.techno in ["FT", "DLBHC", "TLOSS2", "SSDH1", "SSDH2"]:
                    val_acc = outputs["accuracies"]["acc_cls"]
                elif self.techno in ["Region-DH"]:
                    val_acc = outputs["accuracies"]["im_acc_cls"]
                                
                #self.snapshot(sess, iter)
                self.snapshot(sess, iter, val_acc)
                
                rate *= self.cfg.TRAIN_DEFAULT_GAMMA
                sess.run(tf.assign(lr, rate))
                next_stepsize = stepsizes.pop()
            
            timer.tic()
            # Get training data, one batch at a time
            blobs = self.train_gen.get_next_minibatch()

            now = time.time()
            if iter == 1 or now - last_summary_time > self.cfg.TRAIN_DEFAULT_SUMMARY_INTERVAL:
                # Compute the graph with summary
                outputs = self.net.train_step_with_summary(sess, blobs, train_op)               
                summary = outputs["summary"]
                
                #self.writer.add_summary(summary, float(iter)) # by iter
                self.writer.add_summary(summary, float(epoch)) # by epoch

                # Also check the summary on the validation set
                blobs_val = self.val_gen.get_next_minibatch()
                summary_val = self.net.get_summary(sess, blobs_val)            
                
                #self.valwriter.add_summary(summary_val, float(iter)) # by iter
                self.valwriter.add_summary(summary_val, float(epoch)) # by epoch
                last_summary_time = now
            else:
                # Compute the graph without summary                
                outputs = self.net.train_step(sess, blobs, train_op)
                    
            timer.toc()            
            
            """ Get outputs based on techno """
            if self.techno in ["FT", "DLBHC", "TLOSS2"]:
                # get classification outputs
                loss_cls     = outputs["losses"]["loss_cls"]
                acc_cls      = outputs["accuracies"]["acc_cls"]
            
            elif self.techno in ["SSDH1", "SSDH2"]:
                # get classification and encoding outputs
                loss_E1     = outputs["losses"]["loss_E1"]
                loss_E2     = outputs["losses"]["loss_E2"]
                loss_E3     = outputs["losses"]["loss_E3"]
                acc_cls      = outputs["accuracies"]["acc_cls"]
                
            elif self.techno in ["TLOSS1", "TLOSS2"]:
                triplet_loss = outputs["losses"]["triplet_loss"]
                                
            elif self.techno in ["Region-DH"]:
                # get detection, classification and encoding outputs
                L_reg1       = outputs["losses"]["L_reg1"]
                L_reg2       = outputs["losses"]["L_reg2"]
                L_cls1       = outputs["losses"]["L_cls1"]
                L_cls2       = outputs["losses"]["L_cls2"]
                L_cls3       = outputs["losses"]["L_cls3"]
                L_H1_E1      = outputs["losses"]["L_H1_E1"]
                L_H1_E2      = outputs["losses"]["L_H1_E2"]
                L_H2_E1      = outputs["losses"]["L_H2_E1"]
                L_H2_E2      = outputs["losses"]["L_H2_E2"]
                im_acc_cls   = outputs["accuracies"]["im_acc_cls"]
                bbox_acc_cls = outputs["accuracies"]["bbox_acc_cls"]
                
            else:
                raise NotImplementedError
                
            total_loss   = outputs["losses"]["total_loss"]          
            
            # Display training information
            if iter % (self.cfg.TRAIN_DEFAULT_DISPLAY) == 0:
                max_steps = iters_per_epoch
                cur_step = iter - int(epoch)*max_steps
                if cur_step <= 0:
                    cur_step = 0                    
                
                # based on epochs & steps
                print("epochs: [{}/{}], steps: [{}/{}] | total_loss: {:.6f}".\
                      format(int(epoch)+1, int(max_epochs), cur_step, max_steps, total_loss))
                
                if self.techno in ["FT", "DLBHC", "TLOSS2"]:
                    print(" >>> loss_cls: {:.6f} \n >>> acc_cls: {:.6f}".\
                      format(loss_cls, acc_cls))
                
                elif self.techno in ["SSDH1", "SSDH2"]:
                    print(" >>> loss_E1: {:.6f} \n >>> loss_E2: {:.6f} \n >>> loss_E3: {:.6f} \n >>> acc_cls: {:.6f}".\
                      format(loss_E1, loss_E2, loss_E3, acc_cls))
                    
                elif self.techno in ["TLOSS1", "TLOSS2"]:
                    print(" >>> triplet_loss: {:.6f}".format(triplet_loss))
                    
                elif self.techno in ["Region-DH"]:
                    print(" >>> L_reg1: {:.6f} \n >>> L_reg2: {:.6f}".format(L_reg1, L_reg2))
                    print(" >>> L_cls1: {:.6f} \n >>> L_cls2: {:.6f} \n >>> L_cls3: {:.6f}".\
                          format(L_cls1, L_cls2, L_cls3))
                    print(" >>> L_H1_E1: {:.6f}, L_H1_E2: {:.6f}".format(L_H1_E1, L_H1_E2))
                    print(" >>> L_H2_E1: {:.6f}, L_H2_E2: {:.6f}".format(L_H2_E1, L_H2_E2))
                    print(" >>> bbox_acc_cls: {:.6f} >>> im_acc_cls: {:.6f}".format(bbox_acc_cls, im_acc_cls))
                    
                print(" >>> lr: {:.6f}".format(lr.eval()))                
                print("cur_iter: {} \nmax_iters: {} \nsnapshot_iters: {}".format(iter, max_iters, snapshot_iters))
                print("stepsizes: ", sorted(self.stepsizes))
                print("config: ", self.cfg.FILE)
                print("speed: {:.3f}s / iter".format(timer.average_time))
                print("----------------------------")
    
            # Snapshotting       
            #if iter % self.cfg.TRAIN_DEFAULT_SNAPSHOT_ITERS == 0: snapshot_iters
            #if iter % snapshot_iters == 0:
            if iter % snapshot_iters == 0 and iter != (next_stepsize + 1):
                print("Snapshotting iter. {}...".format(iter))
                
                blobs_val = self.val_gen.get_next_minibatch()
                outputs = self.net.train_step(sess, blobs_val, train_op)              
                
                if self.techno in ["FT", "DLBHC", "TLOSS2", "SSDH1", "SSDH2"]:
                    val_acc = outputs["accuracies"]["acc_cls"]
                elif self.techno in ["Region-DH"]:
                    val_acc = outputs["accuracies"]["im_acc_cls"]
                
                last_snapshot_iter = iter
                #ss_path, np_path = self.snapshot(sess, iter)
                ss_path, np_path = self.snapshot(sess, iter, val_acc)
                np_paths.append(np_path)
                ss_paths.append(ss_path)
    
            # Remove the old snapshots if there are too many
            if len(np_paths) > self.cfg.TRAIN_DEFAULT_SNAPSHOT_KEPT:
                self.remove_snapshot(np_paths, ss_paths)
    
            iter += 1
    
        if last_snapshot_iter != iter - 1:
            blobs_val = self.val_gen.get_next_minibatch()   
            outputs = self.net.train_step(sess, blobs_val, train_op)
           
            if self.techno in ["FT", "DLBHC", "TLOSS2", "SSDH1", "SSDH2"]:
                val_acc = outputs["accuracies"]["acc_cls"]
            elif self.techno in ["Region-DH"]:
                val_acc = outputs["accuracies"]["im_acc_cls"]
                
            self.snapshot(sess, iter - 1, val_acc)
    
        self.writer.close()
        self.valwriter.close()
        
        
    def find_previous(self):
        sfiles = os.path.join(self.output_dir, self.cfg.TRAIN_DEFAULT_SNAPSHOT_PREFIX + '_iter_*.ckpt.meta')
        #print("self.output_dir: ", self.output_dir)
        #print("self.cfg.TRAIN_DEFAULT_SNAPSHOT_PREFIX: ", self.cfg.TRAIN_DEFAULT_SNAPSHOT_PREFIX )
        
        sfiles = glob.glob(sfiles)
        sfiles.sort(key=os.path.getmtime)
        # Get the snapshot name in TensorFlow
        redfiles = []
        for stepsize in self.stepsizes:
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
        stepsizes = [s for s in self.stepsizes]
    
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
        for stepsize in self.stepsizes:
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
            if self.techno in ["Region-DH"]:
                layers = self.net.create_architecture('TRAIN', self.dataset.num_cls, tag='default',
                                                      anchor_scales=self.cfg.TRAIN_BATCH_DET_ANCHOR_SCALES,
                                                      anchor_ratios=self.cfg.TRAIN_BATCH_DET_ANCHOR_RATIOS)
            else:
                layers = self.net.create_architecture('TRAIN', self.dataset.num_cls, tag='default')
                
            # Define the loss
            loss = layers['total_loss']
            # Set learning rate and momentum
            #lr = tf.Variable(self.cfg.TRAIN_DEFAULT_LEARNING_RATE, trainable=False, name="lr") # if not ckpt
            lr = tf.Variable(self.cfg.TRAIN_DEFAULT_LEARNING_RATE, trainable=False, name="lr_ft") # if ckpt used
            
            top_opt = tf.train.MomentumOptimizer(lr, self.cfg.TRAIN_DEFAULT_MOMENTUM) # normal learning
            bot_opt = tf.train.MomentumOptimizer(lr*self.cfg.TRAIN_BATCH_DET_BOTTOM_RATE,
                                              self.cfg.TRAIN_DEFAULT_MOMENTUM) # slow/no learning
            
            # Compute the gradients with regard to the loss            
            # & apply some changes on the gradients
            
            if self.cfg.TRAIN_BATCH_DET_BOTTOM_RATE == 1.0:
                # same learning rate for both top and bottom
                gvs = top_opt.compute_gradients(loss)
                train_op = top_opt.apply_gradients(gvs)
            else:
                train_vars = tf.trainable_variables()
                train_names = [v.name for v in train_vars]

                top_names = []
                if self.techno in ["SSDH1", "DLBHC","Region-DH"]:
                    # single fc_emb, multi-cfc
                    if self.techno == "Region-DH":
                        top_names.append(self.net._scope+'/fc6/weights:0')
                        top_names.append(self.net._scope+'/fc6/biases:0')
                        top_names.append(self.net._scope+'/fc7/weights:0')
                        top_names.append(self.net._scope+'/fc7/biases:0')
                        top_names.append(self.net._scope+'/embs_H1/weights:0')
                        top_names.append(self.net._scope+'/embs_H1/biases:0')
                        top_names.append(self.net._scope+'/embs_H2/weights:0')
                        top_names.append(self.net._scope+'/embs_H2/biases:0')
                        top_names.append(self.net._scope+'/bbox_score/weights:0')
                        top_names.append(self.net._scope+'/bbox_score/biases:0')
                    else:
                        top_names.append(self.net._scope+'/fc_emb/weights:0')
                        top_names.append(self.net._scope+'/fc_emb/biases:0')

                    for _id in range(self.dataset.num_cls-1):
                        top_names.append(self.net._scope+'/cls_score'+str(_id)+'/weights:0')
                        top_names.append(self.net._scope+'/cls_score'+str(_id)+'/biases:0')

                elif self.techno in ["SSDH2"]:
                    # multiple fc_emb, multi-cfc
                    
                    if self.techno == "Region-DH":
                        top_names.append(self.net._scope+'/embs_H1/weights:0')
                        top_names.append(self.net._scope+'/embs_H1/biases:0')
                        top_names.append(self.net._scope+'/bbox_score/weights:0')
                        top_names.append(self.net._scope+'/bbox_score/biases:0')
                        
                        for _id in range(self.dataset.num_cls-1):
                            top_names.append(self.net._scope+'/embs_H2'+str(_id)+'/weights:0')
                            top_names.append(self.net._scope+'/embs_H2'+str(_id)+'/biases:0')
                    else:
                        for _id in range(self.dataset.num_cls-1):
                            top_names.append(self.net._scope+'/fc_emb'+str(_id)+'/weights:0')
                            top_names.append(self.net._scope+'/fc_emb'+str(_id)+'/biases:0')
                            
                    for _id in range(self.dataset.num_cls-1):        
                        top_names.append(self.net._scope+'/cls_score'+str(_id)+'/weights:0')
                        top_names.append(self.net._scope+'/cls_score'+str(_id)+'/biases:0') 
                else:
                    raise NotImplementedError

                bottom_names = [v.name for v in train_vars if v.name not in top_names]

                print("-------------------------------------------")
                for name in top_names:
                    print("[INFO] top_layer: ", name)
                print("-------------------------------------------")
                for name in bottom_names:
                    print("[INFO] bot_layer: ", name)
                print("-------------------------------------------")
                print("[INFO] BOTTOM_RATE: ", self.cfg.TRAIN_BATCH_DET_BOTTOM_RATE)
                #raise NotImplementedError
                
                bottom_vars = [v for v in train_vars if v.name in bottom_names]
                top_vars = [v for v in train_vars if v.name in top_names]

                gvs = tf.gradients(loss, bottom_vars+top_vars)
                gvs_bot = gvs[:len(bottom_vars)]
                gvs_top = gvs[len(bottom_vars):] 

                top_op = top_opt.apply_gradients(zip(gvs_top, top_vars))
                bot_op = bot_opt.apply_gradients(zip(gvs_bot, bottom_vars))
                train_op = tf.group(top_op, bot_op)            
            
            # We will handle the snapshots ourselves
            self.saver = tf.train.Saver(max_to_keep=100000)
            # Write the train and validation information to tensorboard
            
            train_tb_dir = osp.join(self.tb_dir, "plot_train")
            if not osp.exists(train_tb_dir):
                os.makedirs(train_tb_dir)
                
            val_tb_dir = osp.join(self.tb_dir, "plot_val")
            if not osp.exists(val_tb_dir):
                os.makedirs(val_tb_dir)
            
            self.writer = tf.summary.FileWriter(train_tb_dir, sess.graph)
            self.valwriter = tf.summary.FileWriter(val_tb_dir)
    
        return lr, train_op
    
    def snapshot(self, sess, iter, val_acc):
        net = self.net
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Store the model snapshot
        #filename = self.cfg.TRAIN_DEFAULT_SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.ckpt'
        filename = self.cfg.TRAIN_DEFAULT_SNAPSHOT_PREFIX + '_iter_{:d}_acc_{:.3f}'.format(iter, val_acc) + '.ckpt'
        filename = os.path.join(self.output_dir, filename)
        self.saver.save(sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))
        
        # Also store some meta information, random state, etc.
        #nfilename = self.cfg.TRAIN_DEFAULT_SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.pkl'
        nfilename = self.cfg.TRAIN_DEFAULT_SNAPSHOT_PREFIX + '_iter_{:d}_acc_{:.3f}'.format(iter, val_acc) + '.pkl'
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
            self.train_gen.cur_idx = cur
            self.train_gen.perm_ids = perm
            self.val_gen.cur_idx = cur_val
            self.val_gen.perm_ids = perm_val
    
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