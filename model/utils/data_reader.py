import numpy as np
import h5py
import random
import os
import glob

class H53DDataLoader(object):
    #data_dir = '/Users/bene/Dropbox/Dokumente/Promotion/PROJECTS/multiSCAT/PYTHON/Unet_3D-master/h5_data_sa'

    def __init__(self, data_dir, patch_size):
        data_files_list = glob.glob(os.path.join(data_dir+'/*.h5'))
       
        self.num_files = len(data_files_list)
        self.current_batchnum = 0
        
        data_files = []
        for i_file in range(0, self.num_files):
            data_files.append(h5py.File(data_files_list[i_file], 'r'))
            print(data_files_list[i_file])
        
        
        inputs = [np.array(data_files[i]['x']) for i in range(self.num_files)]
        labels = [np.array(data_files[i]['y']) for i in range(self.num_files)]


        self.t_c = 1
        self.t_n, self.t_d, self.t_h, self.t_w = inputs[0].shape
        self.d, self.h, self.w = patch_size, patch_size, patch_size

        
        self.train_inputs = [inputs[i] for i in range(self.num_files)]
        self.train_labels = [labels[i] for i in range(self.num_files)]


    def get_batch(self, batch_num):

        input_batches = []
        label_batches = []
        
        
        input_batches.append(self.train_inputs[batch_num])
        label_batches.append(self.train_labels[batch_num])
        
        inputs = np.stack(input_batches, axis=0)
        labels = np.stack(label_batches, axis=0)
        labels = labels[:,:,:,:,0]
        
        return inputs, labels

    def next_batch(self, batch_size):

        input_batches = []
        label_batches = []
        
        
        input_batches.append(self.train_inputs[self.current_batchnum])
        label_batches.append(self.train_labels[self.current_batchnum])
        
        inputs = np.stack(input_batches, axis=0)
        labels = np.stack(label_batches, axis=0)
        labels = labels[:,:,:,:,0]
        
        self.current_batchnum = self.current_batchnum + 1
        if(self.current_batchnum>self.num_files-1):
            self.current_batchnum = 0
        
        return inputs, labels

    def prepare_validation(self, overlap_stepsize):
        self.patches_ids = []
        self.drange = list(range(0, self.t_d-self.d+1, overlap_stepsize))
        self.hrange = list(range(0, self.t_h-self.h+1, overlap_stepsize))
        self.wrange = list(range(0, self.t_w-self.w+1, overlap_stepsize))
        if (self.t_d-self.d) % overlap_stepsize != 0:
            self.drange.append(self.t_d-self.d)
        if (self.t_h-self.h) % overlap_stepsize != 0:
            self.hrange.append(self.t_h-self.h)
        if (self.t_w-self.w) % overlap_stepsize != 0:
            self.wrange.append(self.t_w-self.w)
        for d in self.drange:
            for h in self.hrange:
                for w in self.wrange:
                    self.patches_ids.append((d, h, w))
        
    def reset(self):
        self.valid_patch_id = 0

    def valid_next_batch(self):
        input_batches = []
        label_batches = []
        # self.num_of_valid_patches = len(self.patches_ids)
        d, h, w = self.patches_ids[self.valid_patch_id]
        input_batches.append(self.valid_inputs[d:d+self.d, h:h+self.h, w:w+self.w, :])
        label_batches.append(self.valid_labels[d:d+self.d, h:h+self.h, w:w+self.w])
        inputs = np.stack(input_batches, axis=0)
        labels = np.stack(label_batches, axis=0)
        self.valid_patch_id += 1
        if self.valid_patch_id == self.num_of_valid_patches:
            self.reset()
        return inputs, labels, (d, h, w)
