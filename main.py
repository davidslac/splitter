from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

import numpy as np
import h5py

from psmlearn.pipeline import Pipeline
from psmlearn import h5util

sys.path.append('/home/davidsch/rel/slaclab/vae')
import batchgen
from partitioninfo import PartitionInfo
import splitter_model as sm

class DataGen(object):
    def __init__(self, data_dir, batch_size=32, epochs=0):
        self.data_dir = data_dir

        self.data_fnames = {'train': os.path.join(self.data_dir, 'xtcav_autoencoder_dense_prep_train.h5'),
                            'validation': os.path.join(self.data_dir, 'xtcav_autoencoder_dense_prep_validation.h5'),
                            'test': os.path.join(self.data_dir, 'xtcav_autoencoder_dense_prep_validation.h5')
                            }
        self.balance_samples = {'train': '/scratch/davidsch/dataprep/vae_balance_samples_train.h5',
                                'validation': '/scratch/davidsch/dataprep/vae_balance_samples_validation.h5',
                                'test': '/scratch/davidsch/dataprep/vae_balance_samples_test.h5'
        }

        self.split2num_samples = {'train': 0, 'validation': 0, 'test': 0}
        for split in self.split2num_samples:
            h5 = h5py.File(self.data_fnames[split], 'r')
            self.split2num_samples[split] = len(h5['imgs'])
            h5.close()

        self.batchgens = {'train': None, 'validation': None, 'test': None}
        for split in self.batchgens.keys():
            partition_fname = self.balance_samples[split]
            assert os.path.exists(partition_fname), "%s doesn't exist" % partition_fname
            gen = batchgen.gen_batches(partition_fname,
                                       x_keys=['imgs'],
                                       y_dset2output={'imgs': 'imgs',
                                                      'gasdet': 'gasdet',
                                                      'labels_onehot': 'labels_onehot',
                                                      'labels': 'labels'},
                                       batch_size=batch_size,
                                       epochs=epochs
                                       )
            self.batchgens[split]=gen
        
class MySteps(object):
    def __init__(self):
        self.data_dir = '/scratch/davidsch/dataprep'
        self.datagen = DataGen(self.data_dir)
        self.split2pos = {'train': 0, 'validation': 1, 'test': 2}

    def init(self, config, pipeline):
        pass

    def grand_mean(self, config, pipeline, step2h5list, output_files):
        for split, fname in self.data_fnames.iteritems():
            h5in = h5py.File(fname, 'r')
            imgs = h5in['imgs'][:]
            grand_mean = np.mean(imgs, axis=0)
            pos = self.split2pos[split]
            h5out = h5py.File(output_files[pos],'w')
            h5out['img_grand_mean']=grand_mean
            
    def view_grand_mean(self, plot, pipeline, plotFigH, config, step2h5list):
        split = 'train'
        pos = self.split2pos[split]
        img_fname = self.data_fnames[split]
        imgs = h5py.File(img_fname, 'r')['imgs'][0:100]
        grand_mean = h5py.File(step2h5list['grand_mean'][pos],'r')['img_grand_mean'][:]
        imgs -= grand_mean
        vmin = np.min(imgs)
        vmax = np.max(imgs)
        plt = pipeline.plt
        plt.figure(plotFigH)
        for img in imgs:
            plt.clf()
            img = np.reshape(img, (100, 50))
            plt.imshow(img, interpolation='none', vmin=vmin, vmax=vmax)
            plt.colorbar()
            plt.pause(1)

    def train(self, config, pipeline, step2h5list, output_files):
        t0 = time.time()
        dg = self.datagen
        gen = dg.batchgens['train']
        dropout_rate = config.dropout
        
        f_input = Input(shape=(108,54,1))
        H = Convolution2D(32,5,5,border_mode='same', init='glorot_uniform')(f_input)
        H = BatchNormalization(mode=2)(H)
        H = Activation('relu')
        H = Convolution2D(32,3,3,border_mode='same', init='glorot_uniform')(H)
        H = BatchNormalization(mode=2)(H)
        H = Activation('relu')
        H = Convolution2D(32,3,3,border_mode='same', init='glorot_uniform')(H)
        H = BatchNormalization(mode=2)(H)
        H = Activation('relu')
        H = Convolution2D(1,1,1, border_mode='same', init='glorot_uniform')(H)
        f_out = Activation('sigmoid')(H)
        
        d_input = Input(shape=(108,54,1))
        H = Convolution2D(32, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(d_input)
        H = LeakyReLU(0.2)(H)
        H = Dropout(dropout_rate)(H)
        H = Convolution2D(16, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(H)
        H = LeakyReLU(0.2)(H)
        H = Dropout(dropout_rate)(H)
        H = Flatten()(H)
        H = Dense(256)(H)
        H = LeakyReLU(0.2)(H)
        H = Dropout(dropout_rate)(H)
        d_V = Dense(4,activation='softmax')(H)

        discriminator = Model(d_input, d_V)
        discriminator.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3))
        
        f0_input = Input(shape=(108,54,1), name='f0_input')
        f1_input = Input(shape=(108,54,1), name='f1_input')

        f0_out = F(f0_input)
        f1_out = F(f1_input)

        for batch in gen:
            imgs = batch[0][0]
            gasdet = np.mean(batch[1]['gasdet'], axis=1)
            labels = batch[1]['labels']
            labels_onehot = batch[1]['labels_onehot']

            idx0 = labels==0
            idx1 = labels==1
            imgs0 = imgs[idx0]
            imgs1 = imgs[idx1]
            break
            

        
def main(argv):
    pipeline = Pipeline(stepImpl=MySteps(),
                        defprefix='splitter',
                        outputdir='/scratch/davidsch/dataprep')

    pipeline.parser.add_argument('--batch_size', type=int, help='batch size', default=100)
    pipeline.parser.add_argument('--train_epochs', type=int, help='number of training epochs', default=20)
    pipeline.parser.add_argument('--dropout', type=float, help='D dropout rate', default=0.25)

    pipeline.add_step_method(name='grand_mean', output_files=['_train', '_validation', '_test'])
    pipeline.add_step_method_plot(name='view_grand_mean')
    pipeline.add_step_method(name='train', output_files=['_model','_hist'])

#    pipeline.add_step_method(name='test_balance_samples')
#    pipeline.add_step_method(name='edge_weights', output_files=['_train', '_validation', '_test'])
#    pipeline.add_step_method(name='test_batch_read')

    pipeline.run(argv[1:])
    
if __name__ == '__main__':
    main(sys.argv)
