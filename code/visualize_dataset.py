import cPickle
import gzip
import time
import PIL.Image
import math
import numpy

import theano
import theano.tensor as T
import os

def visualize_dataset(dataset, n_samples):
    '''
    Pick some of the samples in the dataset and visualize them
    type dataset: string
    param1 dataset: the path to the dataset

    type n_samples: int
    param2 n_samples: the number of images to show
    
    '''
    print '... loading data'
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()


if __name__ == '__main__':
    visualize_dataset('../data/mnist2framed.pkl.gz')
