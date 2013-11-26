
import cPickle
import gzip
import time
import PIL.Image
import math

import numpy

import theano
import theano.tensor as T
import os
from logistic_sgd import load_data

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,dtype=theano.config.floatX),borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,dtype=theano.config.floatX),borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32')
    
def add_frame(dataset='../data/mnist.pkl.gz', width=2):
    ''' Loads the dataset, then a frame to all the images in the datasets

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    :type width: int
    :param width: the width of the frame, specified by pixels
    
    '''        

    # the current folder is '/Users/STW/Documents/DeepLearningTutorials/code'

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    #The MNIST dataset is seperated to 50000,10000,10000
    print 'The dimension of training set is ', train_set[0].shape
    print 'The dimension of validation set is', valid_set[0].shape
    print 'The dimension of test set is ', test_set[0].shape

    '''
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)
    
    # return variables in Theano format
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval
    '''
    #assume that the images are square, the length of image is
    imageLength = int(math.sqrt(train_set[0].shape[1]))
    print 'the length of image is ', imageLength
    for i in xrange(train_set[0].shape[0]):
        #add a white frame
        for j in xrange(width):
            for k in xrange(imageLength):
                train_set[0][i][j*imageLength+k] = 1.0
                train_set[0][i][(imageLength-1-j)*imageLength+k] = 1.0
                train_set[0][i][k*imageLength+j] = 1.0
                train_set[0][i][k*imageLength+(imageLength-1-j)] = 1.0
    for i in xrange(valid_set[0].shape[0]):
        for j in xrange(width):
            for k in xrange(imageLength):
                valid_set[0][i][j*imageLength+k] = 1.0
                valid_set[0][i][(imageLength-1-j)*imageLength+k] = 1.0
                valid_set[0][i][k*imageLength+j] = 1.0
                valid_set[0][i][k*imageLength+(imageLength-1-j)] = 1.0
    for i in xrange(test_set[0].shape[0]):
        for j in xrange(width):
            for k in xrange(imageLength):
                test_set[0][i][j*imageLength+k] = 1.0
                test_set[0][i][(imageLength-1-j)*imageLength+k] = 1.0
                test_set[0][i][k*imageLength+j] = 1.0
                test_set[0][i][k*imageLength+(imageLength-1-j)] = 1.0
    #save noisy dataset
    os.chdir('../data')
    noisy_data = (train_set,valid_set,test_set)
    print 'Saving the noisy dataset'
    data_filename = 'mnist'+str(width)+'framed.pkl'
    noisy_data_file = file(data_filename,'wb')
    cPickle.dump(noisy_data, noisy_data_file, protocol = cPickle.HIGHEST_PROTOCOL)
    noisy_data_file.close()

    #compress the noisy data file with gzip
    f_in = open(data_filename, 'rb')
    f_out = gzip.open(data_filename+'.gz', 'wb')
    f_out.writelines(f_in)
    f_out.close()
    f_in.close()
    
    
if __name__ == '__main__':
    add_frame('../data/mnist.pkl.gz',2)
