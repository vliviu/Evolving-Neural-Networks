import cPickle
import gzip
import time
import PIL.Image
import math
import numpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import random


import theano
import theano.tensor as T
import os

def visualize_dataset(dataset, width, height):
    '''
    Pick some of the samples in the dataset and visualize them
    type dataset: string
    param1 dataset: the path to the dataset

    type width: int
    param2 width: the number of images in a row

    type height: int
    param3 height: the number of images in a column
    
    '''
    print '... loading data'
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    number_of_images = width * height;
    image_idx = random.sample(range(1000), number_of_images)
    imageLength = int(math.sqrt(train_set[0].shape[1]))
    print imageLength
    image = numpy.zeros((height * (imageLength+1) + 1, width * (imageLength+1) +1), dtype=float)
    for i in xrange(height):
        for j in xrange(width):
            num = i * width + j
            image[1+i*(imageLength+1):1+i*(imageLength+1)+imageLength, 1+j*(imageLength+1):1+j*(imageLength+1)+imageLength]=\
                    test_set[0][num].reshape(imageLength, imageLength)
    plt.imshow(image,cmap=cm.Greys_r)
    plt.show()

if __name__ == '__main__':
    print 'visualize framed noise dataset'
    visualize_dataset('../data/mnist2framed.pkl.gz', 5, 5)

    print 'visualize random noise dataset'
    visualize_dataset('../data/mnist100random.pkl.gz', 5, 5)
