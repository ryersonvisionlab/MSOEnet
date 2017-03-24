import tensorflow as tf
from src.components import *


class MSOE(object):

    def __init__(self, name, input, reuse=None):
        self.name = name
        self.temporal_extent = 2
        self.nbands = 2
        self.input_shape = input.get_shape().as_list()

        with tf.get_default_graph().name_scope(self.name):
            """
            Construct the MSOE network graph structure
            """
            # first convolution (2x11x11x1x32)
            conv1 = conv3d('MSOE_conv1', input, 11,
                           self.temporal_extent, 32, reuse)
            # activation
            h_conv1 = eltwise_square('square', conv1)
            # average pooling (1x3x3x1x1)
            pool1 = avg_pool3d('avg_pool', h_conv1, 3, 1)
            # second convolution (1x1x1x32x64)
            conv2 = conv3d('MSOE_conv2', pool1, 1, 1, 64, reuse)
            # channel-wise l1 normalization (batchx1xHxWx64)
            l1_norm = l1_normalize('l1_norm', conv2)

            self.output = l1_norm
