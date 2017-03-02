import tensorflow as tf
from components import *


class GatingNetwork(object):

    def __init__(self, name, input, reuse=None):
        self.name = name

        with tf.get_default_graph().name_scope(self.name):
            """
            Construct the gating network graph structure
            """
            # first convolution (1x5x5x1x32)
            conv1 = conv3d('Gate_conv1', input[:, :1], 5, 1, 32, reuse)
            # activation
            h_conv1 = eltwise_square('square', conv1)

            self.output = h_conv1
