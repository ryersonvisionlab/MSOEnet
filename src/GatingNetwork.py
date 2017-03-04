import tensorflow as tf
from components import *


class GatingNetwork(object):

    def __init__(self, name, input, reuse=None):
        self.name = name

        with tf.get_default_graph().name_scope(self.name):
            """
            Construct the gating network graph structure
            """
            # first convolution (1x5x5x1x4)
            conv1 = conv3d('Gate_conv1', input[:, :1], 5, 1, 4, reuse)

            # activation
            h_conv1 = eltwise_square('square', conv1)

            gate_output = conv3d('Gate_conv2', h_conv1, 1, 1, 1, reuse)

            self.output = gate_output
