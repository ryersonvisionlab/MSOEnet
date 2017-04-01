import tensorflow as tf
from src.graph_components import *


class GatingNetwork(object):

    def __init__(self, name, input, reuse=None):
        self.name = name

        with tf.get_default_graph().name_scope(self.name):
            """
            Construct the gating network graph structure
            """
            # first convolution (1x3x3x1x4)
            conv1 = conv3d('Gate_conv1', input[:, :1], 3, 1, 4, reuse)
            # first activation
            h_conv1 = relu(conv1)
            # second convolution (1x3x3x1x8)
            conv2 = conv3d('Gate_conv2', h_conv1, 3, 1, 8, reuse)
            # second activation
            h_conv2 = relu(conv2)
            # decode to final gate output (1x1x1x1xnum_input_channels)
            num_output = input.get_shape().as_list()[-1]
            gate_output = conv3d('Gate_conv3', h_conv2, 1, 1, num_output,
                                 reuse)
            # final activation [0,1]
            h_gate_output = tf.sigmoid(gate_output)

            self.output = h_gate_output
