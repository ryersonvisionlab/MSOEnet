import tensorflow as tf
from src.dataset import read_data_sets
from src.dataset import QueueRunner
from src.components import *


def data_layer():
    with tf.name_scope('data_layer'):
        num_channels = 1
        temporal_extent = 5
        input_shape = [temporal_extent, 256, 256, num_channels]
        target_shape = [256, 256, 2]

        # read UCF-101 image sequences with ground truth flows
        ucf101 = read_data_sets('/home/mtesfald/UCF-101-gt', temporal_extent)

        # read validation data
        x_val, y_val = ucf101.validation_data()

        with tf.device("/cpu:0"):
            queue_runner = QueueRunner(ucf101, input_shape, target_shape,
                                       batch_size, num_threads)
            x, y_ = queue_runner.get_inputs()

        return x, y_, ucf101, queue_runner


def loss_layer(y, y_):
    with tf.name_scope('loss_layer'):
        epe = l1_loss(y, y_)
        return epe


def solver(learning_rate, loss):
    with tf.name_scope('solver'):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return train_step


def summaries(epe, y, y_):
    with tf.name_scope('summaries'):
        # graph loss
        tf.scalar_summary('loss', epe)

        # visualize target and predicted flows
        tf.image_summary('flow 1', flowToColor(y), max_images=1)
        tf.image_summary('flow 2', flowToColor(y_), max_images=1)

        # merge summaries
        merged = tf.merge_all_summaries()

        return merged


def architecture(x, dataset):
    with tf.name_scope('architecture'):
        """first convolutional layer"""
        kernel_height = 5
        kernel_width = 5
        num_filters = 32
        W_conv1 = weight_variable([dataset.temporal_extent,
                                   kernel_height,
                                   kernel_width,
                                   num_channels,
                                   num_filters])
        b_conv1 = bias_variable([num_filters])

        # activation node
        h_conv1 = eltwise_square(conv3d(x, W_conv1) + b_conv1)

        # avg pooling node
        h_pool1 = avg_pool_3x3x3(h_conv1)

        """second convolutional layer"""
        temporal_extent = 1
        kernel_width = 1
        kernel_height = 1
        num_channels = num_filters
        num_filters = 64
        W_conv2 = weight_variable([temporal_extent,
                                   kernel_height,
                                   kernel_width,
                                   num_channels,
                                   num_filters])
        b_conv2 = bias_variable([num_filters])

        # pre-activation node
        conv2 = conv3d(h_pool1, W_conv2) + b_conv2

        # channel-wise l1 normalization
        conv2_l1norm = l1_normalize(conv2, 4)

        """flow-out (decode) layer"""
        temporal_extent = 1
        kernel_width = 1
        kernel_height = 1
        num_channels = num_filters
        num_filters = 2
        W_conv3 = weight_variable([temporal_extent,
                                   kernel_height,
                                   kernel_width,
                                   num_channels,
                                   num_filters])
        b_conv3 = bias_variable([num_filters])

        # flow-out pre-activation node
        y = conv3d(conv2_l1norm, W_conv3) + b_conv3

        # reshape node
        y = tf.reshape(y, [-1, target_shape[0],
                           target_shape[1],
                           target_shape[2]])

        return y


def build_net(batch_size, learning_rate, num_threads):
    # attach data layer
    x, y_, dataset, queue_runner = data_layer()
    # attach architecture
    y = architecture(x, dataset)
    # attach loss
    loss = loss_layer(y, y_)
    # attach solver
    solver = solver(learning_rate, loss)
    # attach summaries
    summaries = summaries(loss, y, y_)

    return summaries, solver, loss, queue_runner
