import tensorflow as tf
from src.dataset import *
from src.util import draw_hsv_ocv


def data_layer(name, path, batch_size, temporal_extent, num_threads):
    with tf.get_default_graph().name_scope(name):
        # read image sequences with ground truth flows
        d = load_FlyingChairs(path)

        # read validation data
        x_val, y_val_ = d.validation_data()

        input_shape = x_val.shape[1:]
        target_shape = y_val_.shape[1:]

        with tf.device("/cpu:0"):
            queue_runner = QueueRunner(d, input_shape, target_shape,
                                       batch_size, num_threads)
            x, y_ = queue_runner.get_inputs()

        return x, y_, x_val, y_val_, queue_runner


def conv3d(name, input_layer, kernel_spatial_size,
           kernel_temporal_size, out_channels, reuse=None):
    with tf.get_default_graph().name_scope(name):
        with tf.variable_scope(name, reuse=reuse):
            in_channels = input_layer.get_shape().as_list()[-1]

            # going to be sharing variables
            weights = tf.get_variable('weights',
                                      [kernel_temporal_size,
                                       kernel_spatial_size,
                                       kernel_spatial_size,
                                       in_channels,
                                       out_channels],
                                      initializer=tf.truncated_normal_initializer(stddev=0.3))
            biases = tf.get_variable('biases',
                                     [out_channels],
                                     initializer=tf.constant_initializer(0.0))

            # weight decay
            if name == 'conv1':
                reg = 0.5 * tf.nn.l2_loss(weights) * 4e-10
                tf.add_to_collection('weight_regs', reg)

            # spatially pad the image, but not temporally
            input_layer = tf.pad(input_layer,
                                 [[0, 0], [0, 0],
                                  [kernel_spatial_size / 2,
                                   kernel_spatial_size / 2],
                                  [kernel_spatial_size / 2,
                                   kernel_spatial_size / 2],
                                  [0, 0]], 'SYMMETRIC')

            conv_output = tf.nn.conv3d(input_layer, weights,
                                       strides=[1, 1, 1, 1, 1],
                                       padding='VALID')

        return tf.nn.bias_add(conv_output, biases)


def avg_pool3d(name, input_layer, kernel_spatial_size,
               kernel_temporal_size, spatial_stride=1):
    with tf.get_default_graph().name_scope(name):
        return tf.nn.avg_pool3d(input_layer,
                                ksize=[1, kernel_temporal_size,
                                       kernel_spatial_size,
                                       kernel_spatial_size, 1],
                                strides=[1, 1, spatial_stride,
                                         spatial_stride, 1],
                                padding='SAME')


def eltwise_square(name, input_layer):
    with tf.get_default_graph().name_scope(name):
        return tf.square(input_layer)


def l1_normalize(name, input_layer, axis=4, eps=1e-12):
    with tf.get_default_graph().name_scope(name):
        abs_sum = tf.reduce_sum(tf.abs(input_layer), axis, keep_dims=True)
        input_layer_inv_norm = tf.inv(tf.maximum(abs_sum, eps))
        return tf.mul(input_layer, input_layer_inv_norm)


def l2_loss(name, input_layer, target):
    with tf.get_default_graph().name_scope(name):
        return tf.reduce_mean(tf.square(tf.sub(input_layer, target)))


def l1_loss(name, input_layer, target):
    with tf.get_default_graph().name_scope(name):
        return tf.reduce_mean(tf.abs(tf.sub(input_layer, target)))


def reshape(name, input_layer, output_shape):
    with tf.get_default_graph().name_scope(name):
        return tf.reshape(input_layer, output_shape)


def area_resample(name, input_layer, output_shape):
    with tf.get_default_graph().name_scope(name):
        return tf.image.resize_area(input_layer, output_shape)


def area_resample3d(name, input_layer, output_shape, axis=1):
    with tf.get_default_graph().name_scope(name):
        unpacked = tf.unpack(input_layer, axis=axis)
        for i in range(len(unpacked)):
            unpacked[i] = area_resample('area_resample', unpacked[i],
                                        output_shape)
        return tf.pack(unpacked, axis=axis)


def bilinear_resample(name, input_layer, output_shape):
    with tf.get_default_graph().name_scope(name):
        return tf.image.resize_bilinear(input_layer, output_shape)


def bilinear_resample3d(name, input_layer, output_shape, axis=1):
    with tf.get_default_graph().name_scope(name):
        unpacked = tf.unpack(input_layer, axis=axis)
        for i in range(len(unpacked)):
            unpacked[i] = bilinear_resample('bilinear_resample', unpacked[i],
                                            output_shape)
        return tf.pack(unpacked, axis=axis)


def channel_concat3d(name, input_layer, axis=4):
    with tf.get_default_graph().name_scope(name):
        return tf.concat(axis, input_layer)


def flow_to_colour(name, input_layer, norm=True):
    with tf.get_default_graph().name_scope(name):
        return draw_hsv_ocv(input_layer, norm)


def put_kernels_on_grid(name, kernel, grid_Y, grid_X, pad=1, norm=True):
    '''
    Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.

    Args:
      kernel: Tensor of shape [Y, X, NumChannels, NumKernels]

      (grid_Y, grid_X): Shape of the grid.
                        Require: NumKernels == grid_Y * grid_X
                        User is responsible of how to break into two multiples.

      pad: Number of black pixels around each filter (between them)

    Return:
      Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
    '''
    with tf.get_default_graph().name_scope(name):
        x_min = tf.reduce_min(kernel)
        x_max = tf.reduce_max(kernel)

        if norm:
            kernel1 = (kernel - x_min) / (x_max - x_min)
        else:
            kernel1 = 0.5 + tf.clip_by_value(kernel, -1.0, 1.0) / 2.0

        # pad X and Y
        x1 = tf.pad(kernel1, tf.constant([[pad, pad],
                                          [pad, pad],
                                          [0, 0], [0, 0]]), mode='CONSTANT')

        # X and Y dimensions, w.r.t. padding
        Y = kernel1.get_shape()[0] + 2 * pad
        X = kernel1.get_shape()[1] + 2 * pad

        channels = kernel1.get_shape()[2]

        # put NumKernels to the 1st dimension
        x2 = tf.transpose(x1, (3, 0, 1, 2))
        # organize grid on Y axis
        x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, channels]))  # 3

        # switch X and Y axes
        x4 = tf.transpose(x3, (0, 2, 1, 3))
        # organize grid on X axis
        x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, channels]))  # 3

        # back to normal order (not combining with the next step for clarity)
        x6 = tf.transpose(x5, (2, 1, 3, 0))

        # to tf.image_summary order [batch_size, height, width, channels],
        #   where in this case batch_size == 1
        x7 = tf.transpose(x6, (3, 0, 1, 2))

        # scale to [0, 255] and convert to uint8
        return tf.image.convert_image_dtype(x7, dtype=tf.uint8)
