import tensorflow as tf
from src.dataset import read_data_sets
from src.dataset import QueueRunner
import math


def data_layer(name, path, batch_size, temporal_extent, num_threads):
    with tf.get_default_graph().name_scope(name):
        # read UCF-101 image sequences with ground truth flows
        ucf101 = read_data_sets(path, temporal_extent)

        # read validation data
        x_val, y_val_ = ucf101.validation_data()

        input_shape = x_val.shape[1:]
        target_shape = y_val_.shape[1:]

        with tf.device("/cpu:0"):
            queue_runner = QueueRunner(ucf101, input_shape, target_shape,
                                       batch_size, num_threads)
            x, y_ = queue_runner.get_inputs()

        return x, y_, tf.pack(x_val), tf.pack(y_val_), queue_runner


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
                                      initializer=tf.truncated_normal_initializer())
            biases = tf.get_variable('biases',
                                     [out_channels], 
                                    initializer=tf.constant_initializer(0.0))

            # spatially pad the image, but not temporally
            input_layer = tf.pad(input_layer,
                                 [[0, 0], [0, 0],
                                  [kernel_spatial_size/2, kernel_spatial_size/2],
                                  [kernel_spatial_size/2, kernel_spatial_size/2],
                                  [0, 0]], 'CONSTANT')

            conv_output = tf.nn.conv3d(input_layer, weights,
                                       strides=[1, 1, 1, 1, 1],
                                       padding='VALID')

        return tf.nn.bias_add(conv_output, biases)


def avg_pool3d(name, input_layer, kernel_spatial_size, kernel_temporal_size):
    with tf.get_default_graph().name_scope(name):
        return tf.nn.avg_pool3d(input_layer,
                                ksize=[1, kernel_temporal_size,
                                       kernel_spatial_size,
                                       kernel_spatial_size, 1],
                                strides=[1, 1, 1, 1, 1], padding='SAME')


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


def area_resample_volume(name, input_layer, output_shape, axis=1):
    with tf.get_default_graph().name_scope(name):
        unpacked = tf.unpack(input_layer, axis=axis)
        for i in range(len(unpacked)):
            unpacked[i] = area_resample('area_resample', unpacked[i],
                                        output_shape)
        return tf.pack(unpacked, axis=axis)


def bilinear_resample(name, input_layer, output_shape):
    with tf.get_default_graph().name_scope(name):
        return tf.image.resize_bilinear(input_layer, output_shape)


def bilinear_resample_volume(name, input_layer, output_shape, axis=1):
    with tf.get_default_graph().name_scope(name):
        unpacked = tf.unpack(input_layer, axis=axis)
        for i in range(len(unpacked)):
            unpacked[i] = bilinear_resample('bilinear_resample', unpacked[i],
                                            output_shape)
        return tf.pack(unpacked, axis=axis)


def flow_to_colour(name, flow):
    with tf.get_default_graph().name_scope(name):
        #constants
        flow_shape = flow.get_shape()

        oor2 = 1 / math.sqrt(2)  # use tf math functions instead?
        sqrt3h = math.sqrt(3) / 2
        k0 = tf.transpose(tf.constant([[[[1, 0]]]], dtype=tf.float32),
                          perm=[0, 1, 3, 2])
        k120 = tf.transpose(tf.constant([[[[-oor2, oor2]]]], dtype=tf.float32),
                            perm=[0, 1, 3, 2])
        k240 = tf.transpose(tf.constant([[[[-oor2, -oor2]]]], dtype=tf.float32),
                            perm=[0, 1, 3, 2])
        k60 = tf.transpose(tf.constant([[[[sqrt3h, -1./2.]]]], dtype=tf.float32),
                           perm=[0, 1, 3, 2])
        k180 = tf.transpose(tf.constant([[[[-1, 0]]]], dtype=tf.float32),
                            perm=[0, 1, 3, 2])
        k300 = tf.transpose(tf.constant([[[[sqrt3h, 1./2.]]]], dtype=tf.float32),
                            perm=[0, 1, 3, 2])

        k0c = tf.transpose(tf.constant([[[[1, 0, 0]]]], dtype=tf.float32),
                           perm=[0, 1, 3, 2])
        k120c = tf.transpose(tf.constant([[[[0, 1, 0]]]], dtype=tf.float32),
                           perm=[0, 1, 3, 2])
        k240c = tf.transpose(tf.constant([[[[0, 0, 1]]]], dtype=tf.float32),
                             perm=[0, 1, 3, 2])
        k60c = tf.transpose(tf.constant([[[[1, 1, 0]]]], dtype=tf.float32),
                            perm=[0, 1, 3, 2])
        k180c = tf.transpose(tf.constant([[[[0, 1, 1]]]], dtype=tf.float32),
                             perm=[0, 1, 3, 2])
        k300c = tf.transpose(tf.constant([[[[1, 0, 1]]]], dtype=tf.float32),
                             perm=[0, 1, 3, 2])

        #find max flow and scale
        flow = flow + 0.0000000001
        flow_sq = flow * flow
        flow_mag = tf.sqrt(tf.reduce_sum(flow_sq,
                                         reduction_indices=[3],
                                         keep_dims=True))
        max_mag = tf.reduce_max(flow_mag,
                                reduction_indices=[1, 2],
                                keep_dims=True)
        scaled_flow = flow / max_mag

        #calculate coefficients
        coef0 = tf.maximum(tf.nn.conv2d(scaled_flow, k0, [1, 1, 1, 1],
                                        padding='SAME'), 0) / 2
        coef120 = tf.maximum(tf.nn.conv2d(scaled_flow, k120, [1, 1, 1, 1],
                                          padding='SAME'), 0) / 2
        coef240 = tf.maximum(tf.nn.conv2d(scaled_flow, k240, [1, 1, 1, 1],
                                          padding='SAME'), 0) / 2
        coef60 = tf.maximum(tf.nn.conv2d(scaled_flow, k60, [1, 1, 1, 1],
                                         padding='SAME'), 0) / 2
        coef180 = tf.maximum(tf.nn.conv2d(scaled_flow, k180, [1, 1, 1, 1],
                                          padding='SAME'), 0) / 2
        coef300 = tf.maximum(tf.nn.conv2d(scaled_flow, k300, [1, 1, 1, 1],
                                          padding='SAME'), 0) / 2

        #combine color components
        comp0 = tf.nn.conv2d_transpose(coef0, k0c,
                                       tf.pack([flow_shape[0],
                                                flow_shape[1],
                                                flow_shape[2], 3]),
                                       [1, 1, 1, 1], padding='SAME')
        comp120 = tf.nn.conv2d_transpose(coef120, k120c,
                                         tf.pack([flow_shape[0],
                                                  flow_shape[1],
                                                  flow_shape[2], 3]),
                                         [1, 1, 1, 1], padding='SAME')
        comp240 = tf.nn.conv2d_transpose(coef240, k240c,
                                         tf.pack([flow_shape[0],
                                                  flow_shape[1],
                                                  flow_shape[2], 3]),
                                         [1, 1, 1, 1], padding='SAME')
        comp60 = tf.nn.conv2d_transpose(coef60, k60c,
                                        tf.pack([flow_shape[0],
                                                 flow_shape[1],
                                                 flow_shape[2], 3]),
                                        [1, 1, 1, 1], padding='SAME')
        comp180 = tf.nn.conv2d_transpose(coef180, k180c,
                                         tf.pack([flow_shape[0],
                                                  flow_shape[1],
                                                  flow_shape[2], 3]),
                                         [1, 1, 1, 1], padding='SAME')
        comp300 = tf.nn.conv2d_transpose(coef300, k300c,
                                         tf.pack([flow_shape[0],
                                                  flow_shape[1],
                                                  flow_shape[2], 3]),
                                         [1, 1, 1, 1], padding='SAME')

        return comp0 + comp120 + comp240 + comp60 + comp180 + comp300


def put_kernels_on_grid(name, kernel, grid_Y, grid_X, pad=1):
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

        kernel1 = (kernel - x_min) / (x_max - x_min)

        # pad X and Y
        x1 = tf.pad(kernel1, tf.constant([[pad, pad],
                                          [pad, pad],
                                          [0, 0],[0,0]]), mode='CONSTANT')

        # X and Y dimensions, w.r.t. padding
        Y = kernel1.get_shape()[0] + 2 * pad
        X = kernel1.get_shape()[1] + 2 * pad

        channels = kernel1.get_shape()[2]

        # put NumKernels to the 1st dimension
        x2 = tf.transpose(x1, (3, 0, 1, 2))
        # organize grid on Y axis
        x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, channels])) #3

        # switch X and Y axes
        x4 = tf.transpose(x3, (0, 2, 1, 3))
        # organize grid on X axis
        x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, channels])) #3

        # back to normal order (not combining with the next step for clarity)
        x6 = tf.transpose(x5, (2, 1, 3, 0))

        # to tf.image_summary order [batch_size, height, width, channels],
        #   where in this case batch_size == 1
        x7 = tf.transpose(x6, (3, 0, 1, 2))

        # scale to [0, 255] and convert to uint8
        return tf.image.convert_image_dtype(x7, dtype=tf.uint8) 