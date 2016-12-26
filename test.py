from src.dataset import read_data_sets
from src.util import draw_hsv
from src.util import draw_hsv_ocv
from src.util import readFlowFile
import cv2
import tensorflow as tf
import math
import numpy as np


def flow_to_colour(flow):
    # constants
    flow_shape = flow.get_shape()

    oor2 = 1 / math.sqrt(2)  # use tf math functions instead?
    sqrt3h = math.sqrt(3) / 2
    k0 = tf.transpose(tf.constant([[[[1, 0]]]], dtype=tf.float32),
                      perm=[0, 1, 3, 2])
    k120 = tf.transpose(tf.constant([[[[-oor2, oor2]]]], dtype=tf.float32),
                        perm=[0, 1, 3, 2])
    k240 = tf.transpose(tf.constant([[[[-oor2, -oor2]]]],
                        dtype=tf.float32),
                        perm=[0, 1, 3, 2])
    k60 = tf.transpose(tf.constant([[[[sqrt3h, -1./2.]]]],
                    dtype=tf.float32),
                    perm=[0, 1, 3, 2])
    k180 = tf.transpose(tf.constant([[[[-1, 0]]]], dtype=tf.float32),
                     perm=[0, 1, 3, 2])
    k300 = tf.transpose(tf.constant([[[[sqrt3h, 1./2.]]]],
                     dtype=tf.float32),
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

    # find max flow and scale
    flow = flow + 0.0000000001
    flow_sq = flow * flow
    flow_mag = tf.sqrt(tf.reduce_sum(flow_sq,
                                  reduction_indices=[3],
                                  keep_dims=True))
    max_mag = tf.reduce_max(flow_mag,
                         reduction_indices=[1, 2],
                         keep_dims=True)
    scaled_flow = flow / max_mag

    # calculate coefficients
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

    # combine color components
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


dataset = read_data_sets('/home/mtesfald/UCF-101-gt', 2)
data, gt = dataset.next_batch(10)

# img1 = '/home/mtesfald/MSOEnet/img3.png'
# img2 = '/home/mtesfald/MSOEnet/img4.png'
# flo = '/home/mtesfald/MSOEnet/img3.flo'
# chunk = [img1, img2]
# scale_factor = 1.0 / 255.0
# png_chunk = np.expand_dims(
#                 np.stack([cv2.imread(
#                           img,
#                           cv2.IMREAD_GRAYSCALE).
#                           astype(np.float32) * scale_factor
#                           for img in chunk]), axis=3)
# gt_flo = readFlowFile(flo)
# gt_flo[:, :, 1] *= -1
# # to emulate batches
# data = np.expand_dims(png_chunk, axis=0)
# gt = np.expand_dims(gt_flo, axis=0)

# config
config_proto = tf.ConfigProto()
config_proto.gpu_options.allow_growth = True
config_proto.allow_soft_placement = True
config_proto.log_device_placement = False

with tf.Session(config=config_proto) as sess:
    dataY = tf.placeholder(dtype=tf.float32, shape=[1, 256, 256, 2])
    tf.image_summary('flow',
                     draw_hsv_ocv(dataY),
                     max_images=1)
    merged = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter('augment_test')

    for i in range(data.shape[0]):
        for j in range(data[i].shape[0]):
            cv2.imshow('image', data[i][j])
            cv2.waitKey()
        cv2.imshow('flow opencv', draw_hsv(gt[i]))
        cv2.waitKey()
        res = sess.run([flow_to_colour(dataY), draw_hsv_ocv(dataY), merged], feed_dict={dataY: np.expand_dims(gt[i], axis=0)})
        cv2.imshow('flow jason', res[0][0][..., [2, 1, 0]])
        cv2.waitKey()
        cv2.imshow('flow mine', res[1][0][..., [2, 1, 0]])
        cv2.waitKey()
        summary_writer.add_summary(res[2])
        summary_writer.flush()
