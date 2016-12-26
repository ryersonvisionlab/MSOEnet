import tensorflow as tf
import numpy as np
import cv2
from src.util import readFlowFile
from src.util import draw_hsv
from src.util import draw_hsv_ocv


def rotate(flow, theta):
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    sin = tf.sin(theta)
    cos = tf.cos(theta)
    fxp = tf.sub(tf.mul(fx, cos), tf.mul(fy, sin))
    fyp = tf.add(tf.mul(fx, sin), tf.mul(fy, cos))
    return tf.pack([fxp, fyp], 2)


def discrete_rotate(input_layer, k=1):
    if k == 1:
        # 90
        flipped = tf.image.flip_left_right(input_layer)
        return rotate(tf.transpose(flipped, perm=[1, 0, 2]), (np.pi / 2.0))
    elif k == 2:
        # 180
        flipped = tf.image.flip_left_right(input_layer)
        return rotate(tf.image.flip_up_down(flipped), np.pi)
    elif k == 3:
        # 270
        flipped = tf.image.flip_up_down(input_layer)
        return rotate(tf.transpose(flipped, perm=[1, 0, 2]), (3 * np.pi) / 2.0)


def augment(dataY, axis=0):
    # k = np.random.randint(1, 5)
    k = 3
    if k > 0:
        unpacked = tf.unpack(dataY, axis=axis)
        for i in range(len(unpacked)):
            unpacked[i] = discrete_rotate(unpacked[i], k)
        return tf.pack(unpacked, axis=axis)
    else:
        return dataY


def np_discrete_rotate(input, k, flow=False):
    if flow:
        rad = k * (np.pi / 2.0)
        fx, fy = np.copy(input[:, :, 0]), np.copy(input[:, :, 1])
        sin = np.sin(rad)
        cos = np.cos(rad)
        input[..., 0] = (fx * cos) - (fy * sin)
        input[..., 1] = (fx * sin) + (fy * cos)
    return np.rot90(input, k)


def augment_cpu(dataX, dataY):
    for i in range(dataX.shape[0]):
        k = np.random.randint(0, 4)
        if k > 0:
            for j in range(dataX.shape[1]):
                dataX[i][j] = np_discrete_rotate(dataX[i][j], k)
            dataY[i] = np_discrete_rotate(dataY[i], k, flow=True)
    return dataX, dataY


img1 = '/home/mtesfald/MSOEnet/img3.png'
img2 = '/home/mtesfald/MSOEnet/img4.png'
flo = '/home/mtesfald/MSOEnet/img3.flo'
chunk = [img1, img2]

# config
config_proto = tf.ConfigProto()
config_proto.gpu_options.allow_growth = True
config_proto.allow_soft_placement = True
config_proto.log_device_placement = False

scale_factor = 1.0 / 255.0
png_chunk = np.expand_dims(
                np.stack([cv2.imread(
                          img,
                          cv2.IMREAD_GRAYSCALE).
                          astype(np.float32) * scale_factor
                          for img in chunk]), axis=3)
gt_flo = readFlowFile(flo)

# to emulate batches
png_chunk = np.expand_dims(png_chunk, axis=0)
gt_flo = np.expand_dims(gt_flo, axis=0)

# augment data (CPU)
png_chunk_a, gt_flo_a = augment_cpu(np.copy(png_chunk), np.copy(gt_flo))
# print gt_flo_a[0][:, :, 0]
# print gt_flo_a[0][:, :, 1]

cv2.imshow('image', png_chunk_a[0][0])
cv2.waitKey()
cv2.imshow('image', png_chunk_a[0][1])
cv2.waitKey()
cv2.imshow('image', draw_hsv(gt_flo_a[0]))
cv2.waitKey()

with tf.Session(config=config_proto) as sess:
    dataX = tf.placeholder(dtype=tf.float32, shape=[1, 2, 256, 256, 1])
    dataY = tf.placeholder(dtype=tf.float32, shape=[1, 256, 256, 2])

    augmented = augment(dataY)
    tf.image_summary('flow',
                     draw_hsv_ocv(augmented),
                     max_images=1)

    merged = tf.merge_all_summaries()

    summary_writer = tf.train.SummaryWriter('augment_test')

    res = sess.run([merged, augmented, draw_hsv_ocv(augmented)], feed_dict={dataY: gt_flo})
    summary_writer.add_summary(res[0])
    summary_writer.flush()

    hsv_ocv = draw_hsv(res[1][0])
    hsv_mine = res[2][0][..., [2, 1, 0]]

    # print res[1][0][:, :, 0]
    # print res[1][0][:, :, 1]

    print (res[1][0][:, :, 0] - gt_flo_a[0][:, :, 0]).sum()
    print (res[1][0][:, :, 1] - gt_flo_a[0][:, :, 1]).sum()

    cv2.imshow('image', hsv_ocv)
    cv2.waitKey()
    cv2.imshow('image', hsv_mine)
    cv2.waitKey()
