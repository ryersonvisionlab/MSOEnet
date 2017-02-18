import tensorflow as tf
import cv2
import numpy as np
from src.architecture import MSOEPyramid
from src.util import draw_hsv
from subprocess import call

# config
config_proto = tf.ConfigProto()
config_proto.gpu_options.allow_growth = True
config_proto.allow_soft_placement = True
config_proto.log_device_placement = False
my_config = {}
my_config['train'] = False
my_config['num_scales'] = 4

with tf.device('/gpu:1'):
    scale_factor = 1.0 / 255.0

    for i in range(1, 40):
        input1 = cv2.imread('test_images/water_2/frame_%d.jpeg' % (i),
                            cv2.IMREAD_GRAYSCALE).astype(np.float32) * \
            scale_factor
        input2 = cv2.imread('test_images/water_2/frame_%d.jpeg' % (i+1),
                            cv2.IMREAD_GRAYSCALE).astype(np.float32) * \
            scale_factor
        stacked = np.expand_dims(
                      np.expand_dims(np.stack([input1, input2]), axis=3),
                      axis=0)
        if i != 1:
            input = np.concatenate((input, stacked), axis=0)
        else:
            input = stacked

    print input.shape

    net = MSOEPyramid(config={'tf': config_proto,
                              'user': my_config}, input=input)
    result = net.run_test()

    for i in range(0, 39):
        cv2.imwrite('test_images/water_2/img_' + str(i) + '.jpeg',
                    draw_hsv(result[i]))

    call('convert -delay 10 -loop 0 -alpha set -dispose previous '
         '`ls -v test_images/water_2/img_*.jpeg` '
         'test_images/water_2/img.gif',
         shell=True)
