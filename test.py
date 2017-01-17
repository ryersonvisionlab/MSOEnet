import tensorflow as tf
import cv2
import numpy as np
from src.architecture import MSOEPyramid

# config
config_proto = tf.ConfigProto()
config_proto.gpu_options.allow_growth = True
config_proto.allow_soft_placement = True
config_proto.log_device_placement = True
my_config = {}
my_config['train'] = False
my_config['num_scales'] = 4

with tf.device('/gpu:1'):
    scale_factor = 1.0 / 255.0
    input1 = cv2.imread('frame1.png',
                        cv2.IMREAD_GRAYSCALE).astype(np.float32) * scale_factor
    input2 = cv2.imread('frame2.png',
                        cv2.IMREAD_GRAYSCALE).astype(np.float32) * scale_factor
    input = np.expand_dims(np.expand_dims(np.stack([input1, input2]), axis=3), axis=0)

    net = MSOEPyramid(config={'tf': config_proto,
                              'user': my_config},
                      input=input)
    result = net.run_test()
