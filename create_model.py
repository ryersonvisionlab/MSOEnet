import tensorflow as tf
import cv2
import numpy as np
from src.architecture import MSOEPyramid

# config
config_proto = tf.ConfigProto()
config_proto.gpu_options.allow_growth = True
config_proto.allow_soft_placement = True
config_proto.log_device_placement = False
my_config = {}
my_config['train'] = False
my_config['num_scales'] = 4

with tf.device('/gpu:1'):
    net = MSOEPyramid(config={'tf': config_proto,
                              'user': my_config})
    result = net.save_model()
