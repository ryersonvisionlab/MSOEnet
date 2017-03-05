import tensorflow as tf
from dataset import *
import cv2
from util import draw_hsv

config_proto = tf.ConfigProto()
config_proto.gpu_options.allow_growth = True
config_proto.allow_soft_placement = True
config_proto.log_device_placement = False
sess = tf.InteractiveSession(config=config_proto)

with tf.device('/gpu:1'):
    d = load_FlyingChairs('/home/mtesfald/FlyingChairs/FlyingChairs_release/data')
    imgs, gts = d.next_batch(10)

for i in range(gts.get_shape().as_list()[0]):
    cv2.imshow('im', imgs[i][0].eval()[..., ::-1]); cv2.waitKey()
    cv2.imshow('im', imgs[i][1].eval()[..., ::-1]); cv2.waitKey()
    cv2.imshow('im', draw_hsv(gts[i].eval())); cv2.waitKey()
