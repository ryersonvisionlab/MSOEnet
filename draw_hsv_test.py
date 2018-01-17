import tensorflow as tf
from src.Dataset import *
import cv2
from src.utilities import draw_hsv
# from imgaug import augmenters as iaa
# from imgaug import parameters as iap
# import time
# import matplotlib
# matplotlib.use('tkagg')
# import matplotlib.pyplot as plt

config_proto = tf.ConfigProto()
config_proto.gpu_options.allow_growth = True
config_proto.allow_soft_placement = True
config_proto.log_device_placement = False
sess = tf.InteractiveSession(config=config_proto)


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow[:, :, 1] *= -1  # since I flipped it in next_batch
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


# def augment_batch(images, flows, aug):
#     ad = aug.to_deterministic()
#     for i in xrange(images.shape[0]):
#         for j in xrange(images[i].shape[0]):  # augment frame pairs equally
#             images[i][j] = ad.augment_image(images[i][j] * 255.0,
#                                             extra=flows[i]) / 255.0
#         ad.reseed(deterministic_too=True)  # reseed for next pair in batch

with tf.device('/gpu:1'):
    d = load_FlyingChairs('/local/ssd/mtesfald/FlyingChairs/data')
    imgs_old, gts_old, masks_old = d.next_batch(5, augment_batch=False)
    imgs, gts, masks = d.augment_batch((imgs_old, gts_old))

# sum1 = 0
# for i in range(10):
#     tic = time.clock()
#     augment_batch(imgs, gts, seq)
#     toc = time.clock()
#     sum1 += toc - tic
# print sum1 / 10.0

for i in range(gts.shape[0]):
    # plt.imshow(imgs[i][0][:, :, 0], cmap='gray')
    # plt.show()
    cv2.imshow('im1', imgs[i][0][..., ::-1])
    cv2.imshow('im2', imgs[i][1][..., ::-1])
    cv2.imshow('im3', draw_hsv(gts[i]))
    # cv2.imshow('im4', masks[i][..., ::-1] * 255.0)
    # cv2.imshow('im5', imgs_old[i][0][..., ::-1])
    # cv2.imshow('im6', imgs_old[i][1][..., ::-1])
    # cv2.imshow('im7', draw_hsv(gts_old[i]))
    # cv2.imshow('im8', masks_old[i][..., ::-1] * 255.0)
    # warped = \
    #     warp_flow(imgs_old[i][0][..., ::-1], gts_old[i]) * masks_old[i][..., 0]
    # warped_aug = warp_flow(imgs[i][0][..., ::-1], gts[i]) * masks[i][..., 0]
    # cv2.imshow('warped', warped)
    # cv2.imshow('warped_aug', warped_aug)
    # print str(i) + '. rmse', str(np.square(warped - imgs_old[i][1][..., ::-1][..., 0]).mean())
    # print str(i) + '. rmse warped', str(np.square(warped_aug - imgs[i][1][..., ::-1][..., 0]).mean())
    cv2.waitKey()
