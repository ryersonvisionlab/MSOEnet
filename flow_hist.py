import os
import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from src.dataset import read_data_sets
import cv2
import time
import itertools


def draw_hist(flow_dirs):
    N = 50

    ax = plt.subplot(111, polar=True)
    n, bins, patches = plt.hist(flow_dirs, N, facecolor='green', alpha=0.75)

    ax.yaxis.set_major_formatter(FuncFormatter(adjust_y_axis(flow_dirs)))

    # Use custom colors and opacity
    cm = plt.cm.get_cmap('hsv')
    for b, p in zip(bins, patches):
        plt.setp(p, 'facecolor', cm(b / (2*np.pi)))
        plt.setp(p, linewidth=0)

    plt.savefig('hist_aug.png', bbox_inches='tight')
    plt.close()


def adjust_y_axis(mydata):
    return lambda x, pos: round(x / (len(mydata) * 1.0), 2)


def main(argv=None):
    flow_dirs = []
    dataset = read_data_sets('/home/mtesfald/UCF-101-gt/', 2)
    data, gt = dataset.validation_data()

    pair_count = 1
    for flow in gt:
        fx, fy = flow[:, :, 0], flow[:, :, 1]
        v, ang = cv2.cartToPolar(fx, fy)
        ang = ang[np.where(v > 0.1)]
        tic = time.time()
        flow_dirs.append(np.ravel(ang).tolist())
        toc = time.time()
        print 'frame ' + str(pair_count) + ' time: ' + str(toc - tic)
        pair_count += 1

    flow_dirs = list(itertools.chain.from_iterable(flow_dirs))
    draw_hist(flow_dirs)


if __name__ == "__main__":
    sys.exit(main())
