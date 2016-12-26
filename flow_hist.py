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
import operator
import gc


def draw_hist(flow_dirs, i):
    N = 50

    ax = plt.subplot(111, polar=True)
    n, bins, patches = plt.hist(flow_dirs, N, facecolor='green', alpha=0.75)

    ax.yaxis.set_major_formatter(FuncFormatter(adjust_y_axis(flow_dirs)))

    # Use custom colors and opacity
    cm = plt.cm.get_cmap('hsv')
    for b, p in zip(bins, patches):
        plt.setp(p, 'facecolor', cm(b / (2*np.pi)))

    plt.savefig('pics/hist' + str(i) + '.png', bbox_inches='tight')
    plt.close()


def adjust_y_axis(mydata):
    return lambda x, pos: round(x / (len(mydata) * 1.0), 2)


def main(argv=None):
    flow_dirs = []
    data = read_data_sets('/home/mtesfald/UCF-101-gt/', 2)
    sequences = data._validation
    count = 0
    hist_count = 1
    for sequence in sequences:
        limit = 10
        prefix = sequence['prefix'] + '/'
        fc = 0
        for frame in sequence['frames']:
            if fc == limit:
                break
            gt_flo = readFlowFile(prefix + frame[0])
            fx, fy = gt_flo[:, :, 0], gt_flo[:, :, 1] * -1
            v, ang = cv2.cartToPolar(fx, fy)
            ang = ang[np.where(v > 0.1)]
            print str(count) + ': frame' + str(fc)
            tic = time.time()
            flow_dirs.append(np.ravel(ang).tolist())
            toc = time.time()
            print str(count) + ': append' + str(fc) + ' time: ' + str(toc - tic)
            fc += 1
        count += 1
        # if count % 10 == 0:
        #     print 'count: ' + str(count)
        #     flow_dirs = list(itertools.chain.from_iterable(flow_dirs))
        #     print 'number of flows: ' + str(len(flow_dirs))
        #     draw_hist(flow_dirs, hist_count)
        #     hist_count += 1
        #     del flow_dirs
        #     gc.collect()
        #     flow_dirs = []
    flow_dirs = list(itertools.chain.from_iterable(flow_dirs))
    draw_hist(flow_dirs, 1)


if __name__ == "__main__":
    sys.exit(main())
