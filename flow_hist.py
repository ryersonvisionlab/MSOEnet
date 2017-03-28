import os
import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from src.Dataset import load_FlyingChairs
import cv2
import time
import itertools


def draw_hist(flow_dirs, flow_mags, batch_size):
    N_dirs = 50
    N_mags = 100
    flow_mags_min = np.min(flow_mags)
    flow_mags_max = 60

    # breaking it up into chunks to deal with OOM issues
    total_dir_hist = np.zeros(N_dirs)
    total_mag_hist = np.zeros(N_mags)
    num_chunks = len(flow_dirs) / batch_size  # same if used flow_mags
    for i in range(num_chunks):
        print 'Processing chunk ' + str(i+1)
        start = i*batch_size
        end = (i+1)*batch_size

        # direction
        dir_hist, dir_bin_edges = np.histogram(flow_dirs[start:end], bins=N_dirs, range=(0.0, 2*np.pi))
        total_dir_hist += dir_hist

        # magnitude
        mag_hist, mag_bin_edges = np.histogram(flow_mags[start:end], bins=N_mags, range=(flow_mags_min, flow_mags_max))
        total_mag_hist += mag_hist

    # gather rest (if any)
    if len(flow_dirs) % batch_size != 0:
        print 'Processing chunk rest'
        start = num_chunks*batch_size

        # direction
        dir_hist, dir_bin_edges = np.histogram(flow_dirs[start:], bins=N_dirs, range=(0.0, 2*np.pi))
        total_dir_hist += dir_hist

        # magnitude
        mag_hist, mag_bin_edges = np.histogram(flow_mags[start:], bins=N_mags, range=(flow_mags_min, flow_mags_max))
        total_mag_hist += mag_hist
    
    # Radial histogram of flow directions
    print 'Processing radial histogram of flow directions...'
    unity_hist = total_dir_hist / total_dir_hist.sum()
    widths = dir_bin_edges[:-1] - dir_bin_edges[1:]
    ax = plt.subplot(111, polar=True)
    cm = plt.cm.get_cmap('hsv')
    bars = plt.bar(dir_bin_edges[:-1], unity_hist, width=widths)
    for bar in bars:
        color = cm(bar._x/(2*np.pi))
        bar.set_facecolor(color)
    plt.savefig('flow_dir_hist.png', bbox_inches='tight')
    plt.close()

    # Histogram of flow magnitudes
    print 'Processing histogram of flow magnitudes...'
    unity_hist = total_mag_hist / total_mag_hist.sum()
    widths = mag_bin_edges[:-1] - mag_bin_edges[1:]
    plt.bar(mag_bin_edges[:-1], unity_hist, width=widths)
    plt.savefig('flow_mag_hist.png', bbox_inches='tight')
    plt.close()

def main(argv=None):
    flow_dirs = []
    flow_mags = []
    dataset = load_FlyingChairs('/home/mtesfald/Datasets/FlyingChairs/FlyingChairs_release/data')
    images, flows = dataset.validation_data()

    pair_count = 1
    for flow in flows:
        fx, fy = flow[:, :, 0], flow[:, :, 1]
        v, ang = cv2.cartToPolar(fx, fy)
        ang = ang[np.where(v > 0.1)]
        v = v[v > 0.1]
        tic = time.time()
        flow_dirs.append(np.ravel(ang).tolist())
        flow_mags.append(np.ravel(v).tolist())
        toc = time.time()
        print 'frame ' + str(pair_count) + ' time: ' + str(toc - tic)
        pair_count += 1

    flow_dirs = list(itertools.chain.from_iterable(flow_dirs))
    flow_mags = list(itertools.chain.from_iterable(flow_mags))
    print 'flows shape:'
    print flows.shape
    print str(len(flow_dirs)) + ' flow directions...'
    print str(len(flow_mags)) + ' flow magnitudes...'
    time.sleep(3)
    draw_hist(flow_dirs, flow_mags, 10000)


if __name__ == "__main__":
    sys.exit(main())
