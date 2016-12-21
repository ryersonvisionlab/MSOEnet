import os
import numpy as np
import cv2
import tensorflow as tf


def readFlowFile(filename):
    """
    readFlowFile read a flow file FILENAME into 2-band image IMG

    According to the c++ source code of Daniel Scharstein
    Contact: schar@middlebury.edu

    Author: Deqing Sun, Department of Computer Science, Brown University
    Contact: dqsun@cs.brown.edu
    Date: 2007-10-31 16:45:40 (Wed, 31 Oct 2006)

    Copyright 2007, Deqing Sun.

                        All Rights Reserved

    Permission to use, copy, modify, and distribute this software and its
    documentation for any purpose other than its incorporation into a
    commercial product is hereby granted without fee, provided that the
    above copyright notice appear in all copies and that both that
    copyright notice and this permission notice appear in supporting
    documentation, and that the name of the author and Brown University not be
    used in advertising or publicity pertaining to distribution of the software
    without specific, written prior permission.

    THE AUTHOR AND BROWN UNIVERSITY DISCLAIM ALL WARRANTIES WITH REGARD TO THIS
    SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
    FOR ANY PARTICULAR PURPOSE.  IN NO EVENT SHALL THE AUTHOR OR BROWN
    UNIVERSITY BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR
    ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER
    IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
    OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
    """

    TAG_FLOAT = 202021.25  # check for this when READING the file

    # sanity check
    if len(filename) == 0:
        raise ValueError('readFlowFile: empty filename')

    try:
        idx = filename.index('.')
    except ValueError:
        raise ValueError('readFlowFile: extension required in filename %s' %
                         (filename))

    if filename[idx:] != '.flo':
        raise ValueError('readFlowFile: filename %s should have extension '
                         '\'.flo\'' % (filename))

    with open(filename, 'rb') as fid:
        tag = np.fromfile(fid, count=1, dtype=np.float32).item()
        width, height = np.fromfile(fid, count=2, dtype=np.int32)

        # sanity check
        if tag != TAG_FLOAT:
            raise ValueError('readFlowFile(%s): wrong tag (possibly due to'
                             ' big-endian machine?)' % (filename))

        if width < 1 or width > 99999:
            raise ValueError('readFlowFile(%s): illegal width %d' % (filename,
                                                                     width))

        if height < 1 or height > 99999:
            raise ValueError('readFlowFile(%s): illegal height %d' % (filename,
                                                                      height))

        nBands = 2

        # arrange into matrix form
        tmp = np.fromfile(fid, count=nBands*width*height, dtype=np.float32)
        img = tmp.reshape((height, width, nBands))

        return img


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    v, ang = cv2.cartToPolar(fx, fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang*(180/np.pi/2)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr


def draw_flow(img, flow, step=16):
    img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T*5
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)

    return vis


def warp_flow(img, flow):
    img = cv2.imread(img)
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_CUBIC)

    return res


def check_snapshots(root_folder='', train=True):
    snapshots_folder = root_folder + 'snapshots/'
    logs_folder = root_folder + 'logs/'

    checkpoint = tf.train.latest_checkpoint(snapshots_folder)

    if train:
        resume = False
        start_iteration = 0

        if checkpoint:
            choice = ''
            while choice != 'y' and choice != 'n':
                print 'Snapshot file detected (' + checkpoint + \
                      ') would you like to resume? (y/n)'
                choice = raw_input().lower()

                if choice == 'y':
                    resume = checkpoint
                    start_iteration = int(checkpoint.split(snapshots_folder)
                                          [1][5:-5])
                    print 'resuming from iteration ' + str(start_iteration)
                else:
                    print 'removing old snapshots and logs, training from' \
                          ' scratch'
                    resume = False
                    for file in os.listdir(snapshots_folder):
                        os.remove(snapshots_folder + file)
                    for file in os.listdir(logs_folder + 'train/'):
                        os.remove(logs_folder + 'train/' + file)
                    for file in os.listdir(logs_folder + 'val/'):
                        os.remove(logs_folder + 'val/' + file)
        else:
            print "No snapshots found, training from scratch"

        return resume, start_iteration
    else:
        return checkpoint
