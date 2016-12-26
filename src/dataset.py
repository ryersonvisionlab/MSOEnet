import os
import random
import numpy as np
import cv2
import tensorflow as tf
import threading
from util import readFlowFile


class DataSet(object):

    def __init__(self, train, validation, temporal_extent):
        count = 0
        for sequence in train:
            count += len(sequence['frames'])
        # inevitable rounding
        self._num_examples = count / temporal_extent
        self._train = train
        self._validation = validation
        self._temporal_extent = temporal_extent
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def training_data(self):
        return self._train

    def validation_data(self):
        # Pick all validation sequences
        indices = np.arange(len(self._validation))
        sequences = [self._validation[i] for i in indices]
        # Pick limit chunks (of length temporal_extent) of frames
        # from each sequence (1000 data points)
        packaged_data = []
        packaged_gts = []
        for sequence in sequences:
            limit = 10
            for i in range(len(sequence['frames']) -
                           self._temporal_extent + 1):
                if i == limit:
                    break
                chunk = sequence['frames'][i:i+self._temporal_extent]
                # Read images in sequence and stack them into a single volume
                prefix = sequence['prefix'] + '/'
                scale_factor = 1.0 / 255.0
                png_chunk = np.expand_dims(
                                np.stack([cv2.imread(
                                          prefix + frame[1],
                                          cv2.IMREAD_GRAYSCALE).
                                          astype(np.float32) * scale_factor
                                          for frame in chunk]), axis=3)
                # Pick flow for middle frame
                gt_flo = prefix + chunk[int(np.ceil(len(chunk)/2.0) - 1)][0]
                # Read flo file
                gt_flo = readFlowFile(gt_flo)
                gt_flo[:, :, 1] *= -1  # fy is in opposite direction, must flip
                # package chunks of images as data points
                # and provide the ground truth flow for the middle frame
                packaged_data.append(png_chunk)
                packaged_gts.append(gt_flo)

        # wrap in numpy array (easier to work with)
        packaged_data = np.array(packaged_data)
        packaged_gts = np.array(packaged_gts)

        return packaged_data, packaged_gts

    @property
    def temporal_extent(self):
        return self._temporal_extent

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """
        1. Pick batch_size amount of sequences at random.
        2. For each sequence, pick a random consecutive chunk of frames of
        length temporal_extent.
        3. Package the (full) image paths in each chunk as a data point,
        e.g. [path1, path2, path3, path4], [path1, path2, path3, path4], ...
        4. For each data point, provide the ground truth flow for the middle
        frame.
        5. Resolve all paths to their files.
        5. Return the batch of data points and labels.
        """
        # Pick a random batch of sequences
        indices = np.random.choice(len(self._train), batch_size)
        sequences = [self._train[i] for i in indices]
        # Pick one consecutive chunk (length temporal_extent) of frames from
        # each sequence at random
        packaged_data = []
        packaged_gts = []
        for sequence in sequences:
            i = random.randint(0, len(sequence['frames']) -
                               self._temporal_extent)
            chunk = sequence['frames'][i:i+self._temporal_extent]
            # Read images in sequence and stack them into a single volume
            prefix = sequence['prefix'] + '/'
            scale_factor = 1.0 / 255.0
            png_chunk = np.expand_dims(
                            np.stack([cv2.imread(
                                      prefix + frame[1],
                                      cv2.IMREAD_GRAYSCALE).
                                      astype(np.float32) * scale_factor
                                      for frame in chunk]), axis=3)
            # pick flow for middle frame
            gt_flo = prefix + chunk[int(np.ceil(len(chunk)/2.0) - 1)][0]
            # Read flo file
            gt_flo = readFlowFile(gt_flo)
            gt_flo[:, :, 1] *= -1  # fy is in opposite direction, must flip
            # package chunks of images as data points
            # and provide the ground truth flow for the middle frame
            packaged_data.append(png_chunk)
            packaged_gts.append(gt_flo)
        # update epoch count
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            # start next epoch
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

        # wrap in numpy array (easier to work with)
        packaged_data = np.array(packaged_data)
        packaged_gts = np.array(packaged_gts)

        return augment(packaged_data, packaged_gts)


def discrete_rotate(input, k, flow=False):
    if flow:
        rad = k * (np.pi / 2.0)
        fx, fy = np.copy(input[:, :, 0]), np.copy(input[:, :, 1])
        sin = np.sin(rad)
        cos = np.cos(rad)
        input[..., 0] = (fx * cos) - (fy * sin)
        input[..., 1] = (fx * sin) + (fy * cos)
    return np.rot90(input, k)


def augment(dataX, dataY, k=None):
    for i in range(dataX.shape[0]):
        if k is None:
            k = np.random.randint(0, 4)
        if k > 0:
            for j in range(dataX.shape[1]):
                dataX[i][j] = discrete_rotate(dataX[i][j], k)
            dataY[i] = discrete_rotate(dataY[i], k, flow=True)
    return dataX, dataY


def read_data_sets(train_dir,
                   temporal_extent=5):

    train_images = []
    validation_images = []

    got_png = False
    got_flo = False

    train_count = 0
    validation_count = 0

    print 'Creating dataset filename structure...'
    for group in get_immediate_subdirectories(train_dir):
        i = 0
        for sequence_path in get_immediate_subdirectories(group):
            sequence_name = os.path.split(sequence_path)[1]
            if i != 0:
                train_images.append({
                    'prefix': sequence_path,
                    'frames': []
                })
                for frame_path in get_immediate_subfiles(sequence_path):
                    frame_name = os.path.split(frame_path)[1]
                    # assuming frame_%8d.{png,flo}
                    frame_number = int(frame_name[6:14])
                    if frame_name[-3:] == 'png':
                        got_png = True
                        png_number = frame_number
                        png_name = frame_name
                    elif frame_name[-3:] == 'flo':
                        got_flo = True
                        flo_number = frame_number
                        flo_name = frame_name
                    if got_png and got_flo and png_number == flo_number:
                        got_png = False
                        got_flow = False
                        train_images[train_count]['frames'].append((flo_name,
                                                                    png_name))
                train_count += 1
            elif i == 0:
                validation_images.append({
                    'prefix': sequence_path,
                    'frames': []
                })
                for frame_path in get_immediate_subfiles(sequence_path):
                    frame_name = os.path.split(frame_path)[1]
                    # assuming frame_%8d.{png,flo}
                    frame_number = int(frame_name[6:14])
                    if frame_name[-3:] == 'png':
                        got_png = True
                        png_number = frame_number
                        png_name = frame_name
                    elif frame_name[-3:] == 'flo':
                        got_flo = True
                        flo_number = frame_number
                        flo_name = frame_name
                    if got_png and got_flo and png_number == flo_number:
                        got_png = False
                        got_flow = False
                        validation_images[validation_count]['frames'].\
                            append((flo_name, png_name))
                validation_count += 1
            i += 1

    return DataSet(train=train_images,
                   validation=validation_images,
                   temporal_extent=temporal_extent)


def load_UCF101(train_dir='UCF-101-gt'):
    return read_data_sets(train_dir)


def get_immediate_subdirectories(a_dir):
    return sorted([os.path.join(a_dir, name) for name in os.listdir(a_dir)
                   if os.path.isdir(os.path.join(a_dir, name))])


def get_immediate_subfiles(a_dir):
    return sorted([os.path.join(a_dir, name) for name in os.listdir(a_dir)
                   if os.path.isfile(os.path.join(a_dir, name))])


def data_iterator(dataset, batch_size):
    while True:
        x_batch, y_batch = dataset.next_batch(batch_size)
        yield x_batch, y_batch


class QueueRunner(object):
    """
    This class manages the the background threads needed to fill
        a queue full of data.
    """
    def __init__(self, dataset,
                 input_shape, target_shape,
                 batch_size, n_threads=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_threads = n_threads
        self.dataX = tf.placeholder(dtype=tf.float32, shape=[None,
                                                             input_shape[0],
                                                             input_shape[1],
                                                             input_shape[2],
                                                             input_shape[3]])
        self.dataY = tf.placeholder(dtype=tf.float32, shape=[None,
                                                             target_shape[0],
                                                             target_shape[1],
                                                             target_shape[2]])
        # The actual queue of data. The queue contains a vector for
        # the mnist features, and a scalar label.
        self.queue = tf.RandomShuffleQueue(shapes=[[input_shape[0],
                                                    input_shape[1],
                                                    input_shape[2],
                                                    input_shape[3]],
                                                   [target_shape[0],
                                                    target_shape[1],
                                                    target_shape[2]]],
                                           dtypes=[tf.float32, tf.float32],
                                           capacity=batch_size*n_threads,
                                           min_after_dequeue=batch_size)

        # The symbolic operation to add data to the queue
        # we could do some preprocessing here or do it in numpy.
        # In this example we do the scaling in numpy
        self.enqueue_op = self.queue.enqueue_many([self.dataX, self.dataY])

    def get_inputs(self):
        """
        Returns tensors containing a batch of images and labels
        """
        images_batch, labels_batch = self.queue.dequeue_many(self.batch_size)
        return images_batch, labels_batch

    def thread_main(self, sess):
        """
        Function run on alternate thread. Basically, keep adding data to the
        queue.
        """
        for dataX, dataY in data_iterator(self.dataset, self.batch_size):
            sess.run(self.enqueue_op, feed_dict={self.dataX: dataX,
                                                 self.dataY: dataY})

    def start_threads(self, sess):
        """ Start background threads to feed queue """
        threads = []
        for n in range(self.n_threads):
            t = threading.Thread(target=self.thread_main, args=(sess,))
            t.daemon = True  # thread will close when parent quits
            t.start()
            threads.append(t)
        return threads
