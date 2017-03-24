import os
import random
import numpy as np
import tensorflow as tf
import threading
from util import readFlowFile, load_image, rgb2gray
from augmentation import *


class DataSet(object):

    def __init__(self, train, validation, temporal_extent):
        count = 0
        for sequence in train:
            count += len(sequence['flows'])
        self._num_examples = count
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

        packaged_data = []
        packaged_gts = []
        for sequence in sequences:
            limit = 2**31 - 1
            for i in range(len(sequence['images']) -
                           self._temporal_extent + 1):

                if i == limit:
                    break

                img_names = sequence['images'][i:i+self._temporal_extent]
                flo_names = sequence['flows'][i:i+self._temporal_extent-1]
                prefix = sequence['prefix'] + '/'

                # Read images in sequence and stack them into a single volume
                images = np.stack([load_image(prefix + name)
                                   for name in img_names])

                # pick flow for middle frame
                gt_flo = prefix + flo_names[int(np.ceil(len(img_names)/2.0) -
                                            1)]

                # Read flo file
                gt_flo = readFlowFile(gt_flo)
                gt_flo[:, :, 1] *= -1  # fy is in opposite direction, must flip

                # package images as data points and provide the ground
                # truth flow for the middle frame
                packaged_data.append(images)
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
        # Pick a random batch of sequences
        indices = np.random.choice(len(self._train), batch_size)
        sequences = [self._train[i] for i in indices]

        packaged_data = []
        packaged_gts = []
        for sequence in sequences:
            # Pick one consecutive chunk (length temporal_extent) of image
            # names and corresponding flo names from each sequence at random
            i = random.randint(0, len(sequence['images']) -
                               self._temporal_extent)
            img_names = sequence['images'][i:i+self._temporal_extent]
            flo_names = sequence['flows'][i:i+self._temporal_extent-1]
            prefix = sequence['prefix'] + '/'

            # Read images in sequence and stack them into a single volume
            images = np.stack([load_image(prefix + name)
                               for name in img_names])

            # pick flow for middle frame
            gt_flo = prefix + flo_names[int(np.ceil(len(img_names)/2.0) - 1)]

            # Read flo file
            gt_flo = readFlowFile(gt_flo)
            gt_flo[:, :, 1] *= -1  # fy is in opposite direction, must flip

            # package images as data points and provide the ground
            # truth flow for the middle frame
            packaged_data.append(images)
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

        return packaged_data, packaged_gts


def augment(dataX, dataY, batchSize):
    frame0 = dataX[:, 0]
    frame1 = dataX[:, 1]
    flow = dataY

    # random crop

    # geo augmentation
    globalAffine = geoAugTransform(batchSize, 0.1, 0.1, 0.17, 0.9, 1.1, True)
    localAffine = geoAugTransform(batchSize, 0.1, 0.1, 0.17, 0.9, 1.1, False)
    globalLocalAffine = tf.matmul(localAffine, globalAffine)
    frame0Geo = geoAug(frame0, globalAffine)  # resize here
    frame1Geo = geoAug(frame1, globalLocalAffine)  # resize here

    # augment flow
    flowGeo = geoAugFlow(flow, globalAffine, globalLocalAffine)
    flowGeo = geoAug(flowGeo, globalAffine)

    # image augmentation
    IMAGENET_MEAN = np.array([123.68, 116.779, 103.939],
                             dtype='float32').reshape((1, 1, 3)) / 255.0
    IMAGENET_MEAN_GRAY = tf.expand_dims(
                    rgb2gray(IMAGENET_MEAN).astype('float32'), 0)
    photoParam = photoAugParam(batchSize, 0.7, 1.3, 0.2, 0.9,
                               1.1, 0.7, 1.5, 0.04)
    frame0photo = photoAug(frame0Geo, photoParam) - IMAGENET_MEAN_GRAY
    frame1photo = photoAug(frame1Geo, photoParam) - IMAGENET_MEAN_GRAY

    dataX = tf.stack([frame0photo, frame1photo], axis=1)
    dataY = flowGeo

    return dataX, dataY


# TODO: rewrite
def load_UCF101(train_dir, temporal_extent=5):

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


def load_FlyingChairs(data_dir):
    train_sequences = []
    validation_sequences = []

    got_ppm1 = False
    got_ppm2 = False
    got_flo = False

    train_count = 0
    validation_count = 0

    file_paths = get_immediate_subfiles(data_dir)
    count = 0

    split_filepath = os.path.split(data_dir)[0] + '/FlyingChairs_train_val.txt'
    split_file = open(split_filepath, 'r')

    print 'Creating dataset filename structure...'
    while count < 22872 * 3:
        flo_name = os.path.split(file_paths[count])[1]
        img1_name = os.path.split(file_paths[count+1])[1]
        img2_name = os.path.split(file_paths[count+2])[1]
        split = split_file.readline().splitlines()[0]
        # assuming %5d_{img1,img2,flow}.{ppm,flo}
        if flo_name[-3:] == 'flo':
            got_flo = True
        if img1_name[-3:] == 'ppm':
            got_ppm1 = True
        if img2_name[-3:] == 'ppm':
            got_ppm2 = True
        if got_ppm1 and got_ppm2 and got_flo:
            got_ppm1 = False
            got_ppm2 = False
            got_flow = False
            if split == '1':
                train_sequences.append({
                    'prefix': data_dir,
                    'images': [img1_name, img2_name],
                    'flows': [flo_name]
                })
            elif split == '2':
                validation_sequences.append({
                    'prefix': data_dir,
                    'images': [img1_name, img2_name],
                    'flows': [flo_name]
                })
        count += 3

    return DataSet(train=train_sequences,
                   validation=validation_sequences,
                   temporal_extent=2)


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
        # The actual queue of data.
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
