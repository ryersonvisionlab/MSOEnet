import os
import numpy as np
from src.utilities import *


class DataSet(object):

    def __init__(self, train, validation):
        self._num_examples = len(train)
        self._train = train
        self._validation = validation
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def validation_data(self):
        packaged_images = []
        packaged_flows = []
        for sequence in self._validation:
            # getting file paths
            img_names = sequence['image_names']
            flow_name = sequence['flow_name']
            prefix = sequence['prefix'] + '/'

            # Read images in sequence and stack them into a single volume
            images = np.stack([load_image(prefix + name)
                               for name in img_names])

            # Read flo file
            flow = readFlowFile(prefix + flow_name)
            flow[:, :, 1] *= -1  # fy is in opposite direction, must flip
                                   # EpicFlow problems.

            # package images as data points and provide the ground truth flow
            packaged_images.append(images)
            packaged_flows.append(flow)

        # wrap in numpy array (easier to work with)
        packaged_images = np.array(packaged_images)
        packaged_flows = np.array(packaged_flows)

        return packaged_images, packaged_flows

    # TODO: reduce code reuse between this and validation_data
    def next_batch(self, batch_size):
        # sampling with replacement
        indices = np.random.choice(len(self._train), batch_size)
        sequences = [self._train[i] for i in indices]

        packaged_images = []
        packaged_flows = []
        for sequence in self._validation:
            # getting file paths
            img_names = sequence['image_names']
            flow_name = sequence['flow_name']
            prefix = sequence['prefix'] + '/'

            # Read images in sequence and stack them into a single volume
            images = np.stack([load_image(prefix + name)
                               for name in img_names])

            # Read flo file
            flow = readFlowFile(prefix + flow_name)
            flow[:, :, 1] *= -1  # fy is in opposite direction, must flip
                                   # EpicFlow problems.

            # package images as data points and provide the ground truth flow
            packaged_images.append(images)
            packaged_flows.append(flow)

        # update epoch count
        assert batch_size <= self._num_examples
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            # start next epoch
            self._index_in_epoch = self._index_in_epoch - self._num_examples

        # wrap in numpy array (easier to work with)
        packaged_images = np.array(packaged_images)
        packaged_flows = np.array(packaged_flows)

        return packaged_images, packaged_flows


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
                    'image_names': [img1_name, img2_name],
                    'flow_name': flo_name
                })
            elif split == '2':
                validation_sequences.append({
                    'prefix': data_dir,
                    'image_names': [img1_name, img2_name],
                    'flow_name': flo_name
                })
        count += 3

    return DataSet(train=train_sequences,
                   validation=validation_sequences)
