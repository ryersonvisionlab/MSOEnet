import os
import numpy as np
from src.utilities import *
import glob


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

  
    def discrete_rotate(self, input, k, flow=False):
        if flow:
            rad = k * (np.pi / 2.0)
            fx, fy = np.copy(input[:, :, 0]), np.copy(input[:, :, 1])
            sin = np.sin(rad)
            cos = np.cos(rad)
            input[..., 0] = (fx * cos) - (fy * sin)
            input[..., 1] = (fx * sin) + (fy * cos)
        return np.rot90(input, k)


    def augment(self, dataX, dataY):
        for i in range(dataX.shape[0]):
            k = np.random.randint(0, 4)
            if k > 0:
                for j in range(dataX.shape[1]):
                    dataX[i][j] = self.discrete_rotate(dataX[i][j], k)
                dataY[i] = self.discrete_rotate(dataY[i], k, flow=True)
        return dataX, dataY


    # TODO: reduce code reuse between this and validation_data
    def next_batch(self, batch_size, augment_batch=False):
        # sampling with replacement
        indices = np.random.choice(len(self._train), batch_size)
        sequences = [self._train[i] for i in indices]

        packaged_images = []
        packaged_flows = []
        for sequence in sequences:
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
            self._index_in_epoch -= self._num_examples

        # wrap in numpy array (easier to work with)
        packaged_images = np.array(packaged_images)
        packaged_flows = np.array(packaged_flows)

        if augment_batch:
            self.augment(packaged_images, packaged_flows)

        return packaged_images, packaged_flows


def load_FlyingChairs(data_dir):
    train_sequences = []
    validation_sequences = []

    got_ppm1 = False
    got_ppm2 = False
    got_flo = False

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


def load_UCF101(data_dir):
    train_sequences = []
    validation_sequences = []

    print 'Creating dataset filename structure...'
    for sequence_group in get_immediate_subdirectories(data_dir):
        i = 0
        for sequence_subgroup in get_immediate_subdirectories(sequence_group):
            flo_paths = sorted(glob.glob(sequence_subgroup + '/*.flo'))
            count = 1
            for flo_path in flo_paths:
                flo_name = os.path.split(flo_path)[1]
                # assuming frame_%08d.{png,flo}
                flo_number = int(flo_name[6:14])
                img1_name = flo_name[:14] + '.png'
                img2_name = 'frame_%08d.png' % (flo_number + 1)
                if count > 10:
                    train_sequences.append({
                        'prefix': sequence_subgroup,
                        'image_names': [img1_name, img2_name],
                        'flow_name': flo_name
                    })
                elif count <= 10 and i == 0:
                    validation_sequences.append({
                        'prefix': sequence_subgroup,
                        'image_names': [img1_name, img2_name],
                        'flow_name': flo_name
                    })
                count += 1
            i += 1

    return DataSet(train=train_sequences,
                   validation=validation_sequences)


