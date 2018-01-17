import os
import numpy as np
from src.utilities import *
import glob
from imgaug import augmenters as iaa
from imgaug import parameters as iap


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

    def augment_batch(self, batches):
        masks = []
        aug = iaa.Sequential([
            iaa.FlowAffine(#scale={'x': (0.9, 1.1), 'y': (0.9, 1.1)},
                           rotate=(-360, 360),
                           p_fliplr=0.5,
                           p_flipud=0.5,
                           masks=masks, global_only=True),
            iaa.AdditiveGaussianNoise(scale=(0.0, 0.04 * 255.0),
                                      global_only=True),
            iaa.GammaCorrect(gamma=(0.7, 1.5), global_only=True),  # gamma
            iaa.Add(value=iap.Normal(loc=0, scale=0.2 * 255.0), global_only=True),  # additive brightness change
            iaa.Multiply(mul=(0.2, 1.4), global_only=True)  # contrast
        ], random_order=True)
        images, flows = batches
        images_result = []
        flows_result = []
        for i in range(images.shape[0]):
            imgs, flow = aug.augment_images(images[i] * 255.0,
                                            extra=flows[i])
            imgs = imgs / 255.0
            images_result.append(imgs)
            flows_result.append(flow)
        # deal with irregularly shaped images/flows (not relevant yet)
        first_shape = images_result[0].shape
        if all([image.shape == first_shape for image in images_result[1:]]):
            images_result = np.stack(images_result)
            flows_result = np.stack(flows_result)

        if len(masks) > 0:
            masks = np.stack(masks)
        else:
            batch_size = flows_result.shape[0]
            height = flows_result.shape[1]
            width = flows_result.shape[2]
            masks = np.ones((batch_size, height, width, 1))

        return images_result.astype(np.float32), \
            flows_result.astype(np.float32), \
            masks.astype(np.float32)

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
            return self.augment_batch((packaged_images, packaged_flows))
        else:
            height = packaged_flows.shape[1]
            width = packaged_flows.shape[2]
            return packaged_images, packaged_flows, np.ones((batch_size,
                                                             height, width, 1))


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
