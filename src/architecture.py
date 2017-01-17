import tensorflow as tf
from src.components import *
from src.util import check_snapshots
from src.MSOE import MSOE
import time
import datetime
import numpy as np


class MSOEPyramid(object):

    def __init__(self, config, input=None):
        # import config
        self.user_config = config['user']
        self.tf_config = config['tf']

        self.graph = tf.Graph()
        with self.graph.as_default():
            # TODO: verify if gpu 1 being used and if this is redundant
            with tf.device('/gpu:1'):
                """
                Construct the MSOE pyramid graph structure
                """
                if self.user_config['train']:
                    # retrieve data (validation data are in numpy arrays, this
                    # is because you can't feed Tensors into feed_dict, which
                    # is what's being done for validation)
                    self.input_layer, self.target, \
                        self.val_input_layer, self.val_target, \
                        self.queue_runner = data_layer('input',
                                                       self.user_config
                                                       ['dataset_path'],
                                                       self.user_config
                                                       ['batch_size'],
                                                       self.user_config
                                                       ['temporal_extent'],
                                                       self.user_config
                                                       ['num_threads'])
                    """ Create pyramid """
                    self.output = self.build_pyramid('train', self.input_layer)
                    # build placeholders for validation input and target.
                    # this will be used as a hack to get over memory
                    # constraints of having a large tensor of validation data,
                    # since we'll be breaking it up into chunks and validating
                    # on each
                    self.build_validation_placeholders()
                    self.val_output = self.build_pyramid('val',
                                                         self.
                                                         val_input_placeholder,
                                                         reuse=True)

                    # attach losses
                    self.loss = l1_loss('l1_loss', self.output, self.target)
                    self.val_loss = l1_loss('l1_loss_val', self.val_output,
                                            self.val_target_placeholder)

                    # attach summaries
                    self.attach_summaries('summaries')
                else:
                    """ feed-forward-only """
                    # user-given input
                    self.input_layer = tf.pack(input)

                    """ Create pyramid """
                    self.output = self.build_pyramid('MSOE_Pyramid',
                                                     self.input_layer)

    def build_validation_placeholders(self):
        input_shape = self.val_input_layer.shape
        target_shape = self.val_target.shape
        self.val_input_placeholder = tf.placeholder(dtype=tf.float32,
                                                    shape=[None,
                                                           input_shape[1],
                                                           input_shape[2],
                                                           input_shape[3],
                                                           input_shape[4]])
        self.val_target_placeholder = tf.placeholder(dtype=tf.float32,
                                                     shape=[None,
                                                            target_shape[1],
                                                            target_shape[2],
                                                            target_shape[3]])

    def attach_summaries(self, name):
        with tf.name_scope(name):
            # fetch shared weights for conv1
            with tf.variable_scope('conv1', reuse=True):
                W_conv1 = tf.get_variable('weights')

            # graph loss
            tf.scalar_summary('loss', self.loss)
            self.average_val_loss = tf.placeholder(tf.float32)
            tf.scalar_summary('val_loss', self.average_val_loss,
                              collections=['val'])

            # visualize target and predicted flows
            tf.image_summary('flow predicted',
                             flow_to_colour('flow_visualization',
                                            self.output),
                             max_images=1)
            tf.image_summary('flow target',
                             flow_to_colour('flow_visualization',
                                            self.target),
                             max_images=1)
            tf.image_summary('image',
                             self.input_layer[0],
                             max_images=5)

            # visualize filters
            viz0 = W_conv1[0, :, :, :, :]
            viz1 = W_conv1[1, :, :, :, :]
            grid0 = put_kernels_on_grid('kernel_visualization',
                                        viz0, 8, 4)
            grid1 = put_kernels_on_grid('kernel_visualization',
                                        viz1, 8, 4)
            tf.image_summary('filter conv1 0', grid0)
            tf.image_summary('filter conv1 1', grid1)

            # merge summaries
            self.summaries = tf.merge_all_summaries()
            self.val_summaries = tf.merge_all_summaries(key='val')

    def build_pyramid(self, name, input_layer, reuse=None):
        with tf.get_default_graph().name_scope(name):
            # assuming square input
            input_hw = [input_layer.get_shape().as_list()[2],
                        input_layer.get_shape().as_list()[3]]

            # initial MSOE on original input size (batchxHxWx2)
            initial = MSOE('MSOE_0', input_layer, reuse).output

            # initialize pyramid
            msoe_array = [initial]
            for scale in range(1, self.user_config['num_scales']):
                # big to small
                spatial_stride = 2**scale

                # downsample data (batchx2xhxwx1)
                small_input = avg_pool3d('downsample',
                                         input_layer, 3, 1,
                                         spatial_stride)

                # create MSOE and insert data
                small_output = MSOE('MSOE_' + str(scale),
                                    small_input, reuse=True).output

                # upsample flow output (batchx1xHxWx64)
                output_layer = bilinear_resample3d('upsample',
                                                   small_output,
                                                   tf.pack(input_hw))

                msoe_array.append(output_layer)

            # channel concatenate outputs (batchx1xHxWx64*num_scales)
            concatenated = channel_concat3d('concat', msoe_array)

            # fourth convolution (flow out i.e. decode) (1x1x1x64*num_scalesx2)
            output = conv3d('conv4', concatenated, 1, 1, 2, reuse)

            # reshape (batch x H x W x 2)
            output_shape = output.get_shape().as_list()
            output = reshape('reshape', output,
                             [-1, output_shape[2],
                              output_shape[3], 2])

            return output

    def run_train(self):
        # for cleanliness
        iterations = self.user_config['iterations']
        base_lr = self.user_config['base_lr']
        lr_gamma = self.user_config['lr_gamma']
        lr_policy_start = self.user_config['lr_policy_start']
        lr_stepsize = self.user_config['lr_stepsize']
        snapshot_frequency = self.user_config['snapshot_frequency']
        print_frequency = self.user_config['print_frequency']
        validation_frequency = self.user_config['validation_frequency']

        with self.graph.as_default():
            learning_rate = tf.placeholder(tf.float32, shape=[])

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                               beta1=0.9, beta2=0.999,
                                               epsilon=1e-08,
                                               use_locking=False, name='Adam')

            train_step = optimizer.minimize(self.loss)

            """
            Train over iterations, printing loss at each one
            """
            saver = tf.train.Saver(max_to_keep=0,
                                   write_version=tf.train.SaverDef.V2)
            with tf.Session(config=self.tf_config) as sess:

                # check snapshots
                resume, start_iteration = check_snapshots()

                # start summary writers
                summary_writer = tf.train.SummaryWriter('logs/train',
                                                        sess.graph)
                summary_writer_val = tf.train.SummaryWriter('logs/val')

                # start the tensorflow QueueRunners
                tf.train.start_queue_runners(sess=sess)

                # start the data queue runner's threads
                threads = self.queue_runner.start_threads(sess)

                if resume:
                    saver.restore(sess, resume)
                else:
                    sess.run(tf.initialize_all_variables())

                last_print = time.time()
                for i in range(start_iteration, iterations):
                    # learning rate step policy
                    # initial lr update
                    if (i + 1) == lr_policy_start:
                        base_lr *= lr_gamma
                    # subsequent lr updates
                    if ((i + 1) - lr_policy_start) % lr_stepsize == 0 and \
                       (i + 1) > lr_policy_start:
                        base_lr *= lr_gamma

                    # run a train step
                    results = sess.run([train_step, self.loss,
                                        self.summaries, learning_rate],
                                       feed_dict={learning_rate: base_lr})

                    # print training information
                    if (i + 1) % print_frequency == 0:
                        time_diff = time.time() - last_print
                        it_per_sec = print_frequency / time_diff
                        remaining_it = iterations - i
                        eta = remaining_it / it_per_sec
                        print 'Iteration %d: loss: %f lr: %f ' \
                              'iter per/s: %f ETA: %s' \
                              % (i + 1, results[1], results[3], it_per_sec,
                                 str(datetime.timedelta(seconds=eta)))
                        summary_writer.add_summary(results[2], i + 1)
                        summary_writer.flush()
                        last_print = time.time()

                    # print validation information
                    if (i + 1) % validation_frequency == 0:
                        print 'Validating...'

                        # breaking up large validation data into chunks to
                        # prevent out of memory issues
                        avg_val_loss, val_summary = self.validate_chunks(sess)

                        print 'Validation loss: %f' % (avg_val_loss)
                        summary_writer_val.add_summary(val_summary, i + 1)
                        summary_writer_val.flush()

                    # save snapshot
                    if (i + 1) % snapshot_frequency == 0:
                        print 'Saving snapshot...'
                        saver.save(sess, 'snapshots/iter_' +
                                   str(i + 1).zfill(16) + '.ckpt')

    def validate_chunks(self, sess):
        batch_size = self.user_config['batch_size']
        total_val_loss = []
        num_val = self.val_input_layer.shape[0]
        num_chunks = num_val / batch_size

        for j in range(num_chunks):
            start = batch_size * j
            end = start + batch_size
            val_input = self.val_input_layer[start:end]
            val_target = self.val_target[start:end]
            val_results = sess.run([self.val_loss],
                                   feed_dict={self.val_input_placeholder:
                                              val_input,
                                   self.val_target_placeholder: val_target})
            total_val_loss.append(val_results[0])

        avg_val_loss = np.mean(total_val_loss)
        val_summary = sess.run([self.val_summaries],
                               feed_dict={self.average_val_loss:
                                          avg_val_loss})[0]

        return avg_val_loss, val_summary

    def run_test(self):
        with self.graph.as_default():
            # TODO: switch to tf.train.import_meta_graph
            saver = tf.train.Saver(max_to_keep=0,
                                   write_version=tf.train.SaverDef.V2)
            with tf.Session(config=self.tf_config) as sess:
                # load model
                model = check_snapshots(train=False)
                saver.restore(sess, model)

                result = sess.run([self.output])[0]
                return result
