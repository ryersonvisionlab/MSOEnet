import tensorflow as tf
from src.components import *
from src.util import check_snapshots
from src.MSOE import MSOE
from src.GatingNetwork import GatingNetwork
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
            with tf.device('/gpu:' + str(self.user_config['gpu'])):
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
                    self.loss = l2_loss('l2_loss', self.output, self.target)
                    self.loss += tf.add_n(tf.get_collection('weight_regs'))
                    self.val_loss = l2_loss('epe_val', self.val_output,
                                            self.val_target_placeholder)

                    # attach summaries
                    self.attach_summaries('summaries')
                else:
                    """ feed-forward-only """
                    # user-given input
                    if input is None:
                        self.input_layer = tf.placeholder(tf.float32,
                                                          [None, 2, 256, 256,
                                                           1], name='images')
                    else:
                        self.input_layer = tf.pack(input)

                    """ Create pyramid """
                    self.output = self.build_pyramid('MSOEnet',
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
            with tf.variable_scope('MSOE_conv1', reuse=True):
                W_conv1 = tf.get_variable('weights')

            # fetch weights for conv3
            # with tf.variable_scope('conv3', reuse=True):
            #     W_conv3 = tf.get_variable('weights')

            # fetch shared weights for gate conv1
            with tf.variable_scope('Gate_conv1', reuse=True):
                gW_conv1 = tf.get_variable('weights')

            # graph loss
            tf.scalar_summary('loss', self.loss)
            self.average_val_loss = tf.placeholder(tf.float32)
            tf.scalar_summary('val_loss', self.average_val_loss,
                              collections=['val'])

            # visualize target and predicted flows
            tf.image_summary('flow predicted norm',
                             flow_to_colour('flow_visualization',
                                            self.output),
                             max_images=1)
            tf.image_summary('flow target norm',
                             flow_to_colour('flow_visualization',
                                            self.target),
                             max_images=1)
            tf.image_summary('flow predicted',
                             flow_to_colour('flow_visualization',
                                            self.output, norm=False),
                             max_images=1)
            tf.image_summary('flow target',
                             flow_to_colour('flow_visualization',
                                            self.target, norm=False),
                             max_images=1)
            tf.image_summary('image',
                             self.input_layer[0],
                             max_images=5)

            # visualize filters
            viz0 = W_conv1[0, :, :, :, :]
            viz1 = W_conv1[1, :, :, :, :]
            grid0n = put_kernels_on_grid('kernel_visualization_norm',
                                         viz0, 8, 4)
            grid1n = put_kernels_on_grid('kernel_visualization_norm',
                                         viz1, 8, 4)
            grid0 = put_kernels_on_grid('kernel_visualization',
                                        viz0, 8, 4, norm=False)
            grid1 = put_kernels_on_grid('kernel_visualization',
                                        viz1, 8, 4, norm=False)
            tf.image_summary('filter conv1 0 norm', grid0n)
            tf.image_summary('filter conv1 1 norm', grid1n)
            tf.image_summary('filter conv1 0', grid0)
            tf.image_summary('filter conv1 1', grid1)

            viz0 = gW_conv1[0, :, :, :, :]
            grid0n = put_kernels_on_grid('gate_kernel_visualization_norm',
                                         viz0, 4, 1)
            grid0 = put_kernels_on_grid('gate_kernel_visualization',
                                        viz0, 4, 1, norm=False)
            tf.image_summary('filter gate_conv1 0 norm', grid0n)
            tf.image_summary('filter gate_conv1 0', grid0)

            # histogram of filters
            for i in range(32):
                drange = tf.reduce_max(W_conv1[:, :, :, :, i]) - \
                         tf.reduce_min(W_conv1[:, :, :, :, i])
                tf.scalar_summary('drange filter conv1 ' + str(i), drange)
                tf.histogram_summary('histogram filter conv1 ' + str(i),
                                     W_conv1[:, :, :, :, i])

            # visualize gates
            for scale in range(self.user_config['num_scales']):
                tf.image_summary('gate_' + str(scale),
                                 self.gates[..., scale:scale+1],
                                 max_images=1)

            # visualize queue usage
            data_queue = self.queue_runner
            data_queue_capacity = data_queue.batch_size * data_queue.n_threads
            tf.scalar_summary('queue saturation',
                              data_queue.queue.size() / data_queue_capacity)

            # merge summaries
            self.summaries = tf.merge_all_summaries()
            self.val_summaries = tf.merge_all_summaries(key='val')

    def build_pyramid(self, name, input_layer, reuse=None):
        with tf.get_default_graph().name_scope(name):
            # initial MSOE on original input size (batchx1xHxWx64)
            initial_msoe = MSOE('MSOE_0', input_layer, reuse).output

            # initial GatingNetwork on original input size (batchx1xHxWx1)
            initial_gate = GatingNetwork('Gate_0', input_layer, reuse).output

            # initialize pyramid
            msoe_array = [initial_msoe]
            gate_array = [initial_gate]

            num_scales = self.user_config['num_scales']
            for scale in range(1, num_scales):
                # big to small
                spatial_stride = 2**scale

                # downsample data (batchx2xhxwx1)
                small_input = blur_downsample3d('input_downsample_' +
                                                str(scale),
                                                input_layer, 5,
                                                spatial_stride, sigma=2)

                # create MSOE and insert data (batchx1xhxwx64)
                small_msoe = MSOE('MSOE_' + str(scale), small_input,
                                  reuse=True).output

                # create GatingNetwork and insert data (batchx1xhxwx1)
                small_gate = GatingNetwork('Gate_' + str(scale), small_input,
                                           reuse=True).output

                # upsample flow output (batchx1xHxWx64)
                msoe = bilinear_resample3d('MSOE_upsample_' + str(scale),
                                           small_msoe, tf.shape(input_layer)
                                           [2:4])

                # upsample gate output (batchx1xHxWx1)
                gate = bilinear_resample3d('Gate_upsample_' + str(scale),
                                           small_gate, tf.shape(input_layer)
                                           [2:4])

                msoe_array.append(msoe)
                gate_array.append(gate)

            # channel concatenate gate outputs (batchx1xHxWxnum_scales)
            concatenated_gates = channel_concat3d('Gates_concat', gate_array)

            # channel-wise softmax the gate outputs (batchx1xHxWxnum_scales)
            gates = softmax('Gates_softmax', concatenated_gates)

            # for image summary visualization
            if name == 'train':
                self.gates = gates[:, 0, :, :, :]

            # apply per-scale gating to msoe outputs
            gated_msoes = [gates[..., :1] * msoe_array[0]]
            for scale in range(1, num_scales):
                # scale responses relative to the scale
                weight = 2**scale

                msoe_at_scale = msoe_array[scale]
                gate_at_scale = gates[..., scale:scale+1]

                # slow
                gated_msoe_at_scale = (gate_at_scale * weight) * msoe_at_scale

                gated_msoes.append(gated_msoe_at_scale)

            # sum up gated responses over all scales (batchx1xHxWx64)
            gated_msoe = tf.add_n(gated_msoes)

            # fourth convolution (flow out i.e. decode) (1x1x1x64x2)
            output = conv3d('conv3', gated_msoe, 1, 1, 2, reuse)

            # reshape (batch x H x W x 2)
            output = reshape('reshape', output,
                             [-1, tf.shape(output)[2],
                              tf.shape(output)[3], 2])

            return output

    def run_train(self):
        # for cleanliness
        iterations = self.user_config['iterations']
        base_lr = self.user_config['base_lr']
        snapshot_frequency = self.user_config['snapshot_frequency']
        print_frequency = self.user_config['print_frequency']
        validation_frequency = self.user_config['validation_frequency']

        with self.graph.as_default():
            with tf.device('/gpu:' + str(self.user_config['gpu'])):
                optimizer = tf.train.AdamOptimizer(learning_rate=base_lr)
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
                    # run a train step
                    results = sess.run([train_step, self.loss,
                                        self.summaries])

                    # print training information
                    if (i + 1) % print_frequency == 0:
                        time_diff = time.time() - last_print
                        it_per_sec = print_frequency / time_diff
                        remaining_it = iterations - i
                        eta = remaining_it / it_per_sec
                        print 'Iteration %d: loss: %f lr: %f ' \
                              'iter per/s: %f ETA: %s' \
                              % (i + 1, results[1], base_lr, it_per_sec,
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
                model = check_snapshots(folder='final_model', train=False)
                saver.restore(sess, model)

                result = sess.run([self.output])[0]
                return result

    def save_model(self):
        with self.graph.as_default():
            saver = tf.train.Saver(max_to_keep=0,
                                   write_version=tf.train.SaverDef.V2)
            with tf.Session(config=self.tf_config) as sess:
                # load model
                model = check_snapshots(train=False)
                saver.restore(sess, model)
                saver.save(sess, 'final_model/MSOEnet.ckpt')
                for op in self.graph.get_operations():
                    print op.name
