import tensorflow as tf
from src.components import *
from src.util import check_snapshots
from src.MSOE import MSOE
import time
import datetime


class MSOEPyramid(object):

    def __init__(self, config):
        # import config
        self.user_config = config['user']
        self.tf_config = config['tf']
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.device('/gpu:1'):
                """
                Construct the MSOE pyramid graph structure
                """
                # retrieve data
                self.input_layer, self.target, \
                self.val_input_layer, self.val_target, \
                self.queue_runner = data_layer('input',
                                               self.user_config['dataset_path'],
                                               self.user_config['batch_size'],
                                               self.user_config['temporal_extent'],
                                               self.user_config['num_threads'])

                """ Create pyramid """
                self.output = self.build_pyramid('train', self.input_layer)
                self.val_output = self.build_pyramid('val', self.val_input_layer,
                                                     reuse=True)

                # attach loss
                self.loss = l1_loss('l1_loss', self.output, self.target)
                self.val_loss = l1_loss('l1_loss_val', self.val_output, self.val_target)

                # attach summaries
                with tf.name_scope('summaries'):
                    # fetch shared weights for conv1
                    with tf.variable_scope('conv1', reuse=True):
                        W_conv1 = tf.get_variable('weights')

                    # graph loss
                    tf.scalar_summary('loss', self.loss)
                    tf.scalar_summary('val_loss', self.val_loss,
                                      collections='val')

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

                # start summary writer
                summary_writer = tf.train.SummaryWriter('logs/train', sess.graph)
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
                              % (i + 1, results[1], results[3],
                                 it_per_sec, str(datetime.timedelta(seconds=eta))) 
                        summary_writer.add_summary(results[2], i + 1)
                        summary_writer.flush()
                        last_print = time.time()

                    # print validation information
                    if (i + 1) % validation_frequency == 0:
                        val_results = sess.run([self.val_loss,
                                                self.val_summaries])
                        print 'Validation loss: %f' % (val_results[0])
                        summary_writer_val.add_summary(val_results[1], i + 1)
                        summary_writer_val.flush()

                    # save snapshot
                    if (i + 1) % snapshot_frequency == 0:
                        print 'Saving snapshot...'
                        saver.save(sess, 'snapshots/iter_' +
                           str(i + 1).zfill(16) + '.ckpt')