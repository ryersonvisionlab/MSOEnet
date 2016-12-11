import tensorflow as tf
import time
import datetime
from src.architecture import build_net
from src.util import check_snapshots


"""-----settings-------"""
print_frequency = 1
snapshot_frequency = 500

batch_size = 10
iterations = 200000
start_iteration = 0
resume = False
learning_rate = 3e-4

num_threads = 4
"""---------------------"""

# check for snapshots
check_snapshots()

# build network
summaries, solver, loss, loss_val, queue_runner = build_net(batch_size,
                                                            learning_rate,
                                                            num_threads)

# config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = True
config.intra_op_parallelism_threads = 12

# saver
saver = tf.train.Saver(max_to_keep=0)

# start a tensorflow session
with tf.Session(config=config) as sess:

    # start summary writer
    summary_writer = tf.train.SummaryWriter('logs', sess.graph)

    with tf.device('/gpu:0'):
        """train and evaluate model"""
        sess.run(tf.initialize_all_variables())

        # start the tensorflow QueueRunners
        tf.train.start_queue_runners(sess=sess)

        # start the data queue runner's threads
        threads = queue_runner.start_threads(sess)

        last_print = time.time()
        for i in range(iterations):
            results = sess.run([summaries, solver, loss, loss_val])

            if (i + 1) % print_frequency == 0:
                time_diff = time.time() - last_print
                it_per_sec = print_frequency / time_diff
                remaining_it = iterations - i
                eta = remaining_it / it_per_sec
                print('Iteration ' + str(i + 1) + ': loss: ' +
                      str(results[2]) + ' loss (val): ' +
                      str(results[3]) + ", iterations per second: " +
                      str(it_per_sec) + ', ETA: ' +
                      str(datetime.timedelta(seconds=eta)))
                summary_writer.add_summary(results[0], i + 1)
                summary_writer.flush()
                last_print = time.time()

            if (i + 1) % snapshot_frequency == 0:
                saver.save(sess, 'snapshots/iter_' +
                           str(i + 1).zfill(16) + '.ckpt')
