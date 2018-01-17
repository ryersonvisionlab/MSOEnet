import tensorflow as tf
import threading


class QueueRunner(object):
    """
    This class manages the the background threads needed to fill
        a queue full of data.
    """
    def __init__(self, dataset, input_shape, target_shape, masks_shape,
                 batch_size, n_threads=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_threads = n_threads
        self.augment_data = False

        input_shape = [None] + input_shape
        target_shape = [None] + target_shape
        masks_shape = [None] + masks_shape
        self.dataX = tf.placeholder(dtype=tf.float32, shape=input_shape)
        self.dataY = tf.placeholder(dtype=tf.float32, shape=target_shape)
        self.masks = tf.placeholder(dtype=tf.float32, shape=masks_shape)

        # The actual queue of data.
        self.queue = tf.RandomShuffleQueue(shapes=[input_shape[1:],
                                                   target_shape[1:],
                                                   masks_shape[1:]],
                                           dtypes=[tf.float32,
                                                   tf.float32,
                                                   tf.float32],
                                           capacity=batch_size*n_threads,
                                           min_after_dequeue=batch_size)

        # The symbolic operation to add data to the queue
        self.enqueue_op = self.queue.enqueue_many([self.dataX,
                                                   self.dataY,
                                                   self.masks])

    def get_inputs(self):
        """
        Returns tensors containing a batch of images and labels
        """
        images_batch, labels_batch, masks_batch = \
            self.queue.dequeue_many(self.batch_size)
        return images_batch, labels_batch, masks_batch

    def thread_main(self, sess):
        """
        Function run on alternate thread. Basically, keep adding data to the
        queue.
        """
        for dataX, dataY, masks in self._data_iterator():
            sess.run(self.enqueue_op, feed_dict={self.dataX: dataX,
                                                 self.dataY: dataY,
                                                 self.masks: masks})

    def start_threads(self, sess):
        """ Start background threads to feed queue """
        threads = []
        for n in range(self.n_threads):
            t = threading.Thread(target=self.thread_main, args=(sess,))
            t.daemon = True  # thread will close when parent quits
            t.start()
            threads.append(t)
        return threads

    def _data_iterator(self):
        while True:
            x_batch, y_batch, mask_batch = \
                self.dataset.next_batch(self.batch_size, self.augment_data)
            yield x_batch, y_batch, mask_batch
