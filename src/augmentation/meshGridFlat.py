import tensorflow as tf

def meshGridFlat(batchSize, height, width):
    with tf.get_default_graph().name_scope('meshGridFlat'):
		# This should be equivalent to:
		#  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
		#				 np.linspace(-1, 1, height))
		#  ones = np.ones(np.prod(x_t.shape))
		#  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
		x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
				tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
		y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
				tf.ones(shape=tf.stack([1, width])))

		x_t_flat = tf.reshape(x_t, (1, -1))
		y_t_flat = tf.reshape(y_t, (1, -1))

		ones = tf.ones_like(x_t_flat)
		grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat, ones])

		baseGrid = tf.expand_dims(grid,0)
		grids = []
		for i in range(batchSize):
			grids.append(baseGrid)
		identityGrid = tf.concat(axis=0,values=grids)

		return identityGrid
