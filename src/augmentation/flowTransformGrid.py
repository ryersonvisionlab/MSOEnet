import tensorflow as tf
from meshGridFlat import *

def flowTransformGrid(flow):
	with tf.get_default_graph().name_scope("flowTransformGrid"):
		flowShape = flow.get_shape()
		batchSize = flowShape[0]
		height = flowShape[1]
		width = flowShape[2]

		identityGrid = meshGridFlat(batchSize,height,width)

		flowU = tf.slice(flow,[0,0,0,0],[-1,-1,-1,1])
		flowV = tf.slice(flow,[0,0,0,1],[-1,-1,-1,1])

		#scale it to normalized range [-1,1]
		flowU = tf.reshape((flowU*2)/tf.cast(width, tf.float32),shape=tf.stack([batchSize,1,-1]))
		flowV = tf.reshape((flowV*2)/tf.cast(height, tf.float32),shape=tf.stack([batchSize,1,-1]))
		zeros = tf.zeros(shape=tf.stack([batchSize,1, height*width]))

		flowScaled = tf.concat(axis=1,values=[flowU,flowV,zeros])

		return identityGrid + flowScaled
