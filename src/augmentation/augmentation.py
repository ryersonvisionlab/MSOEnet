import math
import tensorflow as tf
from meshGridFlat import *
from bilinearSampler import *
from flowTransformGrid import *

def geoAug(image,transform,size=None):
	with tf.get_default_graph().name_scope("geoAug"):
		imshape = image.get_shape()
		batchSize = imshape[0]
		height = imshape[1]
		width = imshape[2]
		if size:
			outshape = tf.pack([size[0], size[1]])
		else:
			outshape = tf.pack([height,width])

		theta = tf.slice(transform,[0,0,0],[-1,2,-1])

		identityGrid = meshGridFlat(batchSize,height,width)
		transformGrid = tf.batch_matmul(theta,identityGrid)

		out = bilinearSampler(image,transformGrid,outshape)
		return tf.reshape(out,image.get_shape())

def photoAug(image, augParams):
	'''
	constrast (multiplicative brightness)
	additive brightness
	gamma
	gaussian noise

	expects image values from [0,1]
	'''
	with tf.get_default_graph().name_scope("photoAug"):
		# generate random values
		contrast = augParams[0]
		brightness = augParams[1]
		gamma = augParams[2]
		noiseStd = augParams[3]

		noise = tf.random_normal(image.get_shape(), 0, noiseStd)

		#transform
		image = image*contrast + brightness
		image = tf.maximum(tf.minimum(image, 1), 0)  # clamp between 0 and 1
		image = tf.pow(image, 1/gamma)
		image = image + noise

		return image

def geoAugTransform(batchSize,translateXMax,translateYMax,rotateMax,scaleMin,scaleMax,flipping):
	with tf.get_default_graph().name_scope("geoAugParam"):
		translateX = tf.random_uniform([batchSize,1,1],-translateXMax,translateXMax)
		translateY = tf.random_uniform([batchSize,1,1],-translateYMax,translateYMax)
		rotate = tf.random_uniform([batchSize,1,1],-rotateMax,rotateMax)
		scaleY = tf.random_uniform([batchSize,1,1],scaleMin,scaleMax)

		if(flipping):
			flip = tf.round(tf.random_uniform([batchSize,1,1],0,1,dtype=tf.float32))*2 - 1
			scaleX = scaleY*flip #flipping is just negative scaling
		else:
			scaleX = scaleY

		#build translation rotation matrix
		sinComponent = tf.sin(rotate)
		cosComponent = tf.cos(rotate)
		zeros = tf.zeros([batchSize,1,1])
		ones = tf.ones([batchSize,1,1])

		r1 = tf.concat(2,[cosComponent,-sinComponent,translateX])
		r2 = tf.concat(2,[sinComponent,cosComponent,translateY])
		r3 = tf.concat(2,[zeros,zeros,ones])

		translateRotateMatrix = tf.concat(1,[r1,r2,r3])

		#build scaling and flipping matrix
		r1 = tf.concat(2,[scaleX,zeros,zeros])
		r2 = tf.concat(2,[zeros,scaleY,zeros])
		r3 = tf.concat(2,[zeros,zeros,ones])

		scalingMatrix = tf.concat(1,[r1,r2,r3])

		return tf.batch_matmul(translateRotateMatrix,scalingMatrix)


def geoAugFlow(flow,transform1,transform2):
	with tf.get_default_graph().name_scope("geoAugFlow"):
		flowShape = flow.get_shape()
		batchSize = flowShape[0].value
		height = flowShape[1].value
		width = flowShape[2].value
		postScale = tf.cast(tf.expand_dims(tf.expand_dims([width/2.0, height/2.0],0),2),tf.float32)

		#invert transformations
		transform1 = tf.matrix_inverse(transform1)
		transform2 = tf.matrix_inverse(transform2)

		#pre transform grid
		grid0 = flowTransformGrid(tf.zeros_like(flow))
		grid1 = flowTransformGrid(flow)

		#post transform grid
		grid0 = tf.batch_matmul(transform1,grid0)
		grid1 = tf.batch_matmul(transform2,grid1)

		#scale back to pixel space
		grid0 = (grid0[:,0:2,:]+1)*postScale
		grid1 = (grid1[:,0:2,:]+1)*postScale

		#unflatten
		augFlowFlat = grid1 - grid0
		augFlowFlat = tf.transpose(augFlowFlat,[0,2,1])
		augFlow = tf.reshape(augFlowFlat,flowShape)

		return augFlow

def photoAugParam(batchSize,contrastMin,contrastMax,brightnessStd,gammaMin,gammaMax,noiseStd):
	with tf.get_default_graph().name_scope("photoAugParam"):
		contrast = tf.random_uniform([batchSize,1,1,1],contrastMin,contrastMax)
		brightness = tf.random_normal([batchSize,1,1,1],0,brightnessStd)
		gamma = tf.random_uniform([batchSize,1,1,1],gammaMin,gammaMax)

		noise = noiseStd

		return [contrast, brightness, gamma, noise]
