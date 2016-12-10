import tensorflow as tf
import math


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


def conv3d(x, W):
    with tf.name_scope('3D_conv'):
        k_h = int(W.get_shape()[1])
        k_w = int(W.get_shape()[2])
        x = tf.pad(x, [[0, 0], [0, 0],
                       [k_h/2, k_h/2],
                       [k_w/2, k_w/2],
                       [0, 0]], 'CONSTANT')
        return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='VALID')


def avg_pool_3x3x3(x):
    with tf.name_scope('avg_pool_3x3x3'):
        return tf.nn.avg_pool3d(x, ksize=[1, 3, 3, 3, 1],
                                strides=[1, 1, 1, 1, 1], padding='SAME')


def eltwise_square(x):
    with tf.name_scope('eltwise_square'):
        return tf.square(x)


def l1_normalize(x, dim, eps=1e-12):
    with tf.name_scope('l1_normalization'):
        abs_sum = tf.reduce_sum(tf.abs(x), dim, keep_dims=True)
        x_inv_norm = tf.inv(tf.maximum(abs_sum, eps))
        return tf.mul(x, x_inv_norm)


def l2_loss(x, y):
    with tf.name_scope('l2_loss'):
        return tf.reduce_mean(tf.square(tf.sub(x, y)))


def l1_loss(x, y):
    with tf.name_scope('l1_loss'):
        return tf.reduce_mean(tf.abs(tf.sub(x, y)))


def flowToColor(flow):
	with tf.name_scope("flow_visualization"):
		#constants
		flowShape = flow.get_shape()

		oor2 = 1/math.sqrt(2)
		sqrt3h = math.sqrt(3)/2
		k0 = tf.transpose(tf.constant([ [ [ [1,0] ] ] ],dtype=tf.float32),perm=[0,1,3,2])
		k120 = tf.transpose(tf.constant([ [ [ [-oor2,oor2] ] ] ],dtype=tf.float32),perm=[0,1,3,2])
		k240 = tf.transpose(tf.constant([ [ [ [-oor2,-oor2] ] ] ],dtype=tf.float32),perm=[0,1,3,2])
		k60 = tf.transpose(tf.constant([ [ [ [sqrt3h,-1./2.] ] ] ],dtype=tf.float32),perm=[0,1,3,2])
		k180 = tf.transpose(tf.constant([ [ [ [-1,0] ] ] ],dtype=tf.float32),perm=[0,1,3,2])
		k300 = tf.transpose(tf.constant([ [ [ [sqrt3h,1./2.] ] ] ],dtype=tf.float32),perm=[0,1,3,2])

		k0c = tf.transpose(tf.constant([ [ [ [1,0,0] ] ] ],dtype=tf.float32),perm=[0,1,3,2])
		k120c = tf.transpose(tf.constant([ [ [ [0,1,0] ] ] ],dtype=tf.float32),perm=[0,1,3,2])
		k240c = tf.transpose(tf.constant([ [ [ [0,0,1] ] ] ],dtype=tf.float32),perm=[0,1,3,2])
		k60c = tf.transpose(tf.constant([ [ [ [1,1,0] ] ] ],dtype=tf.float32),perm=[0,1,3,2])
		k180c = tf.transpose(tf.constant([ [ [ [0,1,1] ] ] ],dtype=tf.float32),perm=[0,1,3,2])
		k300c = tf.transpose(tf.constant([ [ [ [1,0,1] ] ] ],dtype=tf.float32),perm=[0,1,3,2])

		#find max flow and scale
		flow = flow + 0.0000000001
		flowSq = flow*flow;
		flowMag = tf.sqrt(tf.reduce_sum(flowSq,reduction_indices=[3],keep_dims=True))
		maxMag = tf.reduce_max(flowMag,reduction_indices=[1,2],keep_dims=True)
		scaledFlow = flow/maxMag

		#calculate coefficients
		coef0 = tf.maximum(tf.nn.conv2d(scaledFlow,k0,[1,1,1,1],padding="SAME"),0)/2
		coef120 = tf.maximum(tf.nn.conv2d(scaledFlow,k120,[1,1,1,1],padding="SAME"),0)/2
		coef240 = tf.maximum(tf.nn.conv2d(scaledFlow,k240,[1,1,1,1],padding="SAME"),0)/2
		coef60 = tf.maximum(tf.nn.conv2d(scaledFlow,k60,[1,1,1,1],padding="SAME"),0)/2
		coef180 = tf.maximum(tf.nn.conv2d(scaledFlow,k180,[1,1,1,1],padding="SAME"),0)/2
		coef300 = tf.maximum(tf.nn.conv2d(scaledFlow,k300,[1,1,1,1],padding="SAME"),0)/2

		#combine color components
		comp0 = tf.nn.conv2d_transpose(coef0,k0c,tf.pack([flowShape[0],flowShape[1],flowShape[2],3]),[1,1,1,1],padding="SAME")
		comp120 = tf.nn.conv2d_transpose(coef120,k120c,tf.pack([flowShape[0],flowShape[1],flowShape[2],3]),[1,1,1,1],padding="SAME")
		comp240 = tf.nn.conv2d_transpose(coef240,k240c,tf.pack([flowShape[0],flowShape[1],flowShape[2],3]),[1,1,1,1],padding="SAME")
		comp60 = tf.nn.conv2d_transpose(coef60,k60c,tf.pack([flowShape[0],flowShape[1],flowShape[2],3]),[1,1,1,1],padding="SAME")
		comp180 = tf.nn.conv2d_transpose(coef180,k180c,tf.pack([flowShape[0],flowShape[1],flowShape[2],3]),[1,1,1,1],padding="SAME")
		comp300 = tf.nn.conv2d_transpose(coef300,k300c,tf.pack([flowShape[0],flowShape[1],flowShape[2],3]),[1,1,1,1],padding="SAME")
		return comp0 + comp120 + comp240 + comp60 + comp180 + comp300
