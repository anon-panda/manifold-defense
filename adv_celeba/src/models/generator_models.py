import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from .generator_utils import *
import tensorflow.contrib.layers as tcl
slim = tf.contrib.slim
# Copied for WGAN
from lib_external import tflib as lib
import lib_external.tflib.ops.linear as linear
import lib_external.tflib.ops.conv2d as conv2d
import lib_external.tflib.ops.batchnorm as batchnorm
import lib_external.tflib.ops.deconv2d as deconv2d

'''
def celebA_generator(z, hidden_num=128, output_num=3, repeat_num=4, data_format='NCHW', reuse=False):
    with tf.variable_scope("G", reuse=reuse) as vs:
        num_output = int(np.prod([8, 8, hidden_num]))
        x = slim.fully_connected(z, num_output, activation_fn=None)
        x = reshape(x, 8, 8, hidden_num, data_format)       
        for idx in range(repeat_num):
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            if idx < repeat_num - 1:
                x = upscale(x, 2, data_format)
        out = slim.conv2d(x, 3, 3, 1, activation_fn=None, data_format=data_format)
    variables = tf.contrib.framework.get_variables(vs)
    out = tf.transpose(out, [0, 2, 3, 1])
    return out, variables
'''
def celebA_generator(z, n_z=100, DIM=64,reuse=False):
    with tf.variable_scope('G', reuse=reuse) as scope:
        z = tcl.fully_connected(z, 4*4*1024, activation_fn=tf.identity, scope='z')
        z = tf.reshape(z, [-1, 4, 4, 1024])
        z = tcl.batch_norm(z)
        z = tf.nn.relu(z)

        conv1 = tcl.convolution2d_transpose(z, 512, 5, 2, normalizer_fn=tcl.batch_norm, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv1')

        conv2 = tcl.convolution2d_transpose(conv1, 256, 5, 2, normalizer_fn=tcl.batch_norm, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv2')

        conv3 = tcl.convolution2d_transpose(conv2, 128, 5, 2, normalizer_fn=tcl.batch_norm, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv3')

        conv4 = tcl.convolution2d_transpose(conv3, 3, 5, 2, activation_fn=tf.nn.tanh, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv4')

    variables = tf.contrib.framework.get_variables(scope)
    variables = {var.op.name.replace("G/", "g_"): var for var in variables}
    variables['BatchNorm/beta'] = variables.pop('g_BatchNorm/beta')
    variables['BatchNorm/moving_mean'] = variables.pop('g_BatchNorm/moving_mean')
    variables['BatchNorm/moving_variance'] = variables.pop('g_BatchNorm/moving_variance')
    out = tf.reshape(conv4, [-1, 64,64,3])
    out = out*0.5 + 0.5
    return out, variables

def mnist_generator(z, n_z=32, DIM=32, reuse=False):
    with tf.variable_scope("G", reuse=reuse) as scope:
        output = linear.Linear('Generator.Input', n_z, 4*4*4*DIM, z)
        output = tf.nn.relu(output)
        output = tf.reshape(output, [-1, 4*DIM, 4, 4])

        output = deconv2d.Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output)
        output = tf.nn.relu(output)
        output = output[:,:,:7,:7]
        output = deconv2d.Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
        output = tf.nn.relu(output)

        output = deconv2d.Deconv2D('Generator.5', DIM, 1, 5, output)
        output = tf.nn.sigmoid(output)
    variables = tf.contrib.framework.get_variables(scope)
    variables = {var.op.name.replace("G/", ""): var for var in variables}
    out = tf.reshape(output, [-1, 28, 28, 1])
    return out, variables

def cifar_generator(z, n_z=128, DIM=64, reuse=False):
    with tf.variable_scope("G", reuse=reuse) as scope:
        output = linear.Linear('Generator.Input', n_z, 4*4*4*DIM, z)
        output = batchnorm.Batchnorm('Generator.BN1', [0], output)
        output = tf.nn.relu(output)
        output = tf.reshape(output, [-1, 4*DIM, 4, 4])

        output = deconv2d.Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output)
        output = batchnorm.Batchnorm('Generator.BN2', [0,2,3], output)
        output = tf.nn.relu(output)

        output = deconv2d.Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
        output = batchnorm.Batchnorm('Generator.BN3', [0,2,3], output)
        output = tf.nn.relu(output)

        output = deconv2d.Deconv2D('Generator.5', DIM, 3, 5, output)

        output = tf.tanh(output)

        variables = tf.contrib.framework.get_variables(scope)
        variables = {var.op.name.replace("G/", ""): var for var in variables}
        temp = tf.reshape(output, [-1, 3, 32, 32])
        return tf.transpose(temp,[0,2,3,1]), variables
