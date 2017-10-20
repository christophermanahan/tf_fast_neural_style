#Import tensorflow, slim, and vgg net

import tensorflow as tf
import tensorflow.contrib.slim as slim

# fast-neural-style-cnn
# 1) input placeholder of size 256x256x3
# 2) 40 pixel reflection padding to reduce edge artifacts and match size
# 3) 9x9 convolution with 64 filters to increase receptive field of subsequent layers
# 4) 5 rescropblocks
# 5) 2 deconv blocks
# 6) 9x9 convolution with 3 filters followed by tanh activation and 0-255 normalization

def cnn(image_batch):
    #define cnn model
    with tf.variable_scope('network'):
        x = tf.pad(image_batch, [[0, 0], [40, 40], [40, 40], [0, 0]], 'reflect')
        x = slim.conv2d(x, 64, [9, 9], activation_fn = None)
        x = slim.conv2d(x, 64, [3, 3], 2, activation_fn = None)
        x = slim.conv2d(x, 64, [3, 3], 2, activation_fn = None)
        for i in range(5):
            x = res_crop_block(x, i)
        for j in range(2):
            x = deconv_block(x, j)
        x = slim.conv2d(x, 3, [9, 9], activation_fn = None)
        x = tf.tanh(x)
        cnn_outp = tf.multiply(tf.add(x, 1), 127.5)
        return cnn_outp

def res_crop_block(inp, i):
    with tf.variable_scope('resBlock' + str(i)):
        x = slim.batch_norm(inp, activation_fn = None)
        x = slim.conv2d(x, 64, [3, 3], padding = 'valid', activation_fn = None)
        x = tf.nn.relu(x)
        x = slim.batch_norm(x, activation_fn = None)
        x = slim.conv2d(x, 64, [3, 3], padding = 'valid', activation_fn = None)
        x = tf.nn.relu(x)
        b, h, w, d = [i.value for i in inp.get_shape()]
        inp = tf.image.resize_image_with_crop_or_pad(inp, h - 4, w - 4)
        outp = tf.add(x, inp)
        return outp

def deconv_block(inp, i):
    with tf.variable_scope('deconvBlock' + str(i)):
        x = tf.layers.conv2d_transpose(inp, 64, [3, 3], 2, padding = 'same')
        x = slim.batch_norm(x, activation_fn = None)
        outp = tf.nn.relu(x)
        return outp
