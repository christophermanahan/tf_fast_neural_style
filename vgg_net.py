import tensorflow as tf
import scipy.io

class VGG:
    def __init__(self, filepath):
        vgg = scipy.io.loadmat(filepath)
        self.vgg_layers = vgg['layers']

    def _weights(self, layer, expected_layer_name):
        W = self.vgg_layers[0][layer][0][0][0][0][0]
        b = self.vgg_layers[0][layer][0][0][0][0][1]
        layer_name = self.vgg_layers[0][layer][0][0][-2]
        assert layer_name == expected_layer_name
        return W, b

    def _relu(self, conv2d_layer):
        return tf.nn.relu(conv2d_layer)

    def _conv2d(self, prev_layer, layer, layer_name):
        W, b = self._weights(layer, layer_name)
        W = tf.constant(W)
        b = tf.constant(b.reshape(-1))
        return tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b

    def _conv2d_relu(self, prev_layer, layer, layer_name):
        return self._relu(self._conv2d(prev_layer, layer, layer_name))

    def _avgpool(self, prev_layer):
        return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Constructs the graph model.
    def net(self, image_batch):
        graph = {}
        graph['input']    = image_batch
        graph['conv1_1']  = self._conv2d_relu(graph['input'], 0, 'conv1_1')
        graph['conv1_2']  = self._conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
        graph['avgpool1'] = self._avgpool(graph['conv1_2'])
        graph['conv2_1']  = self._conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
        graph['conv2_2']  = self._conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
        graph['avgpool2'] = self._avgpool(graph['conv2_2'])
        graph['conv3_1']  = self._conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
        graph['conv3_2']  = self._conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
        graph['conv3_3']  = self._conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
        graph['conv3_4']  = self._conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
        graph['avgpool3'] = self._avgpool(graph['conv3_4'])
        graph['conv4_1']  = self._conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
        graph['conv4_2']  = self._conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
        graph['conv4_3']  = self._conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
        graph['conv4_4']  = self._conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
        graph['avgpool4'] = self._avgpool(graph['conv4_4'])
        graph['conv5_1']  = self._conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
        graph['conv5_2']  = self._conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
        graph['conv5_3']  = self._conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
        graph['conv5_4']  = self._conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
        graph['avgpool5'] = self._avgpool(graph['conv5_4'])
        return graph
