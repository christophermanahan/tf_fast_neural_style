#Import tensorflow, slim, and vgg net

import tensorflow as tf
import tensorflow.contrib.slim as slim
import vgg_net
import network
import utils
import argparse
import sys
 
class Trainer:
	def __init__(
		self,
		batch_size,
		epochs,
		content_weight,
		style_weight, 
		dataset_filepath,
		style_image_filepath,
		vgg_filepath,
		checkpoint_filepath,
		summary_filepath):
		#bind arguments
		self.batch_size = batch_size
		self.epochs = epochs
		self.content_weight = content_weight
		self.style_weight = style_weight
		self.dataset_filepath = dataset_filepath
		self.style_image_filepath = style_image_filepath
		self.checkpoint_filepath = checkpoint_filepath
		self.summary_filepath = summary_filepath
		#lazy load operations
		self._loss = None
		self._optimize = None
		#init placeholders
		self.image_batch = tf.placeholder(tf.float32, shape = [None, 256, 256, 3], name = 'image_batch')
		self.style_image = tf.placeholder(tf.float32, shape = [1, 256, 256, 3], name = 'style_image')
		#load cnn and vgg outputs
		self.cnn_outp = network.cnn(self.image_batch)
		self.vgg = vgg_net.VGG(vgg_filepath)
		with tf.name_scope('vgg_cnn'):
			self.vgg_cnn_outp = self.vgg.net(self.cnn_outp)
		with tf.name_scope('vgg_image_batch'):
			self.vgg_image_batch = self.vgg.net(self.image_batch)
		with tf.name_scope('vgg_style_image'):
			self.vgg_style_image = self.vgg.net(self.style_image)

	def train(self):
		#load dataset
		iterator = utils.load_dataset(self.dataset_filepath, self.batch_size) 
		batch = iterator.get_next()

		#load style image
		style_image = utils.load_style_image(self.style_image_filepath)

		#optimizer
		with tf.name_scope('total_loss'):
			total_loss = self.loss()
		tf.summary.scalar('total_loss', total_loss)

		with tf.name_scope('train'):
			update = self.optimize()

		#inits
		init_op = tf.global_variables_initializer()
		merged = tf.summary.merge_all()
		saver = tf.train.Saver()
		   
		with tf.Session() as sess:
			step = 0
			for epoch in range(self.epochs):
				#init variables
				sess.run(init_op)
				sess.run(iterator.initializer)
				summary_writer = tf.summary.FileWriter(self.summary_filepath, sess.graph)
				tf.train.start_queue_runners()
				while True:
					try:
						step += 1
						#update
						_, loss_val, summary = sess.run([update, total_loss, merged], feed_dict = {self.image_batch: batch.eval(), self.style_image: style_image.eval()})
						if step % 100 == 0:
							saver.save(sess, self.checkpoint_filepath + '/fast_neural_style_final.ckpt')
							summary_writer.add_summary(summary, step)
							print(step, ' loss: ', loss_val)
					except tf.errors.OutOfRangeError:
						print('epoch ' + epoch)
						break
		summary_writer.close()

	def optimize(self):
		if not self._optimize:
			self._optimize = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(self._loss)
		return self._optimize

	def loss(self):
		if not self._loss:
			content_loss = self.content_loss_calc()
			style_loss = self.style_loss_calc()
			self._loss = self.content_weight * content_loss + self.style_weight * style_loss
		return self._loss

	def content_loss_calc(self):
		image_batch_content = self.vgg_image_batch['conv4_2']
		cnn_content = self.vgg_cnn_outp['conv4_2']
		return self.content_layer_loss(cnn_content, image_batch_content)

	def content_layer_loss(self, outp, inp):
		b, h, w, d = [i.value for i in outp.get_shape()]
		M = h * w 
		N = d
		K = 1. / (2. * N**0.5 * M**0.5)
		loss = K * tf.reduce_sum(tf.pow((outp - inp), 2))
		return loss

	def style_loss_calc(self):
		style_loss = 0
		for l in [2, 3, 4, 5]:
			layer = 'conv' + str(l) + '_2'
			image_batch_content = self.vgg_image_batch[layer]
			cnn_content = self.vgg_cnn_outp[layer]
			style_loss += self.style_layer_loss(cnn_content, image_batch_content)
		return style_loss

	def style_layer_loss(self, outp, inp):
		b, h, w, d = [i.value for i in outp.get_shape()]
		M = h * w 
		N = d 
		A = self.gram_matrix(outp, M, N)
		G = self.gram_matrix(inp, M, N)
		loss = (1./(4 * N**2 * M**2)) * tf.reduce_sum(tf.pow((G - A), 2))
		return loss

	def gram_matrix(self, x, M, N):
		F = tf.reshape(x, (M, N))                   
		G = tf.matmul(tf.transpose(F), F)
		return G

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'Train fast style transfer network')
	parser.add_argument('-bs', '--batch_size', type = int, metavar = '', help = 'images per training batch')
	parser.add_argument('-e', '--epochs', type = int, metavar = '', help = 'runs through entire training set')
	parser.add_argument('-cw', '--content_weight', type = float, metavar = '', help = 'multiplier applied to content loss')
	parser.add_argument('-sw', '--style_weight', type = float, metavar = '', help = 'multiplier applied to style loss')
	parser.add_argument('-dfp', '--dataset_filepath', type = str, metavar = '', help = 'filepath to tfrecord dataset')
	parser.add_argument('-sifp', '--style_image_filepath', type = str, metavar = '', help = 'filepath to style image')
	parser.add_argument('-vfp', '--vgg_filepath', type = str, metavar = '', help = 'filepath to vgg19.mat checkpoint')
	parser.add_argument('-cfp', '--checkpoint_filepath', type = str, metavar = '', help = 'filepath to save network checkpoints during training')
	parser.add_argument('-sfp', '--summary_filepath', type = str, metavar = '', help = 'filepath to save network summaries during training')
	args = parser.parse_args()
	Trainer(args.batch_size, 
			args.epochs,
			args.content_weight,
			args.style_weight,
			args.dataset_filepath,
			args.style_image_filepath,
			args.vgg_filepath,
			args.checkpoint_filepath,
			args.summary_filepath).train()

# http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat
# -bs 1 -e 10 -cw 0.8 -sw 0.1 -dfp '/home/chris/tf/models/pascal_train.record' -sifp '/home/chris/tf/fast_neural_style/tree_painting.jpeg' -vfp '/home/chris/tf/checkpoints/imagenet-vgg-verydeep-19.mat' -cfp '/home/chris/tf/fast_neural_style/checkpoints' -sfp '/home/chris/tf/fast_neural_style/summaries'