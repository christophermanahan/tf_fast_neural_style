import tensorflow as tf

def load_dataset(filepath, batch_size):
	dataset = tf.contrib.data.TFRecordDataset(filepath)
	dataset = dataset.map(parser)
	dataset = dataset.batch(batch_size)
	iterator = dataset.make_initializable_iterator()
	return iterator

def parser(record):
	keys_to_features = {"image/encoded": tf.FixedLenFeature((), tf.string, default_value = "")}

	parsed = tf.parse_single_example(record, keys_to_features)

	# Perform additional preprocessing on the parsed data.
	image = tf.image.decode_jpeg(parsed["image/encoded"])
	image = tf.image.resize_images(image, [256, 256])
	vgg_mean = tf.constant([123.68, 116.779, 103.939], dtype = tf.float32, name = 'img_mean')
	vgg_mean = tf.reshape(vgg_mean, [1, 1, 3])
	image = tf.subtract(image, vgg_mean)
	return image

def load_style_image(filepath, h = 256, w = 256):
	filename = tf.train.string_input_producer([filepath])
	image_reader = tf.WholeFileReader()
	_, image_file = image_reader.read(filename)
	style_image = tf.image.decode_jpeg(image_file)
	style_image = tf.image.resize_images(style_image, [h, w])
	return tf.reshape(style_image, [1, h, w, 3])