import tensorflow as tf
import network
import utils
import argparse
import sys

def style_image(
    style_image_filepath,
    outp_image_filepath,
    checkpoint_filepath,
    height = 256,
    width = 256):
    #load style image
    image_file = utils.load_style_image(style_image_filepath, height, width)

    #image to style placeholder
    image_to_style = tf.placeholder(tf.float32, shape = [1, height, width, 3], name = 'image_to_style')

    #cnn model
    cnn_outp = network.cnn(image_to_style)
    cnn_outp = tf.reshape(cnn_outp, [height, width, 3])
    cnn_outp = tf.image.convert_image_dtype(cnn_outp, tf.uint8, saturate = True)

    #define styled image ops and write file
    encoded_image = tf.image.encode_jpeg(cnn_outp)
    filepath = tf.constant(outp_image_filepath)
    styled_image = tf.write_file(filepath, encoded_image)

    #init
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        sess.run(init_op)
        threads = tf.train.start_queue_runners(coord = coord)

        checkpoint = tf.train.get_checkpoint_state(checkpoint_filepath)

        if checkpoint:
            saver.restore(sess, checkpoint.model_checkpoint_path)
        else:
            raise Exception("No checkpoint found...")

        image_to_write = sess.run(styled_image, feed_dict = {image_to_style: image_file.eval()})
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Output a style image from trained network')
    parser.add_argument('-sifp', '--style_image_filepath', type = str, metavar = '', help = 'filepath to style image')
    parser.add_argument('-oifp', '--outp_image_filepath', type = str, metavar = '', help = 'filepath to output image')
    parser.add_argument('-cfp', '--checkpoint_filepath', type = str, metavar = '', help = 'filepath to save network checkpoints during training')
    parser.add_argument('-ht', '--outp_image_height', type = int, metavar = '', help = 'image height in px of output image')
    parser.add_argument('-wt', '--outp_image_width', type = int, metavar = '', help = 'image width in px of output image')
    args = parser.parse_args()
    style_image(
        args.style_image_filepath,
        args.outp_image_filepath,
        args.checkpoint_filepath,
        args.outp_image_height,
        args.outp_image_width)

#python style_image.py -sifp '/home/chris/tf/fast_neural_style/tree_painting.jpeg' -oifp '/home/chris/tf/fast_neural_style_final/test.jpeg' -cfp '/home/chris/tf/fast_neural_style_final/checkpoints' -ht 660 -wt 660