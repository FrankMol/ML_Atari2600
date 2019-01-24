import tempfile
import moviepy.editor as mpy
import numpy as np
import tensorflow as tf
import os

def convert_tensor_to_gif_summary(summ, fps=4):
    if isinstance(summ, bytes):
        summary_proto = tf.Summary()
        summary_proto.ParseFromString(summ)
        summ = summary_proto

    summary = tf.Summary()
    for value in summ.value:
        tag = value.tag
        images_arr = tf.make_ndarray(value.tensor)

        if len(images_arr.shape) == 5:
            # concatenate batch dimension horizontally
            images_arr = np.concatenate(list(images_arr), axis=-2)
        if len(images_arr.shape) != 4:
            raise ValueError('Tensors must be 4-D or 5-D for gif summary.')
        if images_arr.shape[-1] != 3:
            raise ValueError('Tensors must have 3 channels.')

        # encode sequence of images into gif string
        clip = mpy.ImageSequenceClip(list(images_arr), fps=fps)
        with tempfile.NamedTemporaryFile() as f:
            filename = f.name + '.gif'
        clip.write_gif(filename, verbose=False, progress_bar=False)
        with open(filename, 'rb') as f:
            encoded_image_string = f.read()

        image = tf.Summary.Image()
        image.height = images_arr.shape[-3]
        image.width = images_arr.shape[-2]
        image.colorspace = 3  # code for 'RGB'
        image.encoded_image_string = encoded_image_string
        summary.value.add(tag=tag, image=image)
    return summary

def py_encode_gif(im_thwc, tag, fps=4):
    """
    Given a 4D numpy tensor of images, encodes as a gif.
    """
    with tempfile.NamedTemporaryFile() as f: fname = f.name + '.gif'
    clip = mpy.ImageSequenceClip(list(im_thwc), fps=fps)
    clip.write_gif(fname, verbose=False, progress_bar=False)
    with open(fname, 'rb') as f: enc_gif = f.read()
    os.remove(fname)
    # create a tensorflow image summary protobuf:
    thwc = im_thwc.shape
    im_summ = tf.Summary.Image()
    im_summ.height = thwc[1]
    im_summ.width = thwc[2]
    im_summ.colorspace = 3 # fix to 3 == RGB
    im_summ.encoded_image_string = enc_gif
    # create a summary obj:
    summ = tf.Summary()
    summ.value.add(tag=tag, image=im_summ)
    summ_str = summ.SerializeToString()
    return summ_str

def write_gif_summ(sess, summ_writer, gif_frames):
    images = tf.convert_to_tensor(gif_frames)
    print(images.get_shape())
    print("here?????")
    exit(0)
    tensor_summ = tf.summary.tensor_summary('images_gif', images)
    tensor_value = sess.run(tensor_summ)
    # summ_writer.add_summary(convert_tensor_to_gif_summary(tensor_value), 0)
    summ_writer.add_summary(py_encode_gif(tensor_value), 0)
    summ_writer.flush()



# sess = tf.Session()
# summary_writer = tf.summary.FileWriter('logs/image_summary2', graph=tf.get_default_graph())
#
# images_shape = (1, 12, 64, 64, 3)  # batch, time, height, width, channels
# images = np.random.randint(256, size=images_shape).astype(np.uint8)
# images = tf.convert_to_tensor(images)
#
# tensor_summ = tf.summary.tensor_summary('images_gif', images)
# tensor_value = sess.run(tensor_summ)
# summary_writer.add_summary(convert_tensor_to_gif_summary(tensor_value), 0)
#
# summ = tf.summary.image("images", images[:, 0])  # first time-step only
# value = sess.run(summ)
# summary_writer.add_summary(value, 0)
#
# summary_writer.flush()

# sess = tf.Session()
# summ_writer = tf.summary.FileWriter('logs/image_summary2', graph=tf.get_default_graph())
# images_shape = (1, 12, 64, 64, 3)  # batch, time, height, width, channels
# images = np.random.randint(256, size=images_shape).astype(np.uint8)
#
# write_gif_summ(sess, summ_writer, images)