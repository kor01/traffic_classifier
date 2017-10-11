import numpy as np
import tensorflow as tf

FLAGS = tf.flags.FLAGS


def maybe_augment_image(image, fns, pred):

  origin = image
  for rd, fn in zip(pred[1:], fns):
    image = tf.cond(
      rd, lambda: fn(image), lambda: image)
  return tf.cond(pred[0], lambda: origin, lambda: image)


def augment_single_image(img):

  def brightness(image):
    return tf.image.random_brightness(image, max_delta=0.5)

  def contrast(image):
    return tf.image.random_contrast(image, 0.2, 1.2)

  def saturation(image):
    return tf.image.random_saturation(image, 1.0, 1.5)

  def hue(image):
    return tf.image.random_hue(image, max_delta=0.125)

  def rotation(image):
    angle = tf.random_uniform((), -np.pi / 15, np.pi / 15)
    return tf.contrib.image.rotate(image, angle)

  # fns = [brightness, contrast, saturation, hue, rotation]
  # fns = [brightness, contrast, saturation, hue]
  fns = [brightness, contrast, saturation, hue, rotation]
  rd = tf.random_uniform((len(fns) + 1,), 0, 1.0)

  pred = [rd[0] < FLAGS.keep_prob]
  for i in range(len(fns)):
    pred.append(rd[i + 1] < FLAGS.transform_prob)

  img = maybe_augment_image(img, fns, pred)

  return img


def augment_image(image, label, batch_size):
  with tf.device('/cpu:0'):

    image = augment_single_image(image)

  if FLAGS.shuffle:
    x_batch, y_batch = tf.train.shuffle_batch(
      [image, label], batch_size,
      capacity=10000 * batch_size + FLAGS.min_queue,
      num_threads=FLAGS.data_threads,
      min_after_dequeue=FLAGS.min_queue)

  else:
    x_batch, y_batch = tf.train.shuffle_batch(
      [image, label], batch_size,
      capacity=100 * batch_size,
      num_threads=FLAGS.data_threads, min_after_dequeue=0)

  return x_batch, y_batch
