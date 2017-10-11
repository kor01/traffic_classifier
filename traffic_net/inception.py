import tensorflow as tf


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_boolean(
  'inception_use_residue', True, 'use residue connection')


def two_layer_3_by_3(tensor, dim):
  tensor = tf.layers.conv2d(
    tensor, dim, 3, activation=tf.nn.relu,
    name='conv5A', padding='SAME')
  tensor = tf.layers.conv2d(
    tensor, dim, 3, name='conv5B', padding='SAME')
  return tensor


def inception_module(
    tensor, reduce_dim, dim, residue=False):

  def reduce_tensor(t, d=reduce_dim):
    return tf.layers.conv2d(
      t, d, 1, activation=tf.nn.relu)

  first = tf.layers.conv2d(
    reduce_tensor(tensor), dim, 1, padding='SAME', name='conv1')
  second = tf.layers.conv2d(
    reduce_tensor(tensor), dim, 3, padding='SAME', name='conv3')
  third = two_layer_3_by_3(reduce_tensor(tensor), dim)
  fourth = tf.layers.max_pooling2d(
      tensor, 3, 1, padding='SAME', name='max_pool')
  fourth = tf.layers.conv2d(fourth, dim, 1)

  ret = tf.concat([first, second, third, fourth], axis=-1)
  if residue:
    ret = ret + tensor
  return tf.nn.relu(ret)


def apply_inception(
    batch_x, num_classes,
    is_train=True, reuse=None):

  def dropout(tensor):
    if is_train and FLAGS.dropout_rate < 1.0:
      tensor = tf.nn.dropout(tensor, FLAGS.dropout_rate)
    return tensor

  with tf.variable_scope('inception', reuse=reuse), \
    tf.device('/gpu:0'):

      conv0 = tf.layers.conv2d(
        batch_x, 6, (5, 5),
        activation=tf.nn.relu, name='conv0')

      pool0 = tf.layers.max_pooling2d(conv0, 2, 2)

      conv1 = pool0
      first_depth, second_depth = 2, 2

      use_residue = FLAGS.inception_use_residue
      for i in range(first_depth):
        with tf.variable_scope('inception-%d' % i):
          residue = use_residue if i > 1 else False
          conv1 = inception_module(conv1, 48, 24, residue)

      pool1 = tf.layers.max_pooling2d(conv1, 2, 2)
      conv2 = pool1

      for i in range(second_depth):
        with tf.variable_scope('inception-%d' % (i + first_depth)):
          conv2 = inception_module(conv2, 48, 24, use_residue)

      pool2 = tf.layers.max_pooling2d(conv2, 2, 2)

      features = [tf.contrib.layers.flatten(x)
                  for x in (pool0, pool1, pool2)]

      features = tf.concat(features, axis=-1)
      fc_1 = dropout(tf.layers.dense(
        features, 384, activation=tf.nn.relu))

      fc_2 = dropout(tf.layers.dense(
        fc_1, 192, activation=tf.nn.relu))

      logits = tf.layers.dense(fc_2, num_classes)
      return logits
