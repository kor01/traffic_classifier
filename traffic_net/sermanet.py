import tensorflow as tf


FLAGS = tf.flags.FLAGS


def apply_sermanet(
    batch_x, num_classes,
    is_train=True, reuse=None):

  def dropout(tensor):
    if is_train and FLAGS.dropout_rate < 1.0:
      tensor = tf.nn.dropout(tensor, FLAGS.dropout_rate)
    return tensor

  def l2_regularize(factor):
    if is_train and factor > 0:
      return tf.contrib.layers.l2_regularizer(factor)
    else:
      return None

  with tf.variable_scope('sermanet', reuse=reuse):
    conv_regularizer = l2_regularize(FLAGS.conv_decay)
    fc_regularizer = l2_regularize(FLAGS.fc_decay)
    # stage 1
    conv1 = tf.layers.conv2d(
      batch_x, filters=108, kernel_size=5,
      activation=tf.nn.relu,
      kernel_regularizer=conv_regularizer)
    pool1 = tf.layers.max_pooling2d(
      conv1, pool_size=2, strides=2)

    # stage 2
    conv2 = tf.layers.conv2d(
      pool1, filters=200, kernel_size=5,
      activation=tf.nn.relu,
      kernel_regularizer=conv_regularizer)

    pool2 = tf.layers.max_pooling2d(conv2, 2, 2)

    flatten1 = tf.contrib.layers.flatten(pool2)
    flatten2 = tf.contrib.layers.flatten(pool1)
    flatten = tf.concat([flatten1, flatten2], axis=-1)

    fc_1 = dropout(tf.layers.dense(
      flatten, 384, activation=tf.nn.relu,
      kernel_regularizer=fc_regularizer))

    fc_2 = dropout(tf.layers.dense(
      fc_1, 192, activation=tf.nn.relu,
      kernel_regularizer=fc_regularizer))

    logits = tf.layers.dense(fc_2, num_classes)
    return logits
