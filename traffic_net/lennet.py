import tensorflow as tf

FLAGS = tf.flags.FLAGS


def apply_lennet(batch_x, num_classes,
                 is_train=True, reuse=None):

  def dropout(tensor):
    if is_train and FLAGS.dropout_rate < 1.0:
      tensor = tf.nn.dropout(tensor, FLAGS.dropout_rate)
    return tensor

  def l2_gen(rate):
    if is_train and rate > 0:
      return tf.contrib.layers.l2_regularizer(rate)
    return None

  with tf.variable_scope('traffic_lennet', reuse=reuse):
    conv_regularizer = l2_gen(FLAGS.conv_decay)
    fc_regularizer = l2_gen(FLAGS.fc_decay)
    conv1 = tf.layers.conv2d(
      batch_x, filters=6, kernel_size=5,
      activation=tf.nn.relu,
      kernel_regularizer=conv_regularizer)

    pool1 = tf.layers.max_pooling2d(
      conv1, pool_size=2, strides=2)
    conv2 = tf.layers.conv2d(
      pool1, filters=16, kernel_size=5,
      activation=tf.nn.relu,
      kernel_regularizer=conv_regularizer)

    pool2 = tf.layers.max_pooling2d(conv2, 2, 2)
    flatten = tf.contrib.layers.flatten(pool2)

    fc_1 = dropout(tf.layers.dense(
      flatten, 256, activation=tf.nn.relu,
      kernel_regularizer=fc_regularizer))

    fc_2 = dropout(tf.layers.dense(
      fc_1, 168, activation=tf.nn.relu,
      kernel_regularizer=fc_regularizer))

    logits = tf.layers.dense(fc_2, num_classes)
    return logits
