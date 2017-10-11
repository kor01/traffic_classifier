import tensorflow as tf

FLAGS = tf.flags.FLAGS


def apply_alexnet(
    batch_x, num_classes,
    is_train=True, reuse=None):

  def dropout(tensor):
    if is_train and FLAGS.dropout_rate < 1.0:
      tensor = tf.nn.dropout(tensor, FLAGS.dropout_rate)
    return tensor

  with tf.variable_scope('traffic_alexnet', reuse=reuse):

    conv1 = tf.layers.conv2d(
      batch_x, 64, 5, padding='SAME',
      activation=tf.nn.relu, name='conv1')

    pool1 = tf.layers.max_pooling2d(
      conv1, 3, 2, padding='SAME', name='pool1')

    norm1 = tf.nn.local_response_normalization(
      pool1, bias=1.0, alpha=1e-3 / 9.0, beta=0.75, name='norm1')

    conv2 = tf.layers.conv2d(
      norm1, 64, 5, 1, 'SAME', activation=tf.nn.relu, name='conv2')

    norm2 = tf.nn.local_response_normalization(
      conv2, bias=1.0, alpha=1e-3 / 9.0, beta=0.75, name='norm2')

    pool2 = tf.layers.max_pooling2d(norm2, 3, 2, 'SAME', name='pool2')

    flatten = tf.contrib.layers.flatten(pool2)

    hidden_dim = 384

    linear1 = dropout(tf.layers.dense(
      flatten, hidden_dim, tf.nn.relu, name='linear1'))

    linear2 = dropout(tf.layers.dense(
      linear1, hidden_dim / 2, tf.nn.relu, name='linear2'))

    logits = tf.layers.dense(
      linear2, num_classes, tf.nn.relu, name='linear3')

    return logits
