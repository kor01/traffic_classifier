import tensorflow as tf

FLAGS = tf.flags.FLAGS


def loss_xent(logits, batch_y, num_classes):
  labels = tf.one_hot(batch_y, num_classes)
  xent = tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=labels)
  loss = tf.reduce_mean(xent, axis=0)
  xent = tf.reduce_sum(xent)
  return loss, xent


def eval_precision(images, labels, network, batch_size):

  total_size = tf.shape(images)[0]

  iter_size = tf.to_int32(
    tf.round(total_size / batch_size + 0.5))

  def cond(_, __, i):
    return i < iter_size

  def body(c, l, i):
    batch_x = images[i * batch_size: (i + 1) * batch_size]
    batch_y = labels[i * batch_size: (i + 1) * batch_size]

    logits = network(
      batch_x, FLAGS.num_classes, is_train=False, reuse=True)
    c += is_correct_one_batch(logits, batch_y)
    loss, xent = loss_xent(logits, batch_y, FLAGS.num_classes)
    l += xent
    i += 1
    return c, l, i

  is_correct, total_xent, _ = \
    tf.while_loop(cond, body, [0.0, 0.0, 0], back_prop=False)

  total_size = tf.to_float(total_size)
  return is_correct / total_size, total_xent / total_size


def is_correct_one_batch(logits, labels):

  predictions = tf.to_int32(tf.argmax(logits, axis=-1))
  equals = tf.equal(predictions, labels)
  equals = tf.to_int32(equals)
  non_zeros = tf.count_nonzero(equals, dtype=tf.float32)
  return non_zeros
