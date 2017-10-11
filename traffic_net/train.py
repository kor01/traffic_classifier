import os
import sys
import time
import threading
import datetime
from traffic_net.flags import *
from traffic_net.dataset import store_to_local_variable
from traffic_net.dataset import augment_dataset_queue
from traffic_net.dataset import normalize_image
from traffic_net.dataset import OutputTensors
from traffic_net.eval_funcs import eval_precision
from traffic_net.eval_funcs import loss_xent


def train_model(name, dataset, network, coord):

  sess = tf.Session()

  with sess.graph.as_default():

    if not FLAGS.augment:
      train_dataset = store_to_local_variable(
        sess, dataset.train, FLAGS.batch_size)
    else:
      train_dataset = augment_dataset_queue(
        sess, dataset.train, FLAGS.batch_size)

    batch_x = normalize_image(train_dataset.features)
    logits = network(batch_x, FLAGS.num_classes, is_train=True)

    loss, xent = loss_xent(
      logits, train_dataset.labels, FLAGS.num_classes)

    net = OutputTensors(
      logits=logits, pred=tf.nn.softmax(logits),
      loss=loss, total_xent=xent)

    reg_loss = sum(tf.get_collection(
      tf.GraphKeys.REGULARIZATION_LOSSES))
    train_op = tf.train.AdamOptimizer(
      learning_rate=FLAGS.adam_lr,
      epsilon=FLAGS.adam_epsilon).minimize(net.loss + reg_loss)

    valid_dataset = store_to_local_variable(
      sess, dataset.valid, 0, do_slice=False)

    precision, xent = eval_precision(
      normalize_image(valid_dataset.features),
      valid_dataset.labels, network, 512)

    test_dataset = store_to_local_variable(
      sess, dataset.test, 0, do_slice=False)
    test_precision, test_xent = eval_precision(
      normalize_image(test_dataset.features),
      test_dataset.labels, network, 512)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    if FLAGS.augment:
      tf.train.start_queue_runners(sess, coord)

    saver = tf.train.Saver(var_list=tf.global_variables())

    batch_size = FLAGS.batch_size
    num_examples = len(dataset.train['features'])

    total_loss = 0.0
    epoch_idx = 0
    current_idx = 0

    current_precision = None
    if not os.path.exists(FLAGS.save_path):
      os.mkdir(FLAGS.save_path)

    if not FLAGS.eval_to_stderr:
      test_record_file = os.path.join(
        FLAGS.save_path, 'evaluation.txt')
      test_record_file = open(test_record_file, 'w')
    else:
      test_record_file = sys.stderr

    while not coord.should_stop():

      if not FLAGS.augment:
        feed_dict = {train_dataset.start: current_idx * batch_size}
        _, loss = sess.run([train_op, net.loss], feed_dict=feed_dict)
      else:
        _, loss = sess.run([train_op, net.loss])

      total_loss += loss
      current_idx += 1
      if current_idx * batch_size > num_examples:
        timestamp = str(datetime.datetime.now().ctime())
        prec_val, xent_val = sess.run([precision, xent])

        print('[%s] [%d] epoch average loss [%f] validate '
              'precision [%f] xent [%f]'
              % (timestamp, epoch_idx,
                 total_loss / current_idx,
                 prec_val, xent_val), file=sys.stderr)

        sys.stderr.flush()
        current_idx, total_loss = 0, 0.0
        if current_precision is None or prec_val > current_precision:
          current_precision = prec_val
          prec_val, xent_val = sess.run([test_precision, test_xent])
          timestamp = str(datetime.datetime.now().ctime())

          print('[%s] [%d] epoch precision [%f] xent [%f]'
                % (timestamp, epoch_idx, prec_val, xent_val),
                file=test_record_file)
          test_record_file.flush()
          save_path = os.path.join(FLAGS.save_path, name)
          saver.save(sess, save_path, global_step=epoch_idx,
                     write_meta_graph=False)

        epoch_idx += 1
        if epoch_idx >= FLAGS.max_epoch:
          coord.request_stop()
          break

    if not FLAGS.eval_to_stderr:
      test_record_file.close()
    return sess


def start_train_loop(name, dataset, network, coord):
  thread = threading.Thread(
    target=train_model,
    args=(name, dataset, network, coord))
  thread.start()
  return thread


def wait_for_stop(threads, coord):
  try:
    while not coord.should_stop():
      time.sleep(1)
  except KeyboardInterrupt:
    coord.request_stop()
    coord.join(threads)
