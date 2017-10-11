import os
import pickle
import numpy as np
from traffic_net.flags import *
from collections import namedtuple
from traffic_net.augment import augment_image


TrafficDataset = namedtuple(
  'TrafficDataset', ('train', 'valid', 'test'))

SlicedDataset = namedtuple('SlicedDataset', ('features', 'labels', 'start'))


OutputTensors = namedtuple(
  'OutputTensors', ('logits', 'pred', 'loss', 'total_xent'))


def normalize_image(batch_x):
  if FLAGS.normalize:
    batch_x = tf.image.rgb_to_grayscale(batch_x)
    batch_x = tf.to_float(batch_x)
    batch_x = (batch_x - 128.0) / 128.0
  else:
    batch_x = tf.to_float(batch_x)
  return batch_x


def shuffle_dataset(dataset):
  length = len(dataset['features'])
  perm = np.random.permutation(length)
  features = dataset['features'][perm]
  labels = dataset['labels'][perm]
  sizes = dataset['sizes'][perm]
  coords = dataset['coords'][perm]
  return {'features': features, 'labels': labels,
          'sizes': sizes, 'coords': coords}


def load_traffic_dataset(path=None):

  path = path or FLAGS.dataset

  def read(p):
    return open(p, 'rb')

  train = pickle.load(read(os.path.join(path, 'train.p')))
  valid = pickle.load(read(os.path.join(path, 'valid.p')))
  test = pickle.load(read(os.path.join(path, 'test.p')))

  return TrafficDataset(
    train=shuffle_dataset(train), valid=valid, test=test)


def store_to_local_variable(
    sess, dataset, batch_size, do_slice=True):

  with tf.device('/cpu:0'):
    features = tf.Variable(
      initial_value=dataset['features'],
      trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES],
      dtype=tf.float32)

    labels = tf.Variable(
      initial_value=dataset['labels'],
      trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES],
      dtype=tf.int32)

    if do_slice:
      start = tf.placeholder(dtype=tf.int32, shape=())
      features_slice = features[start: start + batch_size]
      labels_slice = labels[start: start + batch_size]

      sess.run([features.initializer, labels.initializer])

      return SlicedDataset(
        features=features_slice, labels=labels_slice, start=start)
    else:
      return SlicedDataset(
        features=features, labels=labels, start=None)


def augment_dataset_queue(
    sess, dataset, batch_size):
  with tf.device('/cpu:0'):
    # use random index to produce an example queue
    length = len(dataset['features'])
    dataset = store_to_local_variable(sess, dataset, 0, False)

    idx = tf.random_uniform(
      (), maxval=length - 1, dtype=tf.int32)
    index = idx % length
    feature = dataset.features[index]
    label = dataset.labels[index]

  x_batch, y_batch = augment_image(feature, label, batch_size)
  return SlicedDataset(
    features=x_batch, labels=y_batch, start=idx)
