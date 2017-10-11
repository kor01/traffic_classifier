import numpy as np
from sklearn import metrics
from traffic_net.flags import *
from traffic_net.model import get_model
from traffic_net.dataset import normalize_image
from traffic_net.augment import augment_single_image

FLAGS = tf.flags.FLAGS

SINGLE_IMG_PH = None
NORM = None
AUGMENT_NORM = None
SESS = None
LOGITS = None
IMAGES = None
PROB = None


def create_cpu_sess():
  graph = tf.Graph()
  config = tf.ConfigProto(device_count={'GPU': 0})
  sess = tf.Session(config=config, graph=graph)
  return sess


def load_model(
    network=None, path=None,
    sess=None, reuse=None,
    set_global=True):

  path = path or FLAGS.save_path
  network = get_model(network)

  ckpt = tf.train.get_checkpoint_state(path)
  assert ckpt is not None, 'check point does not exists'
  path = ckpt.model_checkpoint_path

  if sess is None:
    sess = create_cpu_sess()
    graph = sess.graph
    reuse = False
  else:
    graph = sess.graph

  with graph.as_default():
    img = tf.placeholder(
      dtype=tf.uint8, shape=(None, 32, 32, 3))
    logits = network(
      normalize_image(img),
      FLAGS.num_classes, is_train=False, reuse=reuse)
    prob = tf.nn.softmax(logits)
    saver = tf.train.Saver(var_list=tf.trainable_variables())

  saver.restore(sess, path)
  if set_global:
    global SESS, LOGITS, IMAGES, PROB
    SESS, LOGITS, PROB, IMAGES = sess, logits, prob, img

  return img, logits, prob, sess


def load_preprocess(sess=None, set_global=True):
  if sess is None:
    sess = create_cpu_sess()
    graph = sess.graph
  else:
    graph = sess.graph

  with graph.as_default():
    img = tf.placeholder(
      dtype=tf.uint8, shape=(32, 32, 3))
    augment = augment_single_image(img)
    batch = tf.expand_dims(img, axis=0)
    augment_batch = tf.expand_dims(augment, axis=0)
    norm = normalize_image(batch)
    augment_norm = normalize_image(augment_batch)
    norm = tf.squeeze(norm, axis=0)
    norm = tf.squeeze(norm, axis=-1)
    augment_norm = tf.squeeze(augment_norm, axis=0)
    augment_norm = tf.squeeze(augment_norm, axis=-1)

  if set_global:
    global SESS, NORM, AUGMENT_NORM, SINGLE_IMG_PH
    SINGLE_IMG_PH = img
    SESS, NORM, AUGMENT_NORM = sess, norm, augment_norm

  return sess, img, norm, augment_norm


def evaluate(img):
  if len(img.shape) == 3:
    img = np.expand_dims(img, axis=0)
    ret = SESS.run(PROB, {IMAGES: img})
    ret = np.squeeze(ret, axis=0)
  else:
    ret = SESS.run(PROB, {IMAGES: img})
  return ret


def evaluate_normalize(img):
  ret = SESS.run(NORM, {SINGLE_IMG_PH: img})
  return ret


def evaluate_augment(img):
  ret = SESS.run(AUGMENT_NORM, {SINGLE_IMG_PH: img})
  return ret


def bad_cases(dataset):
  images = dataset['features']
  labels = dataset['labels']
  p = evaluate(images)
  pred = p.argmax(axis=-1)
  cases = np.not_equal(pred, labels)
  labels = labels[cases]
  p = p[cases, :]
  return labels, p, np.where(cases)[0]


def one_hot(labels):
  batch_size = len(labels)
  zeros = np.zeros((batch_size, FLAGS.num_classes),
                   dtype='int32')
  zeros[range(batch_size), labels] = 1
  return zeros


def confusion_matrix(labels, pred):
  return metrics.confusion_matrix(labels, pred)
