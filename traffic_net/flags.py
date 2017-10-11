import tensorflow as tf

tf.flags.DEFINE_boolean(
  'normalize', True, 'normalize image to gray and std stat')
tf.flags.DEFINE_float('dropout_rate', 0.5, 'set to 1.0 to stop dropout')
tf.flags.DEFINE_integer('num_filters', 128, 'number of filters')
tf.flags.DEFINE_float('conv_decay', 4e-5,
                      'conv kernel l2 regularization factor')
tf.flags.DEFINE_float(
  'fc_decay', 4e-3, 'fc layer l2 regularization factor')

tf.flags.DEFINE_string('arch', 'lennet', 'model architecture')

tf.flags.DEFINE_float('transform_prob', 0.5,
                      'individual augment transform probability')

tf.flags.DEFINE_float('keep_prob', 0.5,
                      'probability to keep original data')

tf.flags.DEFINE_boolean(
  'shuffle', True, 'shuffle training dataset')

tf.flags.DEFINE_integer(
  'data_threads', 16, 'number of data preprocess thread')

tf.flags.DEFINE_integer(
  'min_queue', 10000, 'min examples in queue to ensure randomness')

tf.flags.DEFINE_string('dataset', None, 'traffic dataset dir')

tf.flags.DEFINE_string('save_path', None, 'save path')
tf.flags.DEFINE_integer('max_epoch', 250, 'max train epoch')
tf.flags.DEFINE_integer('num_classes', 43, 'number of classes')
tf.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.flags.DEFINE_float('adam_lr', 1e-3, 'learning rate in adam mode')
tf.flags.DEFINE_float('adam_epsilon', 1e-3, 'epsilon in adam')
tf.flags.DEFINE_boolean('augment', False, 'use data augmentation')
tf.flags.DEFINE_boolean('eval_to_stderr', True, 'output evaluate to stderr')
tf.flags.DEFINE_string('label_names', './signnames.csv', 'label names')

FLAGS = tf.flags.FLAGS
