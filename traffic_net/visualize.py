import cv2
from traffic_net.flags import *
from traffic_net.infer import evaluate
import numpy as np
import matplotlib.pyplot as mplt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

NAMES = None


def plot_confusion_matrix(
    cm, plt=None, title='Confusion', classes=None,
    normalize=False, cmap=mplt.cm.Blues):

  plt = plt or mplt
  classes = classes or ['' for _ in range(FLAGS.num_classes)]

  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')


def analysis_confusion(cm):

  global NAMES
  if NAMES is None:
    NAMES = load_names()

  ret = ''
  for idx, row in enumerate(cm):
    non_zeros = list(np.where(row != 0)[0])
    if len(non_zeros) == 0:
      continue
    ret += (NAMES[idx] + ': ')
    counts = list(row[non_zeros])
    records = list(zip(non_zeros, counts))
    records = sorted(
      records, key=lambda x: x[1], reverse=True)
    for k, v in records:
      ret += '(%s, %d) ' % (NAMES[k], v)
    ret += '\n'
  return ret


def load_names(path=None):
  path = path or FLAGS.label_names
  names = []
  for i, l in enumerate(open(path)):
    if i == 0:
      continue
    row = l[:-1].split(',')
    names.append(row[1])
  return names


def plot_image_row(imgs, labels, pred=None):
  global NAMES
  if NAMES is None:
    NAMES = load_names()

  mplt.figure(figsize=(12, 12))
  gs = gridspec.GridSpec(
    nrows=1, ncols=len(imgs), left=0.1,
    bottom=0.25, right=0.95, top=0.95,
    wspace=0.05, hspace=0., width_ratios=[1] * len(imgs))
  for i in range(len(imgs)):
    name = NAMES[labels[i]]
    if len(name) > 15:
      name = name[:15] + '...'
    ax = mplt.subplot(gs[i])
    ax.imshow(imgs[i])
    red_patch = mpatches.Patch(color='red', label=name)
    if pred is not None:
      p_name = NAMES[pred[i]]
      if len(p_name) > 15:
        p_name = p_name[:15] + '...'
      blue_patch = mpatches.Patch(color='blue', label=p_name)
      handles = [red_patch, blue_patch]
    else:
      handles = [red_patch]

    ax.legend(handles=handles)
  mplt.show()


def random_pick(idx, num):
  sample = np.random.choice(
    idx, num, replace=False)
  return sample


def predict_class(image, target):
  pred = evaluate(image)
  label = pred.argmax(axis=-1)
  ret, score = NAMES[label], pred[label]
  target_cls = NAMES.index(target)
  target_score = pred[target_cls]
  return ret, score, target_score


def read_and_resize(img_path):
  image = cv2.imread(img_path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = cv2.resize(src=image, dsize=(32, 32))
  return image


def get_names():
  global NAMES
  if NAMES is None:
    NAMES = load_names()
  return NAMES


def plot_histogram(ax, name, labels):
  ax.hist(labels)
  mplt.title('%s Histgram' % name)
  mplt.xlabel("Value")
  mplt.ylabel("Frequency")


def sample_images_each_cls(features, labels, num_samples):
  images = [[] for _ in range(FLAGS.num_classes)]
  for img, label in zip(features, labels):
    images[label].append(img)

  samples = []

  for cls in images:
    sample = np.random.choice(
      len(cls), num_samples, replace=False)
    sample = [cls[x] for x in sample]
    samples.append(sample)

  return samples
