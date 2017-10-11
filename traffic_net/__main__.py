import tensorflow as tf
from traffic_net.model import get_model
from traffic_net.train import wait_for_stop
from traffic_net.train import start_train_loop
from traffic_net.dataset import load_traffic_dataset


FLAGS = tf.flags.FLAGS


def main(_):
  dataset = load_traffic_dataset()
  coord = tf.train.Coordinator()
  network = get_model()
  train_thread = start_train_loop(
    FLAGS.arch, dataset, network, coord)
  wait_for_stop([train_thread], coord)


if __name__ == '__main__':
    tf.app.run()
