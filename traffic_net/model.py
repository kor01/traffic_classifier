from traffic_net.flags import *
from traffic_net.lennet import apply_lennet
from traffic_net.alexnet import apply_alexnet
from traffic_net.sermanet import apply_sermanet
from traffic_net.inception import apply_inception


NETS = {'alexnet': apply_alexnet,
        'lennet': apply_lennet,
        'sermanet': apply_sermanet,
        'inception': apply_inception}


def get_model(name=None):
  name = name or FLAGS.arch
  assert name in NETS, '[%s] not implemented' % name
  return NETS[name]
