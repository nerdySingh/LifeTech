try:
  import caffe
except:
  k=0
import numpy as np
import cPickle as pickle
import copy
from timeit import default_timer as timer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def extract_params(net):
  ip0 = [net.params['ip0'][0].data, net.params['ip0'][1].data]
  ip0_next = [net.params['ip0_next'][0].data, net.params['ip0_next'][1].data]
  ip1 = [net.params['ip1'][0].data, net.params['ip1'][1].data]
  params = [ip0, ip0_next, ip1]
  return params

if __name__ == '__main__':
  net = caffe.Net('net.prototxt', 'FINAL_64_32_2_relu_nn.caffemodel', caffe.TRAIN)
  params = extract_params(net)

  k = 0