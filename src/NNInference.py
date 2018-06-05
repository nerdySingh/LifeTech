import numpy as np
import cPickle as pickle
import copy
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class InferenceNet:
  ip0_w = None
  ip0_b = None
  ip0_next_w = None
  ip0_next_b = None
  ip1_w = None
  ip1_b = None
  outs = None

  def __init__(self, params):
    self.ip0_w = params[0][0]
    self.ip0_b = params[0][1]
    self.ip0_next_w = params[1][0]
    self.ip0_next_b = params[1][1]
    self.ip1_w = params[2][0]
    self.ip1_b = params[2][1]

  def relu(self, x):
    x[x<0] = 0
    return x

  def predict(self, x):
    fwd = self.relu(np.dot(x, self.ip0_w.T) + self.ip0_b)
    fwd = self.relu(np.dot(fwd, self.ip0_next_w.T) + self.ip0_next_b)
    fwd = np.dot(fwd, self.ip1_w.T) + self.ip1_b
    num = np.exp(fwd)
    den = np.sum(num, axis=1)
    softmax_out = np.array([num[i] / den[i] for i in xrange(den.shape[0])])
    outs = softmax_out.argmax(axis=1)
    return outs


if __name__ == '__main__':
  net_params = pickle.load(open('inference_net.pickle'))
  nn = InferenceNet(net_params)
  x, y = pickle.load(open('bal_dataset.pickle', 'rb'))
  preds = nn.predict(x)

  acc = accuracy_score(y, preds)
  prec, rec, fsc, _ = precision_recall_fscore_support(y, preds)



#def get_balanced_dataset(dataset, lim=500):
#  x = dataset[:lim, :-1]
#  y = dataset[:lim, -1]
#  idx = np.arange(y.shape[0])
#  one_idx_filt = y==1
#  zero_idx_filt = y==0
#  one_idx = idx[one_idx_filt]
#  zero_idx = idx[zero_idx_filt]
#  test_idx = np.hstack((one_idx, zero_idx[:one_idx.shape[0]]))
#  return x[test_idx], y[test_idx]