try:
  import caffe
except:
  k=0
import numpy as np
import cPickle as pickle
import copy
from timeit import default_timer as timer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def eval_solver(solver, x_test, y_test):
  solver.net.blobs['data'].data[:] = x_test
  solver.net.forward()
  preds = solver.net.blobs['prob'].data.argmax(axis=1)
  prec, rec, fsc, _ = precision_recall_fscore_support(y_test, preds)
  acc = accuracy_score(y_test, preds)
  return acc, prec, rec, fsc


if __name__ == "__main__":
  #x_train, y_train = get_mnist_trainset()
  #x_test, y_test = get_mnist_testset()

  solver = caffe.SGDSolver('solver.prototxt')
  dataset = pickle.load(open('all_data.pickle', 'rb'))
  np.random.shuffle(dataset)
  caffe.set_mode_gpu()

  batch_size = 512
  labels = 2
  epochs = 10000
  learn_rate = .05
  reg = .01

  test_set_size = batch_size

  x_train = dataset[:-test_set_size, :-1]
  y_train = dataset[:-test_set_size, -1]
  x_test = dataset[-test_set_size:, :-1]
  y_test = dataset[-test_set_size:, -1]
  
  #solver.net.params['ip0'][0].data[:] = np.random.uniform(-.05, .05, solver.net.params['ip0'][0].data.shape)
  #solver.net.params['ip1'][0].data[:] = np.random.uniform(-.05, .05, solver.net.params['ip1'][0].data.shape)
  #solver.net.params['ip0'][0].data[:] = np.ones(solver.net.params['ip0'][0].data.shape)
  #solver.net.params['ip1'][0].data[:] = np.zeros(solver.net.params['ip1'][0].data.shape)

  #solver.net.params['ip0'][0].data[:] -= solver.net.params['ip0'][0].data.mean()
  #solver.net.params['ip1'][0].data[:] -= solver.net.params['ip1'][0].data.mean()
  #solver.net.params['ip0'][0].data[:] = np.ones(solver.net.params['ip0'][0].data.shape)
  #solver.net.params['ip1'][0].data[:] = np.zeros(solver.net.params['ip1'][0].data.shape)
  y_oneHot = np.zeros([y_train.shape[0], labels]) #Converting to one-hot encoding
  for i in xrange(y_oneHot.shape[0]):
    y_oneHot[i][y_train[i]] = 1
  train_limit = (x_train.shape[0] / batch_size) * batch_size
  x_train_all = copy.deepcopy(x_train)
  y_oneHot_all = copy.deepcopy(y_oneHot)
  y_train_all = copy.deepcopy(y_train)
  start_idx = 0
  epoch = 1
  iter = 1
  ts = 0.
  avg_iter_time = 0.
  cnt = 1
  #f = open('caffe_results.txt', 'wb')
  while epoch <= epochs: 
    if (start_idx + 1) * batch_size > train_limit:
        start_idx = 0
        epoch += 1
        iter = 1
    x_train = x_train_all[start_idx * batch_size:batch_size * (start_idx + 1)]
    y_oneHot = y_oneHot_all[start_idx * batch_size:batch_size * (start_idx + 1)]
    y_train = y_train_all[start_idx * batch_size:batch_size * (start_idx + 1)]

    st = timer()
    solver.net.blobs['data'].data[:] = x_train
    solver.net.blobs['labels'].data[:] = y_train
    l = solver.step(1)
    et = timer()
    dur = (et - st)
    ts += dur

    if cnt == 1:
      avg_iter_time = dur
    else:
      avg_iter_time += dur
      avg_iter_time /= 2.

    #solver.net.forward()
    preds = solver.net.blobs['prob'].data
    solver.net.blobs['prob'].diff[:] = (preds - y_oneHot) / 10.
    p = y_oneHot * preds
    my_loss = np.mean(-np.log(p[p>0]))
    wt_loss = reg * .5 * ((solver.net.params['ip0'][0].data**2).sum() + (solver.net.params['ip1'][0].data**2).sum())
    loss = my_loss + wt_loss
    pred_labels = preds.argmax(axis=1)
    acc, prec, rec, fsc = eval_solver(solver, x_test, y_test)
    print 'Batch', iter, 'Epoch =', epoch, 'Loss =', my_loss
    print 'Accuracy', acc
    print 'Precision', prec
    print 'Recall', rec
    print 'F-Score =', fsc
    #f.write(str(ts) + ' ' + str(my_loss) + '\n')
    start_idx += 1
    iter += 1

    if ts >= 1:
      k = cnt

    cnt += 1

f.write('Avg iter time = ' + str(avg_iter_time) + ' seconds\n')
f.close()