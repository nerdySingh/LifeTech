import numpy as np
import csv
import os
import glob
from headers.MySVM import MySVM
import matplotlib.pyplot as plt
import cPickle as pickle
from sklearn.metrics import accuracy_score

read_folder = 'processed_data'

def read_csv(fpath):
  f = open(fpath, 'rb')
  reader = csv.reader(f)
  ans = []
  i = 0
  for row in reader:
    row = np.array(row).astype(np.float)
    ans.append(row)
  f.close()
  return np.array(ans)

def train(dataset, lim, imp_features=None, show=True):
  np.random.shuffle(dataset)
  x = dataset[:lim, :-1]
  y = dataset[:lim, -1]
  if imp_features != None:
    x = x[:, imp_features]
  clf = MySVM()
  clf.fit(x, y)
  coeffs = np.abs(clf.classifier.coef_)[0]
  filter = (coeffs > coeffs.mean())
  idx = np.arange(filter.shape[0])
  chosen_idx = idx[filter]
  beat = x[9]
  beat_chosen = beat[filter]
  beat_excluded = beat[~filter]
  pickle.dump(coeffs, open('coeffs_' + str(lim) + '_datapoints_shuffled.pickle', 'wb'))
  pickle.dump(filter, open('filter_' + str(lim) + '_datapoints_shuffled.pickle', 'wb'))
  if show == True:
    plt.plot(idx, coeffs, label='Feature Importances')
    plt.plot(idx, [coeffs.mean()] * filter.shape[0], label='Mean of Feature Importances')
    plt.plot(idx[filter], beat_chosen, '^', label='Important pulse fraction for classification')
    plt.plot(idx[~filter], beat_excluded, '.', label='Pulse fraction not important for classification')
    plt.xlabel('Feature Index (Sample number)')
    plt.ylabel('SVM (Linear Kernel) coefficient')
    plt.title('Feature importance plots highlighting areas of a pulse important for classification over ' + str(x.shape[0]) + ' samples')
    plt.legend()
    plt.show()
  return clf, coeffs, filter #SAVE CLF

def test_clf(clf, dataset, lim, imp_features=None):
  x = dataset[:, :-1]
  y = dataset[:, -1]
  if imp_features != None:
    x = x[:, imp_features]
  x = x[:lim]
  y = y[:lim]
  idx = np.arange(y.shape[0])
  one_idx_filt = y==1
  zero_idx_filt = y==0
  one_idx = idx[one_idx_filt]
  zero_idx = idx[zero_idx_filt]
  np.random.shuffle(one_idx)
  np.random.shuffle(zero_idx)
  test_idx = np.hstack((one_idx, zero_idx[:one_idx.shape[0]]))
  x_test = x[test_idx]
  y_test = y[test_idx]
  y_pred = clf.predict(x_test)
  return accuracy_score(y_test, y_pred)


if __name__ == '__main__':
  #dataset = read_csv('all_data.csv')
  #pickle.dump(dataset, open('all_data.pickle', 'wb'))

  imp_features = [  0,   1,   2,   4,   7,  10,  14,  88,  92, 103, 110, 111, 114,
       117, 118, 121, 122, 125, 128, 129, 131, 132, 133, 135, 136, 137,
       138, 140, 141, 142, 144, 145, 146, 148, 149, 150, 152, 153, 154,
       156, 157, 158, 159, 161, 162, 165, 166, 169, 172, 173, 174, 176,
       177, 178, 180, 181, 182, 184, 185, 187, 188, 189, 190, 192, 193,
       194, 196, 197, 198, 199, 200, 201, 202, 204, 205, 207, 208, 209,
       211, 212, 213, 215, 216, 218, 219, 220, 222, 223, 224, 226, 227,
       230, 231, 234, 235, 238, 239, 242, 243, 246, 247, 250, 253, 254,
       255, 258, 262, 266, 285, 303, 305, 306, 308, 309, 310]

  dataset_size = 20000

  #imp_features = None
  dataset = pickle.load(open('all_data.pickle', 'rb'))
  np.random.shuffle(dataset)
  clf, coeffs, filter = train(dataset, int(.9*dataset_size), imp_features, False)
  acc = test_clf(clf, dataset, int(.1*dataset_size), imp_features)
  k = 0

  #for lim in range(500, dataset.shape[0], 500):
  #  coeffs, filter = get_feature_importance_plot(dataset, lim)