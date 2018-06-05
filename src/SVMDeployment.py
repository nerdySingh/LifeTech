import numpy as np
import csv
import os
import glob
from headers.MySVM import MySVM
import matplotlib.pyplot as plt
import cPickle as pickle
import json
import urllib
from biosppy.signals import ecg

fetch_link = 'http://ec2-34-208-177-88.us-west-2.compute.amazonaws.com/sen'
sample_rate = 380

ecg_max = 3.0500563847739977
ecg_min = -2.2907600761269711
ecg_range = ecg_max - ecg_min
ecg_mid = np.mean([ecg_min, ecg_max])

def get_data_from_link(link):
  f = urllib.urlopen(fetch_link)
  read_data = np.array(json.loads(''.join(f.readlines()))['data'])[:, 0]
  f.close()
  return read_data

def get_pickled_data(fname):
  return pickle.load(open(fname, 'rb'))

def plot_data(data_set, labels):
  i = 0
  for data in data_set:
    idx = np.arange(data.shape[0])
    plt.plot(idx, data, label=labels[i])
    i += 1
  plt.legend()
  plt.show()

def normalize_signal(sig):
  sig_max = sig.max()
  sig_min = sig.min()
  sig_range = sig_max - sig_min
  sig_upper = None
  if sig_min < 0:
    sig_upper =  sig + np.abs(sig_min)
  else:
    sig_upper = sig - sig_min
  sig_upper = ((sig_upper / sig_upper.max()) * ecg_range) - np.abs(ecg_min)
  return sig_upper
  
def get_waves(sig_filtered, peak_locs, window_size):
  half_window = window_size / 2
  peak_waves = []
  for peak_loc in peak_locs:
    lower_lim = peak_loc - half_window
    if lower_lim < 0:
      lower_lim = 0
    upper_lim = peak_loc + half_window
    if window_size % 2 != 0:
      upper_lim += 1
    if upper_lim > sig_filtered.shape[0]:
      upper_lim = sig_filtered.shape[0]
    sz = upper_lim - lower_lim
    margin = window_size - sz
    margin_l = margin / 2
    margin_r = margin - margin_l
    peak_wave = [0] * margin_l
    peak_wave += list(sig_filtered[lower_lim : upper_lim])
    peak_wave += [0] * margin_r
    peak_waves.append(np.array(peak_wave))
  return np.array(peak_waves)

def process_raw_signal(raw_heartbeat_signal):
  ecg_analysis = ecg.ecg(raw_heartbeat_signal, sample_rate)
  filtered_heartbeat_signal = ecg_analysis['filtered']
  peak_locs = ecg_analysis['rpeaks']
  avg_peak_dur = np.mean(np.diff(peak_locs))
  return filtered_heartbeat_signal, peak_locs

if __name__ == '__main__':
  #raw_heartbeat_signal = get_data_from_link(fetch_link)[-650:-50]
  #pickle.dump(raw_heartbeat_signal, open('read_data.pickle', 'wb'))
  raw_heartbeat_signal = get_pickled_data('read_data.pickle')
  filtered_heartbeat_signal, peak_locs = process_raw_signal(raw_heartbeat_signal)
  clf = get_pickled_data('svm_10000')

  plot_data([raw_heartbeat_signal, filtered_heartbeat_signal], ['raw', 'filtered'])

  peak_waves = get_waves(filtered_heartbeat_signal, peak_locs, 311)

  #for pk in peak_waves:

  y1 = clf.predict(peak_waves)

  plot_data([raw_heartbeat_signal, filtered_heartbeat_signal, normalized_filtered_heartbeat_signal], ['raw', 'filtered', 'filtered_normalized'])
  k=0