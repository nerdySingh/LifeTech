import numpy as np
from biosppy.signals import ecg
from matplotlib import pyplot as plt
import csv
import wfdb
import wolframalpha
import os
import glob

data_folder = 'data'
heartbeat_window_size = 311

def read_heartbeat_data(data_folder, data_name):
  targ_path = os.sep.join([data_folder, data_name])
  sig, fields = wfdb.rdsamp(targ_path)
  sample_rate = fields['fs']
  annsamp, anntype, subtype, chan, num, aux, annfs = wfdb.rdann(targ_path, 'atr')
  ecg_analysis = ecg.ecg(sig[:, 0], sample_rate, False)
  peak_loc_diffs = np.diff(ecg_analysis['rpeaks'])
  avg_peak_dur = np.mean(peak_loc_diffs)
  peak_diff_std = np.std(peak_loc_diffs)
  return (sig[:, 0], ecg_analysis['filtered'], anntype[1:], ecg_analysis['rpeaks'], avg_peak_dur, peak_diff_std)

def create_dataset(sig_filtered, peak_locs, window_size, labels):
  half_window = window_size / 2
  peak_waves = []
  truth_labels = []
  i = 0
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
    if labels[i] == 'N':
      truth_labels.append(0)
    else:
      truth_labels.append(1)
    i += 1
  peak_waves = np.array(peak_waves)
  truth_labels = np.array(truth_labels)
  return peak_waves, truth_labels

def get_mit_bih_dataset(dataset_name):
  sig, sig_filtered, labels, peak_locs, avg_peak_dur, peak_diff_std = read_heartbeat_data(data_folder, dataset_name)
  return create_dataset(sig_filtered, peak_locs, heartbeat_window_size, labels)

def write_to_csv(x, y, fname):
  f = open(fname, 'wb')
  writer = csv.writer(f)
  table = np.zeros([x.shape[0], x.shape[1] + 1])
  table[:, :-1] = x
  table[:, -1] = y
  for row in table:
    writer.writerow(row)
  f.close()

def process_dataset(dataset_name):
  x, y = get_mit_bih_dataset(dataset_name)
  write_to_csv(x, y, '.'.join([dataset_name, 'csv']))

if __name__ == '__main__':
  data_file_paths = glob.glob(os.sep.join([data_folder, '*.dat']))
  dataset_names = [name.split(os.sep)[-1].split('.')[0] for name in data_file_paths]
  for dataset_name in dataset_names:
    print 'Processing dataset', dataset_name
    process_dataset(dataset_name)