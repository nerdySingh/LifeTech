import numpy as np
import csv
import os
import glob

read_folder = 'processed_data'

def read_csv(fpath):
  f = open(fpath, 'rb')
  reader = csv.reader(f)
  ans = []
  for row in reader:
    row = np.array(row).astype(np.float)
    ans.append(row)
  return np.array(ans)

def write_csv(table, fname):
  f = open(fname, 'wb')
  writer = csv.writer(f)
  for row in table:
    writer.writerow(row)
  f.close()

if __name__ == '__main__':
  targ_paths = glob.glob(os.sep.join([read_folder, '*.csv']))
  all_data = []
  for targ_path in targ_paths:
    print 'Processing', targ_path
    all_data.append(read_csv(targ_path))
  all_data = np.vstack(all_data)
  write_csv(all_data, 'all_data.csv')
  k = 0