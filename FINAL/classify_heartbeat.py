import numpy as np
import cPickle as pickle
from biosppy.signals import ecg
import urllib
from twilio.rest import TwilioRestClient
import json

fetch_link = 'http://ec2-34-208-177-88.us-west-2.compute.amazonaws.com/sen'
sample_rate = 380

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

def get_data_from_link(link):
  f = urllib.urlopen(fetch_link)
  read_data = np.array(json.loads(''.join(f.readlines()))['data'])[:, 0]
  f.close()
  return read_data

def plot_data(data_set, labels):
  i = 0
  for data in data_set:
    idx = np.arange(data.shape[0])
    plt.plot(idx, data, label=labels[i])
    i += 1
  plt.legend()
  plt.show()

def process_raw_signal(raw_heartbeat_signal):
  ecg_analysis = ecg.ecg(raw_heartbeat_signal, sample_rate, False)
  filtered_heartbeat_signal = ecg_analysis['filtered']
  peak_locs = ecg_analysis['rpeaks']
  avg_peak_dur = np.mean(np.diff(peak_locs))
  return filtered_heartbeat_signal, peak_locs

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


if __name__ == '__main__':
  net_params = pickle.load(open('inference_net.pickle'))
  nn = InferenceNet(net_params)
  msg_sent = False
  while True:
    try:
      raw_heartbeat_signal = get_data_from_link(fetch_link)[-2000:-50] / 1760.
      filtered_heartbeat_signal, peak_locs = process_raw_signal(raw_heartbeat_signal)
      peak_waves = get_waves(filtered_heartbeat_signal, peak_locs, 311)

      inference = np.array([])
      inference = nn.predict(peak_waves)
      print inference
      if np.sum(inference) > 0:
        print 'Potential Heart failure detected!'
        if msg_sent == False:
          msg_sent = True
          ACCOUNT_SID = "ACae8470efe21db56ba174088d5b64de3d"
          AUTH_TOKEN ="59e58b3349c30c4cb4318d5c61688931"
          client = TwilioRestClient(ACCOUNT_SID, AUTH_TOKEN)
          message = client.messages.create(to="+13128520877", from_="+12178661078",
                                             body="heart attack!")
      else:
        print 'Heart is fine! :D'
        msg_sent = False
    except:
      tmp = 0