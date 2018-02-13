#!/usr/bin/env python
__author__    = 'Danelle Cline'
__copyright__ = '2016'
__license__   = 'GPL v3'
__contact__   = 'dcline at mbari.org'
__doc__ = '''

Combine accuracy by species for all models into a single plot
@var __date__: Date of last svn commit
@undocumented: __doc__ parser
@status: production
@license: GPL
'''

import matplotlib
matplotlib.use('Agg')
from pylab import *
import glob
import os
import matplotlib.patches as mpatches
import sys
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import model_metadata as meta
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

plt.style.use('ggplot')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12
sys.path.append(os.path.join(os.path.dirname(__file__), 'tensorflow_models/research'))

import tensorflow as tf

category = {'POLYCHAETA':'Polychaeta', 'CNIDARIA':'Cnidaria', 'OPHIUROIDEA': 'Ophiuroidea'}  
arch_markers = {'Faster RCNN': 'o', 'SSD':'D', 'R-FCN': '*'}
fe_colors = {'Resnet 101':'Y', 'Inception V2':'B'}
sz_colors = {'500x500':'G', '300':'R', '600':'Y'}
sz_proposals = {'600':'R', '300': 'G','100':'k', '50':'m'}

def aggregate(search_path, tempdir):
  all_files = glob.glob(search_path, recursive=True)
  for f in all_files:
    src = f
    dir, file = os.path.split(src)
    dst = '{0}/{1}'.format(tempdir, file)
    shutil.copy(src, dst)

def wallToGPUTime(x, zero_time):
  return round(int((x - zero_time)/60),0)

def valueTomAP(x):
  return round(int(x*100),0)

def smooth(data, smooth_weight):
  # 1st-order IIR low-pass filter to attenuate the higher-frequency components of the time-series.
  smooth_data = []
  last = 0
  numAccum = 0
  for nextVal in data:
    last = last * smooth_weight + (1 - smooth_weight) * nextVal;
    numAccum+=1
    debiasWeight = 1
    if smooth_weight != 1.0:
          debiasWeight = 1.0 - pow(smooth_weight, numAccum)
    smoothed = last / debiasWeight
    smooth_data.append(smoothed)
  return smooth_data

def model_plot(all_model_index, model, ax):
  data = all_model_index.loc[model.name]
  m = '.'
  c = 'grey'
  if model.meta_arch in arch_markers.keys():
    m = arch_markers[model.meta_arch]
  #if model.image_resolution in sz_colors.keys():
  #  c = sz_colors[model.image_resolution]
  if model.proposals in sz_proposals.keys():
    c = sz_proposals[model.proposals]

  ax.scatter(data.index, data.values, marker=m, color=c, s=5, label=model.meta_arch)
  x = data.index
  y = data.values
  '''The smoothing algorithm is a simple moving average, which, given a
     point p and a window w, replaces p with a simple average of the
     points in the [p - floor(w/2), p + floor(w/2)] range.''' 
  smoothing_weight=0.8
  y_smooth = smooth(data.values, smoothing_weight)
  ax.plot(x, y_smooth, color=c, label=model.meta_arch)
 
def extract_data(d, category, name):

  # Grab all of the accuracy results for each model and put into Pandas dataframe
  event_acc = EventAccumulator(d)
  event_acc.Reload()
  # Show all tags in the log file
  print(event_acc.Tags())

  s = event_acc.Scalars('PASCAL/PerformanceByCategory/AP@0.5IOU/{0}'.format(category))
  df = pd.DataFrame(s)
  if df.empty:
    raise('No {0} data available in {1}'.format(category, d))

  time_start = df.wall_time[0]

  # convert wall time and value to rounded values
  df['wall_time'] = df['wall_time'].apply(wallToGPUTime, args=(time_start,))
  df['value'] = df['value'].apply(valueTomAP)

  # rename columns
  df.columns = ['GPU Time', 'step', 'Overall mAP']
  df['model'] = np.full(len(df), name)
  #print(df)
  return df

def main(_):

  search_path = os.path.join(os.getcwd(), 'models') + '/**/eval'
  all_dirs = glob.glob(search_path, recursive=True)

  for cat_key, cat_value in category.items():
    df_eval = pd.DataFrame()
    all_models = []

    for d in all_dirs:
      dir_name = d.split('eval')[0]
      model_name = dir_name.split('/')[-2]
      a = meta.ModelMetadata(model_name, '500x500')
      try:
        df_new = extract_data(d, cat_key, a.name)
        df_eval = df_eval.append(df_new)
        all_models.append(a)
      except Exception as ex:
        print(ex)

    if not df_eval.empty:

      # drop the step column as it's no longer needed
      df_eval = df_eval.drop(['step'], axis=1)
      df_final = df_eval[df_eval['GPU Time'] < 200 ] 

      all_model_index = df_final.set_index(['model','GPU Time']).sort_index()

      with plt.style.context('ggplot'):
        # start a new figure - size is in inches
        fig = plt.figure(figsize=(6, 4), dpi=400)
        ax1 = plt.subplot(aspect='equal')
        ax1.set_xlim(0, 300)
        ax1.set_ylim(0, 100)

        for model in all_models:
          model_plot(all_model_index, model, ax1)

        ax1.set_ylim([0, 100])
        ax1.set_ylabel('mAP', fontsize=10)
        ax1.set_xlabel('GPU Time (minutes)', fontsize=10)
        ax1.set_title('{0} Mean Average Precision'.format(cat_value), fontstyle='italic')
        markers = []
        names = []
        for name, marker in arch_markers.items():
          s = plt.Line2D((0, 1), (0, 0), color='grey', marker=marker, linestyle='')
          names.append(name)
          markers.append(s)
        ax1.legend(markers, names, loc=1)
        inc = 40
        ax1.text(240, 45, r'Box Proposals', fontsize=8)
        for size, color in sz_proposals.items():
          ax1.text(250, inc - 2, r'{0}'.format(size), fontsize=8)
          c = mpatches.Circle( (240, inc), 2, edgecolor='black', facecolor=color)
          ax1.add_patch(c)
          inc -= 10

        out_file='mAP{0}.png'.format(cat_key)
        plt.savefig(out_file, format='png')
        print('Done creating {0}'.format(out_file))
        plt.show()


if __name__ == '__main__':
  tf.app.run()
