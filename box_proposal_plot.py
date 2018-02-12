#!/usr/bin/env python
__author__    = 'Danelle Cline'
__copyright__ = '2016'
__license__   = 'GPL v3'
__contact__   = 'dcline at mbari.org'
__doc__ = '''

Combine all model data into a single plot
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
plt.rcParams['font.size'] = 30
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 20
#plt.rcParams['xtick.labelsize'] = 8
#plt.rcParams['ytick.labelsize'] = 8
#plt.rcParams['legend.fontsize'] = 20
#plt.rcParams['figure.titlesize'] = 20
sys.path.append(os.path.join(os.path.dirname(__file__), 'tensorflow_models/research'))

import tensorflow as tf

arch_markers = {'Faster RCNN': 'o', 'SSD':'D', 'R-FCN': '*'}
fe_colors = {'Resnet 101':'Y', 'Inception V2':'B'}
sz_colors = {'950x540':'G', '300':'R', '600':'Y'}
sz_proposals = {'600':'k', '300':'m', '100':'c'}

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

def model_plot(all_model_index, model, ax):
  data = all_model_index.loc[model.name]
  m = '.'
  c = 'grey'
  if model.meta_arch in arch_markers.keys():
    m = arch_markers[model.meta_arch]
  if model.image_resolution in sz_colors.keys():
    c = sz_colors[model.image_resolution]
  if model.proposals in sz_proposals.keys():
    c = sz_proposals[model.proposals]

  ax.scatter(data.index, data.values, marker=m, color=c, s=40, label=model.meta_arch)


def main(_):

  search_path = os.path.join(os.getcwd(), 'models') + '/**/eval'
  all_dirs = glob.glob(search_path, recursive=True)
  df_eval = pd.DataFrame()
  all_models = []

  for d in all_dirs:

    # Grab all of the accuracy results for each model and put into Pandas dataframe
    event_acc = EventAccumulator(d)
    event_acc.Reload()
    # Show all tags in the log file
    print(event_acc.Tags())
    try:
      s = event_acc.Scalars('PASCAL/Precision/mAP@0.5IOU')
      df = pd.DataFrame(s)
      if df.empty:
        continue

      dir_name = d.split('eval')[0]
      model_name = dir_name.split('/')[-2]

      a = meta.ModelMetadata(model_name)
      all_models.append(a)

      time_start = df.wall_time[0]

      # convert wall time and value to rounded values
      df['wall_time'] = df['wall_time'].apply(wallToGPUTime, args=(time_start,))
      df['value'] = df['value'].apply(valueTomAP)

      # rename columns
      df.columns = ['GPU Time', 'step', 'Overall mAP']
      df['model'] = np.full(len(df), a.name)
      print(df)
      df_eval = df_eval.append(df)


    except Exception as ex:
      print(ex)
      continue

  # drop the step column as it's no longer needed
  df_eval = df_eval.drop(['step'], axis=1)

  all_model_index = df_eval.set_index(['model','GPU Time']).sort_index()

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
    ax1.set_xlabel('GPU Training Time (minutes)', fontsize=10)
    ax1.set_title('Mean Average Precision per Model', fontstyle='italic')
    markers = []
    names = []
    for name, marker in arch_markers.items():
      s = plt.Line2D((0, 1), (0, 0), color='grey', marker=marker, linestyle='')
      names.append(name)
      markers.append(s)
    ax1.legend(markers, names)
    inc = 30
    ax1.text(150, 35, r'Resolution', fontsize=8)
    for size, color in sz_colors.items():
      ax1.text(160, inc - 2, r'{0}'.format(size), fontsize=8)
      c = mpatches.Circle( (150, inc), 2, edgecolor='black', facecolor=color)
      ax1.add_patch(c)
      inc -= 10
    inc = 30
    ax1.text(240, 35, r'Box Proposals', fontsize=8)
    for size, color in sz_proposals.items():
      ax1.text(250, inc - 2, r'{0}'.format(size), fontsize=8)
      c = mpatches.Circle( (240, inc), 2, edgecolor='black', facecolor=color)
      ax1.add_patch(c)
      inc -= 10

    plt.savefig('mAP.png', format='png', bbox_inches='tight')
    plt.show()

  print('Done creating mAP.png')

if __name__ == '__main__':
  tf.app.run()