#!/usr/bin/env python
__author__    = 'Danelle Cline'
__copyright__ = '2018'
__license__   = 'GPL v3'
__contact__   = 'dcline at mbari.org'
__doc__ = '''

Creates a label map for usin TF from a cvs file
@var __date__: Date of last svn commit
@undocumented: __doc__ parser
@status: production
@license: GPL
'''
import pandas as pd
import os
data = '{0}/data/annotations/M56_categories_and_groups_v13.csv'.format(os.getcwd())
df = pd.read_csv(data, sep=',')
map_out = os.path.join(os.getcwd(), 'data', 'aesa_group_map.pbtxt')
index = 1
grouped = df.groupby('Group')
with open(map_out, 'w') as out:
  for name, group in grouped:
    str = 'item\n{{\n id:{0}\n  name: {1}\n}}\n'.format(index, name.upper())
    out.write(str)
    index += 1
