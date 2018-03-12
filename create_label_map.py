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

def correct(category):
  if 'Ophiuroid' in category:
    return 'Ophiuroidea'
  if 'Tunicate2' in category:
    return 'Tunicata2'
  if 'Polycheate1' in category:
    return 'Polychaeta1'
  return category
 
def convert(df, groupby):
  index = 1
  grouped = df.groupby(groupby)
  map_out = os.path.join(os.getcwd(), 'data', 'aesa_{0}_map.pbtxt'.format(groupby.lower()))
  items = {}
  for name, group in grouped:
    correct_name = name.strip().replace(" ", "")
    correct_key = correct(correct_name)
    correct_key = correct_key.upper()
    g = group.index[0]
    group_key = df.iloc[g].Group.upper()
    print('correct_name:{0} correct_key:{1} group_key:{2}'.format(correct_name, correct_key, group_key))
    if group_key == correct_key and groupby == 'Category' and correct_key != 'OPHIUROIDEA' and correct_key != 'MOLLUSCA': 
    	print('Skipping over category matching group{0}'.format(correct_key))
    	continue 

    if groupby == 'Group' and correct_key == 'UNKNOWN':
    	print('Skipping over unknown category')
    	continue 

    if correct_key not in items.keys():
    	items[correct_key] = 1

  with open(map_out, 'w') as out:
    for key, value in sorted(items.items()):
    	item = "item\n{{\n id:{0}\n  name: '{1}'\n}}\n".format(index, key)
    	out.write(item)
    	index += 1

if __name__ == '__main__':
  data = '{0}/data/annotations/M56_categories_and_groups_v13.csv'.format(os.getcwd())
  df = pd.read_csv(data, sep=',')
  convert(df, 'Group')
  convert(df, 'Category')
