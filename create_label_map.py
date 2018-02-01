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