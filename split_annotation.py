import utils
import os
import conf

if __name__ == '__main__':

    collections = ['M535455_1000x1000_by_group', 'M56_1000x1000_by_group']

    for c in collections:
        dir = os.path.join(conf.DATA_DIR, c)
        train_per = 0.5
        tests_per = 0.5
        utils.split(dir, train_per, tests_per)

    print('Done')




