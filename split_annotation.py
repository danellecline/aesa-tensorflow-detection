import utils
import os
import conf

if __name__ == '__main__':

    collections = [conf.COLLECTION_M56]

    for c in collections:
        dir = os.path.join(conf.DATA_DIR, c)
        train_per = 0.5
        tests_per = 0.5
        utils.split(dir, train_per, tests_per)

    print('Done')




