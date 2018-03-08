#!/usr/bin/env python
__author__    = 'Danelle Cline'
__copyright__ = '2018'
__license__   = 'GPL v3'
__contact__   = 'dcline at mbari.org'
__doc__ = '''

Splits collections into training/test sets
@var __date__: Date of last svn commit
@undocumented: __doc__ parser
@status: production
@license: GPL
'''
import utils
import os
import conf

if __name__ == '__main__':
    collections = ['M535455_500x500_by_group_0pad', 'M56_500x500_by_group_0pad']

    for c in collections:
        dir = os.path.join(conf.DATA_DIR, c)
        train_per = 0.5
        tests_per = 0.5
        utils.split(dir, train_per, tests_per)

    print('Done')




