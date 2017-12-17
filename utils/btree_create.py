#!/usr/bin/env python3

"""
Read a file containing a single "key:value" per line and
create a .btree file equivalent

btree_create.py FILENAME ORDER
"""

import logging
import os
import sys
from btree import BTree


# setup logging
logging.basicConfig(level=logging.INFO,
                format='%(asctime)s %(filename)16s %(levelname)8s: %(message)s')
log = logging.getLogger(__name__)

# Color the errors and warnings in red
logging.addLevelName(logging.ERROR, "\033[91m   %s\033[0m" % logging.getLevelName(logging.ERROR))
logging.addLevelName(logging.WARNING, "\033[91m %s\033[0m" % logging.getLevelName(logging.WARNING))

filename = sys.argv[1]
btree_filename = filename + '.btree'
order = sys.argv[2]

if not os.path.isfile(filename):
    print("ERROR: %s does not exist" % filename)
    sys.exit(1)

if os.path.isfile(btree_filename):
    print("ERROR: %s already exist" % btree_filename)
    sys.exit(1)

if order.isdigit():
    order = int(order)
else:
    print("ERROR: order is %s, it must be a int" % order)
    sys.exit(1)

btree = BTree(order)

with open(filename, 'r') as fh:
    line_number = 0

    log.info("reading file: start")
    for line in fh:
        (key, value) = line.strip().split(':')
        btree.add(key, value)

        line_number += 1

        if line_number % 100000 == 0:
            log.info("Added %d" % line_number)
    log.info("reading file: end")

    log.info("saving to disk: start")
    btree.disk_save(btree_filename)
    log.info("saving to disk: end")
