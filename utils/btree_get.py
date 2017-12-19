#!/usr/bin/env python3

"""
Read a file containing a single "key:value" per line and
create a .btree file equivalent
"""

import logging
import os
import sys
from btree import disk_get


# setup logging
logging.basicConfig(level=logging.INFO,
                format='%(asctime)s %(filename)16s %(levelname)8s: %(message)s')
log = logging.getLogger(__name__)

# Color the errors and warnings in red
logging.addLevelName(logging.ERROR, "\033[91m   %s\033[0m" % logging.getLevelName(logging.ERROR))
logging.addLevelName(logging.WARNING, "\033[91m %s\033[0m" % logging.getLevelName(logging.WARNING))

btree_filename = sys.argv[1]
key = sys.argv[2]

if not btree_filename.endswith('.btree'):
    print("ERROR: %s does not end with .btree" % btree_filename)
    sys.exit(1)

if not os.path.isfile(btree_filename):
    print("ERROR: %s does not exist" % btree_filename)
    sys.exit(1)

with open(btree_filename, 'rb') as fh:
    value = disk_get(fh, key)

print(value)
