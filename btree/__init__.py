#!/usr/bin/env python3

import bisect
import logging
import math
import os
import random
import re
import shutil
import sys
import unittest
from pprint import pformat

log = logging.getLogger(__name__)


class DuplicateKey(Exception):
    pass


class InvalidTree(Exception):
    pass


def file_pad_lines(filename):
    """
    pad all lines in filename with whitespaces so that they are all the same length
    """
    filename_pad = filename + '.pad'
    max_length = 0

    with open(filename, 'r') as fh:
        for line in fh:
            length = len(line.strip())

            if length > max_length:
                max_length = length

    with open(filename_pad, 'w') as fh_pad:
        with open(filename, 'r') as fh:
            for line in fh:
                line = line.strip()
                length = len(line)
                spaces_to_add = max_length - length

                if spaces_to_add:
                    line = line + (' ' * spaces_to_add)
                fh_pad.write(line + '\n')

    shutil.move(filename_pad, filename)


def btree_disk_get_first_node(fh):
    fh.seek(0)

    line = next(fh)
    keys = line.decode('utf-8').rstrip().split(',')

    line = next(fh)
    nodes = line.decode('utf-8').rstrip().split(',')

    line = next(fh)
    values = line.decode('utf-8').rstrip().split(',')

    return (keys, nodes, values)


def btree_disk_get_line_width(fh):
    fh.seek(0)
    line = next(fh)
    return len(line)


def btree_disk_get(fh, key, first_node_keys=None, first_node_values=None, first_node_nodes=None, line_width=None):
    """
    Traverse a text file containing a disk_save() BTree and return the value
    for key or None if the key is not found
    """
    seek_count = 0

    # Read the first line to determine the line width and load the root node
    if line_width is None:
        fh.seek(0)
        line = next(fh)
        line_width = len(line)

    if first_node_keys is not None:
        key_index = bisect.bisect_left(first_node_keys, key)

        try:
            tmp_key = first_node_keys[key_index]
        except IndexError:
            tmp_key = None

        # We found the key!!! Return the value
        if tmp_key == key:
            return first_node_values[key_index]

        # This node is NOT a leaf, keep searching
        # seek to the line number of the next node
        node_index = bisect.bisect_right(first_node_keys, key)
        child_node_line_number = int(first_node_nodes[node_index])

        fh.seek(child_node_line_number * line_width)
        seek_count += 1

    else:
        fh.seek(0)

    depth = 0

    while True:
        # A node is written over three lines:
        # - keys
        # - nodes (children)
        # - values

        # ====
        # keys
        # ====
        # TODO will 'if key in keys_line' work if this is not decoded?
        keys_line = fh.read(line_width).decode('utf-8')

        # uncomment to print a summary of each node through the search
        #keys = keys_line.rstrip().split(',')
        #output = '| %d keys, depth %d |' % (len(keys), depth)
        #tmp_width = len(output) - 2
        #padding = '      ' * depth

        #print("%s+%s+\n%s%s\n%s+%s+" % (
        #    padding, '-' * tmp_width,
        #    padding, output,
        #    padding, '-' * tmp_width))

        # Only bother doing the rstrip, split, etc when 'key' is in the keys_line
        if key in keys_line:
            keys = keys_line.rstrip().split(',')
            key_index = bisect.bisect_left(keys, key)

            try:
                tmp_key = keys[key_index]
            except IndexError:
                tmp_key = None

        else:
            tmp_key = None
            keys = None

        # We found the key!!! Return the value
        if tmp_key == key:

            # ======
            # values
            # ======
            line = next(fh) # nodes
            line = next(fh) # values
            values = line.decode('utf-8').rstrip().split(',')
            #log.info("key %s is in the tree, took %d seeks" % (key, seek_count))
            return values[key_index]

        # =====
        # nodes
        # =====
        line = next(fh) # nodes
        nodes_line = line.decode('utf-8')

        # If nodes is empty then we are a leaf node and our search is done, return None
        if nodes_line.startswith(' '):
            #log.info("key %s is NOT in the tree, took %d seeks" % (key, seek_count))
            return None
        else:
            nodes = nodes_line.rstrip().split(',')

            if keys is None:
                keys = keys_line.rstrip().split(',')

            # This node is NOT a leaf, keep searching
            # seek to the line number of the next node
            node_index = bisect.bisect_right(keys, key)
            child_node_line_number = int(nodes[node_index])

            fh.seek(child_node_line_number * line_width)
            seek_count += 1

        depth += 1

    return None


class BTreeNode(object):

    def __init__(self, parent):
        self.parent = parent
        self.keys = []
        self.values = []
        self.nodes = []
        self.key_count = 0

        # Only used when saving BTree to disk
        self.line_number = None

        if self.parent:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0

        # At some point test 'from sortedcontainers import SortedList'
        # for 'keys' to see if it improves perf when bulk loading

    def __str__(self):
        return "BTreeNode[%s]" % ','.join(map(str, self.keys))

    def is_leaf(self):
        """
        Return True if we are a leaf
        """
        return not self.nodes

    def is_root(self):
        """
        Return True if we are the root
        """
        return self.parent == None

    def has_key(self, key):
        """
        Return True if we have 'key'
        """
        index = bisect.bisect_left(self.keys, key)
        try:
            return self.keys[index] == key
        except IndexError:
            return False

    def key_insert_position(self, key):
        """
        Return the list position in self.keys where 'key' should be inserted
        """
        return bisect.bisect_right(self.keys, key)

    def get(self, key):
        """
        Return the value for 'key' or None if we do not have this key
        """
        index = bisect.bisect_left(self.keys, key)

        try:
            if self.keys[index] == key:
                return self.values[index]
            else:
                return None
        except IndexError:
            return None

    def ascii(self):
        """
        Print this node as need for displaying in an ascii tree
        """
        # keys and a space between each for the pointers
        #output = '| %s |' % ' | | '.join(self.keys)

        # keys only
        output = '| %s |' % ' | '.join(map(str, self.keys))
        #output += "(leaf %s)" % self.is_leaf()
        #output += "(ln %s)" % self.line_number
        line_width = len(output) - 2
        padding = '      ' * self.depth

        return "%s+%s+\n%s%s\n%s+%s+" % (
            padding, '-' * line_width,
            padding, output,
            padding, '-' * line_width)

    def add(self, key, value, left_node=None, right_node=None):
        """
        Add key/value
        """
        i = self.key_insert_position(key)


        if left_node:
            assert right_node is not None, "left_node is %s but right_node is None" % left_node
            #log.info("keys       : %s" % pformat(self.keys))
            #log.info("values     : %s" % pformat(self.values))
            #log.info("nodes      : %s" % ' '.join([str(x) for x in self.nodes]))
            #log.info("insert     : %s" % i)
            #log.info("left_node  : %s" % left_node)
            #log.info("right_node : %s" % right_node)

            if self.nodes[i] != left_node:
                raise InvalidTree("%s nodes[%s] is %s but it should be left_node %s" % (self, i, self.nodes[i], left_node))

            self.nodes.insert(i+1, right_node)

        #log.info("%s: adding %s at index %d" % (self, key, i))
        self.keys.insert(i, key)
        self.values.insert(i, value)
        self.key_count += 1

    def delete(self, key, value=None):
        raise Exception("not implemented yet")

    def sanity_child_parent_relationship(self):
        """
        Verify all of our children list us as their parent
        """
        for child_node in self.nodes:
            if child_node.parent != self:
                raise InvalidTree("%s child_node %s does not list us as their parent, their parent is %s" %
                    (self, child_node, child_node.parent))

    def sanity_parent_child_relationship(self):
        """
        Verify our parent lists us as their child
        """
        for child_node in self.parent.nodes:
            if child_node == self:
                break
        else:
            raise InvalidTree("%s parent is %s but they do not list us as a child" % (self, self.parent))

    def sanity(self):

        # A root node
        if self.is_root():
            if len(self.nodes) != self.key_count + 1:
                raise InvalidTree("%s has %d nodes and %d keys" % (self, len(self.nodes), self.key_count))

            self.sanity_child_parent_relationship()

        # A leaf node
        elif self.is_leaf():
            if self.parent is None:
                raise InvalidTree("%s is a leaf node but parent is None" % self)

            self.sanity_parent_child_relationship()

        # A middle node
        else:
            if len(self.nodes) != self.key_count + 1:
                raise InvalidTree("%s has %d nodes and %d keys" % (self, len(self.nodes), self.key_count))

            self.sanity_child_parent_relationship()
            self.sanity_parent_child_relationship()


class BTree(object):

    def __init__(self, ORDER):
        self.ORDER = ORDER
        self.root = BTreeNode(None)
        self.sanity_check_enabled = False

    def __str__(self):
        return 'BTree'

    def _find_key_node(self, node, key):
        """
        Walk the tree and find the node with 'key'
        """
        if node.has_key(key):
            return node
        else:
            if node.nodes:
                node_index = bisect.bisect_right(node.keys, key)
                child_node = node.nodes[node_index]
                return self._find_key_node(child_node, key)

        return None

    def find_key_node(self, key):
        return self._find_key_node(self.root, key)

    def _find_key_insert_node(self, node, key):
        """
        Walk the tree and find the node where 'key' should be inserted.
        Raise an Exception if this key is already in the tree.  This isn't
        a must for B Trees but for my use case I should never see a
        duplicate key so for now I am going to sanity check for this.
        """
        if node.is_leaf():

            # For now we are not supporting duplicate keys
            if node.has_key(key):
                raise DuplicateKey("%s already has %s" % (node, key))

            return node

        else:
            node_index = bisect.bisect_right(node.keys, key)
            child_node = node.nodes[node_index]
            return self._find_key_insert_node(child_node, key)

        return None

    def find_key_insert_node(self, key):
        return self._find_key_insert_node(self.root, key)

    def get(self, key):
        """
        Return the value for key
        """
        key = str(key)
        node = self.find_key_node(key)

        if node:
            return node.get(key)
        else:
            return None

    def has_key(self, key):
        """
        Return True if 'key' lives in this tree
        """
        return self.get(key) is not None

    def is_overfull(self, node):
        """
        Return True if 'node' is over capacity
        """
        return node.key_count > self.ORDER

    def add(self, key, value):
        key = str(key)

        # Add the key to the appropriate Node
        node = self.find_key_insert_node(key)

        if not node:
            print(self.ascii())
            raise Exception("Could not find the node to take key %s" % key)

        node.add(key, value)

        # If the node was already full it is now overfull and must be split
        if self.is_overfull(node):
            self.split(node)

    def delete(self, key, value):
        raise Exception("deletes have not been implemented...yet")

    def _ascii(self, node):
        """
        Return an ascii string for the tree (recursive)
        """
        output = []
        output.append(node.ascii())

        for child_node in node.nodes:
            output.extend(self._ascii(child_node))

        return output

    def ascii(self):
        return '\n'.join(self._ascii(self.root))

    def set_depth(self, node, depth):
        """
        Starting with 'node', set the depth for all nodes (recursive)
        """
        node.depth = depth

        for child_node in node.nodes:
            self.set_depth(child_node, depth + 1)

    def sanity(self, node):
        """
        Sanity check all nodes (recursive)
        """
        node.sanity()

        for child_node in node.nodes:
            self.sanity(child_node)

    def split(self, left_node):
        """
        The majority of the complexity lives here
        """

        # If we are splitting the root the entry we move up will become the new root
        if left_node.is_root():
            new_root = BTreeNode(None)
        else:
            new_root = None

        move_up_index = int(math.floor(self.ORDER / 2))
        move_right_index = move_up_index + 1

        if new_root:
            right_node = BTreeNode(new_root)
        else:
            right_node = BTreeNode(left_node.parent)
            right_node.depth = left_node.depth

        move_up_key = left_node.keys[move_up_index]
        move_up_value = left_node.values[move_up_index]

        # log.info("%s: splitting %s, move_up_key %s" % (self, left_node, move_up_key))

        # Add the right half of 'node's keys/values to the right_node
        right_node.keys = left_node.keys[move_right_index:]
        right_node.values = left_node.values[move_right_index:]
        right_node.key_count = len(right_node.keys)

        # Remove the move_up_key/value and the right_node keys/values from the left_node
        left_node.keys = left_node.keys[:move_up_index]
        left_node.values = left_node.values[:move_up_index]
        left_node.key_count = len(left_node.keys)

        # Move node pointers
        right_node.nodes = left_node.nodes[move_right_index:]
        left_node.nodes = left_node.nodes[:move_right_index]

        for child_node in right_node.nodes:
            child_node.parent = right_node

        # Move up to a new root
        if new_root:
            new_root.add(move_up_key, move_up_value)
            new_root.nodes.append(left_node)
            new_root.nodes.append(right_node)

            left_node.parent = new_root

            for child_node in left_node.nodes:
                child_node.parent = left_node

            right_node.parent = new_root

            for child_node in right_node.nodes:
                child_node.parent = right_node

            self.root = new_root

            self.set_depth(self.root, 0)

            if self.sanity_check_enabled:
                self.sanity(self.root)

        # Move up
        else:
            left_node.parent.add(move_up_key, move_up_value, left_node, right_node)

            if self.sanity_check_enabled:
                self.sanity(self.root)

            if self.is_overfull(left_node.parent):
                self.split(left_node.parent)

    def _assign_line_number(self, node):
        node.line_number = self.line_number

        for child_node in node.nodes:
            self.line_number += 3
            self._assign_line_number(child_node)

    def assign_line_number(self):
        """
        Assign each node the line_number that will be used to store
        that node when saving the tree to a file. We will store one
        node per line. (recursive)
        """
        self.line_number = 0
        self._assign_line_number(self.root)

    def _stats(self, node):
        self.node_count += 1
        self.key_count += len(node.keys)

        if node.depth > self.max_depth:
            self.max_depth = node.depth

        # Repeat for all of node's children
        for child_node in node.nodes:
            self._stats(child_node)

    def stats(self):
        self.node_count = 0
        self.key_count = 0
        self.max_depth = 0
        self._stats(self.root)

        return (self.node_count, self.key_count, self.max_depth)

    def _disk_save(self, node, fh):
        """
        Write 'node' to fh (recursive). Each node will be written via three lines:
        - keys
        - child nodes (will be a list of the line numbers where the child nodes live in the file)
        - values
        """
        nodes_line_numbers = [x.line_number for x in node.nodes]

        fh.write(','.join(node.keys) + '\n')
        fh.write(','.join(map(str, nodes_line_numbers)) + '\n')
        fh.write(','.join(node.values) + '\n')

        # Repeat for all of node's children
        for child_node in node.nodes:
            self._disk_save(child_node, fh)

    def disk_save(self, filename):
        """
        Save the BTree to a text file.  Saving the tree to disk allows us to
        come back later and traverse the text file copy of the tree via
        filehandle seeks/reads without loading the tree into memory.
        """

        # Assign each node a line number
        self.assign_line_number()

        with open(filename, 'w') as fh:
            self._disk_save(self.root, fh)

        # Now pad the lines with whitespaces so they are all the same length
        file_pad_lines(filename)

    def disk_load(self, filename):
        """
        Load a BTree from a text file. If you only need to get the value of a
        single key or the tree is so large that you cannot hold it in memory
        it would be better to use btree_disk_get().
        """
        self.root = None
        line_number_to_nodes = {}
        line_number = 0

        # Create a BTreeNode object for each line in the file...there is one node per line
        with open(filename, 'rb') as fh:

            while True:
                try:
                    keys_line = next(fh)
                    nodes_line = next(fh)
                    values_line = next(fh)
                except StopIteration:
                    break

                keys_line = keys_line.decode('utf-8').rstrip()
                nodes_line = nodes_line.decode('utf-8').rstrip()
                values_line = values_line.decode('utf-8').rstrip()

                node = BTreeNode(None)
                node.keys = keys_line.split(',')

                if nodes_line:
                    node.nodes = [int(x) for x in nodes_line.split(',')]
                else:
                    node.nodes = []

                node.values = values_line.split(',')
                node.key_count = len(node.keys)
                node.line_number = line_number

                line_number_to_nodes[line_number] = node
                line_number += 3

        # Now convert all of the line_number references to the BTreeNode object that
        # was created for the node on that line_number
        for x in range(0, line_number, 3):
            node = line_number_to_nodes[x]

            for (index, line_number_child_node) in enumerate(node.nodes):
                child_node = line_number_to_nodes[line_number_child_node]
                child_node.parent = node

                node.nodes[index] = child_node

        # Populate self.root
        self.root = line_number_to_nodes[0]

        # Populate the depth for all nodes
        self.set_depth(self.root, 0)

        # sanity check everything
        if self.sanity_check_enabled:
            self.sanity(self.root)


class TestBTreeNode(unittest.TestCase):

    def setUp(self):
        self.node = BTreeNode(None)
        self.node.add('d', '4')
        self.node.add('b', '2')
        self.node.add('a', '1')

    def test_is_root(self):
        self.assertTrue(self.node.is_root())

    def test_is_leaf(self):
        self.assertTrue(self.node.is_leaf())

    def test_has_key(self):
        self.assertTrue(self.node.has_key('d'))
        self.assertFalse(self.node.has_key('z'))

    def test_key_insert_position(self):
        self.assertEquals(self.node.key_insert_position('c'), 2)
        self.assertEquals(self.node.key_insert_position('f'), 3)

    def test_get(self):
        self.assertEquals(self.node.get('d'), '4')
        self.assertEquals(self.node.get('z'), None)


class TestBTree(unittest.TestCase):

    def setUp(self):
        btree = BTree(2)
        btree.add(2, 'a')
        btree.add(4, 'b')
        btree.add(5, 'c')
        btree.add(6, 'd')
        btree.add(8, 'e')
        btree.add(10, 'f')
        btree.add(12, 'g')
        btree.add(14, 'h')
        btree.add(16, 'i')
        btree.add(18, 'j')
        btree.add(20, 'k')
        btree.add(22, 'l')
        btree.add(24, 'l')
        btree.add(26, 'l')
        btree.add(28, 'l')
        self.btree = btree

    def test_find_key_node(self):
        node = self.btree.find_key_node('26')
        self.assertEqual(node.keys, ['26',])

    def test_find_key_insert_node(self):
        node = self.btree.find_key_insert_node('44')
        self.assertEquals(node.keys, ['5',])

    def test_get(self):
        self.assertEquals(self.btree.get('2'), 'a')
        self.assertEquals(self.btree.get('8'), 'e')

    def test_has_key(self):
        self.assertTrue(self.btree.has_key('26'))
        self.assertFalse(self.btree.has_key('1234'))

    def test_is_overfull(self):
        node = self.btree.find_key_node('26')
        self.assertFalse(self.btree.is_overfull(node))

    def test_ascii(self):
        """
        Just test that this doesn't crash
        """
        self.btree.ascii()

    def test_stats(self):
        (node_count, key_count, max_depth) = self.btree.stats()
        self.assertEquals(node_count, 15)
        self.assertEquals(key_count, 15)
        self.assertEquals(max_depth, 3)

    def test_disk_save(self):
        self.btree.disk_save("foo.btree")

    def test_disk_load(self):
        self.btree.disk_save("foo.btree")

        tmp_tree = BTree(2)
        tmp_tree.disk_load("foo.btree")
        self.assertEqual(tmp_tree.get('2'), 'a')
        self.assertEqual(tmp_tree.get('8'), 'e')

    def test_disk_get(self):
        self.btree.disk_save("foo.btree")

        # Search for a key in a BTree that has been saved to disk
        with open('foo.btree', 'rb') as fh:
            self.assertEqual(btree_disk_get(fh, '2'), 'a')
            self.assertEqual(btree_disk_get(fh, '8'), 'e')

    def test_random(self):
        """
        Build a random tree (with sanity checking enabled) to test for crashes
        """
        tmp_tree = BTree(2)
        tmp_tree.sanity_check_enabled = True
        first = None
        last = None
        added = {}

        for x in range(40):
            key = int(random.randint(1,101))

            if key in added:
                continue

            value = int(random.randint(200,300))
            added[key] = value
            tmp_tree.add(key, value)

            if first is None:
                first = key
            last = key

        self.assertEqual(tmp_tree.get(first), added[first])
        self.assertEqual(tmp_tree.get(last), added[last])


if __name__ == '__main__':

    # setup logging
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)16s %(levelname)8s: %(message)s')
    log = logging.getLogger(__name__)

    # Color the errors and warnings in red
    logging.addLevelName(logging.ERROR, "\033[91m   %s\033[0m" % logging.getLevelName(logging.ERROR))
    logging.addLevelName(logging.WARNING, "\033[91m %s\033[0m" % logging.getLevelName(logging.WARNING))

    # run all unittests
    test_suites_to_run = []
    test_suites_to_run.append(TestBTreeNode)
    test_suites_to_run.append(TestBTree)

    for test_suite in test_suites_to_run:
        suite = unittest.TestLoader().loadTestsFromTestCase(test_suite)
        unittest.TextTestRunner(verbosity=2).run(suite)
