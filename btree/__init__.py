#!/usr/bin/env python3

import bisect
import logging
import math
import random
from pprint import pformat

log = logging.getLogger(__name__)


class DuplicateKey(Exception):
    pass


class InvalidTree(Exception):
    pass


class BTreeNode(object):

    def __init__(self, parent):
        self.parent = parent
        self.keys = []
        self.values = []
        self.nodes = []
        self.key_count = 0

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

        TODO this should use bisect to find the entry
        """
        return key in self.keys

    def key_insert_position(self, key):
        """
        Return the list position in self.keys where 'key' should be inserted
        """
        return bisect.bisect_right(self.keys, key)

    def get(self, key):

        if self.has_key(key):
            index = bisect.bisect_left(self.keys, key)
            return self.values[index]
        else:
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

        log.info("%s: adding %s at index %d" % (self, key, i))
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

    def __str__(self):
        return 'BTree'

    def _find_key_node(self, node, key):
        """
        Walk the tree and find the node with 'key'

        TODO this is doing a DFS...make this smarter
        """
        if node.has_key(key):
            return node
        else:
            if node.nodes:
                node_index = bisect.bisect_right(node.keys, key)
                child_node = node.nodes[node_index]

                return self._find_key_node(child_node, key)
                #if self._find_key_node(child_node, key):
                #    return child_node

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

    def has_key(self, key):
        """
        Return True if 'key' lives in this tree
        """
        return self.find_key(key) is not None

    def get(self, key):
        """
        Return the value for key
        """
        node = self.find_key_node(key)

        if node:
            return node.get(key)
        else:
            return None

    def is_overfull(self, node):
        """
        Return True if 'node' is over capacity
        """
        return node.key_count > self.ORDER

    def add(self, key, value):

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

        log.info("%s: splitting %s, move_up_key %s" % (self, left_node, move_up_key))

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
            self.sanity(self.root)

        # Move up
        else:
            left_node.parent.add(move_up_key, move_up_value, left_node, right_node)
            self.sanity(self.root)

            #if left_node.parent.is_overfull():
            if self.is_overfull(left_node.parent):
                self.split(left_node.parent)


if __name__ == '__main__':

    # setup logging
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)16s %(levelname)8s: %(message)s')
    log = logging.getLogger(__name__)

    # Color the errors and warnings in red
    logging.addLevelName(logging.ERROR, "\033[91m   %s\033[0m" % logging.getLevelName(logging.ERROR))
    logging.addLevelName(logging.WARNING, "\033[91m %s\033[0m" % logging.getLevelName(logging.WARNING))

    btree = BTree(3)
    # 2, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24

    try:
        # Test the basics...use this site to verify results
        # https://www.cs.usfca.edu/~galles/visualization/BTree.html
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
        print(btree.ascii())

        print("key %s has value %s" % (8, btree.get(8)))
        #print("key %s has value %s" % (47, btree.get(47)))

        '''
        first = None
        numbers = []

        for x in range(20):
            number = int(random.randint(1,101))

            if number in numbers:
                continue

            numbers.append(number)
            btree.add(number, 'a')

            if first is None:
                first = number

        print(btree.ascii())

        print("key %s has value %s" % (first, btree.get(first)))
        print("added %s" % ' '.join(map(str, numbers)))
        '''


    except (DuplicateKey, InvalidTree):
        print(btree.ascii())
        raise
