import math
import random

import tensorflow as tf

import mlable.shapes

# FILTER #######################################################################

class FilterTests(tf.test.TestCase):
    def setUp(self):
        super(FilterTests, self).setUp()
        self._cases = [
            {
                'args': {
                    'shape': range(4, 9, 3),
                    'axes': [-1],},
                'outputs': [1, 7],},
            {
                'args': {
                    'shape': (4, 4, 4, 4),
                    'axes': [],},
                'outputs': 4 * [1],},
            {
                'args': {
                    'shape': (None, 4, 16),
                    'axes': range(3),},
                'outputs': [0, 4, 16],},
            {
                'args': {
                    'shape': [0, 8],
                    'axes': tuple(range(10)),},
                'outputs': [0, 8],},
            {
                'args': {
                    'shape': tf.ones((2, 16)).shape,
                    'axes': [1.4],},
                'outputs': [1, 1],},
            {
                'args': {
                    'shape': tf.keras.Input((32, 32, 8)).shape,
                    'axes': [0],},
                'outputs': [0, 1, 1, 1],},
            {
                'args': {
                    'shape': tf.keras.Input((32, 32, 8)).shape,
                    'axes': [1, 2],},
                'outputs': [1, 32, 32, 1],},]

    def test_length_unchanged(self):
        for __c in self._cases:
            self.assertEqual(len(__c['args']['shape']), len(mlable.shapes.filter(**__c['args'])))

    def test_unchanged_when_all_axes_included(self):
        for __c in self._cases:
            __s = __c['args']['shape']
            self.assertEqual(mlable.shapes.normalize(__s), mlable.shapes.filter(__s, axes=range(len(__s))))

    def test_only_ones_when_no_axes_included(self):
        for __c in self._cases:
            __s = __c['args']['shape']
            self.assertEqual(len(__s) * [1], mlable.shapes.filter(__s, axes=[]))

    def test_specific_cases(self):
        for __c in self._cases:
            self.assertEqual(__c['outputs'], mlable.shapes.filter(**__c['args']))

# NORM #########################################################################

class NormalizeTests(tf.test.TestCase):
    def setUp(self):
        super(NormalizeTests, self).setUp()
        self._cases = [
            {
                'args': {'shape': range(4, 9, 3),},
                'outputs': [4, 7],},
            {
                'args': {'shape': (None, 4, 16),},
                'outputs': [0, 4, 16],},
            {
                'args': {'shape': [2, -1],},
                'outputs': [2, -1],},
            {
                'args': {'shape': tf.ones((2, 16)).shape,},
                'outputs': [2, 16],},
            {
                'args': {'shape': tf.keras.Input((32, 32, 8)).shape,},
                'outputs': [0, 32, 32, 8],},]

    def test_length_unchanged(self):
        for __c in self._cases:
            self.assertEqual(len(__c['args']['shape']), len(mlable.shapes.normalize(**__c['args'])))

    def test_all_dims_are_integer(self):
        for __c in self._cases:
            __s = mlable.shapes.normalize(**__c['args'])
            assert all(isinstance(__d, int) for __d in __s)

    def test_specific_cases(self):
        for __c in self._cases:
            self.assertEqual(__c['outputs'], mlable.shapes.normalize(**__c['args']))

# SYMBOLIC #####################################################################

class SymbolicTests(tf.test.TestCase):
    def setUp(self):
        super(SymbolicTests, self).setUp()
        self._cases = [
            {
                'args': {'shape': range(4, 9, 3),},
                'outputs': [4, 7],},
            {
                'args': {'shape': (None, 4, 16),},
                'outputs': [None, 4, 16],},
            {
                'args': {'shape': [0, 2, -1],},
                'outputs': [None, 2, -1],},
            {
                'args': {'shape': tf.ones((2, 16)).shape,},
                'outputs': [2, 16],},
            {
                'args': {'shape': tf.keras.Input((32, 32, 8)).shape,},
                'outputs': [None, 32, 32, 8],},]

    def test_length_unchanged(self):
        for __c in self._cases:
            self.assertEqual(len(__c['args']['shape']), len(mlable.shapes.symbolic(**__c['args'])))

    def test_specific_cases(self):
        for __c in self._cases:
            self.assertEqual(__c['outputs'], mlable.shapes.symbolic(**__c['args']))

# DIVIDE #######################################################################

class DivideTests(tf.test.TestCase):
    def setUp(self):
        super(DivideTests, self).setUp()
        self._cases = [
            {
                'args': {
                    'shape': range(4, 9),
                    'axis': -1,
                    'factor': 4,
                    'insert': True,
                    'right': True,},
                'outputs': [4, 5, 6, 7, 2, 4],},
            {
                'args': {
                    'shape': (4, 4, 4, 4),
                    'axis': 1,
                    'factor': 2,
                    'insert': False,
                    'right': True,},
                'outputs': [4, 2, 8, 4],},
            {
                'args': {
                    'shape': (None, 4, 16),
                    'axis': -1,
                    'factor': 8,
                    'insert': True,
                    'right': False,},
                'outputs': [0, 4, 8, 2],},
            {
                'args': {
                    'shape': [-1, 8],
                    'axis': -1,
                    'factor': 4,
                    'insert': False,
                    'right': False,},
                'outputs': [-1, 2],},
            {
                'args': {
                    'shape': tf.ones((2, 16)).shape,
                    'axis': -1,
                    'factor': 4,
                    'insert': True,
                    'right': True,},
                'outputs': [2, 4, 4],},
            {
                'args': {
                    'shape': tf.keras.Input((32, 32, 8)).shape,
                    'axis': -1,
                    'factor': 4,
                    'insert': True,
                    'right': False,},
                'outputs': [0, 32, 32, 4, 2],},
            {
                'args': {
                    'shape': tf.keras.Input((32, 32, 8)).shape,
                    'axis': 0,
                    'factor': 1,
                    'insert': True,
                    'right': True,},
                'outputs': [0, 1, 32, 32, 8],},]

    def test_rank_change_with_and_without_insertion(self):
        for __c in self._cases:
            __s = __c['args']['shape']
            __i = mlable.shapes.divide(__s, axis=0, factor=1, insert=True, right=True)
            __u = mlable.shapes.divide(__s, axis=0, factor=1, insert=False, right=True)
            self.assertEqual(len(__s) + 1, len(__i))
            self.assertEqual(len(__s), len(__u))

    def test_factor_1_equivalent_expand_dims(self):
        for __c in self._cases:
            __s = __c['args']['shape']
            __i = mlable.shapes.divide(__s, axis=0, factor=1, insert=True, right=True)
            __u = mlable.shapes.divide(__s, axis=0, factor=1, insert=False, right=True)
            self.assertEqual(len(__s) + 1, len(__i))
            self.assertEqual(mlable.shapes.normalize(__s), __u)

    def test_specific_cases(self):
        for __c in self._cases:
            self.assertEqual(__c['outputs'], mlable.shapes.divide(**__c['args']))

    def test_divide_merge_reciprocity(self):
        for __c in self._cases:
            __s = list(__c['args']['shape'])
            __d = mlable.shapes.divide(shape=__s, axis=-1, factor=4, insert=True, right=True)
            __dm = mlable.shapes.merge(shape=__d, axis=-1, right=False)
            __m = mlable.shapes.merge(shape=__s, axis=-2, right=True)
            __md = mlable.shapes.divide(shape=__m, axis=-1, factor=__s[-1], insert=True, right=True)
            self.assertEqual(mlable.shapes.normalize(__s), __dm)
            self.assertEqual(mlable.shapes.normalize(__s), __md)

# MERGE ########################################################################

class MergeTests(tf.test.TestCase):
    def setUp(self):
        super(MergeTests, self).setUp()
        self._cases = [
            {
                'args': {
                    'shape': range(4, 9),
                    'axis': -1,
                    'right': False,},
                'outputs': [4, 5, 6, 56],},
            {
                'args': {
                    'shape': (4, 4, 4, 4),
                    'axis': 1,
                    'right': True,},
                'outputs': [4, 16, 4],},
            {
                'args': {
                    'shape': (None, 4, 16),
                    'axis': 0,
                    'right': True,},
                'outputs': [0, 16],},
            {
                'args': {
                    'shape': [-1, 8],
                    'axis': -1,
                    'right': False,},
                'outputs': [-1],},
            {
                'args': {
                    'shape': tf.ones((2, 16)).shape,
                    'axis': -1,
                    'right': True,},
                'outputs': [32],},
            {
                'args': {
                    'shape': tf.keras.Input((32, 32, 8)).shape,
                    'axis': 2,
                    'right': False,},
                'outputs': [0, 1024, 8],},
            {
                'args': {
                    'shape': tf.keras.Input((32, 32, 8)).shape,
                    'axis': 1,
                    'right': True,},
                'outputs': [0, 1024, 8],},]

    def test_rank_lowered_by_one(self):
        for __c in self._cases:
            __s = __c['args']['shape']
            __u = mlable.shapes.merge(__s, axis=0, right=True)
            self.assertEqual(len(__s) - 1, len(__u))

    def test_specific_cases(self):
        for __c in self._cases:
            self.assertEqual(__c['outputs'], mlable.shapes.merge(**__c['args']))

# SWAP #########################################################################

class SwapTests(tf.test.TestCase):
    def setUp(self):
        super(SwapTests, self).setUp()
        self._cases = [
            {
                'args': {
                    'shape': range(4, 9),
                    'left': 1,
                    'right': 2,},
                'outputs': [4, 6, 5, 7, 8],},
            {
                'args': {
                    'shape': (None, 4, 16),
                    'left': 1,
                    'right': 1,},
                'outputs': [0, 4, 16],},
            {
                'args': {
                    'shape': [-1, 8],
                    'left': 10,
                    'right': 11,},
                'outputs': [8, -1],},
            {
                'args': {
                    'shape': tf.ones((2, 16)).shape,
                    'left': 1,
                    'right': 0,},
                'outputs': [16, 2],},
            {
                'args': {
                    'shape': tf.keras.Input((32, 32, 8)).shape,
                    'left': -2,
                    'right': -1,},
                'outputs': [0, 32, 8, 32],},
            {
                'args': {
                    'shape': tf.keras.Input((32, 32, 8)).shape,
                    'left': -10,
                    'right': 0,},
                'outputs': [32, 32, 0, 8],},]

    def test_rank_unchanged(self):
        for __c in self._cases:
            __s = __c['args']['shape']
            __u = mlable.shapes.swap(__s, left=0, right=-1)
            self.assertEqual(len(__s), len(__u))

    def test_unchanged_when_left_equals_right(self):
        for __c in self._cases:
            __s = __c['args']['shape']
            __u = mlable.shapes.swap(__s, left=0, right=0)
            self.assertEqual(mlable.shapes.normalize(__s), __u)

    def test_specific_cases(self):
        for __c in self._cases:
            self.assertEqual(__c['outputs'], mlable.shapes.swap(**__c['args']))

# MOVE #########################################################################

class MoveTests(tf.test.TestCase):
    def setUp(self):
        super(MoveTests, self).setUp()
        self._cases = [
            {
                'args': {
                    'shape': range(4, 9),
                    'before': 1,
                    'after': 2,},
                'outputs': [4, 6, 5, 7, 8],},
            {
                'args': {
                    'shape': range(8),
                    'before': 1,
                    'after': 4,},
                'outputs': [0, 2, 3, 4, 1, 5, 6, 7],},
            {
                'args': {
                    'shape': (None, 4, 16),
                    'before': 1,
                    'after': 1,},
                'outputs': [0, 4, 16],},
            {
                'args': {
                    'shape': [-1, 8],
                    'before': 10,
                    'after': 11,},
                'outputs': [8, -1],},
            {
                'args': {
                    'shape': tf.ones((2, 16)).shape,
                    'before': 1,
                    'after': 0,},
                'outputs': [16, 2],},
            {
                'args': {
                    'shape': range(4),
                    'before': -1,
                    'after': 1,},
                'outputs': [0, 3, 1, 2],},
            {
                'args': {
                    'shape': tf.keras.Input((32, 32, 8)).shape,
                    'before': -10,
                    'after': 0,},
                'outputs': [32, 0, 32, 8],},]

    def test_rank_unchanged(self):
        for __c in self._cases:
            __s = __c['args']['shape']
            __u = mlable.shapes.move(__s, before=0, after=-1)
            self.assertEqual(len(__s), len(__u))

    def test_unchanged_when_source_and_destination_axes_are_equal(self):
        for __c in self._cases:
            __s = __c['args']['shape']
            __u = mlable.shapes.move(__s, before=0, after=0)
            self.assertEqual(mlable.shapes.normalize(__s), __u)

    def test_specific_cases(self):
        for __c in self._cases:
            self.assertEqual(__c['outputs'], mlable.shapes.move(**__c['args']))
