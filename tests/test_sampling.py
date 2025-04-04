import math

import numpy as np
import tensorflow as tf

import mlable.sampling

# FILTERING ####################################################################

class FilterToppTest(tf.test.TestCase):
    def setUp(self):
        super(FilterToppTest, self).setUp()
        self._random_cases = [
            {
                'logits': tf.random.uniform((4, 8, 16), minval=-1.0, maxval=1.0, dtype=tf.bfloat16),
                'threshold': 0.2,},
            {
                'logits': tf.random.uniform((2, 4), minval=-1.0, maxval=1.0, dtype=tf.float16),
                'threshold': 0.5,},
            {
                'logits': tf.random.uniform((1, 8), minval=-1.0, maxval=1.0, dtype=tf.float32),
                'threshold': 0.0,},]
        self._unfiltered_cases = [
            {
                'logits': tf.random.uniform((4, 8, 16), minval=-1.0, maxval=1.0, dtype=tf.float32),
                'threshold': 1.1,},
            {
                'logits': tf.range(16, dtype=tf.float32),
                'threshold': 2.,},]
        self._argmax_cases = [
            {
                'logits': tf.random.uniform((4, 8, 16), minval=-1.0, maxval=1.0, dtype=tf.float32),
                'threshold': 1. / 16.,}, # the largest probability cannot be under 1/N
            {
                'logits': tf.range(16, dtype=tf.float32),
                'threshold': 0.0,},]

    def test_shape_and_dtype_match_the_input(self):
        for __case in self._random_cases + self._argmax_cases + self._unfiltered_cases:
            __outputs = mlable.sampling.filter_top_p(**__case)
            self.assertShapeEqual(__outputs, __case['logits'])
            self.assertEqual(__outputs.dtype, __case['logits'].dtype)

    def test_output_has_at_least_one_unfiltered_element(self):
        for __case in self._random_cases:
            __outputs = mlable.sampling.filter_top_p(**__case)
            __mask = __outputs > 0.0
            __count = tf.reduce_sum(tf.cast(__mask, dtype=tf.int32), axis=-1)
            self.assertAllGreaterEqual(__count, 1)

    def test_output_equals_input_when_p_1(self):
        for __case in self._unfiltered_cases:
            __outputs = mlable.sampling.filter_top_p(**__case)
            self.assertAllClose(__outputs, __case['logits'])

    def test_output_is_argmax_on_low_threshold(self):
        for __case in self._argmax_cases:
            __outputs = mlable.sampling.filter_top_p(**__case)
            __mask = __outputs > 0.0
            __count = tf.reduce_sum(tf.cast(__mask, dtype=tf.int32), axis=-1)
            self.assertAllEqual(__count, tf.ones_like(__count))

class FilterTopkTest(tf.test.TestCase):
    def setUp(self):
        super(FilterTopkTest, self).setUp()
        self._random_cases = [
            {
                'logits': tf.random.uniform((4, 8, 16), minval=-1.0, maxval=1.0, dtype=tf.float32),
                'count': 2,},
            {
                'logits': tf.random.uniform((2, 4), minval=-1.0, maxval=1.0, dtype=tf.float16),
                'count': 1,},
            {
                'logits': tf.random.uniform((1, 8), minval=-1.0, maxval=1.0, dtype=tf.float32),
                'count': 4,},]
        self._unfiltered_cases = [
            {
                'logits': tf.random.uniform((4, 8, 16), minval=-1.0, maxval=1.0, dtype=tf.float32),
                'count': 16,},
            {
                'logits': tf.range(16, dtype=tf.float32),
                'count': 16,},]
        self._argmax_cases = [
            {
                'logits': tf.random.uniform((4, 8, 16), minval=-1.0, maxval=1.0, dtype=tf.float32),
                'count': 1,},
            {
                'logits': tf.range(16, dtype=tf.float32),
                'count': 1,},]

    def test_shape_and_dtype_match_the_input(self):
        for __case in self._random_cases + self._argmax_cases + self._unfiltered_cases:
            __outputs = mlable.sampling.filter_top_k(**__case)
            self.assertShapeEqual(__outputs, __case['logits'])
            self.assertEqual(__outputs.dtype, __case['logits'].dtype)

    def test_count_of_elements_kept_equals_k(self):
        for __case in self._random_cases:
            __outputs = mlable.sampling.filter_top_k(**__case)
            __mask = __outputs > 0.0
            __count = tf.reduce_sum(tf.cast(__mask, dtype=tf.int32), axis=-1)
            self.assertAllEqual(__count, __case['count'] * tf.ones_like(__count))

    def test_output_equals_input_when_k_equals_dim(self):
        for __case in self._unfiltered_cases:
            __outputs = mlable.sampling.filter_top_k(**__case)
            self.assertAllClose(__outputs, __case['logits'])

    def test_output_is_argmax_when_k_equals_1(self):
        for __case in self._argmax_cases:
            __outputs = mlable.sampling.filter_top_k(**__case)
            __mask = __outputs > 0.0
            __count = tf.reduce_sum(tf.cast(__mask, dtype=tf.int32), axis=-1)
            self.assertAllEqual(__count, tf.ones_like(__count))

# CATEGORICAL ##################################################################

# BINARY #######################################################################

# RAW ##########################################################################

# __diag = tf.linalg.diag([4 * [1.], 4 * [1.]])
# _categorical(__diag)
# <tf.Tensor: shape=(2, 4, 1), dtype=int64, numpy=
# array([[[0],
#         [1],
#         [2],
#         [3]],

#        [[0],
#         [1],
#         [2],
#         [3]]])>
