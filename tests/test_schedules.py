import math

import tensorflow as tf

import mlable.schedules

# LINEAR #######################################################################

class LinearRateTest(tf.test.TestCase):
    def setUp(self):
        super(LinearRateTest, self).setUp()
        self._cases = [
            {
                'step': 0,
                'step_min': 8,
                'step_max': 32,
                'rate_min': 0.2,
                'rate_max': 1.3,},
            {
                'step': 16,
                'step_min': 8,
                'step_max': 32,
                'rate_min': 0.2,
                'rate_max': 1.3,},
            {
                'step': 64,
                'step_min': 8,
                'step_max': 32,
                'rate_min': 0.2,
                'rate_max': 1.3,},
            {
                'step': -5,
                'step_min': 8,
                'step_max': 32,
                'rate_min': 0.2,
                'rate_max': 1.3,},
            {
                'step': tf.random.uniform((4, 32, 1), minval=0, maxval=64, dtype=tf.int32),
                'step_min': 8,
                'step_max': 32,
                'rate_min': 0.2,
                'rate_max': 1.3,},]

    def test_shape_and_dtype(self):
        for __case in self._cases:
            __inputs = tf.cast(__case['step'], dtype=tf.float32)
            __outputs = mlable.schedules.linear_rate(**__case)
            self.assertEqual(tuple(__outputs.shape), tuple(__inputs.shape))
            self.assertEqual(__outputs.dtype, tf.float32)

    def test_bounds(self):
        for __case in self._cases:
            __min = tf.cast(__case['rate_min'], dtype=tf.float32)
            __max = tf.cast(__case['rate_max'], dtype=tf.float32)
            __outputs = mlable.schedules.linear_rate(**__case)
            self.assertAllGreaterEqual(__outputs, __min)
            self.assertAllLessEqual(__outputs, __max)

# COSINE #######################################################################

class CosineRatesTest(tf.test.TestCase):
    def setUp(self):
        super(CosineRatesTest, self).setUp()
        self._cases = [
            {
                'args': {
                    'angle_rates': 0.0,
                    'start_rate': 0.9,
                    'end_rate': 0.1,
                    'dtype': None,},
                'outputs': (tf.cast(math.sqrt(0.19), tf.float32), tf.cast(0.9, tf.float32)),},
            {
                'args': {
                    'angle_rates': 1.0,
                    'start_rate': 0.9,
                    'end_rate': 0.1,
                    'dtype': tf.float16,},
                'outputs': (tf.cast(math.sqrt(0.99), tf.float16), tf.cast(0.1, tf.float16)),},
            {
                'args': {
                    'angle_rates': tf.cast([0.3, 0.1, 0.9, 0.8], tf.float16),
                    'start_rate': 0.2,
                    'end_rate': 0.2,
                    'dtype': tf.float32,},
                'outputs': (tf.cast(4 * [math.sqrt(0.96)], tf.float32), tf.cast(4 * [0.2], tf.float32)),},
            {
                'args': {
                    'angle_rates': tf.cast([0.5, 0.8], tf.float32),
                    'start_rate': 1.0,
                    'end_rate': 0.0,
                    'dtype': None,},
                'outputs': (tf.cast([math.sin(0.25 * math.pi), math.sin(0.4 * math.pi)], tf.float32), tf.cast([math.cos(0.25 * math.pi), math.cos(0.4 * math.pi)], tf.float32)),},]

    def test_shape_and_dtype(self):
        for __case in self._cases:
            __sin_c, __cos_c = mlable.schedules.cosine_rates(**__case['args'])
            __sin_e, __cos_e = __case['outputs']
            # shapes
            self.assertEqual(tuple(__sin_c.shape), tuple(__sin_e.shape))
            self.assertEqual(tuple(__cos_c.shape), tuple(__cos_e.shape))
            # types
            self.assertEqual(__sin_c.dtype, __sin_e.dtype)
            self.assertEqual(__cos_c.dtype, __cos_e.dtype)

    def test_powers_sum_to_ones(self):
        for __case in self._cases:
            __rand = tf.random.uniform((32, 1, 1), minval=0.0, maxval=1.0, dtype=tf.float32)
            __sin, __cos = mlable.schedules.cosine_rates(__rand, start_rate=__case['args']['start_rate'], end_rate=__case['args']['end_rate'])
            self.assertAllClose(__sin * __sin + __cos * __cos, tf.cast(32 * [[[1.0]]], tf.float32))

    def test_bounds(self):
        for __case in self._cases:
            __cos_min = tf.cast(min(__case['args']['start_rate'], __case['args']['end_rate']), dtype=tf.float32)
            __cos_max = tf.cast(max(__case['args']['start_rate'], __case['args']['end_rate']), dtype=tf.float32)
            __sin_min = tf.sqrt(1.0 - __cos_max * __cos_max)
            __sin_max = tf.sqrt(1.0 - __cos_min * __cos_min)
            __rand = tf.random.uniform((32, 8, 8), minval=0.0, maxval=1.0, dtype=tf.float32)
            __sin, __cos = mlable.schedules.cosine_rates(__rand, start_rate=__case['args']['start_rate'], end_rate=__case['args']['end_rate'])
            # cos / signal rates
            self.assertAllGreaterEqual(tf.round(100.0 * __cos), tf.round(100.0 * __cos_min)) # deal with computation error
            self.assertAllLessEqual(tf.round(100.0 * __cos), tf.round(100.0 * __cos_max)) # deal with computation error
            # sin / noise rates
            self.assertAllGreaterEqual(tf.round(100.0 * __sin), tf.round(100.0 * __sin_min)) # deal with computation error
            self.assertAllLessEqual(tf.round(100.0 * __sin), tf.round(100.0 * __sin_max)) # deal with computation error

    def test_specific_cases(self):
        for __case in self._cases:
            __sin_c, __cos_c = mlable.schedules.cosine_rates(**__case['args'])
            __sin_e, __cos_e = __case['outputs']
            self.assertAllClose(__sin_c, __sin_e, rtol=1e-3)
            self.assertAllClose(__cos_c, __cos_e, rtol=1e-3)
