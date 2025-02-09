import math

import tensorflow as tf

import mlable.schedules

# FN COMPOSITION ##############################################################

class LinearScheduleTest(tf.test.TestCase):
    def setUp(self):
        super(LinearScheduleTest, self).setUp()
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
            __outputs = mlable.schedules.linear_schedule(**__case)
            self.assertEqual(tuple(__outputs.shape), tuple(__inputs.shape))
            self.assertEqual(__outputs.dtype, tf.float32)

    def test_bounds(self):
        for __case in self._cases:
            __min = tf.cast(__case['rate_min'], dtype=tf.float32)
            __max = tf.cast(__case['rate_max'], dtype=tf.float32)
            __outputs = mlable.schedules.linear_schedule(**__case)
            self.assertAllGreaterEqual(__outputs, __min)
            self.assertAllLessEqual(__outputs, __max)
