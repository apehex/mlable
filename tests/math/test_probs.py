import math

import tensorflow as tf

import mlable.math.probs

# FN COMPOSITION ##############################################################

class LogNormalDistributionTest(tf.test.TestCase):
    def setUp(self):
        super(LogNormalDistributionTest, self).setUp()
        self._cases = [
            {
                'sample': tf.random.normal((4, 8), dtype=tf.float32),
                'mean': tf.random.normal((4, 8), dtype=tf.float32),
                'logvar': tf.random.normal((4, 8), dtype=tf.float32),},
            {
                'sample': tf.random.normal((4, 8), dtype=tf.bfloat16),
                'mean': tf.random.normal((4, 8), dtype=tf.float32),
                'logvar': tf.random.normal((4, 8), dtype=tf.float32),},
            {
                'sample': tf.random.normal((4, 32, 8), dtype=tf.float32),
                'mean': tf.random.normal((8,), dtype=tf.float32),
                'logvar': tf.random.normal((8,), dtype=tf.float32),},
            {
                'sample': tf.zeros((1, 8), dtype=tf.float32),
                'mean': tf.zeros((8,), dtype=tf.float32),
                'logvar': tf.random.normal((8,), dtype=tf.float32),},]

    def test_shape_and_dtype(self):
        for __case in self._cases:
            __outputs = mlable.math.probs.log_normal_pdf(**__case)
            self.assertEqual(tuple(__outputs.shape), tuple(__case['sample'].shape))
            self.assertEqual(__outputs.dtype, __case['sample'].dtype)

    def test_specific_values(self):
        for __case in self._cases:
            __outputs = mlable.math.probs.log_normal_pdf(sample=__case['mean'], mean=__case['mean'], logvar=__case['logvar'])
            __log2pi = tf.cast(tf.math.log(2. * math.pi), dtype=__case['mean'].dtype)
            __logvar = tf.cast(__case['logvar'], dtype=__case['mean'].dtype)
            self.assertAllEqual(__outputs, -0.5 * (__log2pi + __logvar))
