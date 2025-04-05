import math

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

class CategoricalTest(tf.test.TestCase):
    def setUp(self):
        super(CategoricalTest, self).setUp()
        self._random_cases = [
            {
                'inputs': {
                    'logits': tf.random.uniform((4, 8, 16), minval=-1.0, maxval=1.0, dtype=tf.float32, seed=42),
                    'temp': 1.0,
                    'topp': 0.0,
                    'topk': 0,
                    'depth': -1,
                    'seed': 42,},
                'outputs': {
                    'shape': (4, 8),},},
            {
                'inputs': {
                    'logits': tf.random.uniform((2, 512), minval=-2.0, maxval=2.0, dtype=tf.float32, seed=101),
                    'temp': 1.0,
                    'topp': 0.0,
                    'topk': 4,
                    'depth': 32,
                    'seed': 101,},
                'outputs': {
                    'shape': (2, 16),},},
            {
                'inputs': {
                    'logits': tf.random.uniform((256,), minval=-1.0, maxval=1.0, dtype=tf.float32, seed=32),
                    'temp': 1.0,
                    'topp': 0.9,
                    'topk': 0,
                    'depth': -1,
                    'seed': 32,},
                'outputs': {
                    'shape': (),},},]
        self._specific_cases = [
            {
                'inputs': {
                    'logits': tf.where(tf.linalg.diag([256 * [1.]]) == 1.0, 16.0, -16.0),
                    'temp': 1.0,
                    'topp': 0.8,
                    'topk': 0,
                    'depth': -1,
                    'seed': 42,},
                'outputs': {
                    'shape': (1, 256,),
                    'values': tf.expand_dims(tf.range(256), axis=0),},},
            {
                'inputs': {
                    'logits': tf.cast(4 * [[-16.0, -16.0, -16.0, -16.0, -16.0, -16.0, 16.0, -16.0,]], dtype=tf.float32),
                    'temp': 1.0,
                    'topp': 0.0,
                    'topk': 4,
                    'depth': -1,
                    'seed': 42,},
                'outputs': {
                    'shape': (4,),
                    'values': tf.cast(4 * [6], dtype=tf.int32),},},]

    def test_shape_and_dtype(self):
        for __case in self._random_cases:
            __outputs = mlable.sampling.categorical(**__case['inputs'])
            self.assertEqual(tuple(__outputs.shape), __case['outputs']['shape'])
            self.assertEqual(__outputs.dtype, tf.int32)

    def test_greedy_sampling_is_the_same_regardless_of_the_algorithm(self):
        for __case in self._random_cases:
            __args = dict(__case['inputs'])
            __args['topp'] = 0.0
            __args['topk'] = 0
            __argmax = mlable.sampling.categorical(**__args)
            __args['topp'] = 0.0001
            __args['topk'] = 0
            __topp = mlable.sampling.categorical(**__args)
            __args['topp'] = 0.0
            __args['topk'] = 1
            __topk = mlable.sampling.categorical(**__args)
            self.assertAllEqual(__argmax, __topp)
            self.assertAllEqual(__argmax, __topk)

    def test_random_sampling_with_definitive_probabilities(self):
        for __case in self._specific_cases:
            __outputs = mlable.sampling.categorical(**__case['inputs'])
            self.assertAllEqual(__outputs, __case['outputs']['values'])

# BINARY #######################################################################

class BinaryTest(tf.test.TestCase):
    def setUp(self):
        super(BinaryTest, self).setUp()
        self._random_cases = [
            {
                'inputs': {
                    'logits': tf.random.uniform((4, 8, 16), minval=-1.0, maxval=1.0, dtype=tf.float32, seed=42),
                    'temp': 1.0,
                    'topp': 0.0,
                    'topk': 0,
                    'depth': 4,
                    'seed': 42,},
                'outputs': {
                    'shape': (4, 8, 4),},},
            {
                'inputs': {
                    'logits': tf.random.uniform((2, 512), minval=-2.0, maxval=2.0, dtype=tf.float32, seed=101),
                    'temp': 1.0,
                    'topp': 0.0,
                    'topk': 4,
                    'depth': 8,
                    'seed': 101,},
                'outputs': {
                    'shape': (2, 64),},},
            {
                'inputs': {
                    'logits': tf.random.uniform((8,), minval=-1.0, maxval=1.0, dtype=tf.float32, seed=32),
                    'temp': 1.0,
                    'topp': 0.9,
                    'topk': 0,
                    'depth': -1,
                    'seed': 32,},
                'outputs': {
                    'shape': (),},},]
        self._specific_cases = [
            {
                'inputs': {
                    'logits': tf.where(tf.linalg.diag([8 * [1.]]) == 1.0, 1.0, -1.0),
                    'temp': 0.1,
                    'topp': 0.9,
                    'topk': 0,
                    'depth': 4,
                    'seed': 42,},
                'outputs': {
                    'shape': (1, 4, 2,),
                    'values': tf.cast([[[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2], [0, 4], [0, 8]]], dtype=tf.int32),},},
            {
                'inputs': {
                    'logits': tf.cast([[0.55, -0.08,  0.51, -0.10,  0.66, 0.47, -0.58, -0.60]], dtype=tf.float32),
                    'temp': 1.0,
                    'topp': 0.0,
                    'topk': 0,
                    'depth': -1,
                    'seed': 42,},
                'outputs': {
                    'shape': (1,),
                    'values': tf.cast([53], dtype=tf.int32),},},
            {
                'inputs': {
                    'logits': tf.cast(4 * [[-16.0, -16.0, -16.0, -16.0, -16.0, -16.0, 16.0, -16.0,]], dtype=tf.float32),
                    'temp': 1.0,
                    'topp': 0.0,
                    'topk': 4,
                    'depth': -1,
                    'seed': 42,},
                'outputs': {
                    'shape': (4,),
                    'values': tf.cast(4 * [64], dtype=tf.int32),},},]

    def test_shape_and_dtype(self):
        for __case in self._random_cases:
            __outputs = mlable.sampling.binary(**__case['inputs'])
            self.assertEqual(tuple(__outputs.shape), __case['outputs']['shape'])
            self.assertEqual(__outputs.dtype, tf.int32)

    def test_greedy_sampling_is_the_same_regardless_of_the_algorithm(self):
        for __case in self._random_cases:
            __args = dict(__case['inputs'])
            __args['topp'] = 0.0
            __args['topk'] = 0
            __argmax = mlable.sampling.binary(**__args)
            __args['topp'] = 0.0001
            __args['topk'] = 0
            __topp = mlable.sampling.binary(**__args)
            __args['topp'] = 0.0
            __args['topk'] = 1
            __topk = mlable.sampling.binary(**__args)
            self.assertAllEqual(__argmax, __topp)
            self.assertAllEqual(__argmax, __topk)

    def test_sampling_with_definitive_outcome(self):
        for __case in self._specific_cases:
            __outputs = mlable.sampling.binary(**__case['inputs'])
            self.assertAllEqual(__outputs, __case['outputs']['values'])
