import tensorflow as tf

import mlable.layers.embedding

# ROPE ########################################################################

class SwapTest(tf.test.TestCase):
    def setUp(self):
        super(SwapTest, self).setUp()
        self._test_cases = [
            {
                'input': {
                    'rank': 5,
                    'sequence_axis': 1,
                    'feature_axis': 2,},
                'output': {
                    'values': [(2, -1)],}},
            {
                'input': {
                    'rank': 3,
                    'sequence_axis': 1,
                    'feature_axis': 2,},
                'output': {
                    'values': [],}},
            {
                'input': {
                    'rank': 5,
                    'sequence_axis': 3,
                    'feature_axis': 1,},
                'output': {
                    'values': [(3, 1), (3, -1)],}},
            {
                'input': {
                    'rank': 5,
                    'sequence_axis': 3,
                    'feature_axis': 2,},
                'output': {
                    'values': [(3, 1), (2, -1)],}},]

    def test_swaps(self):
        for __case in self._test_cases:
            __outputs = mlable.layers.embedding.swap_to_default(**__case['input'])
            self.assertEqual(__outputs, __case['output']['values'])

class TranspositionTest(tf.test.TestCase):
    def setUp(self):
        super(TranspositionTest, self).setUp()
        self._test_cases = [
            {
                'input': {
                    'tensor': tf.random.uniform(shape=(1, 2, 3, 4), dtype=tf.dtypes.float32),
                    'swaps': [(1, 2)]},
                'output': {
                    'shape': (1, 3, 2, 4),}},
            {
                'input': {
                    'tensor': tf.random.uniform(shape=(1, 2, 3, 4), dtype=tf.dtypes.float32),
                    'swaps': []},
                'output': {
                    'shape': (1, 2, 3, 4),}},
            {
                'input': {
                    'tensor': tf.reshape(tf.range(24, dtype=tf.dtypes.float32), shape=(1, 2, 3, 4)),
                    'swaps': [(1, -1)]},
                'output': {
                    'shape': (1, 4, 3, 2),}},
            {
                'input': {
                    'tensor': tf.reshape(tf.range(24, dtype=tf.dtypes.float32), shape=(2, 3, 4)),
                    'swaps': [(2, 0)]},
                'output': {
                    'shape': (4, 3, 2),}},
            {
                'input': {
                    'tensor': tf.reshape(tf.range(120, dtype=tf.dtypes.float32), shape=(5, 2, 3, 4)),
                    'swaps': [(0, 1), (1, 2), (2, 3)]},
                'output': {
                    'shape': (2, 3, 4, 5),
                    'values': tf.convert_to_tensor([
                        [[[  0,  24,  48,  72,  96],
                          [  1,  25,  49,  73,  97],
                          [  2,  26,  50,  74,  98],
                          [  3,  27,  51,  75,  99]],
                         [[  4,  28,  52,  76, 100],
                          [  5,  29,  53,  77, 101],
                          [  6,  30,  54,  78, 102],
                          [  7,  31,  55,  79, 103]],
                         [[  8,  32,  56,  80, 104],
                          [  9,  33,  57,  81, 105],
                          [ 10,  34,  58,  82, 106],
                          [ 11,  35,  59,  83, 107]]],
                        [[[ 12,  36,  60,  84, 108],
                          [ 13,  37,  61,  85, 109],
                          [ 14,  38,  62,  86, 110],
                          [ 15,  39,  63,  87, 111]],
                         [[ 16,  40,  64,  88, 112],
                          [ 17,  41,  65,  89, 113],
                          [ 18,  42,  66,  90, 114],
                          [ 19,  43,  67,  91, 115]],
                         [[ 20,  44,  68,  92, 116],
                          [ 21,  45,  69,  93, 117],
                          [ 22,  46,  70,  94, 118],
                          [ 23,  47,  71,  95, 119]]]], dtype=tf.dtypes.float32)}},]

    def test_transposition_shape(self):
        for __case in self._test_cases:
            __outputs = mlable.layers.embedding.transpose_axes(**__case['input'])
            if 'shape' in __case['output']:
                self.assertEqual(tuple(__outputs.shape), __case['output']['shape'])
            if 'values' in __case['output']:
                self.assertAllClose(__outputs, __case['output']['values'])

    def test_transposition_invariance(self):
        for __case in self._test_cases:
            __inputs = __case['input']['tensor']
            __swaps = __case['input']['swaps']
            # transpose of transpose
            __outputs = mlable.layers.embedding.transpose_axes(tensor=mlable.layers.embedding.transpose_axes(tensor=__inputs, swaps=__swaps), swaps=__swaps[::-1])
            # should be the identity
            self.assertAllClose(__outputs, __inputs)

class RopeTest(tf.test.TestCase):
    def setUp(self):
        super(RopeTest, self).setUp()
        self._test_cases = [
            {
                'init': {
                    'sequence_axis': 1,
                    'feature_axis': -1,
                    'max_wavelength': 100,
                    'scaling_factor': 1.},
                'input': {
                    'inputs': tf.random.uniform(shape=(2, 1, 2, 4), dtype=tf.dtypes.float32),
                    'offset': 3},
                'output': {
                    'shape': (2, 1, 2, 4),}},
            {
                'init': {},
                'input': {
                    'inputs': tf.ones(shape=(1, 4, 6)),
                    'offset': 0},
                'output': {
                    'shape': (1, 4, 6),
                    'values': tf.convert_to_tensor([[
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [-0.30116868, 0.95252377, 0.99784327, 1.3817732, 1.0453222, 1.0021521],
                        [-1.3254442, 0.9029957, 0.9956818, 0.4931506, 1.0883927, 1.0042995],
                        [-1.1311125, 0.85152256, 0.9935159, -0.8488725, 1.1291188, 1.0064424]]])}},
            {
                'init': {},
                'input': {
                    'inputs': tf.zeros(shape=(4, 32, 8)),
                    'offset': 4},
                'output': {
                    'shape': (4, 32, 8),
                    'values': tf.zeros(shape=(4, 32, 8))}},
            {
                'init': {
                    'scaling_factor': 2.},
                'input': {
                    'inputs': tf.ones(shape=(1, 3, 2, 4)),
                    'offset': 0},
                'output': {
                    'shape': (1, 3, 2, 4),
                    'values': tf.convert_to_tensor([
                        [
                            [
                                [1.000000000, 1.000000000, 1.000000000, 1.000000000],
                                [1.000000000, 1.000000000, 1.000000000, 1.000000000],],
                            [
                                [0.398157001, 0.994987488, 1.357008100, 1.004987478],
                                [0.398157001, 0.994987488, 1.357008100, 1.004987478],],
                            [
                                [-0.301168621, 0.989950180, 1.381773233, 1.009949803],
                                [-0.301168621, 0.989950180, 1.381773233, 1.009949803],],]])}},]

    def test_rope_shape(self):
        for __case in self._test_cases:
            __layer = mlable.layers.embedding.RotaryPositionalEmbedding(**__case['init'])
            __outputs = __layer(**__case['input'])
            if 'shape' in __case['output']:
                self.assertEqual(tuple(__outputs.shape), __case['output']['shape'])
            if 'values' in __case['output']:
                self.assertAllClose(__outputs, __case['output']['values'], rtol=1e-3)

# COSINE #######################################################################

class CosineEmbeddingTest(tf.test.TestCase):
    def setUp(self):
        super(CosineEmbeddingTest, self).setUp()
        self._cases = [
            {
                'inputs': tf.ones((2, 1), dtype=tf.float16),
                'args': {'embed_dim': 32, 'wave_dim': 1000, 'shift_dim': 0,},
                'outputs': (2, 1, 32),},
            {
                'inputs': tf.random.normal((2,), dtype=tf.float32),
                'args': {'embed_dim': 16, 'wave_dim': 10000, 'shift_dim': 1,},
                'outputs': (2, 16),},
            {
                'inputs': tf.random.normal((2, 1, 1), dtype=tf.float32),
                'args': {'embed_dim': 32, 'wave_dim': 1000, 'shift_dim': 0,},
                'outputs': (2, 1, 1, 32),},]

    def test_shape(self):
        for __case in self._cases:
            __layer = mlable.layers.embedding.CosineEmbedding(**__case['args'])
            __outputs = __layer(__case['inputs'], training=False)
            self.assertEqual(tuple(__outputs.shape), tuple(__case['outputs']))

    def test_range(self):
        for __case in self._cases:
            __layer = mlable.layers.embedding.CosineEmbedding(**__case['args'])
            __outputs = __layer(__case['inputs'], training=False)
            self.assertAllGreaterEqual(__outputs, -1.0)
            self.assertAllLessEqual(__outputs, 1.0)

# TOKUN EMBEDDING #############################################################

class TokunEmbeddingTest(tf.test.TestCase):
    def setUp(self):
        super(TokunEmbeddingTest, self).setUp()
        self._test_cases = [
            {
                'init': {
                    'input_dim': 256,
                    'output_dim': 128,},
                'input': {
                    'inputs': tf.random.uniform((2, 32, 4), minval=0, maxval=256),},
                'output': {
                    'shape': (2, 32, 512),}},
            {
                'init': {
                    'input_dim': 256,
                    'output_dim': 128,
                    'embeddings_initializer': 'ones'},
                'input': {
                    'inputs': tf.random.uniform((2, 8), minval=0, maxval=256),},
                'output': {
                    'shape': (2, 1024),
                    'values': tf.ones((2, 1024), dtype=tf.float32),}},]

    def test_embedding_shape(self):
        for __case in self._test_cases:
            __layer = mlable.layers.embedding.TokunEmbedding(**__case['init'])
            __outputs = __layer(**__case['input'])
            if 'shape' in __case['output']:
                self.assertEqual(tuple(__outputs.shape), __case['output']['shape'])
            if 'values' in __case['output']:
                self.assertAllEqual(__outputs, __case['output']['values'])
