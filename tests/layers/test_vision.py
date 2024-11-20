import tensorflow as tf

import mlable.layers.vision

# BOTH #########################################################################

class PatchingTest(tf.test.TestCase):
    def setUp(self):
        super(PatchingTest, self).setUp()
        self._test_cases = [
            {
                'init': {
                    'height_axis': 0,
                    'width_axis': 1,
                    'patch_dim': (3, 2),
                    'transpose': False,},
                'init_unpatching': {
                    'space_height_axis': 0,
                    'space_width_axis': 1,
                    'patch_height_axis': 2,
                    'patch_width_axis': 3,},
                'input': {
                    'inputs': tf.reshape(tf.range(48, dtype=tf.int32), shape=(6, 8)),},
                'output': {
                    'shape': (2, 4, 3, 2),
                    'values': tf.convert_to_tensor(
                        [
                            [
                                [[0, 1], [8, 9], [16, 17]],
                                [[2, 3], [10, 11], [18, 19]],
                                [[4, 5], [12, 13], [20, 21]],
                                [[6, 7], [14, 15], [22, 23]],],
                            [
                                [[24, 25], [32, 33], [40, 41]],
                                [[26, 27], [34, 35], [42, 43]],
                                [[28, 29], [36, 37], [44, 45]],
                                [[30, 31], [38, 39], [46, 47]],],])}},
            {
                'init': {
                    'height_axis': 0,
                    'width_axis': 1,
                    'patch_dim': (3, 2),
                    'transpose': True,},
                'init_unpatching': {
                    'space_height_axis': 2,
                    'space_width_axis': 3,
                    'patch_height_axis': 0,
                    'patch_width_axis': 1,},
                'input': {
                    'inputs': tf.reshape(tf.range(48, dtype=tf.int32), shape=(6, 8)),},
                'output': {
                    'shape': (3, 2, 2, 4),
                    'values': tf.convert_to_tensor(
                        [
                            [
                                [[0, 2, 4, 6], [24, 26, 28, 30]],
                                [[1, 3, 5, 7], [25, 27, 29, 31]]],
                            [
                                [[8, 10, 12, 14], [32, 34, 36, 38]],
                                [[9, 11, 13, 15], [33, 35, 37, 39]]],
                            [
                                [[16, 18, 20, 22], [40, 42, 44, 46]],
                                [[17, 19, 21, 23], [41, 43, 45, 47]],],])}},
            {
                'init': {
                    'height_axis': 1,
                    'width_axis': 0,
                    'patch_dim': (2, 3),
                    'transpose': False,},
                'init_unpatching': {
                    'space_height_axis': 0,
                    'space_width_axis': 1,
                    'patch_height_axis': 2,
                    'patch_width_axis': 3,},
                'input': {
                    'inputs': tf.reshape(tf.stack([tf.range(48, dtype=tf.int32), tf.range(48, dtype=tf.int32)], axis=-1), shape=(6, 8, 2)),},
                'output': {
                    'shape': (2, 4, 3, 2, 2),
                    'values': tf.convert_to_tensor(
                        [
                            [
                                [[[0, 0], [1, 1]], [[8, 8], [9, 9]], [[16, 16], [17, 17]]],
                                [[[2, 2], [3, 3]], [[10, 10], [11, 11]], [[18, 18], [19, 19]]],
                                [[[4, 4], [5, 5]], [[12, 12], [13, 13]], [[20, 20], [21, 21]]],
                                [[[6, 6], [7, 7]], [[14, 14], [15, 15]], [[22, 22], [23, 23]]],],
                            [
                                [[[24, 24], [25, 25]], [[32, 32], [33, 33]], [[40, 40], [41, 41]]],
                                [[[26, 26], [27, 27]], [[34, 34], [35, 35]], [[42, 42], [43, 43]]],
                                [[[28, 28], [29, 29]], [[36, 36], [37, 37]], [[44, 44], [45, 45]]],
                                [[[30, 30], [31, 31]], [[38, 38], [39, 39]], [[46, 46], [47, 47]]],],])}},]

    def test_patching(self):
        for __case in self._test_cases:
            __layer = mlable.layers.vision.Patching(**__case['init'])
            __outputs = __layer(**__case['input'])
            if 'shape' in __case['output']:
                self.assertEqual(tuple(__outputs.shape), __case['output']['shape'])
            if 'values' in __case['output']:
                self.assertAllClose(__outputs, __case['output']['values'])

    def test_reciprocity(self):
        for __case in self._test_cases:
            __patch = mlable.layers.vision.Patching(**__case['init'])
            __unpatch = mlable.layers.vision.Unpatching(**__case['init_unpatching'])
            __outputs = __unpatch(__patch(__case['input']['inputs']))
            self.assertAllEqual(__case['input']['inputs'], __outputs)

# PIXEL SHUFFLE ################################################################

class PixelShuffleTest(tf.test.TestCase):
    def setUp(self):
        super(PixelShuffleTest, self).setUp()
        self._test_cases = [
            {
                'init': {
                    'height_axis': 0,
                    'width_axis': 1,
                    'patch_dim': (3, 2),},
                'init_packing': {
                    'height_axis': 0,
                    'width_axis': 1,
                    'patch_dim': (3, 2),},
                'input': {
                    'inputs': tf.reshape(tf.range(48, dtype=tf.int32), shape=(2, 2, 12)),},
                'output': {
                    'shape': (6, 4, 2),
                    'values': tf.convert_to_tensor(
                        [
                            [[0, 1], [2, 3], [12, 13], [14, 15]],
                            [[4, 5], [6, 7], [16, 17], [18, 19]],
                            [[8, 9], [10, 11], [20, 21], [22, 23]],
                            [[24, 25], [26, 27], [36, 37], [38, 39]],
                            [[28, 29], [30, 31], [40, 41], [42, 43]],
                            [[32, 33], [34, 35], [44, 45], [46, 47]]])}},]

    def test_patching(self):
        for __case in self._test_cases:
            __layer = mlable.layers.vision.PixelShuffle(**__case['init'])
            __outputs = __layer(**__case['input'])
            if 'shape' in __case['output']:
                self.assertEqual(tuple(__outputs.shape), __case['output']['shape'])
            if 'values' in __case['output']:
                self.assertAllClose(__outputs, __case['output']['values'])

    def test_reciprocity(self):
        for __case in self._test_cases:
            __shuffle = mlable.layers.vision.PixelShuffle(**__case['init'])
            __pack = mlable.layers.vision.PixelPacking(**__case['init_packing'])
            __outputs = __pack(__shuffle(__case['input']['inputs']))
            self.assertAllEqual(__case['input']['inputs'], __outputs)
