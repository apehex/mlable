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
                    'merge_patch_axes': False,
                    'merge_space_axes': False,},
                'init_unpatching': {
                    'height_dim': 6,
                    'width_dim': 8,
                    'patch_dim': [3, 2],
                    'space_axes': [0, 1],
                    'patch_axes': [2, 3],},
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
                    'merge_patch_axes': True,
                    'merge_space_axes': True,},
                'init_unpatching': {
                    'height_dim': 6,
                    'width_dim': 8,
                    'patch_dim': (3, 2),
                    'space_axes': 0,
                    'patch_axes': 1,},
                'input': {
                    'inputs': tf.reshape(tf.range(48, dtype=tf.int32), shape=(6, 8)),},
                'output': {
                    'shape': (8, 6),
                    'values': tf.convert_to_tensor(
                        [
                            [0, 1, 8, 9, 16, 17],
                            [2, 3, 10, 11, 18, 19],
                            [4, 5, 12, 13, 20, 21],
                            [6, 7, 14, 15, 22, 23],
                            [24, 25, 32, 33, 40, 41],
                            [26, 27, 34, 35, 42, 43],
                            [28, 29, 36, 37, 44, 45],
                            [30, 31, 38, 39, 46, 47],])}},
            {
                'init': {
                    'height_axis': 1,
                    'width_axis': 0,
                    'patch_dim': (2, 3),
                    'merge_patch_axes': True,
                    'merge_space_axes': True,},
                'init_unpatching': {
                    'height_dim': 6,
                    'width_dim': 8,
                    'patch_dim': (3, 2),
                    'space_axes': [0],
                    'patch_axes': [1],},
                'input': {
                    'inputs': tf.reshape(tf.stack([tf.range(48, dtype=tf.int32), tf.range(48, dtype=tf.int32)], axis=-1), shape=(6, 8, 2)),},
                'output': {
                    'shape': (8, 6, 2),
                    'values': tf.convert_to_tensor(
                        [
                            [[0, 0], [1, 1], [8, 8], [9, 9], [16, 16], [17, 17]],
                            [[2, 2], [3, 3], [10, 10], [11, 11], [18, 18], [19, 19]],
                            [[4, 4], [5, 5], [12, 12], [13, 13], [20, 20], [21, 21]],
                            [[6, 6], [7, 7], [14, 14], [15, 15], [22, 22], [23, 23]],
                            [[24, 24], [25, 25], [32, 32], [33, 33], [40, 40], [41, 41]],
                            [[26, 26], [27, 27], [34, 34], [35, 35], [42, 42], [43, 43]],
                            [[28, 28], [29, 29], [36, 36], [37, 37], [44, 44], [45, 45]],
                            [[30, 30], [31, 31], [38, 38], [39, 39], [46, 46], [47, 47]],])}},]

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
