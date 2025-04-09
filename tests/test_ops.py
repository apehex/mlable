import tensorflow as tf

import mlable.ops

# REDUCE ######################################################################

class MaskingTest(tf.test.TestCase):

    def setUp(self):
        super(MaskingTest, self).setUp()
        # all groups have one non-zero value
        self._ascii = tf.convert_to_tensor([
            '1111111111111111',
            '2222222222222222'])
        # each sample has null groups
        self._padded = tf.convert_to_tensor([
            '11111111111111\x00\x00',
            '\x00\x00\x00\x002222222222\x002'])
        # encode
        self._ascii = tf.strings.unicode_transcode(input=self._ascii, input_encoding='UTF-8', output_encoding='UTF-32-BE')
        self._ascii = tf.io.decode_raw(self._ascii, out_type=tf.int8, fixed_length=64)
        self._padded = tf.strings.unicode_transcode(input=self._padded, input_encoding='UTF-8', output_encoding='UTF-32-BE')
        self._padded = tf.io.decode_raw(self._padded, out_type=tf.int8, fixed_length=64)
        # mask
        self._ascii = tf.not_equal(self._ascii, 0)
        self._padded = tf.not_equal(self._padded, 0)

    def test_reduce_shapes(self):
        # reduced along a single axis
        __m_0_any = mlable.ops._reduce_any(data=self._ascii, axis=0, keepdims=False)
        __m_0_all = mlable.ops._reduce_all(data=self._ascii, axis=0, keepdims=False)
        __m_1_any = mlable.ops._reduce_any(data=self._ascii, axis=-1, keepdims=False)
        __m_1_all = mlable.ops._reduce_all(data=self._ascii, axis=-1, keepdims=False)
        self.assertEqual(list(__m_0_any.shape), [64,])
        self.assertEqual(list(__m_0_all.shape), [64,])
        self.assertEqual(list(__m_1_any.shape), [2,])
        self.assertEqual(list(__m_1_all.shape), [2,])
        # reduced on all axes
        __m_any = mlable.ops._reduce_any(data=self._ascii, axis=None, keepdims=False)
        __m_all = mlable.ops._reduce_all(data=self._ascii, axis=None, keepdims=False)
        self.assertEqual(list(__m_any.shape), [])
        self.assertEqual(list(__m_all.shape), [])
        # same shape as input
        __m_0_any_keep = mlable.ops._reduce_any(data=self._ascii, axis=0, keepdims=True)
        __m_0_all_keep = mlable.ops._reduce_all(data=self._ascii, axis=0, keepdims=True)
        __m_1_any_keep = mlable.ops._reduce_any(data=self._ascii, axis=-1, keepdims=True)
        __m_1_all_keep = mlable.ops._reduce_all(data=self._ascii, axis=-1, keepdims=True)
        __m_any_keep = mlable.ops._reduce_any(data=self._ascii, axis=None, keepdims=True)
        __m_all_keep = mlable.ops._reduce_all(data=self._ascii, axis=None, keepdims=True)
        self.assertEqual(list(__m_0_any_keep.shape), list(self._ascii.shape))
        self.assertEqual(list(__m_0_all_keep.shape), list(self._ascii.shape))
        self.assertEqual(list(__m_1_any_keep.shape), list(self._ascii.shape))
        self.assertEqual(list(__m_1_all_keep.shape), list(self._ascii.shape))
        self.assertEqual(list(__m_any_keep.shape), list(self._ascii.shape))
        self.assertEqual(list(__m_all_keep.shape), list(self._ascii.shape))

    def test_reduce_on_ascii_values(self):
        # reduced along a single axis
        __m_0_any = mlable.ops._reduce_any(data=self._ascii, axis=0, keepdims=False)
        __m_0_all = mlable.ops._reduce_all(data=self._ascii, axis=0, keepdims=False)
        __m_1_any = mlable.ops._reduce_any(data=self._ascii, axis=-1, keepdims=False)
        __m_1_all = mlable.ops._reduce_all(data=self._ascii, axis=-1, keepdims=False)
        self.assertAllEqual(__m_0_any, __m_0_all)
        self.assertAllEqual(__m_1_any, tf.convert_to_tensor([True, True]))
        self.assertAllEqual(__m_1_all, tf.convert_to_tensor([False, False]))
        # reduced on all axes
        __m_any = mlable.ops._reduce_any(data=self._ascii, axis=None, keepdims=False)
        __m_all = mlable.ops._reduce_all(data=self._ascii, axis=None, keepdims=False)
        self.assertEqual(__m_any.numpy(), True)
        self.assertEqual(__m_all.numpy(), False)
        # same shape as input
        __m_0_any_keep = mlable.ops._reduce_any(data=self._ascii, axis=0, keepdims=True)
        __m_0_all_keep = mlable.ops._reduce_all(data=self._ascii, axis=0, keepdims=True)
        __m_1_any_keep = mlable.ops._reduce_any(data=self._ascii, axis=-1, keepdims=True)
        __m_1_all_keep = mlable.ops._reduce_all(data=self._ascii, axis=-1, keepdims=True)
        __m_any_keep = mlable.ops._reduce_any(data=self._ascii, axis=None, keepdims=True)
        __m_all_keep = mlable.ops._reduce_all(data=self._ascii, axis=None, keepdims=True)
        self.assertAllEqual(__m_0_any_keep, __m_0_all_keep)
        self.assertAllEqual(__m_1_any_keep, tf.ones(shape=(2, 64), dtype=tf.dtypes.bool))
        self.assertAllEqual(__m_1_all_keep, tf.zeros(shape=(2, 64), dtype=tf.dtypes.bool))
        self.assertAllEqual(__m_any_keep, tf.ones(shape=(2, 64), dtype=tf.dtypes.bool))
        self.assertAllEqual(__m_all_keep, tf.zeros(shape=(2, 64), dtype=tf.dtypes.bool))

    def test_group_shapes(self):
        # reduced along a single axis
        __m_0_any = mlable.ops._reduce_group_by_group_any(data=self._ascii, group=2, axis=0, keepdims=False)
        __m_0_all = mlable.ops._reduce_group_by_group_all(data=self._ascii, group=2, axis=0, keepdims=False)
        __m_1_any = mlable.ops._reduce_group_by_group_any(data=self._ascii, group=4, axis=-1, keepdims=False)
        __m_1_all = mlable.ops._reduce_group_by_group_all(data=self._ascii, group=4, axis=-1, keepdims=False)
        self.assertEqual(list(__m_0_any.shape), [1, 64])
        self.assertEqual(list(__m_0_all.shape), [1, 64])
        self.assertEqual(list(__m_1_any.shape), [2, 16])
        self.assertEqual(list(__m_1_all.shape), [2, 16])
        # same shape as input
        __m_0_any_keep = mlable.ops._reduce_group_by_group_any(data=self._ascii, group=2, axis=0, keepdims=True)
        __m_0_all_keep = mlable.ops._reduce_group_by_group_all(data=self._ascii, group=2, axis=0, keepdims=True)
        __m_1_any_keep = mlable.ops._reduce_group_by_group_any(data=self._ascii, group=4, axis=-1, keepdims=True)
        __m_1_all_keep = mlable.ops._reduce_group_by_group_all(data=self._ascii, group=4, axis=-1, keepdims=True)
        self.assertEqual(list(__m_0_any_keep.shape), list(self._ascii.shape))
        self.assertEqual(list(__m_0_all_keep.shape), list(self._ascii.shape))
        self.assertEqual(list(__m_1_any_keep.shape), list(self._ascii.shape))
        self.assertEqual(list(__m_1_all_keep.shape), list(self._ascii.shape))

    def test_group_on_ascii_values(self):
        # reduce 4 by 4
        __m_1_any = mlable.ops._reduce_group_by_group_any(data=self._ascii, group=4, axis=-1, keepdims=False)
        __m_1_all = mlable.ops._reduce_group_by_group_all(data=self._ascii, group=4, axis=-1, keepdims=False)
        self.assertAllEqual(__m_1_any, tf.ones(shape=(2, 16), dtype=tf.dtypes.bool))
        self.assertAllEqual(__m_1_all, tf.zeros(shape=(2, 16), dtype=tf.dtypes.bool))
        # same shape as input
        __m_1_any_keep = mlable.ops._reduce_group_by_group_any(data=self._ascii, group=4, axis=-1, keepdims=True)
        __m_1_all_keep = mlable.ops._reduce_group_by_group_all(data=self._ascii, group=4, axis=-1, keepdims=True)
        self.assertAllEqual(__m_1_any_keep, tf.ones(shape=(2, 64), dtype=tf.dtypes.bool))
        self.assertAllEqual(__m_1_all_keep, tf.zeros(shape=(2, 64), dtype=tf.dtypes.bool))

    def test_group_on_padded_values(self):
        # reduce 4 by 4
        __m_1_any = mlable.ops.reduce_any(data=self._padded, group=4, axis=-1, keepdims=False)
        __m_1_all = mlable.ops.reduce_all(data=self._padded, group=4, axis=-1, keepdims=False)
        self.assertAllEqual(__m_1_any, tf.convert_to_tensor([14 * [True] + [False, False], 4 * [False] + 10 * [True] + [False, True]], dtype=tf.dtypes.bool))
        self.assertAllEqual(__m_1_all, tf.zeros(shape=(2, 16), dtype=tf.dtypes.bool))
        # same shape as input
        __m_1_any_keep = mlable.ops.reduce_any(data=self._padded, group=4, axis=-1, keepdims=True)
        __m_1_all_keep = mlable.ops.reduce_all(data=self._padded, group=4, axis=-1, keepdims=True)
        self.assertAllEqual(__m_1_any_keep, tf.convert_to_tensor([14 * 4 * [True] + 2 * 4 * [False], 4 * 4 * [False] + 10 * 4 * [True] + 4 * [False] + 4 * [True]], dtype=tf.dtypes.bool))
        self.assertAllEqual(__m_1_all_keep, tf.zeros(shape=(2, 64), dtype=tf.dtypes.bool))

# BASE ########################################################################

class BaseCompositionTest(tf.test.TestCase):

    def setUp(self):
        super(BaseCompositionTest, self).setUp()
        # base 10
        self._digits_10 = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=tf.int32)
        self._digits_10_flat = tf.reshape(self._digits_10, shape=(-1,))
        # base 2
        self._digits_2 = tf.constant([[1, 0, 1, 0], [0, 1, 1, 0], [0, 1, 0, 0]], dtype=tf.int32)
        self._digits_2_flat = tf.reshape(self._digits_2, shape=(-1,))
        # base 256 (UTF-32-BE bytes)
        self._bytes = tf.constant([[0, 3, 0, 73], [0, 0, 1, 110], [0, 0, 0, 32], [10, 0, 0, 0]], dtype=tf.int32)
        self._bytes_flat = tf.reshape(self._bytes, shape=(-1,))

    def test_bidimensional_compositions(self):
        __number_10 = mlable.ops.reduce_base(self._digits_10, base=10, axis=-1, bigendian=False)
        __number_2 = mlable.ops.reduce_base(self._digits_2, base=2, axis=-1, bigendian=True)
        __number_256 = mlable.ops.reduce_base(self._bytes, base=256, axis=-1, bigendian=True)
        self.assertAllEqual(__number_10, tf.constant([321, 654, 987], dtype=tf.int32))
        self.assertAllEqual(__number_2, tf.constant([10, 6, 4], dtype=tf.int32))
        self.assertAllEqual(__number_256, tf.constant([196681, 366, 32, 167772160], dtype=tf.int32))

    def test_flat_compositions(self):
        __number_10 = mlable.ops.reduce_base(self._digits_10_flat, base=10, axis=-1, bigendian=False)
        __number_2 = mlable.ops.reduce_base(self._digits_2_flat, base=2, axis=-1, bigendian=True)
        self.assertAllEqual(__number_10, tf.constant(987_654_321, dtype=tf.int32))
        self.assertAllEqual(__number_2, tf.constant(2660, dtype=tf.int32))

    def test_grouped_compositions(self):
        __number_10 = mlable.ops.reduce_base(self._digits_10_flat, base=10, axis=-1, group=3, bigendian=False)
        __number_2 = mlable.ops.reduce_base(self._digits_2_flat, base=2, axis=-1, group=3, bigendian=True)
        __number_256 = mlable.ops.reduce_base(self._bytes_flat, base=256, axis=-1, group=2, bigendian=True)
        self.assertAllEqual(__number_10, tf.constant([321, 654, 987], dtype=tf.int32))
        self.assertAllEqual(__number_2, tf.constant([5, 1, 4, 4], dtype=tf.int32))
        self.assertAllEqual(__number_256, tf.constant([3, 73, 0, 366, 0, 32, 2560, 0], dtype=tf.int32))

    def test_compositions_along_axis_0(self):
        __number_10 = mlable.ops.reduce_base(self._digits_10, base=10, axis=0, bigendian=True)
        __number_2 = mlable.ops.reduce_base(self._digits_2, base=2, axis=0, bigendian=True)
        __number_256 = mlable.ops.reduce_base(self._bytes, base=256, axis=0, bigendian=True)
        self.assertAllEqual(__number_10, tf.constant([147, 258, 369], dtype=tf.int32))
        self.assertAllEqual(__number_2, tf.constant([4, 3, 6, 0], dtype=tf.int32))
        self.assertAllEqual(__number_256, tf.constant([10, 50331648, 65536, 1231953920], dtype=tf.int32))

class BaseDecompositionTest(tf.test.TestCase):

    def setUp(self):
        super(BaseDecompositionTest, self).setUp()
        # base 10
        self._digits_10 = tf.constant([123, 456, 789], dtype=tf.int32)
        # base 2
        self._digits_2 = tf.constant([5, 1, 15, 0], dtype=tf.int32)
        # base 256 (UTF-32-BE bytes)
        self._bytes = tf.constant([3, 73, 0, 366, 0, 32, 2560, 0], dtype=tf.int32)

    def test_decomposition_values(self):
        __number_10 = mlable.ops.expand_base(self._digits_10, base=10, depth=3, bigendian=False)
        __number_2 = mlable.ops.expand_base(self._digits_2, base=2, depth=4, bigendian=True)
        __number_256 = mlable.ops.expand_base(self._bytes, base=256, depth=2, bigendian=True)
        self.assertAllEqual(__number_10, tf.constant([[3, 2, 1], [6, 5, 4], [9, 8, 7]], dtype=tf.int32))
        self.assertAllEqual(__number_2, tf.constant([[0, 1, 0, 1], [0, 0, 0, 1], [1, 1, 1, 1], [0, 0, 0, 0]], dtype=tf.int32))
        self.assertAllEqual(__number_256, tf.constant([[0, 3], [0, 73], [0, 0], [1, 110], [0, 0], [0, 32], [10, 0], [0, 0]], dtype=tf.int32))
