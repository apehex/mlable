import functools

import tensorflow as tf

import mlable.shaping.axes
import mlable.text

# ENCODING #####################################################################

class EncodingTest(tf.test.TestCase):
    def test_shape_and_dtype(self):
        __s0 = 'hello world!'
        __x0 = tf.cast([__s0], dtype=tf.string)
        __y0 = mlable.text.encode(__x0, sample_dim=32, output_dtype=tf.uint8, output_encoding='UTF-8')
        __y2 = mlable.text.encode(__x0, sample_dim=64, output_dtype=tf.uint8, output_encoding='UTF-32-BE')
        __y3 = mlable.text.encode(__x0, sample_dim=64, output_dtype=tf.int32, output_encoding='UTF-32-BE')
        # the dimension is fixed regardless of the number of bytes or codepoints
        self.assertEqual((1, 32), tuple(__y0.shape))
        self.assertEqual((1, 64), tuple(__y2.shape))
        self.assertEqual((1, 16), tuple(__y3.shape))
        # the dtypes match the arguments
        self.assertEqual(__y0.dtype, tf.uint8)
        self.assertEqual(__y2.dtype, tf.uint8)
        self.assertEqual(__y3.dtype, tf.int32)

    def test_encode_decode_reciprocal(self):
        __s0 = 'hello world!'
        __x0 = tf.cast([__s0], dtype=tf.string)
        # encode
        __y0 = mlable.text.encode(__x0, sample_dim=16, output_dtype=tf.uint8, output_encoding='UTF-8')
        __y2 = mlable.text.encode(__x0, sample_dim=64, output_dtype=tf.uint8, output_encoding='UTF-32-BE')
        __y3 = mlable.text.encode(__x0, sample_dim=64, output_dtype=tf.int32, output_encoding='UTF-32-BE')
        # decode
        __r0 = mlable.text.decode(__y0, encoding='UTF-8')
        __r2 = mlable.text.decode(__y2, encoding='UTF-32-BE')
        __r3 = mlable.text.decode(mlable.shaping.axes.merge(mlable.maths.ops.expand_base(__y3, base=256, depth=4, bigendian=True), axis=-1, right=False), encoding='UTF-32-BE')
        # unpack
        __o0 = mlable.text.unpack(mlable.text.unpad(__r0))
        __o2 = mlable.text.unpack(mlable.text.unpad(__r2))
        __o3 = mlable.text.unpack(mlable.text.unpad(__r3))
        # check
        assert __o0 == [__s0]
        assert __o2 == [__s0]
        assert __o3 == [__s0]

# PREPROCESSING ################################################################

class PreprocessTest(tf.test.TestCase):
    def test_shapes(self):
        __s0 = 'hello world'
        __y0 = mlable.text.preprocess(__s0, sample_dim=32, output_dtype=tf.uint8, output_encoding='UTF-8')
        __y2 = mlable.text.preprocess(__s0, sample_dim=64, output_dtype=tf.uint8, output_encoding='UTF-32-BE')
        __y3 = mlable.text.preprocess(__s0, sample_dim=64, output_dtype=tf.int32, output_encoding='UTF-32-BE')
        # the dimension is fixed regardless of the number of bytes or codepoints
        self.assertEqual((1, 32), tuple(__y0.shape))
        self.assertEqual((1, 64), tuple(__y2.shape))
        self.assertEqual((1, 16), tuple(__y3.shape))
        # the dtypes match the arguments
        self.assertEqual(__y0.dtype, tf.uint8)
        self.assertEqual(__y2.dtype, tf.uint8)
        self.assertEqual(__y3.dtype, tf.int32)

    def test_padding(self):
        __s0 = 'hello world' # ASCII characters get encoded on a single byte
        __y0 = mlable.text.preprocess(__s0, sample_dim=32, output_dtype=tf.uint8, output_encoding='UTF-8')
        __y1 = mlable.text.preprocess(__s0, sample_dim=64, output_dtype=tf.uint8, output_encoding='UTF-32-BE')
        __y2 = mlable.text.preprocess(__s0, sample_dim=64, output_dtype=tf.int32, output_encoding='UTF-32-BE')
        __p0 = 32 - len(__s0)
        __p1 = 64 - 4 * len(__s0)
        __p2 = 16 - len(__s0)
        self.assertAllClose(__y0[0, -__p0:], tf.zeros(shape=(__p0,), dtype=tf.uint8))
        self.assertAllClose(__y1[0, -__p1:], tf.zeros(shape=(__p1,), dtype=tf.uint8))
        self.assertAllClose(__y2[0, -__p2:], tf.zeros(shape=(__p2,), dtype=tf.int32))

# POSTPROCESSING ###############################################################

class PostprocessTest(tf.test.TestCase):
    def test_reciprocity(self):
        __s0 = 'hello world' # ASCII characters get encoded on a single byte
        __y0 = mlable.text.preprocess(__s0, sample_dim=32, output_dtype=tf.uint8, output_encoding='UTF-8')
        __y1 = mlable.text.preprocess(__s0, sample_dim=64, output_dtype=tf.uint8, output_encoding='UTF-32-BE')
        __y2 = mlable.text.preprocess(__s0, sample_dim=64, output_dtype=tf.int32, output_encoding='UTF-32-BE')
        __o0 = mlable.text.unpack(mlable.text.postprocess(__y0, encoding='UTF-8'))
        __o1 = mlable.text.unpack(mlable.text.postprocess(__y1, encoding='UTF-32-BE'))
        __o2 = mlable.text.unpack(mlable.text.postprocess(mlable.shaping.axes.merge(mlable.maths.ops.expand_base(__y2, base=256, depth=4, bigendian=True), axis=-1, right=False), encoding='UTF-32-BE'))
        assert __o0 == [__s0]
        assert __o1 == [__s0]
        assert __o2 == [__s0]
