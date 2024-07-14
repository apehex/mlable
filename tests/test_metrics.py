import tensorflow as tf

import mlable.metrics

# CATEGORICAL #################################################################

class CategoricalGroupAccuracyTest(tf.test.TestCase):

    def test_special_cases(self):
        __batch_dim, __seq_dim, __embed_dim, __group_dim, __iterations = 3, 16, 16, 4, 128
        # init
        __accuracy = mlable.metrics.CategoricalGroupAccuracy(group=__group_dim)
        # no match
        __accuracy.reset_state()
        __yt = tf.one_hot(indices=tf.zeros(shape=(__batch_dim, __seq_dim), dtype=tf.dtypes.int32), depth=__embed_dim)
        __yp = tf.one_hot(indices=tf.ones(shape=(__batch_dim, __seq_dim), dtype=tf.dtypes.int32), depth=__embed_dim)
        self.assertEqual(__accuracy(y_true=__yt, y_pred=__yp).numpy(), 0.)
        # all match
        __accuracy.reset_state()
        __yt = tf.one_hot(indices=tf.ones(shape=(__batch_dim, __seq_dim), dtype=tf.dtypes.int32), depth=__embed_dim)
        __yp = tf.one_hot(indices=tf.ones(shape=(__batch_dim, __seq_dim), dtype=tf.dtypes.int32), depth=__embed_dim)
        self.assertEqual(__accuracy(y_true=__yt, y_pred=__yp).numpy(), 1.)
        # iterating
        __accuracy.reset_state()
        for _ in range(__iterations):
            __accuracy.update_state(y_true=__yt, y_pred=__yp) # both ones
        self.assertEqual(__accuracy.result().numpy(), 1.)

    def test_specific_values(self):
        __iterations = 16
        # init
        __byte_acc = mlable.metrics.CategoricalGroupAccuracy(group=1)
        __character_acc = mlable.metrics.CategoricalGroupAccuracy(group=4)
        # test on ascii => leading zeroes match but not the characters
        __yt = tf.convert_to_tensor([
            '1111111111111111',
            '2222222222222222'])
        __yp = tf.convert_to_tensor([
            '0111111101111110',
            '0002222222222222'])
        # encode
        __yt = tf.strings.unicode_transcode(input=__yt, input_encoding='UTF-8', output_encoding='UTF-32-BE')
        __yp = tf.strings.unicode_transcode(input=__yp, input_encoding='UTF-8', output_encoding='UTF-32-BE')
        __yt = tf.io.decode_raw(__yt, out_type=tf.int8, fixed_length=64)
        __yp = tf.io.decode_raw(__yp, out_type=tf.int8, fixed_length=64)
        __yt = tf.one_hot(__yt, depth=256)
        __yp = tf.one_hot(__yp, depth=256)
        # one-shot
        __byte_acc.update_state(y_true=__yt, y_pred=__yp)
        __character_acc.update_state(y_true=__yt, y_pred=__yp)
        self.assertEqual(__byte_acc.result().numpy(), (4 * 16 - 3) / (4 * 16))
        self.assertEqual(__character_acc.result().numpy(), (16 - 3) / 16)
        # unchanged when iterating
        __byte_acc.reset_state()
        __character_acc.reset_state()
        for _ in range(__iterations):
            __byte_acc.update_state(y_true=__yt, y_pred=__yp)
            __character_acc.update_state(y_true=__yt, y_pred=__yp)
        self.assertEqual(__byte_acc.result().numpy(), (4 * 16 - 3) / (4 * 16))
        self.assertEqual(__character_acc.result().numpy(), (16 - 3) / 16)

    def test_bounds(self):
        # 0. <= a <= 1.
        __batch_dim, __seq_dim, __embed_dim, __group_dim, __iterations = 3, 16, 16, 4, 128
        # init
        __accuracy = mlable.metrics.CategoricalGroupAccuracy(group=__group_dim)
        # single evaluation
        __accuracy.reset_state()
        __yt = tf.one_hot(indices=tf.random.uniform(shape=(__batch_dim, __seq_dim), minval=0, maxval=__embed_dim, dtype=tf.dtypes.int32), depth=__embed_dim)
        __yp = tf.one_hot(indices=tf.random.uniform(shape=(__batch_dim, __seq_dim), minval=0, maxval=__embed_dim, dtype=tf.dtypes.int32), depth=__embed_dim)
        __accuracy.update_state(y_true=__yt, y_pred=__yp)
        self.assertLessEqual(0., __accuracy.result().numpy())
        self.assertLessEqual(__accuracy.result().numpy(), 1.)
        # iterative updates
        __accuracy.reset_state()
        for _ in range(__iterations):
            __yt = tf.one_hot(indices=tf.random.uniform(shape=(__batch_dim, __seq_dim), minval=0, maxval=__embed_dim, dtype=tf.dtypes.int32), depth=__embed_dim)
            __yp = tf.one_hot(indices=tf.random.uniform(shape=(__batch_dim, __seq_dim), minval=0, maxval=__embed_dim, dtype=tf.dtypes.int32), depth=__embed_dim)
            __accuracy.update_state(y_true=__yt, y_pred=__yp)
        self.assertLessEqual(0., __accuracy.result().numpy())
        self.assertLessEqual(__accuracy.result().numpy(), 1.)
        # all match
        __accuracy.reset_state()
        __yt = tf.one_hot(indices=tf.ones(shape=(__batch_dim, __seq_dim), dtype=tf.dtypes.int32), depth=__embed_dim)
        __yp = tf.one_hot(indices=tf.ones(shape=(__batch_dim, __seq_dim), dtype=tf.dtypes.int32), depth=__embed_dim)
        for _ in range(__iterations):
            __accuracy.update_state(y_true=__yt, y_pred=__yp)
        self.assertEqual(__accuracy.result().numpy(), 1.)

    def test_byte_accuracy_different_from_group_accuracy(self):
        # init
        __byte_acc = mlable.metrics.CategoricalGroupAccuracy(group=1)
        __character_acc = mlable.metrics.CategoricalGroupAccuracy(group=4)
        # test on ascii => leading zeroes match but not the characters
        __yt = tf.convert_to_tensor([
            'hello world this',
            'azertyuiopqsdfgh'])
        __yp = tf.convert_to_tensor([
            'yolo coco toto o',
            '1234567890123456'])
        # encode
        __yt = tf.strings.unicode_transcode(input=__yt, input_encoding='UTF-8', output_encoding='UTF-32-BE')
        __yp = tf.strings.unicode_transcode(input=__yp, input_encoding='UTF-8', output_encoding='UTF-32-BE')
        __yt = tf.io.decode_raw(__yt, out_type=tf.int8, fixed_length=64)
        __yp = tf.io.decode_raw(__yp, out_type=tf.int8, fixed_length=64)
        __yt = tf.one_hot(__yt, depth=256)
        __yp = tf.one_hot(__yp, depth=256)
        # evaluate
        __byte_acc.update_state(y_true=__yt, y_pred=__yp)
        __character_acc.update_state(y_true=__yt, y_pred=__yp)
        # check
        self.assertNotEqual(__byte_acc.result().numpy(), __character_acc.result().numpy())
        self.assertGreaterEqual(__byte_acc.result().numpy(), __character_acc.result().numpy())

# LOSS ########################################################################
