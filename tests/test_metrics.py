import tensorflow as tf

import mlable.metrics

# ACCURACY ####################################################################

class GroupAccuracyTest(tf.test.TestCase):

    def test_special_cases(self):
        __batch_dim, __seq_dim, __embed_dim, __group_dim, __iterations = 3, 16, 16, 4, 128
        # no match
        __yt = tf.one_hot(indices=tf.zeros(shape=(__batch_dim, __seq_dim), dtype=tf.dtypes.int32), depth=__embed_dim)
        __yp = tf.one_hot(indices=tf.ones(shape=(__batch_dim, __seq_dim), dtype=tf.dtypes.int32), depth=__embed_dim)
        __total, __correct = mlable.metrics.group_accuracy(y_true=__yt, y_pred=__yp, group=__group_dim)
        self.assertEqual(__total.numpy(), __batch_dim * __seq_dim // __group_dim)
        self.assertEqual(__correct.numpy(), 0)
        # all match
        __yt = tf.one_hot(indices=tf.ones(shape=(__batch_dim, __seq_dim), dtype=tf.dtypes.int32), depth=__embed_dim)
        __yp = tf.one_hot(indices=tf.ones(shape=(__batch_dim, __seq_dim), dtype=tf.dtypes.int32), depth=__embed_dim)
        __total, __correct = mlable.metrics.group_accuracy(y_true=__yt, y_pred=__yp, group=__group_dim)
        self.assertEqual(__total.numpy(), __batch_dim * __seq_dim // __group_dim)
        self.assertEqual(__correct.numpy(), __total.numpy())
        # iterating
        __acc = mlable.metrics.GroupAccuracy(group=__group_dim)
        for _ in range(__iterations):
            __acc.update_state(y_true=__yt, y_pred=__yp) # both ones
        self.assertEqual(__acc._total.numpy(), __iterations * __batch_dim * __seq_dim // __group_dim)
        self.assertEqual(__acc._correct.numpy(), __acc._total.numpy())

    def test_specific_values(self):
        __iterations = 16
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
        # accuracies
        __byte_acc = mlable.metrics.GroupAccuracy(group=1)
        __character_acc = mlable.metrics.GroupAccuracy(group=4)
        # one-shot
        __byte_acc.update_state(y_true=__yt, y_pred=__yp)
        __character_acc.update_state(y_true=__yt, y_pred=__yp)
        self.assertEqual(__byte_acc._total.numpy(), 2 * 16 * 4)
        self.assertEqual(__character_acc._total.numpy(), 2 * 16)
        self.assertEqual(__byte_acc._correct.numpy(), 2 * (16 * 4 - 3)) # only 3 wrong bytes since the leading 0 stay the same
        self.assertEqual(__character_acc._correct.numpy(), 2 * (16 - 3)) # 3 full character errors
        self.assertEqual(__byte_acc.result(), (4 * 16 - 3) / (4 * 16))
        self.assertEqual(__character_acc.result(), (16 - 3) / 16)
        # unchanged when iterating
        __byte_acc.reset_state()
        __character_acc.reset_state()
        for _ in range(__iterations):
            __byte_acc.update_state(y_true=__yt, y_pred=__yp)
            __character_acc.update_state(y_true=__yt, y_pred=__yp)
        self.assertEqual(__byte_acc._total.numpy(), __iterations * 2 * 16 * 4)
        self.assertEqual(__character_acc._total.numpy(), __iterations * 2 * 16)
        self.assertEqual(__byte_acc._correct.numpy(), __iterations * 2 * (16 * 4 - 3))
        self.assertEqual(__character_acc._correct.numpy(), __iterations * 2 * (16 - 3))
        self.assertEqual(__byte_acc.result(), (4 * 16 - 3) / (4 * 16))
        self.assertEqual(__character_acc.result(), (16 - 3) / 16)

    def test_bounds(self):
        # 0. <= a <= 1.
        __batch_dim, __seq_dim, __embed_dim, __group_dim, __iterations = 3, 16, 16, 4, 128
        # single evaluation
        __yt = tf.one_hot(indices=tf.random.uniform(shape=(__batch_dim, __seq_dim), minval=0, maxval=__embed_dim, dtype=tf.dtypes.int32), depth=__embed_dim)
        __yp = tf.one_hot(indices=tf.random.uniform(shape=(__batch_dim, __seq_dim), minval=0, maxval=__embed_dim, dtype=tf.dtypes.int32), depth=__embed_dim)
        __total, __correct = mlable.metrics.group_accuracy(y_true=__yt, y_pred=__yp, group=__group_dim)
        self.assertEqual(__total.numpy(), __batch_dim * __seq_dim // __group_dim)
        self.assertGreaterEqual(0, __correct.numpy())
        self.assertLessEqual(__correct.numpy(), __total.numpy())
        # iterative updates
        __acc = mlable.metrics.GroupAccuracy(group=__group_dim)
        for _ in range(__iterations):
            __yt = tf.one_hot(indices=tf.random.uniform(shape=(__batch_dim, __seq_dim), minval=0, maxval=__embed_dim, dtype=tf.dtypes.int32), depth=__embed_dim)
            __yp = tf.one_hot(indices=tf.random.uniform(shape=(__batch_dim, __seq_dim), minval=0, maxval=__embed_dim, dtype=tf.dtypes.int32), depth=__embed_dim)
            __acc.update_state(y_true=__yt, y_pred=__yp)
        self.assertEqual(__acc._total.numpy(), __iterations * __batch_dim * __seq_dim // __group_dim)
        self.assertGreaterEqual(0, __acc._correct.numpy())
        self.assertGreaterEqual(0., __acc.result().numpy())
        self.assertLessEqual(__acc._correct.numpy(), __acc._total.numpy())
        self.assertLessEqual(__acc.result().numpy(), 1.)

    def test_byte_accuracy_different_from_group_accuracy(self):
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
        # accuracies
        __byte_acc = mlable.metrics.GroupAccuracy(group=1)
        __character_acc = mlable.metrics.GroupAccuracy(group=4)
        # evaluate
        __byte_acc.update_state(y_true=__yt, y_pred=__yp)
        __character_acc.update_state(y_true=__yt, y_pred=__yp)
        # check
        self.assertNotEqual(__byte_acc.result(), __character_acc.result())
        self.assertGreaterEqual(__byte_acc.result(), __character_acc.result())

# LOSS ########################################################################
