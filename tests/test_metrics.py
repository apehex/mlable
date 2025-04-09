import tensorflow as tf

import mlable.metrics

# CATEGORICAL #################################################################

class CategoricalGroupAccuracyTest(tf.test.TestCase):

    def test_special_cases(self):
        __batch_dim, __seq_dim, __encoding_dim, __group_dim, __iterations = 3, 16, 16, 4, 128
        # init
        __accuracy = mlable.metrics.CategoricalGroupAccuracy(group=__group_dim)
        # no match
        __accuracy.reset_state()
        __yt = tf.one_hot(indices=tf.zeros(shape=(__batch_dim, __seq_dim), dtype=tf.dtypes.int32), depth=__encoding_dim)
        __yp = tf.one_hot(indices=tf.ones(shape=(__batch_dim, __seq_dim), dtype=tf.dtypes.int32), depth=__encoding_dim)
        self.assertEqual(__accuracy(y_true=__yt, y_pred=__yp).numpy(), 0.)
        # all match
        __accuracy.reset_state()
        __yt = tf.one_hot(indices=tf.ones(shape=(__batch_dim, __seq_dim), dtype=tf.dtypes.int32), depth=__encoding_dim)
        __yp = tf.one_hot(indices=tf.ones(shape=(__batch_dim, __seq_dim), dtype=tf.dtypes.int32), depth=__encoding_dim)
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

    def test_flat_predictions(self):
        __iterations = 16
        # init
        __byte_acc = mlable.metrics.CategoricalGroupAccuracy(group=1, depth=256)
        __character_acc = mlable.metrics.CategoricalGroupAccuracy(group=4, depth=256)
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
        # flatten
        __yt = tf.reshape(__yt, (2, -1))
        __yp = tf.reshape(__yp, (2, -1))
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
        __batch_dim, __seq_dim, __encoding_dim, __group_dim, __iterations = 3, 16, 16, 4, 128
        # init
        __accuracy = mlable.metrics.CategoricalGroupAccuracy(group=__group_dim, dtype=tf.int16)
        # single evaluation
        __accuracy.reset_state()
        __yt = tf.one_hot(indices=tf.random.uniform(shape=(__batch_dim, __seq_dim), minval=0, maxval=__encoding_dim, dtype=tf.dtypes.int32), depth=__encoding_dim)
        __yp = tf.one_hot(indices=tf.random.uniform(shape=(__batch_dim, __seq_dim), minval=0, maxval=__encoding_dim, dtype=tf.dtypes.int32), depth=__encoding_dim)
        __accuracy.update_state(y_true=__yt, y_pred=__yp)
        self.assertLessEqual(0., __accuracy.result().numpy())
        self.assertLessEqual(__accuracy.result().numpy(), 1.)
        # iterative updates
        __accuracy.reset_state()
        for _ in range(__iterations):
            __yt = tf.one_hot(indices=tf.random.uniform(shape=(__batch_dim, __seq_dim), minval=0, maxval=__encoding_dim, dtype=tf.dtypes.int32), depth=__encoding_dim)
            __yp = tf.one_hot(indices=tf.random.uniform(shape=(__batch_dim, __seq_dim), minval=0, maxval=__encoding_dim, dtype=tf.dtypes.int32), depth=__encoding_dim)
            __accuracy.update_state(y_true=__yt, y_pred=__yp)
        self.assertLessEqual(0., __accuracy.result().numpy())
        self.assertLessEqual(__accuracy.result().numpy(), 1.)
        # all match
        __accuracy.reset_state()
        __yt = tf.one_hot(indices=tf.ones(shape=(__batch_dim, __seq_dim), dtype=tf.dtypes.int32), depth=__encoding_dim)
        __yp = tf.one_hot(indices=tf.ones(shape=(__batch_dim, __seq_dim), dtype=tf.dtypes.int32), depth=__encoding_dim)
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

    def test_multiple_group_reductions(self):
        __iterations = 16
        # init
        __acc_0 = mlable.metrics.CategoricalGroupAccuracy(depth=256, group=16, axis=-1)
        __acc_1 = mlable.metrics.CategoricalGroupAccuracy(depth=256, group=[4, 2], axis=[-1, -2])
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
        # flatten
        __yt = tf.reshape(__yt, (2, -1))
        __yp = tf.reshape(__yp, (2, -1))
        # one-shot
        __acc_0.update_state(y_true=__yt, y_pred=__yp)
        __acc_1.update_state(y_true=__yt, y_pred=__yp)
        self.assertEqual(__acc_0.result().numpy(), ((0.25 + 0.75) / 2.)) # 1 / 4 on the first sample and 3 / 4 on the second
        self.assertEqual(__acc_1.result().numpy(), ((16. - 5.) / 16.)) # combine the predictions on both axes
        # unchanged when iterating
        __acc_0.reset_state()
        __acc_1.reset_state()
        for _ in range(__iterations):
            __acc_0.update_state(y_true=__yt, y_pred=__yp)
            __acc_1.update_state(y_true=__yt, y_pred=__yp)
        self.assertEqual(__acc_0.result().numpy(), ((0.25 + 0.75) / 2.)) # 1 / 4 on the first sample and 3 / 4 on the second
        self.assertEqual(__acc_1.result().numpy(), ((16. - 5.) / 16.)) # combine the predictions on both axes

# BINARY ######################################################################

class BinaryGroupAccuracyTest(tf.test.TestCase):

    def test_special_cases(self):
        __batch_dim, __seq_dim, __encoding_dim, __group_dim, __iterations = 3, 32, 8, 4, 128
        # init
        __accuracy = mlable.metrics.BinaryGroupAccuracy(group=__group_dim)
        # no match
        __accuracy.reset_state()
        __yt = tf.zeros(shape=(__batch_dim, __seq_dim, __encoding_dim), dtype=tf.dtypes.float32)
        __yp = tf.ones(shape=(__batch_dim, __seq_dim, __encoding_dim), dtype=tf.dtypes.float32)
        self.assertEqual(__accuracy(y_true=__yt, y_pred=__yp).numpy(), 0.)
        # all match
        __accuracy.reset_state()
        __yt = tf.ones(shape=(__batch_dim, __seq_dim, __encoding_dim), dtype=tf.dtypes.float32)
        __yp = tf.ones(shape=(__batch_dim, __seq_dim, __encoding_dim), dtype=tf.dtypes.float32)
        self.assertEqual(__accuracy(y_true=__yt, y_pred=__yp).numpy(), 1.)
        # iterating
        __accuracy.reset_state()
        for _ in range(__iterations):
            __accuracy.update_state(y_true=__yt, y_pred=__yp) # both ones
        self.assertEqual(__accuracy.result().numpy(), 1.)

    def test_specific_values(self):
        __iterations = 16
        # init
        __byte_acc = mlable.metrics.BinaryGroupAccuracy(group=1, dtype=tf.uint8)
        __char_acc = mlable.metrics.BinaryGroupAccuracy(group=4, dtype=tf.uint8)
        # test on ascii => leading zeroes match but not the characters
        __yt = tf.convert_to_tensor([
            [float(__b) for __b in '11111111' + '11111111'+ '11111111'+ '11111111'],
            [float(__b) for __b in '11111111' + '11111111'+ '11111111'+ '11111111']], dtype=tf.dtypes.float32)
        __yp = tf.convert_to_tensor([
            [float(__b) for __b in '10111110' + '11111111'+ '11111111'+ '11111100'],
            [float(__b) for __b in '11111111' + '11111111'+ '11111111'+ '11111111']], dtype=tf.dtypes.float32)
        # reshape
        __yt = tf.reshape(__yt, shape=(2, 4, 8))
        __yp = tf.reshape(__yp, shape=(2, 4, 8))
        # one-shot
        __byte_acc.update_state(y_true=__yt, y_pred=__yp)
        __char_acc.update_state(y_true=__yt, y_pred=__yp)
        self.assertEqual(__byte_acc.result().numpy(), 6 / 8)
        self.assertEqual(__char_acc.result().numpy(), 0.5)
        # unchanged when iterating
        __byte_acc.reset_state()
        __char_acc.reset_state()
        for _ in range(__iterations):
            __byte_acc.update_state(y_true=__yt, y_pred=__yp)
            __char_acc.update_state(y_true=__yt, y_pred=__yp)
        self.assertEqual(__byte_acc.result().numpy(), 6 / 8)
        self.assertEqual(__char_acc.result().numpy(), 0.5)

    def test_flat_predictions(self):
        __iterations = 16
        # init
        __byte_acc = mlable.metrics.BinaryGroupAccuracy(group=1, depth=8)
        __char_acc = mlable.metrics.BinaryGroupAccuracy(group=4, depth=8)
        # test on ascii => leading zeroes match but not the characters
        __yt = tf.convert_to_tensor([
            [float(__b) for __b in '11111111' + '11111111'+ '11111111'+ '11111111'],
            [float(__b) for __b in '11111111' + '11111111'+ '11111111'+ '11111111']], dtype=tf.dtypes.float32)
        __yp = tf.convert_to_tensor([
            [float(__b) for __b in '10111110' + '11111111'+ '11111111'+ '11111100'],
            [float(__b) for __b in '11111111' + '11111111'+ '11111111'+ '11111111']], dtype=tf.dtypes.float32)
        # reshape
        __yt = tf.reshape(__yt, shape=(2, 32))
        __yp = tf.reshape(__yp, shape=(2, 32))
        # one-shot
        __byte_acc.update_state(y_true=__yt, y_pred=__yp)
        __char_acc.update_state(y_true=__yt, y_pred=__yp)
        self.assertEqual(__byte_acc.result().numpy(), 6 / 8)
        self.assertEqual(__char_acc.result().numpy(), 0.5)
        # unchanged when iterating
        __byte_acc.reset_state()
        __char_acc.reset_state()
        for _ in range(__iterations):
            __byte_acc.update_state(y_true=__yt, y_pred=__yp)
            __char_acc.update_state(y_true=__yt, y_pred=__yp)
        self.assertEqual(__byte_acc.result().numpy(), 6 / 8)
        self.assertEqual(__char_acc.result().numpy(), 0.5)

    def test_bounds(self):
        # 0. <= a <= 1.
        __batch_dim, __seq_dim, __encoding_dim, __group_dim, __iterations = 3, 32, 8, 2, 128
        # init
        __accuracy = mlable.metrics.BinaryGroupAccuracy(group=__group_dim)
        # single evaluation
        __accuracy.reset_state()
        __yt = tf.random.uniform(shape=(__batch_dim, __seq_dim, __encoding_dim), minval=0., maxval=1., dtype=tf.dtypes.float32) # probs
        __yp = tf.random.uniform(shape=(__batch_dim, __seq_dim, __encoding_dim), minval=-1., maxval=1., dtype=tf.dtypes.float32) # logits
        __accuracy.update_state(y_true=__yt, y_pred=__yp)
        self.assertLessEqual(0., __accuracy.result().numpy())
        self.assertLessEqual(__accuracy.result().numpy(), 1.)
        # iterative updates
        __accuracy.reset_state()
        for _ in range(__iterations):
            __yt = tf.random.uniform(shape=(__batch_dim, __seq_dim, __encoding_dim), minval=0., maxval=1., dtype=tf.dtypes.float32) # probs
            __yp = tf.random.uniform(shape=(__batch_dim, __seq_dim, __encoding_dim), minval=-1., maxval=1., dtype=tf.dtypes.float32) # logits
            __accuracy.update_state(y_true=__yt, y_pred=__yp)
        self.assertLessEqual(0., __accuracy.result().numpy())
        self.assertLessEqual(__accuracy.result().numpy(), 1.)
        # all match
        __accuracy.reset_state()
        __yt = indices=tf.ones(shape=(__batch_dim, __seq_dim, __encoding_dim), dtype=tf.dtypes.float32)
        __yp = indices=tf.ones(shape=(__batch_dim, __seq_dim, __encoding_dim), dtype=tf.dtypes.float32)
        for _ in range(__iterations):
            __accuracy.update_state(y_true=__yt, y_pred=__yp)
        self.assertEqual(__accuracy.result().numpy(), 1.)

    def test_byte_accuracy_different_from_character_accuracy(self):
        # init
        __batch_dim, __seq_dim, __encoding_dim, __iterations = 3, 32, 4, 128
        # init
        __byte_acc = mlable.metrics.BinaryGroupAccuracy(group=1)
        __char_acc = mlable.metrics.BinaryGroupAccuracy(group=4)
        # single evaluation
        __byte_acc.reset_state()
        __char_acc.reset_state()
        __yt = tf.random.uniform(shape=(__batch_dim, __seq_dim, __encoding_dim), minval=0., maxval=1., dtype=tf.dtypes.float32) # probs
        __yp = tf.random.uniform(shape=(__batch_dim, __seq_dim, __encoding_dim), minval=-1., maxval=1., dtype=tf.dtypes.float32) # logits
        # evaluate
        __byte_acc.update_state(y_true=__yt, y_pred=__yp)
        __char_acc.update_state(y_true=__yt, y_pred=__yp)
        # check
        self.assertNotEqual(__byte_acc.result().numpy(), __char_acc.result().numpy())
        self.assertGreaterEqual(__byte_acc.result().numpy(), __char_acc.result().numpy())
