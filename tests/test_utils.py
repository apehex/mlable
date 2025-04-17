import math
import random

import tensorflow as tf

import mlable.utils

# FN COMPOSITION ##############################################################

class ComposeTest(tf.test.TestCase):
    def setUp(self):
        super(ComposeTest, self).setUp()
        self._random = [random.uniform(-8., 8.) for _ in range(32)]

    def test_identity(self):
        __f = lambda __x: __x
        __g = lambda __x: -__x
        __h = lambda __x: (__x, -__x)
        __i = lambda __t: tuple(reversed(__t))
        __f4 = mlable.utils.compose([__f, __f, __f, __f])
        __g2 = mlable.utils.compose([__g, __g])
        __hi2 = mlable.utils.compose([__h, __i, __i])
        self.assertEqual(self._random, [__f4(__e) for __e in self._random])
        self.assertEqual(self._random, [__g2(__e) for __e in self._random])
        self.assertEqual([__h(__e) for __e in self._random], [__hi2(__e) for __e in self._random])
        self.assertEqual(self._random, __f4(self._random))

# FN MAP ######################################################################

class DistributeTest(tf.test.TestCase):
    def setUp(self):
        super(DistributeTest, self).setUp()
        self._random = [random.uniform(-8., 8.) for _ in range(32)]

    def test_values(self):
        __f = lambda __x: __x ** 2
        __g = lambda __x: -__x
        __fn = mlable.utils.distribute(__f)
        __gn = mlable.utils.distribute(__g)
        self.assertEqual([(__e ** 2, __e ** 2, __e ** 2) for __e in self._random], [__fn(__e, __e, __e) for __e in self._random])
        self.assertEqual([(-__e,) for __e in self._random], [__gn(__e) for __e in self._random])
