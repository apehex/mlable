"""Tensorflow port of the tutorial by Andrej Karpathy: https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/"""

import random

import tensorflow as tf

# META ########################################################################

N_ENCODING = 27
N_CONTEXT = 4
N_EMBEDDING = 32
N_HIDDEN = 256
N_SAMPLE = 32

N_STEPS = 1024
N_BATCH = 4096
TRAINING_RATE = 1

# DATA ########################################################################

USERNAMES = open('.data/usernames.txt', 'r').read().splitlines()
random.shuffle(USERNAMES)

N1 = int(0.8 * len(USERNAMES))
N2 = int(0.9 * len(USERNAMES))

# N-GRAMS #####################################################################

def ngrams(word: str, length: int=N_CONTEXT):
    __context = length * '.'
    for __c in word + '.':
        yield __context
        __context = __context[1:] + __c

# ENCODING ####################################################################

def stoi(c: str) -> int:
    return 0 if c == '.' else (ord(c.lower()) - 96)

def itos(i: int) -> str:
    return '.' if i == 0 else chr(i + 96)

def encode(text: str) -> tf.Tensor:
    return [stoi(__c) for __c in text]

# DATASETS ####################################################################

def dataset(words: list, context: int=N_CONTEXT, depth: int=N_ENCODING) -> tuple:
    __x = [encode(__n) for __w in words for __n in ngrams(word=__w, length=context)]
    __y = [__i for __w in words for __i in encode(__w + '.')]
    return tf.one_hot(indices=__x, depth=depth, dtype=tf.dtypes.float32), tf.one_hot(indices=__y, depth=depth, dtype=tf.dtypes.float32)

# SPLIT #######################################################################

X_TRAIN, Y_TRAIN = dataset(words=USERNAMES[:N1], context=N_CONTEXT)
X_DEV, Y_DEV = dataset(words=USERNAMES[N1:N2], context=N_CONTEXT)
X_TEST, Y_TEST = dataset(words=USERNAMES[N2:], context=N_CONTEXT)

# MODEL #######################################################################

class Model(tf.Module):
    def __init__(self, n_context: int=N_CONTEXT, n_encoding: int=N_ENCODING, n_embedding: int=N_EMBEDDING, n_hidden: int=N_HIDDEN, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # save the parameters
        self._N_CONTEXT = n_context
        self._N_ENCODING = n_encoding
        self._N_EMBEDDING = n_embedding
        self._N_HIDDEN = n_hidden
        # embedding
        self._C = tf.Variable(initial_value=tf.random.normal(shape=(self._N_ENCODING, self._N_EMBEDDING), mean=0., stddev=1., dtype=tf.dtypes.float32), name='C')
        # hidden layer
        self._W1 = tf.Variable(initial_value=0.1 * tf.random.normal(shape=(self._N_EMBEDDING * self._N_CONTEXT, self._N_HIDDEN), mean=0., stddev=1., dtype=tf.dtypes.float32), name='W1')
        self._B1 = tf.Variable(initial_value=0.1 * tf.random.normal(shape=(1, self._N_HIDDEN), mean=0., stddev=1., dtype=tf.dtypes.float32), name='B1')
        # head layer
        self._W2 = tf.Variable(initial_value=0.1 * tf.random.normal(shape=(self._N_HIDDEN, self._N_ENCODING), mean=0., stddev=1., dtype=tf.dtypes.float32), name='W2')
        self._B2 = tf.Variable(initial_value=0.1 * tf.random.normal(shape=(1, self._N_ENCODING), mean=0., stddev=1., dtype=tf.dtypes.float32), name='B2')
        # parameters
        self.parameters = [self._C, self._W1, self._W2, self._B1, self._B2]

    def __call__(self, x):
        __e = tf.reshape(x @ self._C, (x.shape[0], self._N_CONTEXT * self._N_EMBEDDING))
        __h = tf.math.tanh(__e @ self._W1 + self._B1)
        return tf.nn.softmax(__h @ self._W2 + self._B2)

# LOSS ########################################################################

def loss(target_y: tf.Tensor, predicted_y: tf.Tensor):
    __l = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0., axis=-1, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name='L')
    # __p = tf.math.multiply(x=target_y, y=1. - predicted_y) # low probability = high loss
    return __l(target_y, predicted_y)

# SAMPLE ######################################################################

def tensor(ngram: list, depth: int=N_ENCODING) -> tf.Tensor:
    return tf.one_hot(indices=[ngram], depth=depth, dtype=tf.dtypes.float32)

def _next(model: Model, x: tf.Tensor, classes: int=N_ENCODING, rand: bool=True) -> int:
    __prob = model(x)[0]
    __unigrams = tf.cast(x=100. * __prob, dtype=tf.dtypes.int32).numpy().tolist()
    __highest = tf.argmax(__prob, axis=-1).numpy()
    __random, _, _ = tf.nn.fixed_unigram_candidate_sampler(
        true_classes = tf.convert_to_tensor([range(27)], dtype=tf.dtypes.int64),
        num_true = classes,
        num_sampled = 1,
        unique = False,
        range_max = classes,
        unigrams = __unigrams)
    return __random.numpy()[0] if rand else __highest

def sample(model: Model, context: int=N_CONTEXT, depth: int=N_ENCODING, max_length: int=N_SAMPLE) -> str:
    __i = 0
    __start = int(random.uniform(0, 27))
    __result = itos(__start)
    __ngram = (context - 1) * [0,] + [__start]
    __x = tensor(ngram=__ngram, depth=depth)
    __n = _next(model=model, x=__x)
    while __n != 0 and __i < max_length:
        __ngram = __ngram[1:] + [__n]
        __x = tensor(ngram=__ngram, depth=depth)
        __n = _next(model=model, x=__x)
        __result += itos(__n)
        __i += 1
    return __result

# TRAIN #######################################################################

def step(model: Model, x: tf.Tensor, y: tf.Tensor, rate: int=TRAINING_RATE) -> None:
    with tf.GradientTape() as __tape:
        __tape.watch(model.parameters)
        # forward
        __loss = loss(y, model(x))
        # backward
        __grad = __tape.gradient(__loss, model.parameters)
        # update the model
        for __i in range(len(model.parameters)):
            model.parameters[__i].assign_sub(rate * __grad[__i])

def train(model: Model, x_train: tf.Tensor, y_train: tf.Tensor, x_test: tf.Tensor, y_test:tf.Tensor, steps: int=N_STEPS, batch: int=N_BATCH, rate: int=TRAINING_RATE) -> None:
    for __i in range(steps + 1):
        # random batch
        __indices = random.sample(population=range(x_train.shape[0]), k=batch)
        __x = tf.gather(params=x_train, indices=__indices, axis=0)
        __y = tf.gather(params=y_train, indices=__indices, axis=0)
        # update the model
        step(model=model, x=__x, y=__y, rate=rate)
        # compute loss
        if __i % 128 == 0:
            __train_loss = loss(target_y=y_train, predicted_y=model(x_train))
            __test_loss = loss(target_y=y_test, predicted_y=model(x_test))
            # log the progress
            print('[epoch {epoch}] train loss: {train} test loss: {test}'.format(epoch=__i, train=__train_loss, test=__test_loss))

# TEST ########################################################################

# MAIN ########################################################################

MODEL = Model()
train(model=MODEL, x_train=X_TRAIN, y_train=Y_TRAIN, x_test=X_TEST, y_test=Y_TEST, steps=N_STEPS, batch=N_BATCH, rate=TRAINING_RATE)

# VIZ #########################################################################

# tf.argmax(X_TRAIN, axis=-1) # convert one_hot to indices and check the dataset
# tf.argmax(Y_TRAIN, axis=-1)