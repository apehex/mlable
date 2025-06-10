import functools
import itertools
import math
import random

import tensorflow as tf

# METADATA ####################################################################

def _label(c: str) -> str:
    return '#{}'.format(c.encode('utf-32-be').hex())

def label(token: str) -> str:
    return ' '.join(_label(__c) for __c in token)

# SERIALIZATION ###############################################################

def write(data: any, path: str, tsv: bool=True) -> None:
    with open(path, 'w') as __f:
        for __row in data:
            __line = '\t'.join(str(__v) for __v in __row) if tsv else repr(__row)[1:-1]
            __f.write(__line + '\n') # escape special characters

# STATS #######################################################################

def stats(dataset: tf.data.Dataset, count: int=None, features: list=[]) -> dict:
    # init
    __min, __avg, __max, __cnt = 0, 0, 0, 0
    # scan the whole dataset
    for __sample in itertools.islice(dataset, 0, count):
        # preprocess
        __s = tf.strings.join(inputs=[__sample[__f] for __f in features], separator='\x1d').numpy() if features else __sample.numpy()
        __l = len(__s)
        # compute
        __min = min(__min, __l)
        __max = max(__max, __l)
        __avg = __avg + __l
        __cnt = __cnt + 1
    # average
    __avg = __avg // __cnt
    # format
    return {'min': __min, 'avg': __avg, 'max': __max}

# DECOMPOSITION ################################################################

def _random_codepoint_binary(lower_plane: int=0, upper_plane: int=0x323af) -> list:
    __h = '{0:0>32b}'.format(int(random.uniform(lower_plane, upper_plane)))
    return [int(__b) for __b in __h]

def _random_codepoint_bytes(lower_plane: int=0, upper_plane: int=0x323af) -> list:
    __h = '{0:0>8x}'.format(int(random.uniform(lower_plane, upper_plane)))
    return list(bytes.fromhex(__h))

# UNICODE ######################################################################

def random_codepoint(lower_plane: int=0, upper_plane: int=0x323af, binary: bool=False) -> list:
    return (
        _random_codepoint_binary(lower_plane=lower_plane, upper_plane=upper_plane) if binary
        else _random_codepoint_bytes(lower_plane=lower_plane, upper_plane=upper_plane))

def random_character(lower_plane: int=0, upper_plane: int=0x323af) -> str:
    __h = '{0:0>8x}'.format(int(random.uniform(lower_plane, upper_plane)))
    return bytes.fromhex(__h).decode('utf-32-be', errors='replace')

# SAMPLE #######################################################################

def random_encoding(sample_size: int, lower_plane: int=0, upper_plane: int=0x323af, binary: bool=False) -> list:
    __nested = [random_codepoint(lower_plane=lower_plane, upper_plane=upper_plane, binary=binary) for _ in range(sample_size)]
    return list(itertools.chain.from_iterable(__nested))

def random_string(sample_size: int, lower_plane: int=0, upper_plane: int=0x323af) -> list:
    __nested = [random_character(lower_plane=lower_plane, upper_plane=upper_plane) for _ in range(sample_size)]
    return ''.join(__nested)

# DATASET ######################################################################

def random_dataset_of_bytes(sample_count: int, sample_size: int) -> tf.data.Dataset:
    # sample generator
    def __generator() -> iter:
        for _ in range(sample_count):
            yield [random.randint(0, 0xff) for _ in range(sample_size)]
    # wrap into a dataset
    return tf.data.Dataset.from_generator(
        generator=__generator,
        output_signature=tf.TensorSpec(shape=(sample_size,), dtype=tf.int32))

def random_dataset_of_codepoints(sample_count: int, sample_size: int, lower_plane: int=0, upper_plane: int=0x323af, binary: bool=False) -> tf.data.Dataset:
    __factor = 32 if binary else 4
    # sample generator
    def __generator() -> iter:
        for _ in range(sample_count):
            yield random_encoding(sample_size=sample_size, lower_plane=lower_plane, upper_plane=upper_plane, binary=binary)
    # wrap into a dataset
    return tf.data.Dataset.from_generator(
        generator=__generator,
        output_signature=tf.TensorSpec(shape=(__factor * sample_size,), dtype=tf.int32))

def random_dataset_of_strings(sample_count: int, sample_size: int, lower_plane: int=0, upper_plane: int=0x323af) -> tf.data.Dataset:
    # sample generator
    def __generator() -> iter:
        for _ in range(sample_count):
            yield random_string(sample_size=sample_size, lower_plane=lower_plane, upper_plane=upper_plane)
    # wrap into a dataset
    return tf.data.Dataset.from_generator(
        generator=__generator,
        output_signature=tf.TensorSpec(shape=(), dtype=tf.string))
