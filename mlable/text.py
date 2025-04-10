import functools
import itertools
import math

import tensorflow as tf

import mlable.ops
import mlable.sampling
import mlable.shaping

# UNICODE ######################################################################

CODE_STX = b'\x02'
CODE_ETX = b'\x03'
CODE_FS = b'\x1c'
CODE_GS = b'\x1d'
CODE_RS = b'\x1e'
CODE_US = b'\x1f'

# 2D ###########################################################################

def split(data: tf.Tensor, height_dim: int, separator_str: str='\n', padding_str: str='') -> tf.Tensor:
    # add an axis for the substrings
    __shape = tuple(data.shape) + (height_dim,)
    # don't limit the number of splits yet
    __outputs = tf.strings.split(data, sep=separator_str, maxsplit=-1)
    # pad and truncate to enforce the shape
    return __outputs.to_tensor(default_value=padding_str, shape=__shape)

# TARGETS ######################################################################

def offset(data: tf.Tensor, ticks: int=1) -> tf.Tensor:
    return tf.convert_to_tensor([ticks * b'\x00']) + data

# ENCODE #######################################################################

def encode(data: tf.Tensor, sample_dim: int, output_dtype: tf.DType=tf.uint8, output_encoding: str='UTF-32-BE') -> tf.Tensor:
    # decode bytes from UTF-8
    __bytes = tf.strings.unicode_transcode(input=data, input_encoding='UTF-8', output_encoding=output_encoding) # (B,)
    # decode byte strings to arrays of byte integers
    return tf.io.decode_raw(__bytes, out_type=output_dtype, fixed_length=sample_dim, little_endian=False) # (B, 4 * S) or (B, S) depending on the dtype (1 or 4 bytes)

# TRIM #########################################################################

def trim(data: tf.Tensor, count: int=1, outof: int=4) -> tf.Tensor:
    # group the bytes 4 by 4 (one UTF-32 character)
    __outputs = mlable.shaping.divide(data, input_axis=-2, output_axis=-1, factor=outof, insert=True)
    # remove the most significant bytes (most often 0 in UTF-32)
    __outputs = tf.gather(__outputs, indices=range(count, outof), axis=-1)
    # flatten the data back
    return mlable.shaping.merge(__outputs, left_axis=-2, right_axis=-1, left=True)

def untrim(data: tf.Tensor, count: int=1, outof: int=4) -> tf.Tensor:
    # group the bytes codepoint by codepoint (4 bytes minus the ones that were trimmed)
    __outputs = mlable.shaping.divide(data, input_axis=-2, output_axis=-1, factor=outof - count, insert=True)
    # there may be more zeros than data => the data can't just be sliced
    __zeros = tf.zeros(tuple(__outputs.shape)[:-1] + (count,), dtype=__outputs.dtype)
    # add leading 0s to each group / codepoint
    __outputs = tf.concat([__zeros, __outputs], axis=-1)
    # flatten the data back
    return mlable.shaping.merge(__outputs, left_axis=-2, right_axis=-1, left=True)

# DECODE #######################################################################

def codepoint(data: tf.Tensor, bigendian: bool=True) -> tf.Tensor:
    # make sure the dtype is large enough for UTF-32 codepoints
    __data = tf.cast(data, dtype=tf.int32)
    # group the bytes 4 by 4
    __bytes = mlable.shaping.divide(data=__data, input_axis=-2, output_axis=-1, factor=4, insert=True)
    # compute the UTF-32-BE codepoints
    return mlable.ops.reduce_base(data=__bytes, base=256, axis=-1, keepdims=False, bigendian=bigendian)

def decode(data: tf.Tensor, encoding: str='UTF-32-BE') -> tf.Tensor:
    __data = tf.cast(data, dtype=tf.int32)
    # input = array of unicode codepoints
    __data = tf.strings.unicode_encode(__data, output_encoding=encoding)
    # convert to standard UTF-8
    return tf.strings.unicode_transcode(input=__data, input_encoding=encoding, output_encoding='UTF-8')

# CLEAN ########################################################################

def unpack(data: tf.Tensor) -> list:
    __data = data.numpy().tolist()
    return [__s.decode('utf-8') for __s in __data]

def unpad(text: str) -> str:
    return text.strip('\x00')

# > ############################################################################

def preprocess(text: str, token_dim: int, output_dtype: tf.DType=tf.uint8, output_encoding: str='UTF-32-BE') -> tf.Tensor:
    # as tensor
    __data = tf.convert_to_tensor(text, dtype=tf.string)
    # list of bytes / codepoints
    __bytes = encode(data=__data, token_dim=token_dim, sample_dim=4 * len(text), output_dtype=output_dtype, output_encoding=output_encoding)
    # expand with unitary batch dim + cast
    return tf.cast(tf.expand_dims(__bytes, axis=0), dtype=output_dtype)

# < ############################################################################

def postprocess(data: tf.Tensor, encoding: str='UTF-32-BE') -> tf.Tensor:
    # merge the bytes into codepoints
    __outputs = codepoint(data=data) if ('32' in encoding) else data
    # decode the UTF-32-BE codepoints
    return decode(data=__outputs, encoding=encoding)
