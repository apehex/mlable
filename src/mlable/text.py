import functools
import itertools
import math

import numpy as np
import tensorflow as tf

import mlable.maths.ops
import mlable.sampling
import mlable.shapes
import mlable.shaping.axes

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

def codepoint(data: tf.Tensor, bigendian: bool=True) -> tf.Tensor:
    # make sure the dtype is large enough for UTF-32 codepoints
    __data = tf.cast(data, dtype=tf.int32)
    # group the bytes 4 by 4
    __bytes = mlable.shaping.axes.divide(data=__data, axis=-1, factor=4, insert=True, right=True)
    # compute the UTF-32-BE codepoints
    return mlable.maths.ops.reduce_base(data=__bytes, base=256, axis=-1, keepdims=False, bigendian=bigendian)

# TRIM #########################################################################

def trim(data: tf.Tensor, count: int=1, outof: int=4) -> tf.Tensor:
    # group the bytes 4 by 4 (one UTF-32 character)
    __outputs = mlable.shaping.axes.divide(data, axis=-1, factor=outof, insert=True, right=True)
    # remove the most significant bytes (most often 0 in UTF-32)
    __outputs = tf.gather(__outputs, indices=range(count, outof), axis=-1)
    # flatten the data back
    return mlable.shaping.axes.merge(__outputs, axis=-1, right=False)

def untrim(data: tf.Tensor, count: int=1, outof: int=4) -> tf.Tensor:
    # group the bytes codepoint by codepoint (4 bytes minus the ones that were trimmed)
    __outputs = mlable.shaping.axes.divide(data, axis=-1, factor=outof - count, insert=True, right=True)
    # there may be more zeros than data => the data can't just be sliced
    __zeros = tf.zeros(tuple(__outputs.shape)[:-1] + (count,), dtype=__outputs.dtype)
    # add leading 0s to each group / codepoint
    __outputs = tf.concat([__zeros, __outputs], axis=-1)
    # flatten the data back
    return mlable.shaping.axes.merge(__outputs, axis=-1, right=False)

# DECODE #######################################################################

def _decode(data: tf.Tensor, encoding: str='UTF-32-BE', errors: str='replace') -> tf.Tensor:
    return bytes(data).decode(encoding.lower(), errors=errors)

def decode(data: tf.Tensor, encoding: str='UTF-32-BE', errors: str='replace') -> tf.Tensor:
    # clarify the dtype to avoid interpreting the values as codepoints
    __data = tf.cast(data, dtype=tf.uint8).numpy()
    # function operating on a whole axis at once
    __string = functools.partial(_decode, encoding=encoding, errors=errors)
    # keep the spatial dimension, as the text data might be 2D or even 3D
    __text = np.apply_along_axis(__string, axis=-1, arr=__data)
    # enforce dtype
    return tf.cast(__text, tf.string)

# CLEAN ########################################################################

def unpad(data: tf.Tensor) -> tf.Tensor:
    return tf.strings.regex_replace(data, pattern='\x00', rewrite='')

def unpack(data: tf.Tensor) -> list:
    return [__s.decode('utf-8') for __s in data.numpy().tolist()]

# > ############################################################################

def preprocess(text: str, sample_dim: int, output_dtype: tf.DType=tf.uint8, output_encoding: str='UTF-32-BE') -> tf.Tensor:
    # as tensor
    __data = tf.convert_to_tensor(text, dtype=tf.string)
    # list of bytes / codepoints
    __bytes = encode(data=__data, sample_dim=sample_dim, output_dtype=output_dtype, output_encoding=output_encoding)
    # expand with unitary batch dim
    return tf.expand_dims(__bytes, axis=0)

# < ############################################################################

def postprocess(data: tf.Tensor, encoding: str='UTF-32-BE') -> tf.Tensor:
    # decode the UTF-32-BE codepoints
    __outputs = decode(data=data, encoding=encoding)
    # remove the null padding
    return unpad(__outputs)
