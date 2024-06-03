import tensorflow as tf

# SHAPES ######################################################################

def filter_shape(shape: list, axes: list) -> list:
    return [__d if __i in axes else 1 for __i, __d in enumerate(shape)]
