import tensorflow as tf

# PIPELINE ####################################################################

def process(dataset: tf.data.Dataset, pipeline: list, replace: bool=True) -> tf.data.Dataset:
    __dataset = dataset
    # specify how to combine each operation result with the original dataset
    __replace = len(list(pipeline)) * [replace] if isinstance(replace, bool) else replace
    # apply the operation successively  
    for __fn, __repl in zip(pipeline, __replace):
        __new = __dataset.map(__fn)
        __dataset = __new if __repl else __dataset.concatenate(__new)
    return __dataset
