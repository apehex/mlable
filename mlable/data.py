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
