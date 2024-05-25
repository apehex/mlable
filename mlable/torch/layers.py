import math

import torch

# NORMALIZATION ###############################################################

class BatchNorm1d(torch.nn.Module):
    def __init__(self, dim: int, epsilon: float=1e-5, momentum: float=0.1, **kwargs) -> None:
        super(BatchNorm1d, self).__init__(**kwargs)
        self._epsilon = epsilon
        self._momentum = momentum
        # parameters (trained with backprop)
        self._gamma = torch.nn.Parameter(torch.ones(dim), requires_grad=True)
        self._beta = torch.nn.Parameter(torch.zeros(dim), requires_grad=True)
        # buffers (trained with a running 'momentum update')
        self._mean = torch.zeros(dim)
        self._var = torch.ones(dim)
        self.register_buffer("mean", self._mean)
        self.register_buffer("variance", self._var)
  
    def forward(self, x: torch.Tensor, training: bool, **kwargs) -> torch.Tensor:
        # current mean
        if training:
            __axes = list(range(x.ndim - 1)) # reduce all axes except the last one
            with torch.no_grad():
                __mean = x.mean(__axes, keepdim=True) # batch mean
                __var = x.var(__axes, keepdim=True) # batch variance
                self._mean = (1. - self._momentum) * self._mean + self._momentum * __mean
                self._var = (1. - self._momentum) * self._var + self._momentum * __var
        # normalize x
        __x = (x - self._mean) / torch.sqrt(self._var + self._epsilon)
        # scale
        return self._gamma * __x + self._beta

# ACTIVATION ##################################################################

class Tanh(torch.nn.Module):
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.tanh(x)

class NewGELU(torch.nn.Module):
    """Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415"""
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

# RESHAPING ###################################################################

def _normalize_shape(shape: list) -> list:
    return [-1 if __d is None else __d for __d in shape]

def _normalize_dim(dim: int) -> int:
    return -1 if (dim is None or dim < 0) else dim

def _multiply_dim(dim_l: int, dim_r: int) -> int:
    return -1 if (dim_l == -1 or dim_r == -1) else dim_l * dim_r

def _divide_dim(dim_l: int, dim_r: int) -> int:
    return -1 if (dim_l == -1 or dim_r == -1) else dim_l // dim_r

class Divide(torch.nn.Module):
    def __init__(
        self,
        input_axis: int, # relative to the NEW shape / rank
        output_axis: int, # same
        factor: int,
        insert: bool=False,
        **kwargs
    ) -> None:
        super(Divide, self).__init__(**kwargs)
        self._input_axis = input_axis
        self._output_axis = output_axis
        self._factor = factor
        self._insert = insert

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # infer the dimension of the symbolic axis
        __shape = _normalize_shape(list(inputs.shape))
        # rank, according to the new shape
        __rank = len(__shape) + int(self._insert)
        # axes, taken from the new shape
        __axis0 = self._input_axis % __rank
        __axis1 = self._output_axis % __rank
        # option to group data on a new axis
        if self._insert: __shape.insert(__axis1, 1)
        # move data from axis 0 to axis 1
        __shape[__axis0] = _divide_dim(__shape[__axis0], self._factor)
        __shape[__axis1] = _multiply_dim(__shape[__axis1], self._factor)
        return inputs.view(*__shape) #.squeeze(1)

class Merge(torch.nn.Module):
    def __init__(
        self,
        left_axis: int=-2,
        right_axis: int=-1,
        left: bool=True,
        **kwargs
    ) -> None:
        super(Merge, self).__init__(**kwargs)
        self._left_axis = left_axis
        self._right_axis = right_axis
        self._left = left

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # infer the dimension of the symbolic axis
        __shape = _normalize_shape(list(inputs.shape))
        __rank = len(__shape)
        # target axes
        __axis_l = self._left_axis % __rank
        __axis_r = self._right_axis % __rank
        # new axis
        __dim = _multiply_dim(__shape[__axis_l], __shape[__axis_r])
        __axis_k = __axis_l if self._left else __axis_r # kept axis
        __axis_d = __axis_r if self._left else __axis_l # deleted axis
        # new shape
        __shape[__axis_k] = __dim
        __shape.pop(__axis_d)
        # actually merge the two axes
        return inputs.view(*__shape)

# LINEAR ######################################################################

class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool=True, **kwargs) -> None:
        super(Linear, self).__init__(**kwargs)
        self._weight = torch.nn.Parameter(torch.randn((in_features, out_features)) / (in_features ** 0.5), requires_grad=True)
        self._bias = torch.nn.Parameter(torch.zeros(out_features), requires_grad=True) if bias else None

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        __x = torch.matmul(x, self._weight)
        if self._bias is not None:
            __x += self._bias
        return __x

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, **kwargs) -> None:
        super(Embedding, self).__init__(**kwargs)
        self._depth = num_embeddings
        self._weight = torch.nn.Parameter(torch.randn((num_embeddings, embedding_dim)), requires_grad=True)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        __x = torch.nn.functional.one_hot(input=x, num_classes=self._depth)
        return torch.matmul(__x.float(), self._weight)

class PositionalEmbedding(torch.nn.Module):
    def __init__(
        self,
        time_dim: int,
        embed_dim: int,
        input_axis: int=1, # axis of the sequence
        output_axis: int=-1, # axis of the embedding
        **kwargs
    ):
        super(PositionalEmbedding, self).__init__(**kwargs)
        # weights
        self._input_axis = input_axis
        self._output_axis = output_axis
        self._kernel = torch.nn.Parameter(torch.randn((time_dim, embed_dim)), requires_grad=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # shape
        __input_shape = list(inputs.shape)
        __axes = [self._input_axis % len(__input_shape), self._output_axis % len(__input_shape)]
        __output_shape = [(__d if __i in __axes else 1) for __i, __d in enumerate(list(__input_shape))]
        return inputs + self._kernel.view(*__output_shape) # each index in the sequence axis has a dedicated bias (different from dense bias)

# RECURRENT ###################################################################

class RNNCell(torch.nn.Module):
    def __init__(self, embed_dim: int, state_dim: int, **kwargs) -> None:
        super(RNNCell, self).__init__(**kwargs)
        self._weights = Linear(in_features=embed_dim + state_dim, out_features=state_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        __xh = torch.cat([x, h], dim=-1)
        return torch.nn.functional.tanh(self._weights(__xh))

class GRUCell(torch.nn.Module):
    def __init__(self, embed_dim: int, state_dim: int, **kwargs) -> None:
        super(GRUCell, self).__init__(**kwargs)
        # input, forget, output, gate
        self._xh_to_z = Linear(in_features=embed_dim + state_dim, out_features=state_dim)
        self._xh_to_r = Linear(in_features=embed_dim + state_dim, out_features=state_dim)
        self._xh_to_hhat = Linear(in_features=embed_dim + state_dim, out_features=state_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # state
        __xh = torch.cat([x, h], dim=-1)
        # reset gate
        __r = torch.nn.functional.sigmoid(self._xh_to_r(__xh))
        # switch gate
        __z = torch.nn.functional.sigmoid(self._xh_to_z(__xh))
        # reset state
        __xhr = torch.cat([x, __r * h], dim=-1)
        # candidate state
        __hhat = torch.nn.functional.tanh(self._xh_to_hhat(__xhr))
        # combine candidate and previous states
        return (1. - __z) * h + __z * __hhat

# ATTENTION ###################################################################

class CausalSelfAttention(torch.nn.Module):
    def __init__(self, time_dim: int, embed_dim: int, num_heads: int, **kwargs) -> None:
        super(CausalSelfAttention, self).__init__(**kwargs)
        assert embed_dim % num_heads == 0
        # key, query, value projections for all heads, but in a batch
        self._attention = Linear(in_features=embed_dim, out_features=3 * embed_dim)
        # output projection
        self._projection = Linear(in_features=embed_dim, out_features=embed_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self._mask = torch.tril(torch.ones(time_dim, time_dim)).view(1, 1, time_dim, time_dim)
        self.register_buffer("mask", self._mask)
        # save the shape
        self._head_count = num_heads
        self._head_dim = embed_dim

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # insert a new axis to group by attention head
        __shape = list(x.shape)
        __shape.insert(2, self._head_count)
        __shape[-1] = __shape[-1] // self._head_count
        # calculate query, key, values for all heads in batch
        __q, __k, __v  = self._attention(x).split(self._head_dim, dim=-1)
        # group by head rather than time
        __k = __k.view(*__shape).transpose(1, 2) # (B, H, T, E/H)
        __q = __q.view(*__shape).transpose(1, 2) # (B, H, T, E/H)
        __v = __v.view(*__shape).transpose(1, 2) # (B, H, T, E/H)
        # self-attention
        __w = (__q @ __k.transpose(-2, -1)) * (1.0 / math.sqrt(__shape[-1])) # (B, H, T, E/H) x (B, H, E/H, T) -> (B, H, T, T)
        # causal: only attend to past tokens
        __w = __w.masked_fill(self._mask == 0, float('-inf'))
        __w = torch.nn.functional.softmax(__w, dim=-1)
        # values
        __y = __w @ __v # (B, H, T, T) x (B, H, T, E/H) -> (B, H, T, E/H)
        # assemble heads
        __y = __y.transpose(1, 2).contiguous().view(*x.shape) # original shape (B, T, E)
        # output projection
        return self._projection(__y)

# BLOCKS ######################################################################

class Sequential(torch.nn.Module):
    def __init__(self, layers: list, **kwargs) -> None:
        super(Sequential, self).__init__(**kwargs)
        self._layers = layers

    def forward(self, x: torch.Tensor, training: bool=True, **kwargs) -> torch.Tensor:
        __x = x
        # forward
        for __l in self._layers:
            __x = __l(x=__x, training=training, **kwargs)
        # conclude
        return __x

class TransformerBlock(torch.nn.Module):
    def __init__(self, time_dim: int, embed_dim: int, num_heads: int, **kwargs) -> None:
        super(TransformerBlock, self).__init__(**kwargs)
        self._block = torch.nn.Sequential(
            torch.nn.LayerNorm(embed_dim),
            CausalSelfAttention(time_dim=time_dim, embed_dim=embed_dim, num_heads=num_heads),
            torch.nn.LayerNorm(embed_dim),
            Linear(embed_dim, 4 * embed_dim),
            Linear(4 * embed_dim, embed_dim),
            NewGELU())

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._block(x, **kwargs)
