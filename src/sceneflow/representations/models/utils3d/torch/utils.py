from typing import *
from numbers import Number
from itertools import chain
from numbers import Integral

import torch
from torch import Tensor
import torch.nn.functional as F

from .helpers import batched
from ..helpers import no_warnings


__all__ = [
    'sliding_window',
    'masked_min',
    'masked_max',
    'lookup',
    'lookup_get',
    'lookup_set',
    'segment_roll',
    'segment_take',
    'csr_matrix_from_dense_indices',
    'csr_eliminate_zeros',
    'group',
    'group_as_segments'
]


def sliding_window(
    x: Tensor, 
    window_size: Union[int, Tuple[int, ...]], 
    stride: Optional[Union[int, Tuple[int, ...]]] = None, 
    dilation: Optional[Union[int, Tuple[int, ...]]] = None,
    pad_size: Optional[Union[int, Tuple[int, int], Tuple[Tuple[int, int]]]] = None, 
    pad_mode: str = 'constant',
    pad_value: Number = 0,
    dim: Tuple[int, ...] = None
) -> Tensor:
    """
    Get a sliding window of the input array.
    This function is a wrapper of `torch.nn.functional.unfold` with additional support for padding and stride.

    ## Parameters
    - `x` (Tensor): Input tensor.
    - `window_size` (int or Tuple[int,...]): Size of the sliding window. If int
        is provided, the same size is used for all specified axes.
    - `stride` (Optional[Tuple[int,...]]): Stride between the sliding windows. If None,
        no stride is applied. If int is provided, the same stride is used for all specified axes.
    - `dilation` (Optional[Tuple[int,...]]): Dilation in each sliding window. If None,
        no dilation is applied. If int is provided, the same dilation is used for all specified axes.
    - `pad_size` (Optional[Union[int, Tuple[int, int], Tuple[Tuple[int, int]]]]): Size of padding to apply before sliding window.
        Corresponding to `axis`.
        - General format is `((before_1, after_1), (before_2, after_2), ...)`.
        - Shortcut formats: 
            - `int` -> same padding before and after for all axes;
            - `(int, int)` -> same padding before and after for each axis;
            - `((int,), (int,) ...)` -> specify padding for each axis, same before and after.
    - `pad_mode` (str): Padding mode to use. Refer to `numpy.pad` for more details.
    - `pad_value` (Union[int, float]): Value to use for constant padding. Only used
        when `pad_mode` is 'constant'.
    - `axis` (Optional[Tuple[int,...]]): Axes to apply the sliding window. If None, all axes are used.

    ## Returns
    - (Tensor): Sliding window of the input array. 
        - If no padding, the output is a view of the input array with zero copy.
        - Otherwise, the output is no longer a view but a copy of the padded array.
    """
    if dim is None:
        dim = tuple(range(x.ndim))
    if isinstance(dim, Integral):
        dim = (dim,)
    dim = [dim[i] % x.ndim for i in range(len(dim))]
    if isinstance(window_size, Integral):
        window_size = (window_size,) * len(dim)
    if stride is None:
        stride = (1,) * len(dim)
    elif isinstance(stride, Integral):
        stride = (stride,) * len(dim)
    if dilation is None:
        dilation = (1,) * len(dim)
    elif isinstance(dilation, Integral):
        dilation = (dilation,) * len(dim)
    assert len(window_size) == len(stride) == len(dim)

    # Pad the input array if needed
    if pad_size is not None:
        if isinstance(pad_size, Integral):
            pad_size = ((pad_size, pad_size),) * len(dim)
        elif isinstance(pad_size, tuple) and len(pad_size) == 2 and all(isinstance(p, Integral) for p in pad_size):
            pad_size = (pad_size,) * len(dim)
        elif isinstance(pad_size, tuple) and all(isinstance(p, tuple) and 1 <= len(p) <= 2 for p in pad_size):
            if len(pad_size) == 1:
                pad_size = pad_size * len(dim)
            else:
                assert len(pad_size) == len(dim), f"pad_size {pad_size} must match the number of axes {len(dim)}"
            pad_size = tuple(p * 2 if len(p) == 1 else p for p in pad_size)
        else:
            raise ValueError(f"Invalid pad_size {pad_size}")
        full_pad = [(0, 0) if i not in dim else pad_size[dim.index(i)] for i in range(x.ndim)]
        full_pad = tuple(chain(*reversed(full_pad)))
        x = F.pad(x, full_pad, mode=pad_mode, value=pad_value)
    
    for i in range(len(window_size)):
        x = x.unfold(dim[i], (window_size[i] - 1) * dilation[i] + 1, stride[i])[..., ::dilation[i]]
    return x


def masked_min(input: Tensor, mask: torch.BoolTensor, dim: int = None, keepdim: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Similar to torch.min, but with mask
    """
    if dim is None:
        return torch.where(mask, input, torch.tensor(torch.inf, dtype=input.dtype, device=input.device)).min()
    else:
        return torch.where(mask, input, torch.tensor(torch.inf, dtype=input.dtype, device=input.device)).min(dim=dim, keepdim=keepdim)


def masked_max(input: Tensor, mask: torch.BoolTensor, dim: int = None, keepdim: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Similar to torch.max, but with mask
    """
    if dim is None:
        return torch.where(mask, input, torch.tensor(-torch.inf, dtype=input.dtype, device=input.device)).max()
    else:
        return torch.where(mask, input, torch.tensor(-torch.inf, dtype=input.dtype, device=input.device)).max(dim=dim, keepdim=keepdim)
    

def lookup(key: Tensor, query: Tensor) -> torch.LongTensor:
    """Look up `query` in `key` like a dictionary. Useful for COO indexing.

    ## Parameters
    - `key` (Tensor): shape `(K, ...)`, the array to search in
    - `query` (Tensor): shape `(Q, ...)`, the array to search for

    ## Returns
    - `indices` (Tensor): shape `(Q,)` indices of `query` in `key`. If a query is not found in key, the corresponding index will be -1.

    ## NOTE
    `O((Q + K) * log(Q + K))` complexity.
    """
    unique, inverse = torch.unique(
        torch.cat([key, query], dim=0),
        dim=0,
        return_inverse=True
    )
    index = torch.full((unique.shape[0],), -1, dtype=torch.long, device=key.device)
    index.scatter_(0, inverse[:key.shape[0]], torch.arange(key.shape[0], device=key.device))
    result = index.index_select(0, inverse[key.shape[0]:])
    return torch.where(result < key.shape[0], result, -1)


def lookup_get(key: Tensor, value: Tensor, get_key: Tensor, default_value: Union[Number, Tensor] = 0) -> Tensor:
    """Dictionary-like get for arrays

    ## Parameters
    - `key` (Tensor): shape `(N, *key_shape)`, the key array of the dictionary to get from
    - `value` (Tensor): shape `(N, *value_shape)`, the value array of the dictionary to get from
    - `get_key` (Tensor): shape `(M, *key_shape)`, the key array to get for

    ## Returns
        `get_value` (Tensor): shape `(M, *value_shape)`, result values corresponding to `get_key`
    """
    indices = lookup(key, get_key)
    return torch.where(
        (indices >= 0)[(slice(None), *((None,) * (value.ndim - 1)))], 
        value[indices.clip(0, key.shape[0] - 1)], 
        default_value
    )


def lookup_set(key: Tensor, value: Tensor, set_key: Tensor, set_value: Tensor, append: bool = False, inplace: bool = False) -> Tuple[Tensor, Tensor]:
    """Dictionary-like set for arrays.

    ## Parameters
    - `key` (Tensor): shape `(N, *key_shape)`, the key array of the dictionary to set
    - `value` (Tensor): shape `(N, *value_shape)`, the value array of the dictionary to set
    - `set_key` (Tensor): shape `(M, *key_shape)`, the key array to set for
    - `set_value` (Tensor): shape `(M, *value_shape)`, the value array to set as
    - `append` (bool): If True, append the (key, value) pairs in (set_key, set_value) that are not in (key, value) to the result.
    - `inplace` (bool): If True, modify the input `value` array

    ## Returns
    - `result_key` (Tensor): shape `(N_new, *value_shape)`. N_new = N + number of new keys added if append is True, else N.
    - `result_value (Tensor): shape `(N_new, *value_shape)` 
    """
    set_indices = lookup(key, set_key)
    if inplace:
        assert append is False, "Cannot append when inplace is True"
    else:
        value = value.clone()
    hit = torch.where(set_indices >= 0)
    value[set_indices[hit]] = set_value[hit]
    if append:
        missing = torch.where(set_indices < 0)
        key = torch.cat([key, set_key[missing]], axis=0)
        value = torch.cat([value, set_value[missing]], axis=0)
    return key, value


def segment_roll(data: torch.Tensor, offsets: torch.Tensor, shift: int) -> Tensor:
    """Roll the data within each segment.
    """
    lengths = torch.diff(offsets)
    start = offsets[:-1].repeat_interleave(lengths)
    elem_indices = start + (torch.arange(data.shape[0], dtype=offsets.dtype) - start - shift) % lengths.repeat_interleave(lengths)
    data = data.gather(0, elem_indices)
    return data


def segment_take(data: Tensor, offsets: Tensor, taking: Tensor) -> Tuple[Tensor, Tensor]:
    """Take some segments from a segmented array
    """
    lengths = torch.diff(offsets)
    new_lengths = lengths[taking]
    new_offsets = torch.cat([torch.tensor([0], dtype=lengths.dtype, device=lengths.device), torch.cumsum(new_lengths, dim=0)])
    indices = torch.arange(new_offsets[-1]) + torch.repeat_interleave(offsets[taking] - new_offsets[:-1], new_lengths)
    new_data = data.index_select(0, indices)
    return new_data, new_offsets


def csr_matrix_from_dense_indices(indices: Tensor, n_cols: int) -> Tensor:
    """Convert a regular indices array to a sparse CSR adjacency matrix format

    ## Parameters
        - `indices` (Tensor): shape (N, M) dense tensor. Each one in `N` has `M` connections.
        - `values` (Tensor): shape (N, M) values of the connections
        - `n_cols` (int): total number of columns in the adjacency matrix

    ## Returns
        Tensor: shape `(N, n_cols)` sparse CSR adjacency matrix
    """
    return torch.sparse_csr_tensor(
        crow_indices=torch.arange(0, indices.numel() + 1, indices.shape[1], device=indices.device),
        col_indices=indices.view(-1),
        values=torch.ones_like(indices, dtype=torch.bool).view(-1),
        size=(indices.shape[0], n_cols)
    )


def csr_eliminate_zeros(input: Tensor):
    """Remove zero elements from a sparse CSR tensor.
    """
    nonzero = input.values() != 0
    nonzero_element_indices = nonzero.nonzero(as_tuple=False).flatten()
    row_nonzero_count = torch.sparse_csr_tensor(
        input.crow_indices(), 
        input.col_indices(), 
        nonzero, 
        input.size()
    ).long().sum(dim=-1, keepdim=True).to_dense().flatten()
    crow_indices = torch.cat([torch.tensor([0], device=input.device), torch.cumsum(row_nonzero_count, dim=0)])
    col_indices = input.col_indices()[nonzero_element_indices]
    values = input.values()[nonzero_element_indices]
    return torch.sparse_csr_tensor(crow_indices, col_indices, values, input.size())

def csr_roll_col_indices(input: Tensor, shift: int):
    """Roll the order of column indices of a sparse CSR tensor.
    The result is mathematically equivalent to the original matrix, but with a different column order.

    ## Parameters
        - `input` (Tensor): shape `(N, M)`, sparse CSR tensor
        - `shift` (int): number of positions to shift the column indices

    ## Returns
        Tensor: shape `(N, M)` sparse CSR tensor with rolled column indices
    """
    lengths = input.crow_indices()[1:] - input.crow_indices()[:-1]
    start = input.crow_indices()[:-1].repeat_interleave(lengths)
    elem_indices = start + (torch.arange(input.col_indices().shape[0], dtype=input.col_indices().dtype) - start - shift) % lengths.repeat_interleave(lengths)
    col_indices = input.col_indices().gather(0, elem_indices)
    values = input.values().gather(0, elem_indices)
    return torch.sparse_csr_tensor(input.crow_indices(), col_indices, values, input.size())


def group(labels: Tensor, data: Optional[Tensor] = None) -> List[Tuple[Tensor, Tensor]]:
    """
    Split the data into groups based on the provided labels.

    ## Parameters
    - `labels` (Tensor): shape `(N, *label_dims)` array of labels for each data point. Labels can be multi-dimensional.
    - `data` (Tensor, optional): shape `(N, *data_dims)` dense tensor. Each one in `N` has `D` features.
        If None, return the indices in each group instead.

    ## Returns
    - `groups` (List[Tuple[Tensor, Tensor]]): List of each group, a tuple of `(label, data_in_group)`.
        - `label` (Tensor): shape (*label_dims,) the label of the group.
        - `data_in_group` (Tensor): shape (M, *data_dims) the data points in the group.
        If `data` is None, `data_in_group` will be the indices of the data points in the original array.
    """
    group_labels, inv, counts = torch.unique(labels, return_inverse=True, return_counts=True, dim=0)
    if data is None:
        data = torch.arange(labels.shape[0], device=labels.device)
    data_groups = torch.split(data[torch.argsort(inv)], counts.tolist())
    return list(zip(group_labels, data_groups))


def group_as_segments(labels: Tensor, data: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Group as segments by labels

    ## Parameters
    
    - `labels` (Tensor): shape `(N, *label_dims)` array of labels for each data point. Labels can be multi-dimensional.
    - `data` (Tensor, optional): shape `(N, *data_dims)` array.
        If None, return the indices in each group instead.

    ## Returns

    Assuming there are `M` difference labels:

    - `segment_labels`: `(Tensor)` shape `(M, *label_dims)` labels of of each segment
    - `data`: `(Tensor)` shape `(N,)` or `(N, *data_dims)` the rearranged data (or indices) where the same labels are grouped as a continous segment.
    - `offsets`: `(Tensor)` shape `(M + 1,)`
    
    `data[offsets[i]:offsets[i + 1]]` corresponding to the i-th segment whose label is `segment_labels[i]`
    """
    group_labels, inv, counts = torch.unique(labels, return_inverse=True, return_counts=True, dim=0)
    if data is None:
        data = torch.arange(labels.shape[0], device=labels.device)
    offsets = torch.cat([torch.tensor([0], dtype=counts.dtype, device=counts.device), torch.cumsum(counts, dim=0)])
    data = data[torch.argsort(inv)]
    return group_labels, data, offsets