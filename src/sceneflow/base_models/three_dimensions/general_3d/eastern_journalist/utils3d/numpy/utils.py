import numpy as np
from numpy import ndarray
import scipy.sparse as sp
from scipy.sparse import csr_array
from typing import *
from numbers import Number, Integral
import warnings
import functools


__all__ = [
    'sliding_window',
    'pooling',
    'max_pool_2d',
    'lookup',
    'lookup_get',
    'lookup_set',
    'segment_roll',
    'segment_take',
    'segment_concatenate',
    'csr_matrix_from_dense_indices',
    'group',
    'group_as_segments'
]


def sliding_window(
    x: ndarray, 
    window_size: Union[int, Tuple[int, ...]], 
    stride: Optional[Union[int, Tuple[int, ...]]] = None, 
    dilation: Optional[Union[int, Tuple[int, ...]]] = None,
    pad_size: Optional[Union[int, Tuple[int, int], Tuple[Tuple[int, int]]]] = None, 
    pad_mode: str = 'constant',
    pad_value: Number = 0,
    axis: Optional[Tuple[int,...]] = None
) -> ndarray:
    """
    Get a sliding window of the input array. Window axis(axes) will be appended as the last dimension(s).
    This function is a wrapper of `numpy.lib.stride_tricks.sliding_window_view` with additional support for padding and stride.

    ## Parameters
    - `x` (ndarray): Input array.
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
    - (ndarray): Sliding window of the input array. 
        - If no padding, the output is a view of the input array with zero copy.
        - Otherwise, the output is no longer a view but a copy of the padded array.
    """
    # Process axis
    if axis is None:
        axis = tuple(range(x.ndim))
    if isinstance(axis, Integral):
        axis = (axis,)
    axis = [axis[i] % x.ndim for i in range(len(axis))]
    if isinstance(window_size, Integral):
        window_size = (window_size,) * len(axis)
    if dilation is not None:
        if isinstance(dilation, Integral):
            dilation = (dilation,) * len(axis)
    if stride is not None:
        if isinstance(stride, Integral):
            stride = (stride,) * len(axis)

    # Pad the input array if needed
    if pad_size is not None:
        if isinstance(pad_size, Integral):
            pad_size = ((pad_size, pad_size),) * len(axis)
        elif isinstance(pad_size, tuple) and len(pad_size) == 2 and all(isinstance(p, Integral) for p in pad_size):
            pad_size = (pad_size,) * len(axis)
        elif isinstance(pad_size, tuple) and all(isinstance(p, tuple) and 1 <= len(p) <= 2 for p in pad_size):
            if len(pad_size) == 1:
                pad_size = pad_size * len(axis)
            else:
                assert len(pad_size) == len(axis), f"pad_size {pad_size} must match the number of axes {len(axis)}"
        else:
            raise ValueError(f"Invalid pad_size {pad_size}")
        full_pad = [(0, 0) if i not in axis else pad_size[axis.index(i)] for i in range(x.ndim)]
        if pad_mode == 'constant':
            x = np.pad(x, full_pad, mode=pad_mode, constant_values=pad_value)
        else:
            x = np.pad(x, full_pad, mode=pad_mode)
    
    # Apply sliding window
    if dilation is None:
        x = np.lib.stride_tricks.sliding_window_view(x, window_size, axis=axis)
    else:
        window_size_dilated = tuple((window_size[i] - 1) * dilation[i] + 1 for i in range(len(window_size)))
        x = np.lib.stride_tricks.sliding_window_view(x, window_size_dilated, axis=axis)

    # Apply stride if needed
    if stride is not None:
        stride_slice = tuple(slice(None) if i not in axis else slice(None, None, stride[axis.index(i)]) for i in range(x.ndim - len(axis)))
        x = x[stride_slice]
    
    # Apply dilation if needed
    if dilation is not None:
        dilation_slice = tuple(slice(None, None, dilation[i]) for i in range(len(axis)))
        x = x[(..., *dilation_slice)]

    return x


def pooling(
    x: ndarray, 
    kernel_size: Union[int, Tuple[int, ...]], 
    stride: Optional[Union[int, Tuple[int, ...]]] = None, 
    padding: Optional[Union[int, Tuple[int, int], Tuple[Tuple[int, int]]]] = None, 
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    mode: Literal['min', 'max', 'sum', 'mean'] = 'max'
) -> ndarray:
    """Compute the pooling of the input array. 
    NOTE: NaNs will be ignored.

    ## Parameters
        - `x` (ndarray): Input array.
        - `kernel_size` (int or Tuple[int,...]): Size of the pooling window.
        - `stride` (Optional[Tuple[int,...]]): Stride of the pooling window. If None,
            no stride is applied. If int is provided, the same stride is used for all specified axes.
        - `padding` (Optional[Union[int, Tuple[int, int], Tuple[Tuple[int, int]]]]): Size of padding to apply before pooling.
            Corresponding to `axis`.
            - General format is `((before_1, after_1), (before_2, after_2), ...)`.
            - Shortcut formats: 
                - `int` -> same padding before and after for all axes;
                - `(int, int)` -> same padding before and after for each axis;
                - `((int,), (int,) ...)` -> specify padding for each axis, same before and after.
        - `axis` (Optional[Tuple[int,...]]): Axes to apply the pooling. If None, all axes are used.
        - `mode` (str): Pooling mode. One of 'min', 'max', 'sum', 'mean'.

    ## Returns
        - (ndarray): Pooled array with the same number of dimensions as input array.
    """
    if axis is None:
        axis = tuple(range(x.ndim))
    if isinstance(axis, Integral):
        axis = (axis,)
    axis = [axis[i] % x.ndim for i in range(len(axis))]
    if isinstance(kernel_size, Integral):
        kernel_size = (kernel_size,) * len(axis)
    if not isinstance(stride, tuple):
        stride = (stride,) * len(axis)
    if padding is not None:
        if isinstance(padding, Integral):
            padding = ((padding, padding),) * len(axis)
        elif isinstance(padding, tuple) and len(padding) == 2 and all(isinstance(p, Integral) for p in padding):
            padding = (padding,) * len(axis)
        elif isinstance(padding, tuple) and all(isinstance(p, tuple) and 1 <= len(p) <= 2 for p in padding):
            if len(padding) == 1:
                padding = padding * len(axis)
            else:
                assert len(padding) == len(axis), f"padding {padding} must match the number of axes {len(axis)}"
        else:
            raise ValueError(f"Invalid padding {padding}")
    else:
        padding = ((0, 0),) * len(axis)

    if mode == 'max':
        pad_mode = 'constant'
        pad_value = -np.inf if x.dtype.kind == 'f' else np.iinfo(x.dtype).min
        pool_fn = np.nanmax
    elif mode == 'min':
        pad_mode = 'constant'
        pad_value = np.inf if x.dtype.kind == 'f' else np.iinfo(x.dtype).max
        pool_fn = np.nanmin
    elif mode == 'sum':
        pad_mode = 'constant'
        pad_value = 0
        pool_fn = np.sum
        x = np.where(np.isnan(x), 0, x)
    elif mode == 'mean':
        mask = ~np.isnan(x)
        full_pad = [(0, 0) if i not in axis else padding[axis.index(i)] for i in range(x.ndim)]
        x = pooling(np.pad(x, full_pad, mode='edge'), kernel_size, stride, axis=axis, mode='sum')
        x /= pooling(np.pad(mask, full_pad, mode='edge'), kernel_size, stride, axis=axis, mode='sum')
        return x
    else:
        raise ValueError(f"Invalid pooling mode {mode}. Supported modes are 'min', 'max', 'sum', 'mean'.")

    for i in range(len(axis)):
        x = pool_fn(
            sliding_window(x, kernel_size[i], stride[i], 
                           pad_size=padding[i], pad_mode=pad_mode, pad_value=pad_value, 
                           axis=axis[i]), 
            axis=-1
        )
    return x


def max_pool_2d(x: ndarray, kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], padding: Union[int, Tuple[int, int]], axis: Tuple[int, int] = (-2, -1)):
    if isinstance(kernel_size, Number):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, Number):
        stride = (stride, stride)
    if isinstance(padding, Number):
        padding = (padding, padding)
    axis = tuple(axis)
    return pooling(x, kernel_size, stride, padding, axis, 'max')


def lookup(key: ndarray, query: ndarray) -> ndarray:
    """Look up `query` in `key` like a dictionary. Useful for COO indexing.

    ## Parameters
    - `key` (ndarray): shape `(K, ...)`, the array to search in
    - `query` (ndarray): shape `(Q, ...)`, the array to search for

    ## Returns
    - `indices` (ndarray): shape `(Q,)` indices of `query` in `key`. If a query is not found in key, the corresponding index will be -1.

    ## NOTE
    `O((Q + K) * log(Q + K))` complexity.
    """
    _, index, inverse = np.unique(
        np.concatenate([key, query], axis=0),
        axis=0,
        return_index=True,
        return_inverse=True
    )
    result = index[inverse[key.shape[0]:]]
    return np.where(result < key.shape[0], result, -1)


def lookup_get(key: ndarray, value: ndarray, get_key: ndarray, default_value: Union[Number, ndarray] = 0) -> ndarray:
    """Dictionary-like get for arrays

    ## Parameters
    - `key` (ndarray): shape `(N, *key_shape)`, the key array of the dictionary to get from
    - `value` (ndarray): shape `(N, *value_shape)`, the value array of the dictionary to get from
    - `get_key` (ndarray): shape `(M, *key_shape)`, the key array to get for

    ## Returns
        `get_value` (ndarray): shape `(M, *value_shape)`, result values corresponding to `get_key`
    """
    indices = lookup(key, get_key)
    return np.where(
        (indices >= 0)[(slice(None), *((None,) * (value.ndim - 1)))], 
        value[indices.clip(0, key.shape[0] - 1)], 
        default_value
    )


def lookup_set(key: ndarray, value: ndarray, set_key: ndarray, set_value: ndarray, append: bool = False, inplace: bool = False) -> Tuple[ndarray, ndarray]:
    """Dictionary-like set for arrays.

    ## Parameters
    - `key` (ndarray): shape `(N, *key_shape)`, the key array of the dictionary to set
    - `value` (ndarray): shape `(N, *value_shape)`, the value array of the dictionary to set
    - `set_key` (ndarray): shape `(M, *key_shape)`, the key array to set for
    - `set_value` (ndarray): shape `(M, *value_shape)`, the value array to set as
    - `append` (bool): If True, append the (key, value) pairs in (set_key, set_value) that are not in (key, value) to the result.
    - `inplace` (bool): If True, modify the input `value` array

    ## Returns
    - `result_key` (ndarray): shape `(N_new, *value_shape)`. N_new = N + number of new keys added if append is True, else N.
    - `result_value (ndarray): shape `(N_new, *value_shape)` 
    """
    set_indices = lookup(key, set_key)
    if inplace:
        assert append is False, "Cannot append when inplace is True"
    else:
        value = value.copy()
    hit = np.where(set_indices >= 0)
    value[set_indices[hit]] = set_value[hit]
    if append:
        missing = np.where(set_indices < 0)
        key = np.concatenate([key, set_key[missing]], axis=0)
        value = np.concatenate([value, set_value[missing]], axis=0)
    return key, value


def segment_roll(data: ndarray, offsets: ndarray, shift: int) -> ndarray:
    """Roll the data within each segment.
    """
    lengths = offsets[1:] - offsets[:-1]
    start = np.repeat(offsets[:-1], lengths)
    elem_indices = start + (np.arange(data.shape[0], dtype=offsets.dtype) - start - shift) % np.repeat(lengths, lengths)
    data = data[elem_indices]
    return data


def segment_take(data: ndarray, offsets: ndarray, taking: ndarray) -> Tuple[ndarray, ndarray]:
    """Take some segments from a segmented array

    ## Parameters
    - `data`: (ndarray) shape `(N, *data_dims)` the data to take segments from
    - `offsets`: (ndarray) shape `(M + 1,)` the offsets of the segmented data
    - `taking`: (ndarray) the indices of segments to take of shape `(K,)`, or boolean mask of shape `(M,)`

    ## Returns
    - `new_data`: (ndarray) shape `(N_new, *data_dims)`
    - `new_offsets`: (ndarray) shape `(K + 1,)` the offsets of the new segmented data
    """
    lengths = np.diff(offsets)
    new_lengths = lengths[taking]
    new_offsets = np.concatenate([[0], np.cumsum(new_lengths)])
    indices = np.arange(new_offsets[-1]) + np.repeat(offsets[taking] - new_offsets[:-1], new_lengths)
    new_data = data[indices]
    return new_data, new_offsets


def segment_concatenate(segments: List[Tuple[ndarray, ndarray]]) -> Tuple[ndarray, ndarray]:
    """Concatenate a list of segmented arrays into a single segmented array

    ## Parameters
    - `segments`: (List[Tuple[ndarray, ndarray]]) list of segmented arrays to concatenate.
        Each element is a tuple of `(data, offsets)`:
        - `data`: (ndarray) shape `(N_i, *data_dims)` the
        - `offsets`: (ndarray) shape `(M_i + 1,)` the offsets of the segmented data

    ## Returns
    - `data`: (ndarray) shape `(N, *data_dims)` the concatenated data
    - `offsets`: (ndarray) shape `(M + 1,)` the offsets of the concatenated segmented data
    """
    data_list = []
    offsets_list = [np.array([0])]
    for data, offsets in segments:
        if len(offsets) > 1:
            data_list.append(data)
            offsets_list.append(offsets[1:] + offsets_list[-1][-1])
    data = np.concatenate(data_list, axis=0)
    offsets = np.concatenate(offsets_list, axis=0)
    return data, offsets


def csr_matrix_from_dense_indices(indices: ndarray, n_cols: int) -> csr_array:
    """Convert a regular indices array to a sparse CSR adjacency matrix format

    ## Parameters
        - `indices` (ndarray): shape (N, M) dense tensor. Each one in `N` has `M` connections.
        - `n_cols` (int): total number of columns in the adjacency matrix

    ## Returns
        Tensor: shape `(N, n_cols)` sparse CSR adjacency matrix
    """
    return csr_array((
        np.ones_like(indices, dtype=bool).ravel(), 
        indices.ravel(),
        np.arange(0, indices.size + 1, indices.shape[1])
    ), shape=(indices.shape[0], n_cols))


def group(labels: ndarray, data: Optional[np.ndarray] = None) -> List[Tuple[ndarray, ndarray]]:
    """
    Split the data into groups based on the provided labels.

    ## Parameters
    - `labels` `(ndarray)` shape `(N, *label_dims)` array of labels for each data point. Labels can be multi-dimensional.
    - `data`: `(ndarray, optional)` shape `(N, *data_dims)` dense tensor. Each one in `N` has `D` features.
        If None, return the indices in each group instead.

    ## Returns
    - `groups` `(List[Tuple[ndarray, ndarray]])`: List of each group, a tuple of `(label, data_in_group)`.
        - `label` (ndarray): shape `(*label_dims,)` the label of the group.
        - `data_in_group` (ndarray): shape `(length_of_group, *data_dims)` the data points in the group.
        If `data` is None, `data_in_group` will be the indices of the data points in the original array.
    """
    group_labels, inv, counts = np.unique(labels, return_inverse=True, return_counts=True, axis=0)
    if data is None:
        data = np.arange(labels.shape[0])
    sections = np.cumsum(counts, axis=0)[:-1]
    data_groups = np.split(data[np.argsort(inv)], sections)
    return list(zip(group_labels, data_groups))


def group_as_segments(labels: ndarray, data: Optional[np.ndarray] = None) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Group as segments by labels

    ## Parameters
    - `labels` (ndarray): shape `(N, *label_dims)` array of labels for each data point. Labels can be multi-dimensional.
    - `data` (ndarray, optional): shape `(N, *data_dims)` array.
        If None, return the indices in each group instead.

    ## Returns
    Assuming there are `M` difference labels:

    - `segment_labels`: `(ndarray)` shape `(M, *label_dims)` labels of of each segment
    - `data`: `(ndarray)` shape `(N,)` or `(N, *data_dims)` the rearranged data (or indices) where the same labels are grouped as a continous segment.
    - `offsets`: `(ndarray)` shape `(M + 1,)`
    
    `data[offsets[i]:offsets[i + 1]]` corresponding to the i-th segment whose label is `segment_labels[i]`
    """
    group_labels, inv, counts = np.unique(labels, return_inverse=True, return_counts=True, axis=0)
    offsets = np.concatenate([[0], np.cumsum(counts, axis=0)])
    if data is None:
        data = np.argsort(inv)
    else:
        data = data[np.argsort(inv)]
    return group_labels, data, offsets


def take_view(a: ndarray, i: Union[int, slice], axis: int = 0) -> ndarray:
    """Take a view of the input array at the specified index along the given axis."""
    return a[(slice(None),) * (axis % a.ndim) + (i,)]


def lite_sum(a: ndarray, axis: int = -1) -> ndarray:
    """Compute the sum of the input array along the specified small axis.
    """
    result_dtype = np.result_type(a.dtype, 0)
    if a.shape[axis] == 0:
        return np.zeros(a.shape[:axis] + a.shape[axis + 1:], dtype=result_dtype)
    elif a.shape[axis] <= 4:    # Sweet point for python loop vs einsum
        s = take_view(a, 0, axis=axis).astype(result_dtype, copy=True)
        for i in range(1, a.shape[axis]):
            s += take_view(a, i, axis=axis)
        return s
    else:   # Einsum is faster than np.sum in most cases
        return np.einsum('...i->...', np.moveaxis(a, axis, -1))


def lite_prod(a: ndarray, axis: int = -1) -> ndarray:
    """Compute the product of the input array along the specified small axis.
    """
    result_dtype = np.result_type(a.dtype, 1)
    if a.shape[axis] == 0:
        return np.ones(a.shape[:axis] + a.shape[axis + 1:], dtype=result_dtype)
    elif a.shape[axis] <= 8:
        p = take_view(a, 0, axis=axis).astype(result_dtype, copy=True)
        for i in range(1, a.shape[axis]):
            p *= take_view(a, i, axis=axis)
        return p
    else:
        return np.prod(a, axis=axis)


def lite_dot(a: ndarray, b: ndarray, axis: int = -1) -> ndarray:
    """Compute the dot product of two input arrays along the specified small axis.
    """
    if a.shape[axis] == 0:
        return np.zeros(a.shape[:axis] + a.shape[axis + 1:], dtype=np.result_type(a.dtype, b.dtype))
    elif a.shape[axis] <= 3:
        return lite_sum(a * b, axis=axis)
    else:
        return np.einsum('...i,...i->...', np.moveaxis(a, axis, -1), np.moveaxis(b, axis, -1))


def lite_norm(a: ndarray, ord: int = 2, axis: int = -1) -> ndarray:
    """Compute the norm of the input array along the specified small axis.
    """
    if ord == 1:
        return lite_sum(np.abs(a), axis=axis)
    elif ord == 2:
        return np.sqrt(lite_sum(a * a, axis=axis))
    elif ord == np.inf:
        return np.max(np.abs(a), axis=axis)
    else:
        raise ValueError(f"Unsupported norm order {ord}. Supported orders are 1, 2, and inf.")