# Auto-generated interface file
from typing import List, Tuple, Dict, Union, Optional, Any, overload, Literal, Callable
from typing_extensions import Unpack
import numpy as numpy_
import torch as torch_
import nvdiffrast.torch
import numbers
from . import numpy, torch
import utils3d.numpy, utils3d.torch

__all__ = ["sliding_window", 
"pooling", 
"max_pool_2d", 
"lookup", 
"lookup_get", 
"lookup_set", 
"segment_roll", 
"segment_take", 
"segment_concatenate", 
"csr_matrix_from_dense_indices", 
"group", 
"group_as_segments", 
"perspective_from_fov", 
"perspective_from_window", 
"intrinsics_from_fov", 
"intrinsics_from_focal_center", 
"fov_to_focal", 
"focal_to_fov", 
"intrinsics_to_fov", 
"view_look_at", 
"extrinsics_look_at", 
"perspective_to_intrinsics", 
"perspective_to_near_far", 
"intrinsics_to_perspective", 
"extrinsics_to_view", 
"view_to_extrinsics", 
"normalize_intrinsics", 
"denormalize_intrinsics", 
"crop_intrinsics", 
"pixel_to_uv", 
"pixel_to_ndc", 
"uv_to_pixel", 
"depth_linear_to_buffer", 
"depth_buffer_to_linear", 
"unproject_cv", 
"unproject_gl", 
"project_cv", 
"project_gl", 
"project", 
"unproject", 
"screen_coord_to_view_coord", 
"quaternion_to_matrix", 
"axis_angle_to_matrix", 
"matrix_to_quaternion", 
"extrinsics_to_essential", 
"axis_angle_to_quaternion", 
"euler_axis_angle_rotation", 
"euler_angles_to_matrix", 
"matrix_to_axis_angle", 
"matrix_to_euler_angles", 
"quaternion_to_axis_angle", 
"skew_symmetric", 
"rotation_matrix_from_vectors", 
"ray_intersection", 
"make_affine_matrix", 
"random_rotation_matrix", 
"lerp", 
"slerp", 
"slerp_rotation_matrix", 
"interpolate_se3_matrix", 
"piecewise_lerp", 
"piecewise_interpolate_se3_matrix", 
"transform_points", 
"angle_between", 
"vector_outer", 
"procrustes", 
"solve_pose", 
"solve_poses_sequential", 
"triangulate_mesh", 
"compute_face_corner_angles", 
"compute_face_corner_normals", 
"compute_face_corner_tangents", 
"compute_face_normals", 
"compute_face_tangents", 
"compute_vertex_normals", 
"remove_corrupted_faces", 
"merge_duplicate_vertices", 
"remove_unused_vertices", 
"subdivide_mesh", 
"mesh_edges", 
"mesh_half_edges", 
"mesh_connected_components", 
"graph_connected_components", 
"mesh_adjacency_graph", 
"flatten_mesh_indices", 
"create_cube_mesh", 
"create_icosahedron_mesh", 
"create_square_mesh", 
"create_camera_frustum_mesh", 
"merge_meshes", 
"uv_map", 
"pixel_coord_map", 
"screen_coord_map", 
"build_mesh_from_map", 
"build_mesh_from_depth_map", 
"depth_map_edge", 
"depth_map_aliasing", 
"normal_map_edge", 
"point_map_to_normal_map", 
"depth_map_to_point_map", 
"depth_map_to_normal_map", 
"chessboard", 
"masked_nearest_resize", 
"masked_area_resize", 
"colorize_depth_map", 
"colorize_normal_map", 
"flood_fill", 
"perlin_noise", 
"perlin_noise_map", 
"fractal_perlin_noise_map", 
"RastContext", 
"rasterize_triangles", 
"rasterize_triangles_peeling", 
"rasterize_lines", 
"rasterize_point_cloud", 
"sample_texture", 
"test_rasterization", 
"read_extrinsics_from_colmap", 
"read_intrinsics_from_colmap", 
"write_extrinsics_as_colmap", 
"write_intrinsics_as_colmap", 
"read_obj", 
"write_obj", 
"masked_min", 
"masked_max", 
"csr_eliminate_zeros", 
"rotation_matrix_2d", 
"rotate_2d", 
"translate_2d", 
"scale_2d", 
"mesh_dual_graph", 
"compute_boundaries", 
"remove_isolated_pieces", 
"compute_mesh_laplacian", 
"laplacian_smooth_mesh", 
"taubin_smooth_mesh", 
"laplacian_hc_smooth_mesh", 
"bounding_rect_from_mask", 
"texture_composite"]

@overload
def sliding_window(x: numpy_.ndarray, window_size: Union[int, Tuple[int, ...]], stride: Union[int, Tuple[int, ...], NoneType] = None, dilation: Union[int, Tuple[int, ...], NoneType] = None, pad_size: Union[int, Tuple[int, int], Tuple[Tuple[int, int]], NoneType] = None, pad_mode: str = 'constant', pad_value: numbers.Number = 0, axis: Optional[Tuple[int, ...]] = None) -> numpy_.ndarray:
    """Get a sliding window of the input array. Window axis(axes) will be appended as the last dimension(s).
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
    - Otherwise, the output is no longer a view but a copy of the padded array."""
    utils3d.numpy.utils.sliding_window

@overload
def pooling(x: numpy_.ndarray, kernel_size: Union[int, Tuple[int, ...]], stride: Union[int, Tuple[int, ...], NoneType] = None, padding: Union[int, Tuple[int, int], Tuple[Tuple[int, int]], NoneType] = None, axis: Union[int, Tuple[int, ...], NoneType] = None, mode: Literal['min', 'max', 'sum', 'mean'] = 'max') -> numpy_.ndarray:
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
    - (ndarray): Pooled array with the same number of dimensions as input array."""
    utils3d.numpy.utils.pooling

@overload
def max_pool_2d(x: numpy_.ndarray, kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], padding: Union[int, Tuple[int, int]], axis: Tuple[int, int] = (-2, -1)):
    utils3d.numpy.utils.max_pool_2d

@overload
def lookup(key: numpy_.ndarray, query: numpy_.ndarray) -> numpy_.ndarray:
    """Look up `query` in `key` like a dictionary. Useful for COO indexing.

## Parameters
- `key` (ndarray): shape `(K, ...)`, the array to search in
- `query` (ndarray): shape `(Q, ...)`, the array to search for

## Returns
- `indices` (ndarray): shape `(Q,)` indices of `query` in `key`. If a query is not found in key, the corresponding index will be -1.

## NOTE
`O((Q + K) * log(Q + K))` complexity."""
    utils3d.numpy.utils.lookup

@overload
def lookup_get(key: numpy_.ndarray, value: numpy_.ndarray, get_key: numpy_.ndarray, default_value: Union[numbers.Number, numpy_.ndarray] = 0) -> numpy_.ndarray:
    """Dictionary-like get for arrays

## Parameters
- `key` (ndarray): shape `(N, *key_shape)`, the key array of the dictionary to get from
- `value` (ndarray): shape `(N, *value_shape)`, the value array of the dictionary to get from
- `get_key` (ndarray): shape `(M, *key_shape)`, the key array to get for

## Returns
    `get_value` (ndarray): shape `(M, *value_shape)`, result values corresponding to `get_key`"""
    utils3d.numpy.utils.lookup_get

@overload
def lookup_set(key: numpy_.ndarray, value: numpy_.ndarray, set_key: numpy_.ndarray, set_value: numpy_.ndarray, append: bool = False, inplace: bool = False) -> Tuple[numpy_.ndarray, numpy_.ndarray]:
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
- `result_value (ndarray): shape `(N_new, *value_shape)` """
    utils3d.numpy.utils.lookup_set

@overload
def segment_roll(data: numpy_.ndarray, offsets: numpy_.ndarray, shift: int) -> numpy_.ndarray:
    """Roll the data within each segment.
    """
    utils3d.numpy.utils.segment_roll

@overload
def segment_take(data: numpy_.ndarray, offsets: numpy_.ndarray, taking: numpy_.ndarray) -> Tuple[numpy_.ndarray, numpy_.ndarray]:
    """Take some segments from a segmented array

## Parameters
- `data`: (ndarray) shape `(N, *data_dims)` the data to take segments from
- `offsets`: (ndarray) shape `(M + 1,)` the offsets of the segmented data
- `taking`: (ndarray) the indices of segments to take of shape `(K,)`, or boolean mask of shape `(M,)`

## Returns
- `new_data`: (ndarray) shape `(N_new, *data_dims)`
- `new_offsets`: (ndarray) shape `(K + 1,)` the offsets of the new segmented data"""
    utils3d.numpy.utils.segment_take

@overload
def segment_concatenate(segments: List[Tuple[numpy_.ndarray, numpy_.ndarray]]) -> Tuple[numpy_.ndarray, numpy_.ndarray]:
    """Concatenate a list of segmented arrays into a single segmented array

## Parameters
- `segments`: (List[Tuple[ndarray, ndarray]]) list of segmented arrays to concatenate.
    Each element is a tuple of `(data, offsets)`:
    - `data`: (ndarray) shape `(N_i, *data_dims)` the
    - `offsets`: (ndarray) shape `(M_i + 1,)` the offsets of the segmented data

## Returns
- `data`: (ndarray) shape `(N, *data_dims)` the concatenated data
- `offsets`: (ndarray) shape `(M + 1,)` the offsets of the concatenated segmented data"""
    utils3d.numpy.utils.segment_concatenate

@overload
def csr_matrix_from_dense_indices(indices: numpy_.ndarray, n_cols: int) -> scipy.sparse._csr.csr_array:
    """Convert a regular indices array to a sparse CSR adjacency matrix format

## Parameters
    - `indices` (ndarray): shape (N, M) dense tensor. Each one in `N` has `M` connections.
    - `n_cols` (int): total number of columns in the adjacency matrix

## Returns
    Tensor: shape `(N, n_cols)` sparse CSR adjacency matrix"""
    utils3d.numpy.utils.csr_matrix_from_dense_indices

@overload
def group(labels: numpy_.ndarray, data: Optional[numpy_.ndarray] = None) -> List[Tuple[numpy_.ndarray, numpy_.ndarray]]:
    """Split the data into groups based on the provided labels.

## Parameters
- `labels` `(ndarray)` shape `(N, *label_dims)` array of labels for each data point. Labels can be multi-dimensional.
- `data`: `(ndarray, optional)` shape `(N, *data_dims)` dense tensor. Each one in `N` has `D` features.
    If None, return the indices in each group instead.

## Returns
- `groups` `(List[Tuple[ndarray, ndarray]])`: List of each group, a tuple of `(label, data_in_group)`.
    - `label` (ndarray): shape `(*label_dims,)` the label of the group.
    - `data_in_group` (ndarray): shape `(length_of_group, *data_dims)` the data points in the group.
    If `data` is None, `data_in_group` will be the indices of the data points in the original array."""
    utils3d.numpy.utils.group

@overload
def group_as_segments(labels: numpy_.ndarray, data: Optional[numpy_.ndarray] = None) -> Tuple[numpy_.ndarray, numpy_.ndarray, numpy_.ndarray]:
    """Group as segments by labels

## Parameters
- `labels` (ndarray): shape `(N, *label_dims)` array of labels for each data point. Labels can be multi-dimensional.
- `data` (ndarray, optional): shape `(N, *data_dims)` array.
    If None, return the indices in each group instead.

## Returns
Assuming there are `M` difference labels:

- `segment_labels`: `(ndarray)` shape `(M, *label_dims)` labels of of each segment
- `data`: `(ndarray)` shape `(N,)` or `(N, *data_dims)` the rearranged data (or indices) where the same labels are grouped as a continous segment.
- `offsets`: `(ndarray)` shape `(M + 1,)`

`data[offsets[i]:offsets[i + 1]]` corresponding to the i-th segment whose label is `segment_labels[i]`"""
    utils3d.numpy.utils.group_as_segments

@overload
def perspective_from_fov(*, fov_x: Union[float, numpy_.ndarray, NoneType] = None, fov_y: Union[float, numpy_.ndarray, NoneType] = None, fov_min: Union[float, numpy_.ndarray, NoneType] = None, fov_max: Union[float, numpy_.ndarray, NoneType] = None, aspect_ratio: Union[float, numpy_.ndarray, NoneType] = None, near: Union[float, numpy_.ndarray, NoneType], far: Union[float, numpy_.ndarray, NoneType]) -> numpy_.ndarray:
    """Get OpenGL perspective matrix from field of view 

## Returns
    (ndarray): [..., 4, 4] perspective matrix"""
    utils3d.numpy.transforms.perspective_from_fov

@overload
def perspective_from_window(left: Union[float, numpy_.ndarray], right: Union[float, numpy_.ndarray], bottom: Union[float, numpy_.ndarray], top: Union[float, numpy_.ndarray], near: Union[float, numpy_.ndarray], far: Union[float, numpy_.ndarray]) -> numpy_.ndarray:
    """Get OpenGL perspective matrix from the window of z=-1 projection plane

## Returns
    (ndarray): [..., 4, 4] perspective matrix"""
    utils3d.numpy.transforms.perspective_from_window

@overload
def intrinsics_from_fov(fov_x: Union[float, numpy_.ndarray, NoneType] = None, fov_y: Union[float, numpy_.ndarray, NoneType] = None, fov_max: Union[float, numpy_.ndarray, NoneType] = None, fov_min: Union[float, numpy_.ndarray, NoneType] = None, aspect_ratio: Union[float, numpy_.ndarray, NoneType] = None) -> numpy_.ndarray:
    """Get normalized OpenCV intrinsics matrix from given field of view.
You can provide either fov_x, fov_y, fov_max or fov_min and aspect_ratio

## Parameters
    fov_x (float | ndarray): field of view in x axis
    fov_y (float | ndarray): field of view in y axis
    fov_max (float | ndarray): field of view in largest dimension
    fov_min (float | ndarray): field of view in smallest dimension
    aspect_ratio (float | ndarray): aspect ratio of the image

## Returns
    (ndarray): [..., 3, 3] OpenCV intrinsics matrix"""
    utils3d.numpy.transforms.intrinsics_from_fov

@overload
def intrinsics_from_focal_center(fx: Union[float, numpy_.ndarray], fy: Union[float, numpy_.ndarray], cx: Union[float, numpy_.ndarray], cy: Union[float, numpy_.ndarray]) -> numpy_.ndarray:
    """Get OpenCV intrinsics matrix

## Returns
    (ndarray): [..., 3, 3] OpenCV intrinsics matrix"""
    utils3d.numpy.transforms.intrinsics_from_focal_center

@overload
def fov_to_focal(fov: numpy_.ndarray):
    utils3d.numpy.transforms.fov_to_focal

@overload
def focal_to_fov(focal: numpy_.ndarray):
    utils3d.numpy.transforms.focal_to_fov

@overload
def intrinsics_to_fov(intrinsics: numpy_.ndarray) -> Tuple[numpy_.ndarray, numpy_.ndarray]:
    utils3d.numpy.transforms.intrinsics_to_fov

@overload
def view_look_at(eye: numpy_.ndarray, look_at: numpy_.ndarray, up: numpy_.ndarray) -> numpy_.ndarray:
    """Get OpenGL view matrix looking at something

## Parameters
    eye (ndarray): [..., 3] the eye position
    look_at (ndarray): [..., 3] the position to look at
    up (ndarray): [..., 3] head up direction (y axis in screen space). Not necessarily othogonal to view direction

## Returns
    (ndarray): [..., 4, 4], view matrix"""
    utils3d.numpy.transforms.view_look_at

@overload
def extrinsics_look_at(eye: numpy_.ndarray, look_at: numpy_.ndarray, up: numpy_.ndarray) -> numpy_.ndarray:
    """Get OpenCV extrinsics matrix looking at something

## Parameters
    eye (ndarray): [..., 3] the eye position
    look_at (ndarray): [..., 3] the position to look at
    up (ndarray): [..., 3] head up direction (-y axis in screen space). Not necessarily othogonal to view direction

## Returns
    (ndarray): [..., 4, 4], extrinsics matrix"""
    utils3d.numpy.transforms.extrinsics_look_at

@overload
def perspective_to_intrinsics(perspective: numpy_.ndarray) -> numpy_.ndarray:
    """OpenGL perspective matrix to OpenCV intrinsics

## Parameters
    perspective (ndarray): [..., 4, 4] OpenGL perspective matrix

## Returns
    (ndarray): shape [..., 3, 3] OpenCV intrinsics"""
    utils3d.numpy.transforms.perspective_to_intrinsics

@overload
def perspective_to_near_far(perspective: numpy_.ndarray) -> Tuple[numpy_.ndarray, numpy_.ndarray]:
    """Get near and far planes from OpenGL perspective matrix

## Parameters"""
    utils3d.numpy.transforms.perspective_to_near_far

@overload
def intrinsics_to_perspective(intrinsics: numpy_.ndarray, near: Union[float, numpy_.ndarray], far: Union[float, numpy_.ndarray]) -> numpy_.ndarray:
    """OpenCV intrinsics to OpenGL perspective matrix

## Parameters
    intrinsics (ndarray): [..., 3, 3] OpenCV intrinsics matrix
    near (float | ndarray): [...] near plane to clip
    far (float | ndarray): [...] far plane to clip
## Returns
    (ndarray): [..., 4, 4] OpenGL perspective matrix"""
    utils3d.numpy.transforms.intrinsics_to_perspective

@overload
def extrinsics_to_view(extrinsics: numpy_.ndarray) -> numpy_.ndarray:
    """OpenCV camera extrinsics to OpenGL view matrix

## Parameters
    extrinsics (ndarray): [..., 4, 4] OpenCV camera extrinsics matrix

## Returns
    (ndarray): [..., 4, 4] OpenGL view matrix"""
    utils3d.numpy.transforms.extrinsics_to_view

@overload
def view_to_extrinsics(view: numpy_.ndarray) -> numpy_.ndarray:
    """OpenGL view matrix to OpenCV camera extrinsics

## Parameters
    view (ndarray): [..., 4, 4] OpenGL view matrix

## Returns
    (ndarray): [..., 4, 4] OpenCV camera extrinsics matrix"""
    utils3d.numpy.transforms.view_to_extrinsics

@overload
def normalize_intrinsics(intrinsics: numpy_.ndarray, size: Union[Tuple[numbers.Number, numbers.Number], numpy_.ndarray], pixel_convention: Literal['integer-center', 'integer-corner'] = 'integer-center') -> numpy_.ndarray:
    """Normalize intrinsics from pixel cooridnates to uv coordinates

## Parameters
- `intrinsics` (ndarray): `(..., 3, 3)` camera intrinsics to normalize
- `size` (tuple | ndarray): A tuple `(height, width)` of the image size,
    or an array of shape `(..., 2)` corresponding to the multiple image size(s)
- `pixel_convention`: `str`, optional `'integer-center'` or `'integer-corner'`, whether integer coordinates correspond to pixel centers or corners. Defaults to 'integer-center'.
    - For more definitions, please refer to `pixel_coord_map()`

## Returns
    `(ndarray)`: `(..., 3, 3)` normalized camera intrinsics(s)"""
    utils3d.numpy.transforms.normalize_intrinsics

@overload
def denormalize_intrinsics(intrinsics: numpy_.ndarray, size: Union[Tuple[numbers.Number, numbers.Number], numpy_.ndarray], pixel_convention: Literal['integer-center', 'integer-corner'] = 'integer-center') -> numpy_.ndarray:
    """Denormalize intrinsics from uv cooridnates to pixel coordinates

## Parameters
- `intrinsics` (ndarray): `(..., 3, 3)` camera intrinsics to denormalize
- `size` (tuple | ndarray): A tuple `(height, width)` of the image size,
    or an array of shape `(..., 2)` corresponding to the multiple image size(s)
- `pixel_convention`: `str`, optional `'integer-center'` or `'integer-corner'`, whether integer coordinates correspond to pixel centers or corners. Defaults to 'integer-center'.
    - For more definitions, please refer to `pixel_coord_map()`

## Returns
    `(ndarray)`: `(..., 3, 3)` denormalized camera intrinsics in pixel coordinates"""
    utils3d.numpy.transforms.denormalize_intrinsics

@overload
def crop_intrinsics(intrinsics: numpy_.ndarray, size: Union[Tuple[numbers.Number, numbers.Number], numpy_.ndarray], cropped_top: Union[numbers.Number, numpy_.ndarray], cropped_left: Union[numbers.Number, numpy_.ndarray], cropped_height: Union[numbers.Number, numpy_.ndarray], cropped_width: Union[numbers.Number, numpy_.ndarray]) -> numpy_.ndarray:
    """Evaluate the new intrinsics after cropping the image

## Parameters
- `intrinsics` (ndarray): (..., 3, 3) camera intrinsics(s) to crop
- `size` (tuple | ndarray): A tuple `(height, width)` of the image size,
    or an array of shape `(..., 2)` corresponding to the multiple image size(s)
- `cropped_top` (int | ndarray): (...) top pixel index of the cropped image(s)
- `cropped_left` (int | ndarray): (...) left pixel index of the cropped image(s)
- `cropped_height` (int | ndarray): (...) height of the cropped image(s)
- `cropped_width` (int | ndarray): (...) width of the cropped image(s)

## Returns
    (ndarray): (..., 3, 3) cropped camera intrinsics"""
    utils3d.numpy.transforms.crop_intrinsics

@overload
def pixel_to_uv(pixel: numpy_.ndarray, size: Union[Tuple[numbers.Number, numbers.Number], numpy_.ndarray], pixel_convention: Literal['integer-center', 'integer-corner'] = 'integer-center') -> numpy_.ndarray:
    """Convert pixel space coordiantes to UV space coordinates.

## Parameters
- `pixel` (ndarray): `(..., 2)` pixel coordinrates 
- `size` (tuple | ndarray): A tuple `(height, width)` of the image size,
    or an array of shape `(..., 2)` corresponding to the multiple image size(s)
- `pixel_convention`: `str`, optional `'integer-center'` or `'integer-corner'`, whether integer coordinates correspond to pixel centers or corners. Defaults to 'integer-center'.
    - For more definitions, please refer to `pixel_coord_map()`

## Returns
    (ndarray): `(..., 2)` uv coordinrates"""
    utils3d.numpy.transforms.pixel_to_uv

@overload
def pixel_to_ndc(pixel: numpy_.ndarray, size: Union[Tuple[numbers.Number, numbers.Number], numpy_.ndarray], pixel_convention: Literal['integer-center', 'integer-corner'] = 'integer-center') -> numpy_.ndarray:
    """Convert pixel coordinates to NDC (Normalized Device Coordinates).

## Parameters
- `pixel` (ndarray): `(..., 2)` pixel coordinrates.
- `size` (tuple | ndarray): A tuple `(height, width)` of the image size,
    or an array of shape `(..., 2)` corresponding to the multiple image size(s)
- `pixel_convention`: `str`, optional `'integer-center'` or `'integer-corner'`, whether integer coordinates represent pixel centers or corners. Defaults to 'integer-center'.
    - For more definitions, please refer to `pixel_coord_map()`

## Returns
    (ndarray): `(..., 2)` ndc coordinrates, the range is (-1, 1)"""
    utils3d.numpy.transforms.pixel_to_ndc

@overload
def uv_to_pixel(uv: numpy_.ndarray, size: Union[Tuple[numbers.Number, numbers.Number], numpy_.ndarray], pixel_convention: Literal['integer-center', 'integer-corner'] = 'integer-center') -> numpy_.ndarray:
    """Convert UV space coordinates to pixel space coordinates.

## Parameters
- `uv` (ndarray): `(..., 2)` uv coordinrates.
- `size` (tuple | ndarray): A tuple `(height, width)` of the image size,
    or an array of shape `(..., 2)` corresponding to the multiple image size(s)
- `pixel_convention`: `str`, optional `'integer-center'` or `'integer-corner'`, whether integer coordinates correspond to pixel centers or corners. Defaults to 'integer-center'.
    - For more definitions, please refer to `pixel_coord_map()`

## Returns
    (ndarray): `(..., 2)` pixel coordinrates"""
    utils3d.numpy.transforms.uv_to_pixel

@overload
def depth_linear_to_buffer(depth: numpy_.ndarray, near: Union[float, numpy_.ndarray], far: Union[float, numpy_.ndarray]) -> numpy_.ndarray:
    """Project linear depth to depth value in screen space

## Parameters
    depth (ndarray): [...] depth value
    near (float | ndarray): [...] near plane to clip
    far (float | ndarray): [...] far plane to clip

## Returns
    (ndarray): [..., 1] depth value in screen space, value ranging in [0, 1]"""
    utils3d.numpy.transforms.depth_linear_to_buffer

@overload
def depth_buffer_to_linear(depth_buffer: numpy_.ndarray, near: Union[float, numpy_.ndarray], far: Union[float, numpy_.ndarray]) -> numpy_.ndarray:
    """OpenGL depth buffer to linear depth

## Parameters
    depth_buffer (ndarray): [...] depth value
    near (float | ndarray): [...] near plane to clip
    far (float | ndarray): [...] far plane to clip

## Returns
    (ndarray): [..., 1] linear depth"""
    utils3d.numpy.transforms.depth_buffer_to_linear

@overload
def unproject_cv(uv: numpy_.ndarray, depth: numpy_.ndarray, intrinsics: numpy_.ndarray, extrinsics: numpy_.ndarray = None) -> numpy_.ndarray:
    """Unproject uv coordinates to 3D view space following the OpenCV convention

## Parameters
    uv (ndarray): [..., N, 2] uv coordinates, value ranging in [0, 1].
        The origin (0., 0.) is corresponding to the left & top
    depth (ndarray): [..., N] depth value
    extrinsics (ndarray): [..., 4, 4] extrinsics matrix
    intrinsics (ndarray): [..., 3, 3] intrinsics matrix

## Returns
    points (ndarray): [..., N, 3] 3d points"""
    utils3d.numpy.transforms.unproject_cv

@overload
def unproject_gl(uv: numpy_.ndarray, depth: numpy_.ndarray, projection: numpy_.ndarray, view: Optional[numpy_.ndarray] = None) -> numpy_.ndarray:
    """Unproject screen space coordinates to 3D view space following the OpenGL convention (except for row major matrices)

## Parameters
    uv (ndarray): (..., N, 2) screen space XY coordinates, value ranging in [0, 1].
        The origin (0., 0.) is corresponding to the left & bottom
    depth (ndarray): (..., N) linear depth values
    projection (ndarray): (..., 4, 4) projection  matrix
    view (ndarray): (..., 4, 4) view matrix
    
## Returns
    points (ndarray): (..., N, 3) 3d points"""
    utils3d.numpy.transforms.unproject_gl

@overload
def project_cv(points: numpy_.ndarray, intrinsics: numpy_.ndarray, extrinsics: Optional[numpy_.ndarray] = None) -> Tuple[numpy_.ndarray, numpy_.ndarray]:
    """Project 3D points to 2D following the OpenCV convention

## Parameters
    points (ndarray): [..., N, 3]
    extrinsics (ndarray): [..., 4, 4] extrinsics matrix
    intrinsics (ndarray): [..., 3, 3] intrinsics matrix

## Returns
    uv_coord (ndarray): [..., N, 2] uv coordinates, value ranging in [0, 1].
        The origin (0., 0.) is corresponding to the left & top
    linear_depth (ndarray): [..., N] linear depth"""
    utils3d.numpy.transforms.project_cv

@overload
def project_gl(points: numpy_.ndarray, projection: numpy_.ndarray, view: numpy_.ndarray = None) -> Tuple[numpy_.ndarray, numpy_.ndarray]:
    """Project 3D points to 2D following the OpenGL convention (except for row major matrices)

## Parameters
    points (ndarray): [..., N, 3] or [..., N, 4] 3D points to project, if the last 
        dimension is 4, the points are assumed to be in homogeneous coordinates
    view (ndarray): [..., 4, 4] view matrix
    projection (ndarray): [..., 4, 4] projection matrix

## Returns
    scr_coord (ndarray): [..., N, 2] OpenGL screen space XY coordinates, value ranging in [0, 1].
        The origin (0., 0.) is corresponding to the left & bottom
    linear_depth (ndarray): [..., N] linear depth"""
    utils3d.numpy.transforms.project_gl

@overload
def project(points: numpy_.ndarray, *, intrinsics: Optional[numpy_.ndarray] = None, extrinsics: Optional[numpy_.ndarray] = None, view: Optional[numpy_.ndarray] = None, projection: Optional[numpy_.ndarray] = None) -> Tuple[numpy_.ndarray, numpy_.ndarray]:
    """Calculate projection. 
- For OpenCV convention, use `intrinsics` and `extrinsics` matrices. 
- For OpenGL convention, use `view` and `projection` matrices.

## Parameters

- `points`: (..., N, 3) 3D world-space points
- `intrinsics`: (..., 3, 3) intrinsics matrix
- `extrinsics`: (..., 4, 4) extrinsics matrix
- `view`: (..., 4, 4) view matrix
- `projection`: (..., 4, 4) projection matrix

## Returns

- `uv`: (..., N, 2) 2D coordinates. 
    - For OpenCV convention, it is the normalized image coordinate where (0, 0) is the top left corner.
    - For OpenGL convention, it is the screen space XY coordinate where (0, 0) is the bottom left corner.
- `depth`: (..., N) linear depth values, where `depth > 0` is visible.
    - For OpenCV convention, it is the Z coordinate in camera space.
    - For OpenGL convention, it is the -Z coordinate in camera space."""
    utils3d.numpy.transforms.project

@overload
def unproject(uv: numpy_.ndarray, depth: Optional[numpy_.ndarray], *, intrinsics: Optional[numpy_.ndarray] = None, extrinsics: Optional[numpy_.ndarray] = None, projection: Optional[numpy_.ndarray] = None, view: Optional[numpy_.ndarray] = None) -> numpy_.ndarray:
    """Calculate inverse projection. 
- For OpenCV convention, use `intrinsics` and `extrinsics` matrices. 
- For OpenGL convention, use `view` and `projection` matrices.

## Parameters

- `uv`: (..., N, 2) 2D coordinates. 
    - For OpenCV convention, it is the normalized image coordinate where (0, 0) is the top left corner.
    - For OpenGL convention, it is the screen space XY coordinate where (0, 0) is the bottom left corner.
- `depth`: (..., N) linear depth values, where `depth > 0` is visible.
    - For OpenCV convention, it is the Z coordinate in camera space.
    - For OpenGL convention, it is the -Z coordinate in camera space.
- `intrinsics`: (..., 3, 3) intrinsics matrix
- `extrinsics`: (..., 4, 4) extrinsics matrix
- `view`: (..., 4, 4) view matrix
- `projection`: (..., 4, 4) projection matrix

## Returns

- `points`: (..., N, 3) 3D world-space points"""
    utils3d.numpy.transforms.unproject

@overload
def screen_coord_to_view_coord(screen_coord: numpy_.ndarray, projection: numpy_.ndarray) -> numpy_.ndarray:
    """Unproject screen space coordinates to 3D view space following the OpenGL convention (except for row major matrices)

## Parameters
    screen_coord (ndarray): (..., N, 3) screen space XYZ coordinates, value ranging in [0, 1]
        The origin (0., 0.) is corresponding to the left & bottom
    projection (ndarray): (..., 4, 4) projection matrix

## Returns
    points (ndarray): [..., N, 3] 3d points in view space"""
    utils3d.numpy.transforms.screen_coord_to_view_coord

@overload
def quaternion_to_matrix(quaternion: numpy_.ndarray) -> numpy_.ndarray:
    """Converts a batch of quaternions (w, x, y, z) to rotation matrices

## Parameters
    quaternion (ndarray): shape (..., 4), the quaternions to convert

## Returns
    ndarray: shape (..., 3, 3), the rotation matrices corresponding to the given quaternions"""
    utils3d.numpy.transforms.quaternion_to_matrix

@overload
def axis_angle_to_matrix(axis_angle: numpy_.ndarray) -> numpy_.ndarray:
    """Convert axis-angle representation (rotation vector) to rotation matrix, whose direction is the axis of rotation and length is the angle of rotation

## Parameters
    axis_angle (ndarray): shape (..., 3), axis-angle vcetors

## Returns
    ndarray: shape (..., 3, 3) The rotation matrices for the given axis-angle parameters"""
    utils3d.numpy.transforms.axis_angle_to_matrix

@overload
def matrix_to_quaternion(rot_mat: numpy_.ndarray) -> numpy_.ndarray:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)

## Parameters
    rot_mat (ndarray): shape (..., 3, 3), the rotation matrices to convert

## Returns
    ndarray: shape (..., 4), the quaternions corresponding to the given rotation matrices"""
    utils3d.numpy.transforms.matrix_to_quaternion

@overload
def extrinsics_to_essential(extrinsics: numpy_.ndarray):
    """extrinsics matrix `[[R, t] [0, 0, 0, 1]]` such that `x' = R (x - t)` to essential matrix such that `x' E x = 0`

## Parameters
    extrinsics (np.ndaray): [..., 4, 4] extrinsics matrix

## Returns
    (np.ndaray): [..., 3, 3] essential matrix"""
    utils3d.numpy.transforms.extrinsics_to_essential

@overload
def axis_angle_to_quaternion(axis_angle: numpy_.ndarray) -> numpy_.ndarray:
    """Convert axis-angle representation (rotation vector) to quaternion (w, x, y, z)

## Parameters
    axis_angle (ndarray): shape (..., 3), axis-angle vcetors

## Returns
    ndarray: shape (..., 4) The quaternions for the given axis-angle parameters"""
    utils3d.numpy.transforms.axis_angle_to_quaternion

@overload
def euler_axis_angle_rotation(axis: str, angle: numpy_.ndarray) -> numpy_.ndarray:
    """Return the rotation matrices for one of the rotations about an axis
of which Euler angles describe, for each value of the angle given.

## Parameters
    axis: Axis label "X" or "Y or "Z".
    angle: any shape ndarray of Euler angles in radians

## Returns
    Rotation matrices as ndarray of shape (..., 3, 3)."""
    utils3d.numpy.transforms.euler_axis_angle_rotation

@overload
def euler_angles_to_matrix(euler_angles: numpy_.ndarray, convention: str = 'XYZ') -> numpy_.ndarray:
    """Convert rotations given as Euler angles in radians to rotation matrices.

## Parameters
    euler_angles: Euler angles in radians as ndarray of shape (..., 3), XYZ
    convention: permutation of "X", "Y" or "Z", representing the order of Euler rotations to apply.

## Returns
    Rotation matrices as ndarray of shape (..., 3, 3)."""
    utils3d.numpy.transforms.euler_angles_to_matrix

@overload
def matrix_to_axis_angle(rot_mat: numpy_.ndarray) -> numpy_.ndarray:
    """Convert a batch of 3x3 rotation matrices to axis-angle representation (rotation vector)

## Parameters
    rot_mat (ndarray): shape (..., 3, 3), the rotation matrices to convert

## Returns
    ndarray: shape (..., 3), the axis-angle vectors corresponding to the given rotation matrices"""
    utils3d.numpy.transforms.matrix_to_axis_angle

@overload
def matrix_to_euler_angles(matrix: numpy_.ndarray, convention: str) -> numpy_.ndarray:
    """Convert rotations given as rotation matrices to Euler angles in radians.
NOTE: The composition order eg. `XYZ` means `Rz * Ry * Rx` (like blender), instead of `Rx * Ry * Rz` (like pytorch3d)

## Parameters
    matrix: Rotation matrices as tensor of shape (..., 3, 3).
    convention: Convention string of three uppercase letters.

## Returns
    Euler angles in radians as tensor of shape (..., 3), in the order of XYZ (like blender), instead of convention (like pytorch3d)"""
    utils3d.numpy.transforms.matrix_to_euler_angles

@overload
def quaternion_to_axis_angle(quaternion: numpy_.ndarray) -> numpy_.ndarray:
    """Convert a batch of quaternions (w, x, y, z) to axis-angle representation (rotation vector)

## Parameters
    quaternion (ndarray): shape (..., 4), the quaternions to convert

## Returns
    ndarray: shape (..., 3), the axis-angle vectors corresponding to the given quaternions"""
    utils3d.numpy.transforms.quaternion_to_axis_angle

@overload
def skew_symmetric(v: numpy_.ndarray):
    """Skew symmetric matrix from a 3D vector"""
    utils3d.numpy.transforms.skew_symmetric

@overload
def rotation_matrix_from_vectors(v1: numpy_.ndarray, v2: numpy_.ndarray):
    """Rotation matrix that rotates v1 to v2"""
    utils3d.numpy.transforms.rotation_matrix_from_vectors

@overload
def ray_intersection(p1: numpy_.ndarray, d1: numpy_.ndarray, p2: numpy_.ndarray, d2: numpy_.ndarray):
    """Compute the intersection/closest point of two D-dimensional rays
If the rays are intersecting, the closest point is the intersection point.

## Parameters
    p1 (ndarray): (..., D) origin of ray 1
    d1 (ndarray): (..., D) direction of ray 1
    p2 (ndarray): (..., D) origin of ray 2
    d2 (ndarray): (..., D) direction of ray 2

## Returns
    (ndarray): (..., N) intersection point"""
    utils3d.numpy.transforms.ray_intersection

@overload
def make_affine_matrix(M: numpy_.ndarray, t: numpy_.ndarray) -> numpy_.ndarray:
    """Make an affine transformation matrix from a linear matrix and a translation vector.

## Parameters
    M (ndarray): [..., D, D] linear matrix (rotation, scaling or general deformation)
    t (ndarray): [..., D] translation vector

## Returns
    ndarray: [..., D + 1, D + 1] affine transformation matrix"""
    utils3d.numpy.transforms.make_affine_matrix

@overload
def random_rotation_matrix(*size: int, dtype=numpy_.float32) -> numpy_.ndarray:
    """Generate random 3D rotation matrix.

## Parameters
    dtype: The data type of the output rotation matrix.

## Returns
    ndarray: `(*size, 3, 3)` random rotation matrix."""
    utils3d.numpy.transforms.random_rotation_matrix

@overload
def lerp(x1: numpy_.ndarray, x2: numpy_.ndarray, t: numpy_.ndarray) -> numpy_.ndarray:
    """Linear interpolation between two vectors.

## Parameters
    x1 (ndarray): [..., D] vector 1
    x2 (ndarray): [..., D] vector 2
    t (ndarray): [..., N] interpolation parameter. [0, 1] for interpolation between x1 and x2, otherwise for extrapolation.

## Returns
    ndarray: [..., N, D] interpolated vector"""
    utils3d.numpy.transforms.lerp

@overload
def slerp(v1: numpy_.ndarray, v2: numpy_.ndarray, t: numpy_.ndarray) -> numpy_.ndarray:
    """Spherical linear interpolation between two (unit) vectors.

## Parameters
- `v1` (ndarray): `(..., D)` (unit) vector 1
- `v2` (ndarray): `(..., D)` (unit) vector 2
- `t` (ndarray): `(..., N)` interpolation parameter in [0, 1]

## Returns
    ndarray: `(..., N, D)` interpolated unit vector"""
    utils3d.numpy.transforms.slerp

@overload
def slerp_rotation_matrix(R1: numpy_.ndarray, R2: numpy_.ndarray, t: numpy_.ndarray) -> numpy_.ndarray:
    """Spherical linear interpolation between two rotation matrices.

## Parameters
- `R1` (ndarray): [..., 3, 3] rotation matrix 1
- `R2` (ndarray): [..., 3, 3] rotation matrix 2
- `t` (ndarray): [..., N] interpolation parameter in [0, 1]

## Returns
    ndarray: [...,N, 3, 3] interpolated rotation matrix"""
    utils3d.numpy.transforms.slerp_rotation_matrix

@overload
def interpolate_se3_matrix(T1: numpy_.ndarray, T2: numpy_.ndarray, t: numpy_.ndarray) -> numpy_.ndarray:
    """Linear interpolation between two SE(3) matrices.

## Parameters
- `T1` (ndarray): [..., 4, 4] SE(3) matrix 1
- `T2` (ndarray): [..., 4, 4] SE(3) matrix 2
- `t` (ndarray): [..., N] interpolation parameter in [0, 1]

## Returns
    ndarray: [..., N, 4, 4] interpolated SE(3) matrix"""
    utils3d.numpy.transforms.interpolate_se3_matrix

@overload
def piecewise_lerp(x: numpy_.ndarray, t: numpy_.ndarray, s: numpy_.ndarray, extrapolation_mode: Literal['constant', 'linear'] = 'constant') -> numpy_.ndarray:
    """Linear spline interpolation.

## Parameters
- `x`: ndarray, shape (n, d): the values of data points.
- `t`: ndarray, shape (n,): the times of the data points.
- `s`: ndarray, shape (m,): the times to be interpolated.
- `extrapolation_mode`: str, the mode of extrapolation. 'constant' means extrapolate the boundary values, 'linear' means extrapolate linearly.

## Returns
- `y`: ndarray, shape (..., m, d): the interpolated values."""
    utils3d.numpy.transforms.piecewise_lerp

@overload
def piecewise_interpolate_se3_matrix(T: numpy_.ndarray, t: numpy_.ndarray, s: numpy_.ndarray, extrapolation_mode: Literal['constant', 'linear'] = 'constant') -> numpy_.ndarray:
    """Linear spline interpolation for SE(3) matrices.

## Parameters
- `T`: ndarray, shape (n, 4, 4): the SE(3) matrices.
- `t`: ndarray, shape (n,): the times of the data points.
- `s`: ndarray, shape (m,): the times to be interpolated.
- `extrapolation_mode`: str, the mode of extrapolation. 'constant' means extrapolate the boundary values, 'linear' means extrapolate linearly.

## Returns
- `T_interp`: ndarray, shape (..., m, 4, 4): the interpolated SE(3) matrices."""
    utils3d.numpy.transforms.piecewise_interpolate_se3_matrix

@overload
def transform_points(x: numpy_.ndarray, *Ts: numpy_.ndarray) -> numpy_.ndarray:
    """Apply transformation(s) to a point or a set of points.
It is like `(Tn @ ... @ T2 @ T1 @ x[:, None]).squeeze(0)`, but: 
1. Automatically handle the homogeneous coordinate;
        - x will be padded with homogeneous coordinate 1.
        - Each T will be padded by identity matrix to match the dimension. 
2. Using efficient contraction path when array sizes are large, based on `einsum`.

## Parameters
- `x`: ndarray, shape `(..., D)`: the points to be transformed.
- `Ts`: ndarray, shape `(..., D1, D2)`: the affine transformation matrix (matrices)
    If more than one transformation is given, they will be applied in corresponding order.
## Returns
- `y`: ndarray, shape `(..., D)`: the transformed point or a set of points.

## Example Usage

- Just linear transformation

    ```
    y = transform(x_3, mat_3x3) 
    ```

- Affine transformation

    ```
    y = transform(x_3, mat_3x4)
    ```

- Chain multiple transformations

    ```
    y = transform(x_3, T1_4x4, T2_3x4, T3_3x4)
    ```"""
    utils3d.numpy.transforms.transform_points

@overload
def angle_between(v1: numpy_.ndarray, v2: numpy_.ndarray):
    """Calculate the angle between two (batches of) vectors.
Better precision than using the arccos dot product directly.

## Parameters
- `v1`: ndarray, shape (..., D): the first vector.
- `v2`: ndarray, shape (..., D): the second vector.

## Returns
`angle`: ndarray, shape (...): the angle between the two vectors."""
    utils3d.numpy.transforms.angle_between

@overload
def vector_outer(x: numpy_.ndarray, y: Optional[numpy_.ndarray] = None) -> numpy_.ndarray:
    utils3d.numpy.transforms.vector_outer

@overload
def procrustes(cov_yx: numpy_.ndarray, cov_xx: Optional[numpy_.ndarray] = None, cov_yy: Optional[numpy_.ndarray] = None, mean_x: Optional[numpy_.ndarray] = None, mean_y: Optional[numpy_.ndarray] = None, niter: int = 8) -> Tuple[numpy_.ndarray, numpy_.ndarray]:
    """Procrustes analysis to solve for scale `s`, rotation `R` and translation `t` such that `y_i ~= s R x_i + t`.

Parameters
----
- `cov_yx`: (..., 3, 3) covariance matrix between y and x points.
- `cov_xx`: (..., 3, 3) covariance matrix of x points. If None, no scaling is solved.
- `cov_yy`: (..., 3, 3) covariance matrix of y points. If None, no scaling is solved.
- `mean_x`: (..., 3) mean of x points. If None, no translation is solved.
- `mean_y`: (..., 3) mean of y points. If None, no translation is solved.
- `niter`: int, number of Newton iterations for scale solving when both cov_xx and cov_yy are given.

Specifically, based on provided inputs:

- To solve the rotation `R`, `cov_yx` must be given.
- To solve the scale `s`, at least one of `cov_xx` and `cov_yy` must be given.
    - (Recommended) If both `cov_xx` and `cov_yy` are given, the scale will be solved by minimizing a symmetric cost:
        `||s R X + t - Y||_F^2 / ||Y||_F^2 + ||s R^T (Y - t)  - X||_F^2 / ||X||_F^2`
    - If only `cov_xx` is given, the scale will be solved by minimizing forward cost
        `||s R X  + t - Y||_F^2`
    - If only `cov_yy` is given, the scale will be solved by minimizing inverse cost 
        `||s R^T (Y - t)  - X||_F^2`
- To solve the translation `t`, provide `mean_x` and `mean_y`.

Returns
----
- `s`: (...) scale factor. None if both cov_xx and cov_yy are None. 
- `R`: (..., 3, 3) rotation matrix.
- `t`: (..., 3) translation vector. None if mean_x or mean_y is None."""
    utils3d.numpy.transforms.procrustes

@overload
def solve_pose(p: numpy_.ndarray, q: numpy_.ndarray, w: Optional[numpy_.ndarray] = None, offsets: Optional[numpy_.ndarray] = None, mode: Literal['rigid', 'similar', 'affine'] = 'rigid', lam: float = 0.01, niter: int = 5) -> Tuple[numpy_.ndarray, numpy_.ndarray]:
    """Solve for the pose (transformation from p to q) given weighted point correspondences.

Parameters
----
For batch input
- `p`: (..., N, 3) source points
- `q`: (..., N, 3) target points
- `w`: optional (..., N) weights for each point correspondence. If None, uniform weights are used.

For segment input
- `p`: (N, 3) source points
- `q`: (N, 3) target points
- `w`: (N,) weights for each point correspondence
- `offsets`: (S + 1,) segment offsets. Points in each segment belong to the same rigid / affine body.

Optional parameters
----
- `mode`: mode of transformation to apply. Can be 'rigid', 'similar', or 'affine'.
    - For 'rigid', only rotation and translation are allowed.
    - For 'similar', uniform scaling, rotation and translation are allowed.
    - For 'affine', full affine transformation is allowed. Using least squares.
- `lam`: regularization weight for 'affine' mode.
- `niter`: number of iterations for 'affine' mode.

Returns
----
- `pose`: (..., 4, 4) transformations matrix from p to q."""
    utils3d.numpy.transforms.solve_pose

@overload
def solve_poses_sequential(trajectories: numpy_.ndarray, weights: Optional[numpy_.ndarray] = None, offsets: Optional[numpy_.ndarray] = None, accum: Optional[Tuple[numpy_.ndarray, ...]] = None, min_valid_size: int = 0, mode: Literal['rigid', 'similar', 'affine'] = 'rigid', lam: float = 0.01, niter: int = 8) -> Tuple[numpy_.ndarray, Tuple[numpy_.ndarray, ...], Tuple[numpy_.ndarray, numpy_.ndarray, numpy_.ndarray, numpy_.ndarray]]:
    """Given trajectories of points over time, sequentially solve for the poses (transformations from canonical to each frame) of each body at each frame.

Parameters
----
For single or batch input
- `trajectories`: (T, ..., N, 3) posed points. T is number of frames. `...` is optional batch dimensions. N is number of points per group.
- `weights`: (T, ..., N) quardratic error term weights for each point at each frame

For segment input
- `trajectories`: (T, N, 3) posed points.
- `weights`: (T, N) quardratic error term weights for each point at each frame
- `offsets`: (S + 1,) segment offsets. Points in each segment belong to the same rigid / affine body.

Optional parameters
----
- `accum`: accumulated statistics from previous calls. If None, start fresh.
- `min_valid_size`: minimum number of valid points in each frame to consider the segment / group valid.
- `mode`: mode of transformation to apply. Can be 'rigid', 'similar', or 'affine'.
    - For 'rigid', only rotation and translation are allowed.
    - For 'similar', uniform scaling, rotation and translation are allowed. 
    - For 'affine', full affine transformation is allowed. Using least squares.
- `lam`: rigidity regularization weight for 'affine' mode.
- `niter`: number of iterations for 'affine' mode.

Returns
----
- `poses`: (T, ..., 4, 4) transformations from canonical to each frame.
- `valid`: (T, ...) boolean mask indicating valid segments
- `stats`: canonical statistics of each group,
    It is a tuple of:
    - `mu`: (..., 3) weighted mean of points
    - `cov`: (..., 3, 3) weighted covariance of points
    - `tot_w`: (...,) total weight of points
    - `nnz`: (...,) number of non-zero weight points
- `canonical_points`: (..., N, 3) canonical points.
- `err`: (..., N,) per-point RMS error over all time := sqrt(sum_over_time(per_point_weights * per_point_squared_error) / per_point_nnz)
    Use this to filter outliers as needed.
- `accum`: per point accumulated statistics. Just pass it to the next call for incremental solving.
    It is a tuple of:
    - `accum_sqrtw`: (..., N,) sum of sqrt(weights)
    - `accum_sqrtwx`: (..., N, 3) sum of sqrt(weights) * x
    - `accum_sqrtwxx`: (...N, 3, 3) sum of sqrt(weights) * outer(x - mean_sqrtwx, x - mean_sqrtwx)
    - `accum_w`: (..., N,) sum of weights
    - `accum_wx`: (..., N, 3) sum of weights * x
    - `accum_wxx`: (..., N, 3, 3) sum of weights * outer(x - mean_wx, x - mean_wx)
    - `accum_nnz`: (..., N,) number of non-zero weight accumulations

Example
----
```
accum = None
poses, valid = [], []
for new_trajectories_chunk in data_stream:
    # new_trajectories_chunk: (T_chunk, N, 3)
    poses_chunk, valid_chunk, stats, canonical_points, err, accum = solve_poses(
        new_trajectories_chunk,
        accum=accum,
    )
    poses.append(poses_chunk)
    valid.append(valid_chunk)
    # `stats`, `canonical_points` and `err` are returned and updated every chunk.
poses = np.concatenate(poses, axis=0)   # (T_all, 4, 4), poses over all frames
valid = np.concatenate(valid, axis=0)   # (T_all,), poses' validity over all frames"""
    utils3d.numpy.transforms.solve_poses_sequential

@overload
def triangulate_mesh(faces: numpy_.ndarray, vertices: numpy_.ndarray = None, method: Literal['fan', 'strip', 'diagonal'] = 'fan') -> numpy_.ndarray:
    """Triangulate a polygonal mesh.

## Parameters
    faces (ndarray): [L, P] polygonal faces
    vertices (ndarray, optional): [N, 3] 3-dimensional vertices.
        If given, the triangulation is performed according to the distance
        between vertices. Defaults to None.
    backslash (ndarray, optional): [L] boolean array indicating
        how to triangulate the quad faces. Defaults to None.

## Returns
    (ndarray): [L * (P - 2), 3] triangular faces"""
    utils3d.numpy.mesh.triangulate_mesh

@overload
def compute_face_corner_angles(vertices: numpy_.ndarray, faces: Optional[numpy_.ndarray] = None) -> numpy_.ndarray:
    """Compute face corner angles of a mesh

## Parameters
- `vertices` (ndarray): `(..., N, 3)` vertices if `faces` is provided, or `(..., F, P, 3)` if `faces` is None
- `faces` (ndarray, optional): `(F, P)` face vertex indices, where P is the number of vertices per face

## Returns
- `angles` (ndarray): `(..., F, P)` face corner angles"""
    utils3d.numpy.mesh.compute_face_corner_angles

@overload
def compute_face_corner_normals(vertices: numpy_.ndarray, faces: Optional[numpy_.ndarray] = None, normalize: bool = True) -> numpy_.ndarray:
    """Compute the face corner normals of a mesh

## Parameters
- `vertices` (ndarray): `(..., N, 3)` vertices if `faces` is provided, or `(..., F, P, 3)` if `faces` is None
- `faces` (ndarray, optional): `(F, P)` face vertex indices, where P is the number of vertices per face
- `normalize` (bool): whether to normalize the normals to unit vectors. If not, the normals are the raw cross products.

## Returns
- `normals` (ndarray): (..., F, P, 3) face corner normals"""
    utils3d.numpy.mesh.compute_face_corner_normals

@overload
def compute_face_corner_tangents(vertices: numpy_.ndarray, uv: numpy_.ndarray, faces_vertices: Optional[numpy_.ndarray] = None, faces_uv: Optional[numpy_.ndarray] = None, normalize: bool = True) -> numpy_.ndarray:
    """Compute the face corner tangent (and bitangent) vectors of a mesh

## Parameters
- `vertices` (ndarray): `(..., N, 3)` if `faces` is provided, or `(..., F, P, 3)` if `faces_vertices` is None
- `uv` (ndarray): `(..., N, 2)` if `faces` is provided, or `(..., F, P, 2)` if `faces_uv` is None
- `faces_vertices` (ndarray, optional): `(F, P)` face vertex indices
- `faces_uv` (ndarray, optional): `(F, P)` face UV indices
- `normalize` (bool): whether to normalize the tangents to unit vectors. If not, the tangents (dX/du, dX/dv) matches the UV parameterized manifold.

## Returns
- `tangents` (ndarray): `(..., F, P, 3, 2)` face corner tangents (and bitangents), 
    where the last dimension represents the tangent and bitangent vectors."""
    utils3d.numpy.mesh.compute_face_corner_tangents

@overload
def compute_face_normals(vertices: numpy_.ndarray, faces: Optional[numpy_.ndarray] = None) -> numpy_.ndarray:
    """Compute face normals of a mesh

## Parameters
- `vertices` (ndarray): `(..., N, 3)` vertices if `faces` is provided, or `(..., F, P, 3)` if `faces` is None
- `faces` (ndarray, optional): `(F, P)` face vertex indices, where P is the number of vertices per face

## Returns
- `normals` (ndarray): `(..., F, 3)` face normals. Always normalized."""
    utils3d.numpy.mesh.compute_face_normals

@overload
def compute_face_tangents(vertices: numpy_.ndarray, uv: numpy_.ndarray, faces_vertices: Optional[numpy_.ndarray] = None, faces_uv: Optional[numpy_.ndarray] = None, normalize: bool = True) -> numpy_.ndarray:
    """Compute the face corner tangent (and bitangent) vectors of a mesh

## Parameters
- `vertices` (ndarray): `(..., N, 3)` if `faces` is provided, or `(..., F, P, 3)` if `faces_vertices` is None
- `uv` (ndarray): `(..., N, 2)` if `faces` is provided, or `(..., F, P, 2)` if `faces_uv` is None
- `faces_vertices` (ndarray, optional): `(F, P)` face vertex indices
- `faces_uv` (ndarray, optional): `(F, P)` face UV indices

## Returns
- `tangents` (ndarray): `(..., F, 3, 2)` face corner tangents (and bitangents), 
    where the last dimension represents the tangent and bitangent vectors."""
    utils3d.numpy.mesh.compute_face_tangents

@overload
def compute_vertex_normals(vertices: numpy_.ndarray, faces: numpy_.ndarray, weighted: Literal['uniform', 'area', 'angle'] = 'uniform') -> numpy_.ndarray:
    """Compute vertex normals of a triangular mesh by averaging neighboring face normals

## Parameters
    vertices (ndarray): [..., N, 3] 3-dimensional vertices
    faces (ndarray): [T, P] face vertex indices, where P is the number of vertices per face

## Returns
    normals (ndarray): [..., N, 3] vertex normals (already normalized to unit vectors)"""
    utils3d.numpy.mesh.compute_vertex_normals

@overload
def remove_corrupted_faces(faces: numpy_.ndarray) -> numpy_.ndarray:
    """Remove corrupted faces (faces with duplicated vertices)

## Parameters
    faces (ndarray): [T, 3] triangular face indices

## Returns
    ndarray: [T_, 3] triangular face indices"""
    utils3d.numpy.mesh.remove_corrupted_faces

@overload
def merge_duplicate_vertices(vertices: numpy_.ndarray, faces: numpy_.ndarray, tol: float = 1e-06) -> Tuple[numpy_.ndarray, numpy_.ndarray]:
    """Merge duplicate vertices of a triangular mesh. 
Duplicate vertices are merged by selecte one of them, and the face indices are updated accordingly.

## Parameters
    vertices (ndarray): [N, 3] 3-dimensional vertices
    faces (ndarray): [T, 3] triangular face indices
    tol (float, optional): tolerance for merging. Defaults to 1e-6.

## Returns
    vertices (ndarray): [N_, 3] 3-dimensional vertices
    faces (ndarray): [T, 3] triangular face indices"""
    utils3d.numpy.mesh.merge_duplicate_vertices

@overload
def remove_unused_vertices(faces: numpy_.ndarray, *vertice_attrs, return_indices: bool = False) -> Tuple[numpy_.ndarray, ...]:
    """Remove unreferenced vertices of a mesh. 
Unreferenced vertices are removed, and the face indices are updated accordingly.

## Parameters
    faces (ndarray): [T, P] face indices
    *vertice_attrs: vertex attributes

## Returns
    faces (ndarray): [T, P] face indices
    *vertice_attrs: vertex attributes
    indices (ndarray, optional): [N] indices of vertices that are kept. Defaults to None."""
    utils3d.numpy.mesh.remove_unused_vertices

@overload
def subdivide_mesh(vertices: numpy_.ndarray, faces: numpy_.ndarray, level: int = 1) -> Tuple[numpy_.ndarray, numpy_.ndarray]:
    """Subdivide a triangular mesh by splitting each triangle into 4 smaller triangles.
NOTE: All original vertices are kept, and new vertices are appended to the end of the vertex list.

## Parameters
    vertices (ndarray): [N, 3] 3-dimensional vertices
    faces (ndarray): [T, 3] triangular face indices
    level (int, optional): level of subdivisions. Defaults to 1.

## Returns
    vertices (ndarray): [N_, 3] subdivided 3-dimensional vertices
    faces (ndarray): [(4 ** level) * T, 3] subdivided triangular face indices"""
    utils3d.numpy.mesh.subdivide_mesh

@overload
def mesh_edges(faces: Union[numpy_.ndarray, scipy.sparse._csr.csr_array], return_face2edge: bool = False, return_edge2face: bool = False, return_counts: bool = False) -> Tuple[numpy_.ndarray, Union[numpy_.ndarray, scipy.sparse._csr.csr_array], scipy.sparse._csr.csr_array, numpy_.ndarray]:
    """Get undirected edges of a mesh. Optionally return additional mappings.

## Parameters
- `faces` (ndarray): polygon faces
    - `(F, P)` dense array of indices, where each face has `P` vertices.
    - `(F, V)` binary sparse csr array of indices, each row corresponds to the vertices of a face.
- `return_face2edge` (bool): whether to return the face to edge mapping
- `return_edge2face` (bool): whether to return the edge to face mapping
- `return_counts` (bool): whether to return the counts of edges

## Returns
- `edges` (ndarray): `(E, 2)` unique edges' vertex indices

If `return_face2edge`, `return_edge2face`, `return_opposite_edge`, or `return_counts` is True, the corresponding outputs will be appended in order:

- `face2edge` (ndarray | csr_array): mapping from faces to the indices of edges
    - `(F, P)` if input `faces` is a dense array
    - `(F, E)` if input `faces` is a sparse csr array
- `edge2face` (csr_array): `(E, F)` binary sparse CSR matrix of edge to face.
- `counts` (ndarray): `(E,)` counts of each edge"""
    utils3d.numpy.mesh.mesh_edges

@overload
def mesh_half_edges(faces: Union[numpy_.ndarray, scipy.sparse._csr.csr_array], return_face2edge: bool = False, return_edge2face: bool = False, return_twin: bool = False, return_next: bool = False, return_prev: bool = False, return_counts: bool = False) -> Tuple[numpy_.ndarray, Union[numpy_.ndarray, scipy.sparse._csr.csr_array], scipy.sparse._csr.csr_array, numpy_.ndarray, numpy_.ndarray, numpy_.ndarray, numpy_.ndarray]:
    """Get half edges of a mesh. Optionally return additional mappings.

## Parameters
- `faces` (ndarray): polygon faces
    - `(F, P)` dense array of indices, where each face has `P` vertices.
    - `(F, V)` binary sparse csr array of indices, each row corresponds to the vertices of a face.
- `return_face2edge` (bool): whether to return the face to edge mapping
- `return_edge2face` (bool): whether to return the edge to face mapping
- `return_twin` (bool): whether to return the mapping from one edge to its opposite/twin edge
- `return_next` (bool): whether to return the mapping from one edge to its next edge in the face loop
- `return_prev` (bool): whether to return the mapping from one edge to its previous edge in the face loop
- `return_counts` (bool): whether to return the counts of edges

## Returns
- `edges` (ndarray): `(E, 2)` unique edges' vertex indices

If `return_face2edge`, `return_edge2face`, `return_opposite_edge`, or `return_counts` is True, the corresponding outputs will be appended in order:

- `face2edge` (ndarray | csr_array): mapping from faces to the indices of edges
    - `(F, P)` if input `faces` is a dense array
    - `(F, E)` if input `faces` is a sparse csr array
- `edge2face` (csr_array): `(E, F)` binary sparse CSR matrix of edge to face.
- `twin` (ndarray): `(E,)` mapping from edges to indices of opposite edges. -1 if not found. 
- `next` (ndarray): `(E,)` mapping from edges to indices of next edges in the face loop.
- `prev` (ndarray): `(E,)` mapping from edges to indices of previous edges in the face loop.
- `counts` (ndarray): `(E,)` counts of each half edge

NOTE: If the mesh is not manifold, `twin`, `next`, and `prev` can point to arbitrary one of the candidates."""
    utils3d.numpy.mesh.mesh_half_edges

@overload
def mesh_connected_components(faces: Optional[numpy_.ndarray] = None, num_vertices: Optional[int] = None) -> Union[numpy_.ndarray, Tuple[numpy_.ndarray, numpy_.ndarray]]:
    """Compute connected faces of a mesh.

## Parameters
- `faces` (ndarray): polygon faces
    - `(F, P)` dense array of indices, where each face has `P` vertices.
    - `(F, V)` binary sparse csr array of indices, each row corresponds to the vertices of a face.
- `num_vertices` (int, optional): total number of vertices. If given, the returned components will include all vertices. Defaults to None.

## Returns

If `num_vertices` is given, return:
- `labels` (ndarray): (N,) component labels of each vertex

If `num_vertices` is None, return:
- `vertices_ids` (ndarray): (N,) vertex indices that are in the edges
- `labels` (ndarray): (N,) int32 component labels corresponding to `vertices_ids`"""
    utils3d.numpy.mesh.mesh_connected_components

@overload
def graph_connected_components(edges: numpy_.ndarray, num_vertices: Optional[int] = None) -> Union[numpy_.ndarray, Tuple[numpy_.ndarray, numpy_.ndarray]]:
    """Compute connected components of an undirected graph.
Using scipy.sparse.csgraph.connected_components as backend.

## Parameters
- `edges` (ndarray): (E, 2) edge indices

## Returns

If `num_vertices` is given, return:
- `labels` (ndarray): (N,) component labels of each vertex

If `num_vertices` is None, return:
- `vertices_ids` (ndarray): (N,) vertex indices that are in the edges
- `labels` (ndarray): (N,) int32 component labels corresponding to `vertices_ids`"""
    utils3d.numpy.mesh.graph_connected_components

@overload
def mesh_adjacency_graph(adjacency: Literal['vertex2edge', 'vertex2face', 'edge2vertex', 'edge2face', 'face2edge', 'face2vertex', 'vertex2edge2vertex', 'vertex2face2vertex', 'edge2vertex2edge', 'edge2face2edge', 'face2edge2face', 'face2vertex2face'], faces: Union[numpy_.ndarray, scipy.sparse._csr.csr_array, NoneType] = None, edges: Optional[numpy_.ndarray] = None, num_vertices: Optional[int] = None, self_loop: bool = False) -> scipy.sparse._csr.csr_array:
    """Get adjacency graph of a mesh.

## Parameters
- `adjacency` (str): type of adjacency graph. Options:
    - `'vertex2edge'`: vertex to adjacent edges. Returns (V, E) csr
    - `'vertex2face'`: vertex to adjacent faces. Returns (V, F) csr
    - `'edge2vertex'`: edge to adjacent vertices. Returns (E, V) csr
    - `'edge2face'`: edge to its adjacent faces. Returns (E, F) csr
    - `'face2edge'`: face to its adjacent edges. Returns (F, E) csr
    - `'face2vertex'`: face to its adjacent vertices. Returns (F, V) csr
    - `'vertex2edge2vertex'`: vertex to adjacent vertices if they share an edge. Returns (V, V) csr
    - `'vertex2face2vertex'`: vertex to adjacent vertices if they share a face. Returns (V, V) csr
    - `'edge2vertex2edge'`: edge to adjacent edges if they share a vertex. Returns (E, E) csr
    - `'edge2face2edge'`: edge to adjacent edges if they share a face. Returns (E, E) csr
    - `'face2edge2face'`: face to adjacent faces if they share an edge. Returns (F, F) csr
    - `'face2vertex2face'`: face to adjacent faces if they share a vertex. Returns (F, F) csr
- `faces` (ndarray): polygon faces
    - `(F, P)` dense array of indices, where each face has `P` vertices.
    - `(F, V)` binary sparse csr array of indices, each row corresponds to the vertices of a face.
- `edges` (ndarray, optional): (E, 2) edge indices. NOTE: assumed to be undirected edges.
- `num_vertices` (int, optional): total number of vertices.
- `self_loop` (bool): whether to include self-loops in the adjacency graph. Defaults to False.

## Returns
- `graph` (csr_array): adjacency graph in csr format"""
    utils3d.numpy.mesh.mesh_adjacency_graph

@overload
def flatten_mesh_indices(*args: numpy_.ndarray) -> Tuple[numpy_.ndarray, ...]:
    utils3d.numpy.mesh.flatten_mesh_indices

@overload
def create_cube_mesh(tri: bool = False) -> Tuple[numpy_.ndarray, numpy_.ndarray]:
    """Create a cube mesh of size 1 centered at origin.

### Parameters
    tri (bool, optional): return triangulated mesh. Defaults to False, which returns quad mesh.

### Returns
    vertices (ndarray): shape (8, 3) 
    faces (ndarray): shape (12, 3)"""
    utils3d.numpy.mesh.create_cube_mesh

@overload
def create_icosahedron_mesh():
    """Create an icosahedron mesh of centered at origin."""
    utils3d.numpy.mesh.create_icosahedron_mesh

@overload
def create_square_mesh(tri: bool = False) -> Tuple[numpy_.ndarray, numpy_.ndarray]:
    """Create a square mesh of area 1 centered at origin in the xy-plane.

## Returns
    vertices (ndarray): shape (4, 3)
    faces (ndarray): shape (1, 4)"""
    utils3d.numpy.mesh.create_square_mesh

@overload
def create_camera_frustum_mesh(extrinsics: numpy_.ndarray, intrinsics: numpy_.ndarray, depth: float = 1.0) -> Tuple[numpy_.ndarray, numpy_.ndarray, numpy_.ndarray]:
    """Create a triangle mesh of camera frustum."""
    utils3d.numpy.mesh.create_camera_frustum_mesh

@overload
def merge_meshes(meshes: List[Tuple[numpy_.ndarray, ...]]) -> Tuple[numpy_.ndarray, ...]:
    """Merge multiple meshes into one mesh. Vertices will be no longer shared.

## Parameters
    - `meshes`: a list of tuple (faces, vertices_attr1, vertices_attr2, ....)

## Returns
    - `faces`: [sum(T_i), P] merged face indices, contigous from 0 to sum(T_i) * P - 1
    - `*vertice_attrs`: [sum(T_i) * P, ...] merged vertex attributes, where every P values correspond to a face"""
    utils3d.numpy.mesh.merge_meshes

@overload
def uv_map(*size: Union[int, Tuple[int, int]], top: float = 0.0, left: float = 0.0, bottom: float = 1.0, right: float = 1.0, dtype: numpy_.dtype = numpy_.float32) -> numpy_.ndarray:
    """Get image UV space coordinate map, where (0., 0.) is the top-left corner of the image, and (1., 1.) is the bottom-right corner of the image.
This is commonly used as normalized image coordinates in texture mapping (when image is not flipped vertically).

## Parameters
- `*size`: `Tuple[int, int]` or two integers of map size `(height, width)`
- `top`: `float`, optional top boundary in uv space. Defaults to 0.
- `left`: `float`, optional left boundary in uv space. Defaults to 0.
- `bottom`: `float`, optional bottom boundary in uv space. Defaults to 1.
- `right`: `float`, optional right boundary in uv space. Defaults to 1.
- `dtype`: `np.dtype`, optional data type of the output uv map. Defaults to np.float32.

## Returns
- `uv (ndarray)`: shape `(height, width, 2)`

## Example Usage

>>> uv_map(10, 10):
[[[0.05, 0.05], [0.15, 0.05], ..., [0.95, 0.05]],
 [[0.05, 0.15], [0.15, 0.15], ..., [0.95, 0.15]],
  ...             ...                  ...
 [[0.05, 0.95], [0.15, 0.95], ..., [0.95, 0.95]]]"""
    utils3d.numpy.maps.uv_map

@overload
def pixel_coord_map(*size: Union[int, Tuple[int, int]], top: int = 0, left: int = 0, convention: Literal['integer-center', 'integer-corner'] = 'integer-center', dtype: numpy_.dtype = numpy_.float32) -> numpy_.ndarray:
    """Get image pixel coordinates map, where (0, 0) is the top-left corner of the top-left pixel, and (width, height) is the bottom-right corner of the bottom-right pixel.

## Parameters
- `*size`: `Tuple[int, int]` or two integers of map size `(height, width)`
- `top`: `int`, optional top boundary of the pixel coord map. Defaults to 0.
- `left`: `int`, optional left boundary of the pixel coord map. Defaults to 0.
- `convention`: `str`, optional `'integer-center'` or `'integer-corner'`, whether integer coordinates correspond to pixel centers or corners. Defaults to 'integer-center'.
    - `'integer-center'`: `pixel[i][j]` has integer coordinates `(j, i)` as its center, and occupies square area `[j - 0.5, j + 0.5) × [i - 0.5, i + 0.5)`. 
        The top-left corner of the top-left pixel is `(-0.5, -0.5)`, and the bottom-right corner of the bottom-right pixel is `(width - 0.5, height - 0.5)`.
    - `'integer-corner'`: `pixel[i][j]` has coordinates `(j + 0.5, i + 0.5)` as its center, and occupies square area `[j, j + 1) × [i, i + 1)`.
        The top-left corner of the top-left pixel is `(0, 0)`, and the bottom-right corner of the bottom-right pixel is `(width, height)`.
- `dtype`: `np.dtype`, optional data type of the output pixel coord map. Defaults to np.float32.

## Returns
    ndarray: shape (height, width, 2)

>>> pixel_coord_map(10, 10, convention='integer-center', dtype=int):
[[[0, 0], [1, 0], ..., [9, 0]],
 [[0, 1], [1, 1], ..., [9, 1]],
    ...      ...         ...
 [[0, 9], [1, 9], ..., [9, 9]]]

>>> pixel_coord_map(10, 10, convention='integer-corner', dtype=np.float32):
[[[0.5, 0.5], [1.5, 0.5], ..., [9.5, 0.5]],
 [[0.5, 1.5], [1.5, 1.5], ..., [9.5, 1.5]],
  ...             ...                  ...
[[0.5, 9.5], [1.5, 9.5], ..., [9.5, 9.5]]]"""
    utils3d.numpy.maps.pixel_coord_map

@overload
def screen_coord_map(*size: Union[int, Tuple[int, int]], top: float = 1.0, left: float = 0.0, bottom: float = 0.0, right: float = 1.0, dtype: numpy_.dtype = numpy_.float32) -> numpy_.ndarray:
    """Get screen space coordinate map, where (0., 0.) is the bottom-left corner of the image, and (1., 1.) is the top-right corner of the image.
This is commonly used in graphics APIs like OpenGL.

## Parameters
    - `*size`: `Tuple[int, int]` or two integers of map size `(height, width)`
    - `top`: `float`, optional top boundary in the screen space. Defaults to 1.
    - `left`: `float`, optional left boundary in the screen space. Defaults to 0.
    - `bottom`: `float`, optional bottom boundary in the screen space. Defaults to 0.
    - `right`: `float`, optional right boundary in the screen space. Defaults to 1.
    - `dtype`: `np.dtype`, optional data type of the output map. Defaults to np.float32.

## Returns
    (ndarray): shape (height, width, 2)"""
    utils3d.numpy.maps.screen_coord_map

@overload
def build_mesh_from_map(*maps: numpy_.ndarray, mask: Optional[numpy_.ndarray] = None, tri: bool = False) -> Tuple[numpy_.ndarray, ...]:
    """Get a mesh regarding image pixel uv coordinates as vertices and image grid as faces.

## Parameters
    *maps (ndarray): attribute maps in shape (height, width, [channels])
    mask (ndarray, optional): binary mask of shape (height, width), dtype=bool. Defaults to None.

## Returns
    faces (ndarray): faces connecting neighboring pixels. shape (T, 4) if tri is False, else (T, 3)
    *attributes (ndarray): vertex attributes in corresponding order with input maps"""
    utils3d.numpy.maps.build_mesh_from_map

@overload
def build_mesh_from_depth_map(depth: numpy_.ndarray, *other_maps: numpy_.ndarray, intrinsics: numpy_.ndarray, extrinsics: Optional[numpy_.ndarray] = None, atol: Optional[float] = None, rtol: Optional[float] = None, tri: bool = False) -> Tuple[numpy_.ndarray, ...]:
    """Get a mesh by lifting depth map to 3D, while removing depths of large depth difference.

## Parameters
    depth (ndarray): [H, W] depth map
    extrinsics (ndarray, optional): [4, 4] extrinsics matrix. Defaults to None.
    intrinsics (ndarray, optional): [3, 3] intrinsics matrix. Defaults to None.
    *other_maps (ndarray): [H, W, C] vertex attributes. Defaults to None.
    atol (float, optional): absolute tolerance of difference. Defaults to None.
    rtol (float, optional): relative tolerance of difference. Defaults to None.
        triangles with vertices having depth difference larger than atol + rtol * depth will be marked.
    remove_by_depth (bool, optional): whether to remove triangles with large depth difference. Defaults to True.
    return_uv (bool, optional): whether to return uv coordinates. Defaults to False.
    return_indices (bool, optional): whether to return indices of vertices in the original mesh. Defaults to False.

## Returns
    faces (ndarray): [T, 3] faces
    vertices (ndarray): [N, 3] vertices
    *other_attrs (ndarray): [N, C] vertex attributes"""
    utils3d.numpy.maps.build_mesh_from_depth_map

@overload
def depth_map_edge(depth: numpy_.ndarray, atol: Optional[float] = None, rtol: Optional[float] = None, ltol: Optional[float] = None, kernel_size: int = 3, mask: numpy_.ndarray = None) -> numpy_.ndarray:
    """Compute the edge mask from depth map. The edge is defined as the pixels whose neighbors have large difference in depth.

## Parameters
    depth (ndarray): shape (..., height, width), linear depth map
    atol (float): absolute tolerance
    rtol (float): relative tolerance
    ltol (float): relative tolerance of inverse depth laplacian

## Returns
    edge (ndarray): shape (..., height, width) of dtype torch.bool"""
    utils3d.numpy.maps.depth_map_edge

@overload
def depth_map_aliasing(depth: numpy_.ndarray, atol: float = None, rtol: float = None, kernel_size: int = 3, mask: numpy_.ndarray = None) -> numpy_.ndarray:
    """Compute the map that indicates the aliasing of x depth map, identifying pixels which neither close to the maximum nor the minimum of its neighbors.
## Parameters
    depth (ndarray): shape (..., height, width), linear depth map
    atol (float): absolute tolerance
    rtol (float): relative tolerance

## Returns
    edge (ndarray): shape (..., height, width) of dtype torch.bool"""
    utils3d.numpy.maps.depth_map_aliasing

@overload
def normal_map_edge(normals: numpy_.ndarray, tol: float, kernel_size: int = 3, mask: numpy_.ndarray = None) -> numpy_.ndarray:
    """Compute the edge mask from normal map.

## Parameters
    normal (ndarray): shape (..., height, width, 3), normal map
    tol (float): tolerance in degrees

## Returns
    edge (ndarray): shape (..., height, width) of dtype torch.bool"""
    utils3d.numpy.maps.normal_map_edge

@overload
def point_map_to_normal_map(point: numpy_.ndarray, mask: numpy_.ndarray = None, edge_threshold: float = None) -> numpy_.ndarray:
    """Calculate normal map from point map. Value range is [-1, 1]. 

## Parameters
    point (ndarray): shape (height, width, 3), point map
    mask (optional, ndarray): shape (height, width), dtype=bool. Mask of valid depth pixels. Defaults to None.
    edge_threshold (optional, float): threshold for the angle (in degrees) between the normal and the view direction. Defaults to None.

## Returns
    normal (ndarray): shape (height, width, 3), normal map. """
    utils3d.numpy.maps.point_map_to_normal_map

@overload
def depth_map_to_point_map(depth: numpy_.ndarray, intrinsics: numpy_.ndarray, extrinsics: numpy_.ndarray = None) -> numpy_.ndarray:
    """Unproject depth map to 3D points.

## Parameters
    depth (ndarray): [..., H, W] depth value
    intrinsics ( ndarray): [..., 3, 3] intrinsics matrix
    extrinsics (optional, ndarray): [..., 4, 4] extrinsics matrix

## Returns
    points (ndarray): [..., N, 3] 3d points"""
    utils3d.numpy.maps.depth_map_to_point_map

@overload
def depth_map_to_normal_map(depth: numpy_.ndarray, intrinsics: numpy_.ndarray, mask: numpy_.ndarray = None, edge_threshold: float = None) -> numpy_.ndarray:
    """Calculate normal map from depth map. Value range is [-1, 1]. Normal direction in OpenCV identity camera's coordinate system.

## Parameters
    depth (ndarray): shape (height, width), linear depth map
    intrinsics (ndarray): shape (3, 3), intrinsics matrix
    mask (optional, ndarray): shape (height, width), dtype=bool. Mask of valid depth pixels. Defaults to None.
    edge_threshold (optional, float): threshold for the angle (in degrees) between the normal and the view direction. Defaults to None.

## Returns
    normal (ndarray): shape (height, width, 3), normal map. """
    utils3d.numpy.maps.depth_map_to_normal_map

@overload
def chessboard(*size: Union[int, Tuple[int, int]], grid_size: int, color_a: numpy_.ndarray, color_b: numpy_.ndarray) -> numpy_.ndarray:
    """Get a chessboard image

## Parameters
    - `*size`: `Tuple[int, int]` or two integers of map size `(height, width)`
    - `grid_size (int)`: size of chessboard grid
    - `color_a (ndarray)`: color of the grid at the top-left corner
    - `color_b (ndarray)`: color in complementary grid cells

## Returns
    image (ndarray): shape (height, width, channels), chessboard image"""
    utils3d.numpy.maps.chessboard

@overload
def masked_nearest_resize(*image: numpy_.ndarray, mask: numpy_.ndarray, size: Tuple[int, int], return_index: bool = False) -> Tuple[Unpack[Tuple[numpy_.ndarray, ...]], numpy_.ndarray, Tuple[numpy_.ndarray, ...]]:
    """Resize image(s) by nearest sampling with mask awareness. Suitable for sparse maps. ![masked_nearest_resize.png](doc/masked_nearest_resize.png)
- Downsampling: Assign the nearest valid pixel within the target pixel's receptive field.
- Upsampling: Assign the valid pixel to only the nearest pixel in the resized map.

### Parameters
- `*image`: Input image(s) of shape `(..., H, W, C)` or `(... , H, W)` 
    - You can pass multiple images to be resized at the same time for efficiency.
- `mask`: input mask of shape `(..., H, W)`, dtype=bool
- `size`: target size `(H', W')`
- `return_index`: whether to return the nearest neighbor indices in the original map for each pixel in the resized map.
    Defaults to False.

### Returns
- `*resized_image`: resized image(s) of shape `(..., H', W', C)`. or `(..., H', W')`
- `resized_mask`: mask of the resized map of shape `(..., H', W')`
- `nearest_indices`: tuple of shape `(..., H', W')`. The nearest neighbor indices of the resized map of each dimension."""
    utils3d.numpy.maps.masked_nearest_resize

@overload
def masked_area_resize(*image: numpy_.ndarray, mask: numpy_.ndarray, size: Tuple[int, int]) -> Tuple[Unpack[Tuple[numpy_.ndarray, ...]], numpy_.ndarray]:
    """Resize 2D map by area sampling with mask awareness.

### Parameters
- `*image`: Input image(s) of shape `(..., H, W, C)` or `(..., H, W)`
    - You can pass multiple images to be resized at the same time for efficiency.
- `mask`: Input mask of shape `(..., H, W)`
- `size`: target image size `(H', W')`

### Returns
- `*resized_image`: resized image(s) of shape `(..., H', W', C)`. or `(..., H', W')`
- `resized_mask`: mask of the resized map of shape `(..., H', W')`"""
    utils3d.numpy.maps.masked_area_resize

@overload
def colorize_depth_map(depth: numpy_.ndarray, mask: numpy_.ndarray = None, near: Optional[float] = None, far: Optional[float] = None, cmap: str = 'Spectral') -> numpy_.ndarray:
    """Colorize depth map for visualization.

## Parameters
    - `depth` (ndarray): shape (H, W), linear depth map
    - `mask` (ndarray, optional): shape (H, W), dtype=bool. Mask of valid depth pixels. Defaults to None.
    - `near` (float, optional): near plane for depth normalization. If None, use the 0.1% quantile of valid depth values. Defaults to None.
    - `far` (float, optional): far plane for depth normalization. If None, use the 99.9% quantile of valid depth values. Defaults to None.
    - `cmap` (str, optional): colormap name in matplotlib. Defaults to 'Spectral'.

## Returns
    - `colored` (ndarray): shape (H, W, 3), dtype=uint8, RGB [0, 255]"""
    utils3d.numpy.maps.colorize_depth_map

@overload
def colorize_normal_map(normal: numpy_.ndarray, mask: numpy_.ndarray = None, flip_yz: bool = False) -> numpy_.ndarray:
    """Colorize normal map for visualization. Value range is [-1, 1].

## Parameters
    - `normal` (ndarray): shape (H, W, 3), normal
    - `mask` (ndarray, optional): shape (H, W), dtype=bool. Mask of valid depth pixels. Defaults to None.
    - `flip_yz` (bool, optional): whether to flip the y and z. 
        - This is useful when converting between OpenCV and OpenGL camera coordinate systems. Defaults to False.

## Returns
    - `colored` (ndarray): shape (H, W, 3), dtype=uint8, RGB in [0, 255]"""
    utils3d.numpy.maps.colorize_normal_map

@overload
def flood_fill(*image: numpy_.ndarray, mask: numpy_.ndarray, return_index: bool = False) -> numpy_.ndarray:
    """Flooding fill the holes in the image(s) according to the mask. ![doc/flood_fill.png](doc/flood_fill.png)

Parameters
----
- `*image` (ndarray): shape (..., height, width, [C]), input image(s)
- `mask` (ndarray): shape (..., height, width), binary mask indicating valid regions

Returns
----
- `*filled_image` (ndarray): shape (..., height, width, [C]), flood filled map
- `filled_indices` (Tuple[ndarray, ...], optional): tuple of shape (..., height, width). The nearest neighbor indices of each pixel in the original map.
    It satisfies `filled_image = image[filled_indices]`"""
    utils3d.numpy.maps.flood_fill

@overload
def perlin_noise(x: numpy_.ndarray, seed: Optional[int] = None) -> numpy_.ndarray:
    """Generate Perlin noise for the given coordinates.

Parameters
----
- `x` (ndarray): shape (*batch_shape, N_1, ..., N_D, D), coordinates to sample Perlin.
    If input ndim is more than D + 1, the leading dimensions are treated as batch dimensions. Instances in the batch have different noise patterns.
- `seed` (int, optional): random seed. The same seed will generate the same noise pattern(s). Defaults to None.

Returns
----
- `y` (ndarray): shape (*batch_shape, N_1, ..., N_D), Perlin noise value at the given coordinates and seed. Value range is approximately [-1, 1]"""
    utils3d.numpy.maps.perlin_noise

@overload
def perlin_noise_map(size: Tuple[int, ...], frequency: Union[float, numpy_.ndarray], seed: Optional[int] = None, dtype: Optional[numpy_.dtype] = numpy_.float32) -> numpy_.ndarray:
    """Generate Perlin noise map.

Parameters
----
- `size` (Tuple[int, ...]): size of the noise map (..., H, W)
- `frequency` (float | ndarray): frequency relative to map's larger dimension max(H, W) of the Perlin noise. 
    Represents how many periods of noise fit in the larger dimension of the map.
- `seed` (int, optional): random seed. The same seed will generate the same noise pattern(s). Defaults to None.

Returns
----
- `noise_map` (ndarray): shape (..., H, W), Perlin noise map. Value range is approximately [-1, 1]"""
    utils3d.numpy.maps.perlin_noise_map

@overload
def fractal_perlin_noise_map(size: Tuple[int, ...], base_frequency: Union[float, numpy_.ndarray], octaves: int = 4, lacunarity: float = 2.0, gain: float = 0.5, seed: Optional[int] = None, dtype: Optional[numpy_.dtype] = numpy_.float32) -> numpy_.ndarray:
    """Generate fractal Perlin noise map. ![fractal_perlin_base_frequeny2_octaves7_gain0.7.png](doc/fractal_perlin_base_frequeny2_octaves7_gain0.7.png)

Parameters
----
- `size` (Tuple[int, ...]): size of the noise map (..., H, W)
- `base_frequency` (float | ndarray): base frequency (relative to map's larger dimension max(H, W)) of the Perlin noise.
    Represents how many periods of noise fit in the larger dimension of the map.
- `octaves` (int, optional): number of octaves. Defaults to 4.
- `lacunarity` (float, optional): frequency multiplier between octaves. Defaults to 2.0.
- `gain` (float, optional): amplitude multiplier between octaves. Defaults to 0.5.
- `seed` (int, optional): random seed. The same seed will generate the same noise pattern. Defaults to None.

Returns
----
- `noise_map` (ndarray): shape (..., H, W), fractal Perlin noise map. Value range is approximately [-1, 1]"""
    utils3d.numpy.maps.fractal_perlin_noise_map

@overload
def RastContext(*args, **kwargs):
    """Context for numpy-side rasterization. Based on moderngl.
    """
    utils3d.numpy.rasterization.RastContext

@overload
def rasterize_triangles(size: Tuple[int, int], *, vertices: numpy_.ndarray, attributes: Optional[numpy_.ndarray] = None, attributes_domain: Optional[Literal['vertex', 'face']] = 'vertex', faces: Optional[numpy_.ndarray] = None, view: numpy_.ndarray = None, projection: numpy_.ndarray = None, extrinsics: numpy_.ndarray = None, intrinsics: numpy_.ndarray = None, near: float = 0.01, far: float = inf, cull_backface: bool = False, return_depth: bool = False, return_interpolation: bool = False, background: Optional[Dict[str, numpy_.ndarray]] = None, ctx: Optional[utils3d.numpy.rasterization.RastContext] = None) -> Dict[str, numpy_.ndarray]:
    """Rasterize triangles.

Parameters
----
- `size` (Tuple[int, int]): (height, width) of the output image
- `vertices` (ndarray): (N, 3) or (T, 3, 3)
- `faces` (Optional[ndarray]): (T, 3) or None. If `None`, the vertices must be an array with shape (T, 3, 3)
- `attributes` (ndarray): (N, C), (T, 3, C) for vertex domain or (T, C) for face domain
- `attributes_domain` (Literal['vertex', 'face']): domain of the attributes
- `view` | `extrinsics` (ndarray): (4, 4) View matrix or extrinsics matrix. Provide either one of them.
- `projection` | `intrinsics` (ndarray): (4, 4) Projection matrix or (3, 3) Intrinsics matrix. Provide either one of them.
- `cull_backface` (bool): whether to cull backface
- `background` (Optional[Dict[str, ndarray]]): background to composite with. Contains the same keys as the return dictionary, each with shape matching the output.
- `ctx` (RastContext): rasterization context. Created by `RastContext()`. Default to the thread-local default context.

Returns
----
A dictionary containing

- `mask` (ndarray): (H, W) bool mask of valid pixels

if attributes is not None
- `image` (ndarray): (H, W, C) float32 rendered image corresponding to the input attributes

if return_depth is True
- `depth` (ndarray): (H, W) float32 camera space linear depth. Empty pixels are filled with inf.

if return_interpolation is True
- `interpolation_id` (ndarray): (H, W) int32 triangle ID map
- `interpolation_uv` (ndarray): (H, W, 2) float32 triangle UV (first two channels of barycentric coordinates)"""
    utils3d.numpy.rasterization.rasterize_triangles

@overload
def rasterize_triangles_peeling(size: Tuple[int, int], *, vertices: numpy_.ndarray, attributes: numpy_.ndarray, attributes_domain: Literal['vertex', 'face'] = 'vertex', faces: Optional[numpy_.ndarray] = None, view: numpy_.ndarray = None, projection: numpy_.ndarray = None, extrinsics: numpy_.ndarray = None, intrinsics: numpy_.ndarray = None, near: float = 0.01, far: float = inf, cull_backface: bool = False, return_depth: bool = False, return_interpolation: bool = False, ctx: Optional[utils3d.numpy.rasterization.RastContext] = None) -> Iterator[Iterator[Dict[str, numpy_.ndarray]]]:
    """Rasterize triangles with depth peeling.

Parameters
----
- `size` (Tuple[int, int]): (height, width) of the output image
- `vertices` (ndarray): (N, 3) or (T, 3, 3)
- `faces` (Optional[ndarray]): (T, 3) or None. If `None`, the vertices must be an array with shape (T, 3, 3)
- `attributes` (ndarray): (N, C), (T, 3, C) for vertex domain or (T, C) for face domain
- `attributes_domain` (Literal['vertex', 'face']): domain of the attributes
- `view` | `extrinsics` (ndarray): (4, 4) View matrix or extrinsics matrix. Provide either one of them.
- `projection` | `intrinsics` (ndarray): (4, 4) Projection matrix or (3, 3) Intrinsics matrix. Provide either one of them.
- `near` (float): near clipping plane. Only used for intrinsics. Ignored if projection matrix is provided.
- `far` (float): far clipping plane. Only used for intrinsics. Ignored if projection matrix is provided.
- `cull_backface` (bool): whether to cull backfaces
- `ctx` (RastContext): rasterization context. Created by `RastContext()`. Default to the thread-local default context.

Returns
----
A generator that yields dictionaries for each peeling layer containing:

- `mask` (ndarray): (H, W) bool mask of valid pixels in this layer

if attributes is not None
- `image` (ndarray): (H, W, C) float32 rendered image corresponding to the input attributes

if return_depth is True
- `depth` (ndarray): (H, W) float32 camera space linear depth. Empty pixels are filled with inf.

if return_interpolation is True
- `interpolation_id` (ndarray): (H, W) int32 triangle ID map
- `interpolation_uv` (ndarray): (H, W, 2) float32 triangle UV (first two channels of barycentric coordinates)

The last layer yielded will be empty (no pixels covered), then the generator stops.

Example
----
```
for i, layer_output in enumerate(rasterize_triangles_peeling(
    (512, 512), 
    vertices=vertices, 
    faces=faces, 
    attributes=attributes,
    view=view,
    projection=projection
)):
    print(f"Layer {i}:")
    for key, value in layer_output.items():
        print(f"  {key}: {value.shape}")
    if i >= 4:  # Stop after 5 layers at most
        break
```"""
    utils3d.numpy.rasterization.rasterize_triangles_peeling

@overload
def rasterize_lines(size: Tuple[int, int], *, vertices: numpy_.ndarray, lines: numpy_.ndarray, attributes: Optional[numpy_.ndarray], attributes_domain: Literal['vertex', 'line'] = 'vertex', view: Optional[numpy_.ndarray] = None, projection: Optional[numpy_.ndarray] = None, extrinsics: Optional[numpy_.ndarray] = None, intrinsics: Optional[numpy_.ndarray] = None, near: float = 0.01, far: float = inf, line_width: float = 1.0, return_depth: bool = False, return_interpolation: bool = False, background: Optional[Dict[str, numpy_.ndarray]] = None, ctx: Optional[utils3d.numpy.rasterization.RastContext] = None) -> Tuple[numpy_.ndarray, ...]:
    """Rasterize lines.

Parameters
----
- `size` (Tuple[int, int]): (height, width) of the output image
- `vertices` (ndarray): (N, 3) or (T, 3, 3)
- `faces` (Optional[ndarray]): (T, 3) or None. If `None`, the vertices must be an array with shape (T, 3, 3)
- `attributes` (ndarray): (N, C), (T, 3, C) for vertex domain or (T, C) for face domain
- `attributes_domain` (Literal['vertex', 'face']): domain of the attributes
- `view` | `extrinsics` (ndarray): (4, 4) View matrix or extrinsics matrix. Provide either one of them.
- `projection` | `intrinsics` (ndarray): (4, 4) Projection matrix or (3, 3) Intrinsics matrix. Provide either one of them.
- `near` (float): near clipping plane. Only used for intrinsics. Ignored if projection matrix is provided.
- `far` (float): far clipping plane. Only used for intrinsics. Ignored if projection matrix is provided.
- `background` (Optional[Dict[str, ndarray]]): background to composite with. Contains the same keys as the return dictionary, each with shape matching the output.
- `ctx` (RastContext): rasterization context. Created by `RastContext()`. Defaults to the current default context.

Returns
----
A dictionary containing

- `mask` (ndarray): (H, W) bool mask of valid pixels

if attributes is not None
- `image` (ndarray): (H, W, C) float32 rendered image corresponding to the input attributes

if return_depth is True
- `depth` (ndarray): (H, W) float32 camera space linear depth, ranging from 0 to 1.

if return_interpolation is True
- `interpolation_id` (ndarray): (H, W) int32 triangle ID map
- `interpolation_uv` (ndarray): (H, W, 2) float32 triangle UV (first two channels of barycentric coordinates)"""
    utils3d.numpy.rasterization.rasterize_lines

@overload
def rasterize_point_cloud(size: Tuple[int, int], *, points: numpy_.ndarray, point_sizes: Union[float, numpy_.ndarray] = 10, point_size_in: Literal['2d', '3d'] = '2d', point_shape: Literal['triangle', 'square', 'pentagon', 'hexagon', 'circle'] = 'square', attributes: Optional[numpy_.ndarray] = None, view: numpy_.ndarray = None, projection: numpy_.ndarray = None, extrinsics: numpy_.ndarray = None, intrinsics: numpy_.ndarray = None, near: float = 0.01, far: float = inf, return_depth: bool = False, return_point_id: bool = False, background: Optional[Dict[str, numpy_.ndarray]] = None, ctx: Optional[utils3d.numpy.rasterization.RastContext] = None) -> Dict[str, numpy_.ndarray]:
    """Rasterize point cloud.

Parameters
----
- `size` (Tuple[int, int]): (height, width) of the output image
- `points` (ndarray): (N, 3)
- `point_sizes` (ndarray): (N,) or float
- `point_size_in`: Literal['2d', '3d'] = '2d'. Whether the point sizes are in 2D (size in pixels) or 3D (size in world units).
- `point_shape`: Literal['triangle', 'square', 'pentagon', 'hexagon', 'circle'] = 'square'. The visual shape of the points.
- `attributes` (ndarray): (N, C)
- `view` | `extrinsics` (ndarray): (4, 4) View matrix or extrinsics matrix. Provide either one of them.
- `projection` | `intrinsics` (ndarray): (4, 4) Projection matrix or (3, 3) Intrinsics matrix. Provide either one of them.
- `near` (float): near clipping plane. Only used for intrinsics. Ignored if projection matrix is provided.
- `far` (float): far clipping plane. Only used for intrinsics. Ignored if projection matrix is provided.
- `return_depth` (bool): whether to return depth map
- `return_point_id` (bool): whether to return point ID map
- `background` (Optional[Dict[str, ndarray]]): background to composite with. Contains the same keys as the return dictionary, each with shape matching the output.
- `ctx` (RastContext): rasterization context. Created by `RastContext()`. Defaults to the current default context.

Returns
----
A dictionary containing

if attributes is not None
- `image` (ndarray): (H, W, C) float32 rendered image corresponding to the input attributes

if return_depth is True
- `depth` (ndarray): (H, W) float32 camera space linear depth, ranging from 0 to 1.

if return_point_id is True
- `point_id` (ndarray): (H, W) int32 point ID map"""
    utils3d.numpy.rasterization.rasterize_point_cloud

@overload
def sample_texture(uv_map: numpy_.ndarray, texture_map: numpy_.ndarray, interpolation: Literal['linear', 'nearest'] = 'linear', mipmap_level: Union[int, Tuple[int, int]] = 0, repeat: Union[bool, Tuple[bool, bool]] = False, anisotropic: float = 1.0, ctx: Optional[utils3d.numpy.rasterization.RastContext] = None) -> numpy_.ndarray:
    """Sample from a texture map with a UV map."""
    utils3d.numpy.rasterization.sample_texture

@overload
def test_rasterization(ctx: Optional[utils3d.numpy.rasterization.RastContext] = None):
    """Test if rasterization works. It will render a cube with random colors and save it as a CHECKME.png file."""
    utils3d.numpy.rasterization.test_rasterization

@overload
def read_extrinsics_from_colmap(file: Union[str, pathlib._local.Path]) -> Union[numpy_.ndarray, List[int], List[str]]:
    """Read extrinsics from colmap `images.txt` file. 
## Parameters
    file: Path to `images.txt` file.
## Returns
    extrinsics: (N, 4, 4) array of extrinsics.
    camera_ids: List of int, camera ids. Length is N. Note that camera ids in colmap typically starts from 1.
    image_names: List of str, image names. Length is N."""
    utils3d.numpy.io.colmap.read_extrinsics_from_colmap

@overload
def read_intrinsics_from_colmap(file: Union[str, pathlib._local.Path], normalize: bool = False) -> Tuple[List[int], numpy_.ndarray, numpy_.ndarray]:
    """Read intrinsics from colmap `cameras.txt` file.
## Parameters
    file: Path to `cameras.txt` file.
    normalize: Whether to normalize the intrinsics. If True, the intrinsics will be normalized. (mapping coordinates to [0, 1] range)
## Returns
    camera_ids: List of int, camera ids. Length is N. Note that camera ids in colmap typically starts from 1.
    intrinsics: (N, 3, 3) array of intrinsics.
    distortions: (N, 5) array of distortions."""
    utils3d.numpy.io.colmap.read_intrinsics_from_colmap

@overload
def write_extrinsics_as_colmap(file: Union[str, pathlib._local.Path], extrinsics: numpy_.ndarray, image_names: Union[str, List[str]] = 'image_{i:04d}.png', camera_ids: List[int] = None):
    """Write extrinsics to colmap `images.txt` file.
## Parameters
    file: Path to `images.txt` file.
    extrinsics: (N, 4, 4) array of extrinsics.
    image_names: str or List of str, image names. Length is N. 
        If str, it should be a format string with `i` as the index. (i starts from 1, in correspondence with IMAGE_ID in colmap)
    camera_ids: List of int, camera ids. Length is N.
        If None, it will be set to [1, 2, ..., N]."""
    utils3d.numpy.io.colmap.write_extrinsics_as_colmap

@overload
def write_intrinsics_as_colmap(file: Union[str, pathlib._local.Path], intrinsics: numpy_.ndarray, width: int, height: int, normalized: bool = False):
    """Write intrinsics to colmap `cameras.txt` file. Currently only support PINHOLE model (no distortion)
## Parameters
    file: Path to `cameras.txt` file.
    intrinsics: (N, 3, 3) array of intrinsics.
    width: Image width.
    height: Image height.
    normalized: Whether the intrinsics are normalized. If True, the intrinsics will unnormalized for writing."""
    utils3d.numpy.io.colmap.write_intrinsics_as_colmap

@overload
def read_obj(file: Union[str, pathlib._local.Path, _io.TextIOWrapper], encoding: Optional[str] = None, ignore_unknown: bool = False) -> utils3d.numpy.io.obj.WavefrontOBJDict:
    """Read wavefront .obj file.

Parameters
----
- `file` (str, Path, TextIOWrapper): filepath or file object
- `encoding` (str, optional): file encoding
- `ignore_unknown` (bool): whether to ignore unknown keywords in .obj file. Default to False.

Returns
----
A dictionary maybe containing the following fields. Note that some fields may be absent if not present in the .obj file:

Vertices and attributes data:
- `v` (ndarray): (N_v, 3 or 4) vertex coordinates.
- `vt` (ndarray): (N_vt, 2 or 3). vertex texture coordinates.
- `vn` (ndarray): (N_vn, 3). vertex normals.

Primitive definitions (face/line). NOTE: all indices are 0-based, unlike the 1-based indexing in .obj file.
- `f` (ndarray): Vertex indices in each face.
    - If all faces have the same number of vertices: (M, K)
    - If faces have different numbers of vertices: a tuple of segmented array:
        - `data` (ndarray): flattened array of shape (sum(K_i),)
        - `offsets` (ndarray): shape (M + 1,), offsets of each face in the flattened array
- `ft` (ndarray): Vertex texture coordinate indices of each face. The same format as `f`.
- `fn` (ndarray): Vertex normal indices of each face. The same format as `f`.
- `l` (ndarray): Vertex indices of each line segment.
    - If all lines have the same number of vertices: (L, K_line)
    - If lines have different numbers of vertices: a tuple of segmented array:
        - `data` (ndarray): flattened array of shape (sum(K_line_i),)
        - `offsets` (ndarray): shape (L + 1,), offsets of each line in the flattened array
- `lt` (ndarray): Vertex texture coordinate indices of each line. The same format as `l`.
- `p` (ndarray): Vertex indices of each point. 1D array of shape (N_p,)
- `pt` (ndarray): Vertex texture coordinate indices of each point. The same format as `p`.

Object/Group/Material info:
- `o` (Dict[str, Dict[Literal['f', 'l', 'p'], slice]]). The objects and their corresponding faces/lines/points. E.g. `o['object_A']['f']` gives the slice of faces belonging to object_A.
- `g` (Dict[str, Dict[Literal['f', 'l', 'p'], slice]]). The groups and their corresponding faces/lines/points. E.g. `g['group_A']['f']` gives the slice of faces belonging to group_A.
- `usemtl` (Dict[str, Dict[Literal['f', 'l', 'p'], slice]]). The materials and their corresponding faces/lines/points. E.g. `usemtl['material_A']['f']` gives the slice of faces using material_A.

Here, `names` is a list of names, and `offsets` is an array of shape (num_entities + 1,) indicating the start and end indices of faces belonging to each object/group/material.
For example, the second material has name `material_names[1]`, and its faces are `f[material_offsets[1]:material_offsets[2]]`.

Material library:
- `mtllib` (List[str]): list of material library filenames - `mtllib` lines in .obj file."""
    utils3d.numpy.io.obj.read_obj

@overload
def write_obj(file: Union[str, pathlib._local.Path, os.PathLike], obj: utils3d.numpy.io.obj.WavefrontOBJDict, encoding: Optional[str] = None):
    utils3d.numpy.io.obj.write_obj

@overload
def sliding_window(x: torch_.Tensor, window_size: Union[int, Tuple[int, ...]], stride: Union[int, Tuple[int, ...], NoneType] = None, dilation: Union[int, Tuple[int, ...], NoneType] = None, pad_size: Union[int, Tuple[int, int], Tuple[Tuple[int, int]], NoneType] = None, pad_mode: str = 'constant', pad_value: numbers.Number = 0, dim: Tuple[int, ...] = None) -> torch_.Tensor:
    """Get a sliding window of the input array.
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
    - Otherwise, the output is no longer a view but a copy of the padded array."""
    utils3d.torch.utils.sliding_window

@overload
def masked_min(input: torch_.Tensor, mask: torch_.BoolTensor, dim: int = None, keepdim: bool = False) -> Union[torch_.Tensor, Tuple[torch_.Tensor, torch_.Tensor]]:
    """Similar to torch.min, but with mask
    """
    utils3d.torch.utils.masked_min

@overload
def masked_max(input: torch_.Tensor, mask: torch_.BoolTensor, dim: int = None, keepdim: bool = False) -> Union[torch_.Tensor, Tuple[torch_.Tensor, torch_.Tensor]]:
    """Similar to torch.max, but with mask
    """
    utils3d.torch.utils.masked_max

@overload
def lookup(key: torch_.Tensor, query: torch_.Tensor) -> torch_.LongTensor:
    """Look up `query` in `key` like a dictionary. Useful for COO indexing.

## Parameters
- `key` (Tensor): shape `(K, ...)`, the array to search in
- `query` (Tensor): shape `(Q, ...)`, the array to search for

## Returns
- `indices` (Tensor): shape `(Q,)` indices of `query` in `key`. If a query is not found in key, the corresponding index will be -1.

## NOTE
`O((Q + K) * log(Q + K))` complexity."""
    utils3d.torch.utils.lookup

@overload
def lookup_get(key: torch_.Tensor, value: torch_.Tensor, get_key: torch_.Tensor, default_value: Union[numbers.Number, torch_.Tensor] = 0) -> torch_.Tensor:
    """Dictionary-like get for arrays

## Parameters
- `key` (Tensor): shape `(N, *key_shape)`, the key array of the dictionary to get from
- `value` (Tensor): shape `(N, *value_shape)`, the value array of the dictionary to get from
- `get_key` (Tensor): shape `(M, *key_shape)`, the key array to get for

## Returns
    `get_value` (Tensor): shape `(M, *value_shape)`, result values corresponding to `get_key`"""
    utils3d.torch.utils.lookup_get

@overload
def lookup_set(key: torch_.Tensor, value: torch_.Tensor, set_key: torch_.Tensor, set_value: torch_.Tensor, append: bool = False, inplace: bool = False) -> Tuple[torch_.Tensor, torch_.Tensor]:
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
- `result_value (Tensor): shape `(N_new, *value_shape)` """
    utils3d.torch.utils.lookup_set

@overload
def segment_roll(data: torch_.Tensor, offsets: torch_.Tensor, shift: int) -> torch_.Tensor:
    """Roll the data within each segment.
    """
    utils3d.torch.utils.segment_roll

@overload
def segment_take(data: torch_.Tensor, offsets: torch_.Tensor, taking: torch_.Tensor) -> Tuple[torch_.Tensor, torch_.Tensor]:
    """Take some segments from a segmented array
    """
    utils3d.torch.utils.segment_take

@overload
def csr_matrix_from_dense_indices(indices: torch_.Tensor, n_cols: int) -> torch_.Tensor:
    """Convert a regular indices array to a sparse CSR adjacency matrix format

## Parameters
    - `indices` (Tensor): shape (N, M) dense tensor. Each one in `N` has `M` connections.
    - `values` (Tensor): shape (N, M) values of the connections
    - `n_cols` (int): total number of columns in the adjacency matrix

## Returns
    Tensor: shape `(N, n_cols)` sparse CSR adjacency matrix"""
    utils3d.torch.utils.csr_matrix_from_dense_indices

@overload
def csr_eliminate_zeros(input: torch_.Tensor):
    """Remove zero elements from a sparse CSR tensor.
    """
    utils3d.torch.utils.csr_eliminate_zeros

@overload
def group(labels: torch_.Tensor, data: Optional[torch_.Tensor] = None) -> List[Tuple[torch_.Tensor, torch_.Tensor]]:
    """Split the data into groups based on the provided labels.

## Parameters
- `labels` (Tensor): shape `(N, *label_dims)` array of labels for each data point. Labels can be multi-dimensional.
- `data` (Tensor, optional): shape `(N, *data_dims)` dense tensor. Each one in `N` has `D` features.
    If None, return the indices in each group instead.

## Returns
- `groups` (List[Tuple[Tensor, Tensor]]): List of each group, a tuple of `(label, data_in_group)`.
    - `label` (Tensor): shape (*label_dims,) the label of the group.
    - `data_in_group` (Tensor): shape (M, *data_dims) the data points in the group.
    If `data` is None, `data_in_group` will be the indices of the data points in the original array."""
    utils3d.torch.utils.group

@overload
def group_as_segments(labels: torch_.Tensor, data: Optional[torch_.Tensor] = None) -> Tuple[torch_.Tensor, torch_.Tensor, torch_.Tensor]:
    """Group as segments by labels

## Parameters

- `labels` (Tensor): shape `(N, *label_dims)` array of labels for each data point. Labels can be multi-dimensional.
- `data` (Tensor, optional): shape `(N, *data_dims)` array.
    If None, return the indices in each group instead.

## Returns

Assuming there are `M` difference labels:

- `segment_labels`: `(Tensor)` shape `(M, *label_dims)` labels of of each segment
- `data`: `(Tensor)` shape `(N,)` or `(N, *data_dims)` the rearranged data (or indices) where the same labels are grouped as a continous segment.
- `offsets`: `(Tensor)` shape `(M + 1,)`

`data[offsets[i]:offsets[i + 1]]` corresponding to the i-th segment whose label is `segment_labels[i]`"""
    utils3d.torch.utils.group_as_segments

@overload
def perspective_from_fov(*, fov_x: Union[float, torch_.Tensor, NoneType] = None, fov_y: Union[float, torch_.Tensor, NoneType] = None, fov_min: Union[float, torch_.Tensor, NoneType] = None, fov_max: Union[float, torch_.Tensor, NoneType] = None, aspect_ratio: Union[float, torch_.Tensor, NoneType] = None, near: Union[float, torch_.Tensor, NoneType], far: Union[float, torch_.Tensor, NoneType]) -> torch_.Tensor:
    """Get OpenGL perspective matrix from field of view 

## Returns
    (Tensor): [..., 4, 4] perspective matrix"""
    utils3d.torch.transforms.perspective_from_fov

@overload
def perspective_from_window(left: Union[float, torch_.Tensor], right: Union[float, torch_.Tensor], bottom: Union[float, torch_.Tensor], top: Union[float, torch_.Tensor], near: Union[float, torch_.Tensor], far: Union[float, torch_.Tensor]) -> torch_.Tensor:
    """Get OpenGL perspective matrix from the window of z=-1 projection plane

## Returns
    (Tensor): [..., 4, 4] perspective matrix"""
    utils3d.torch.transforms.perspective_from_window

@overload
def intrinsics_from_fov(fov_x: Union[float, torch_.Tensor, NoneType] = None, fov_y: Union[float, torch_.Tensor, NoneType] = None, fov_max: Union[float, torch_.Tensor, NoneType] = None, fov_min: Union[float, torch_.Tensor, NoneType] = None, aspect_ratio: Union[float, torch_.Tensor, NoneType] = None) -> torch_.Tensor:
    """Get normalized OpenCV intrinsics matrix from given field of view.
You can provide either fov_x, fov_y, fov_max or fov_min and aspect_ratio

## Parameters
    fov_x (float | Tensor): field of view in x axis
    fov_y (float | Tensor): field of view in y axis
    fov_max (float | Tensor): field of view in largest dimension
    fov_min (float | Tensor): field of view in smallest dimension
    aspect_ratio (float | Tensor): aspect ratio of the image

## Returns
    (Tensor): [..., 3, 3] OpenCV intrinsics matrix"""
    utils3d.torch.transforms.intrinsics_from_fov

@overload
def intrinsics_from_focal_center(fx: Union[float, torch_.Tensor], fy: Union[float, torch_.Tensor], cx: Union[float, torch_.Tensor], cy: Union[float, torch_.Tensor]) -> torch_.Tensor:
    """Get OpenCV intrinsics matrix

## Parameters
    focal_x (float | Tensor): focal length in x axis
    focal_y (float | Tensor): focal length in y axis
    cx (float | Tensor): principal point in x axis
    cy (float | Tensor): principal point in y axis

## Returns
    (Tensor): [..., 3, 3] OpenCV intrinsics matrix"""
    utils3d.torch.transforms.intrinsics_from_focal_center

@overload
def focal_to_fov(focal: torch_.Tensor):
    utils3d.torch.transforms.focal_to_fov

@overload
def fov_to_focal(fov: torch_.Tensor):
    utils3d.torch.transforms.fov_to_focal

@overload
def intrinsics_to_fov(intrinsics: torch_.Tensor) -> Tuple[torch_.Tensor, torch_.Tensor]:
    """NOTE: approximate FOV by assuming centered principal point"""
    utils3d.torch.transforms.intrinsics_to_fov

@overload
def view_look_at(eye: torch_.Tensor, look_at: torch_.Tensor, up: torch_.Tensor) -> torch_.Tensor:
    """Get OpenGL view matrix looking at something

## Parameters
    eye (Tensor): [..., 3] the eye position
    look_at (Tensor): [..., 3] the position to look at
    up (Tensor): [..., 3] head up direction (y axis in screen space). Not necessarily othogonal to view direction

## Returns
    (Tensor): [..., 4, 4], view matrix"""
    utils3d.torch.transforms.view_look_at

@overload
def extrinsics_look_at(eye: torch_.Tensor, look_at: torch_.Tensor, up: torch_.Tensor) -> torch_.Tensor:
    """Get OpenCV extrinsics matrix looking at something

## Parameters
    eye (Tensor): [..., 3] the eye position
    look_at (Tensor): [..., 3] the position to look at
    up (Tensor): [..., 3] head up direction (-y axis in screen space). Not necessarily othogonal to view direction

## Returns
    (Tensor): [..., 4, 4], extrinsics matrix"""
    utils3d.torch.transforms.extrinsics_look_at

@overload
def perspective_to_intrinsics(perspective: torch_.Tensor) -> torch_.Tensor:
    """OpenGL perspective matrix to OpenCV intrinsics

## Parameters
    perspective (Tensor): [..., 4, 4] OpenGL perspective matrix

## Returns
    (Tensor): shape [..., 3, 3] OpenCV intrinsics"""
    utils3d.torch.transforms.perspective_to_intrinsics

@overload
def intrinsics_to_perspective(intrinsics: torch_.Tensor, near: Union[float, torch_.Tensor], far: Union[float, torch_.Tensor]) -> torch_.Tensor:
    """OpenCV intrinsics to OpenGL perspective matrix
NOTE: not work for tile-shifting intrinsics currently

## Parameters
    intrinsics (Tensor): [..., 3, 3] OpenCV intrinsics matrix
    near (float | Tensor): [...] near plane to clip
    far (float | Tensor): [...] far plane to clip
## Returns
    (Tensor): [..., 4, 4] OpenGL perspective matrix"""
    utils3d.torch.transforms.intrinsics_to_perspective

@overload
def extrinsics_to_view(extrinsics: torch_.Tensor) -> torch_.Tensor:
    """OpenCV camera extrinsics to OpenGL view matrix

## Parameters
    extrinsics (Tensor): [..., 4, 4] OpenCV camera extrinsics matrix

## Returns
    (Tensor): [..., 4, 4] OpenGL view matrix"""
    utils3d.torch.transforms.extrinsics_to_view

@overload
def view_to_extrinsics(view: torch_.Tensor) -> torch_.Tensor:
    """OpenGL view matrix to OpenCV camera extrinsics

## Parameters
    view (Tensor): [..., 4, 4] OpenGL view matrix

## Returns
    (Tensor): [..., 4, 4] OpenCV camera extrinsics matrix"""
    utils3d.torch.transforms.view_to_extrinsics

@overload
def normalize_intrinsics(intrinsics: torch_.Tensor, size: Union[Tuple[numbers.Number, numbers.Number], torch_.Tensor], pixel_convention: Literal['integer-corner', 'integer-center'] = 'integer-center') -> torch_.Tensor:
    """Normalize camera intrinsics to uv space

## Parameters
- `intrinsics` (Tensor): `(..., 3, 3)` camera intrinsics to normalize
- `size` (tuple | Tensor): A tuple `(height, width)` of the image size,
    or an array of shape `(..., 2)` corresponding to the multiple image size(s)
- `pixel_convention`: `str`, optional `'integer-center'` or `'integer-corner'`, whether integer coordinates correspond to pixel centers or corners. Defaults to 'integer-center'.
    - For more definitions, please refer to `pixel_coord_map()`

## Returns
    (Tensor): [..., 3, 3] normalized camera intrinsics"""
    utils3d.torch.transforms.normalize_intrinsics

@overload
def denormalize_intrinsics(intrinsics: torch_.Tensor, size: Union[Tuple[numbers.Number, numbers.Number], torch_.Tensor], pixel_convention: Literal['integer-center', 'integer-corner'] = 'integer-center') -> torch_.Tensor:
    """Denormalize camera intrinsics(s) from uv space to pixel space

## Parameters
- `intrinsics` (Tensor): `(..., 3, 3)` camera intrinsics
- `size` (tuple | Tensor): A tuple `(height, width)` of the image size,
    or an array of shape `(..., 2)` corresponding to the multiple image size(s)
- `pixel_convention`: `str`, optional `'integer-center'` or `'integer-corner'`, whether integer coordinates correspond to pixel centers or corners. Defaults to 'integer-center'.
    - For more definitions, please refer to `pixel_coord_map()`

## Returns
    (Tensor): [..., 3, 3] denormalized camera intrinsics in pixel space"""
    utils3d.torch.transforms.denormalize_intrinsics

@overload
def crop_intrinsics(intrinsics: torch_.Tensor, size: Union[Tuple[numbers.Number, numbers.Number], torch_.Tensor], cropped_top: Union[numbers.Number, torch_.Tensor], cropped_left: Union[numbers.Number, torch_.Tensor], cropped_height: Union[numbers.Number, torch_.Tensor], cropped_width: Union[numbers.Number, torch_.Tensor]) -> torch_.Tensor:
    """Evaluate the new intrinsics after cropping the image

## Parameters
    intrinsics (Tensor): (..., 3, 3) camera intrinsics(s) to crop
    height (int | Tensor): (...) image height(s)
    width (int | Tensor): (...) image width(s)
    cropped_top (int | Tensor): (...) top pixel index of the cropped image(s)
    cropped_left (int | Tensor): (...) left pixel index of the cropped image(s)
    cropped_height (int | Tensor): (...) height of the cropped image(s)
    cropped_width (int | Tensor): (...) width of the cropped image(s)

## Returns
    (Tensor): (..., 3, 3) cropped camera intrinsics"""
    utils3d.torch.transforms.crop_intrinsics

@overload
def pixel_to_uv(pixel: torch_.Tensor, size: Union[Tuple[numbers.Number, numbers.Number], torch_.Tensor], pixel_convention: Literal['integer-corner', 'integer-center'] = 'integer-center') -> torch_.Tensor:
    """## Parameters
- `pixel` (Tensor): `(..., 2)` pixel coordinrates 
- `size` (tuple | Tensor): A tuple `(height, width)` of the image size,
    or an array of shape `(..., 2)` corresponding to the multiple image size(s)
- `pixel_convention`: `str`, optional `'integer-center'` or `'integer-corner'`, whether integer coordinates correspond to pixel centers or corners. Defaults to 'integer-center'.
    - For more definitions, please refer to `pixel_coord_map()`

## Returns
    (Tensor): `(..., 2)` uv coordinrates"""
    utils3d.torch.transforms.pixel_to_uv

@overload
def pixel_to_ndc(pixel: torch_.Tensor, size: Union[Tuple[numbers.Number, numbers.Number], torch_.Tensor], pixel_convention: Literal['integer-corner', 'integer-center'] = 'integer-center') -> torch_.Tensor:
    """Convert pixel coordinates to NDC (Normalized Device Coordinates).

## Parameters
- `pixel` (Tensor): `(..., 2)` pixel coordinrates.
- `size` (tuple | Tensor): A tuple `(height, width)` of the image size,
    or an array of shape `(..., 2)` corresponding to the multiple image size(s)
- `pixel_convention`: `str`, optional `'integer-center'` or `'integer-corner'`, whether integer coordinates correspond to pixel centers or corners. Defaults to 'integer-center'.
    - For more definitions, please refer to `pixel_coord_map()`

## Returns
    (Tensor): `(..., 2)` ndc coordinrates, the range is (-1, 1)"""
    utils3d.torch.transforms.pixel_to_ndc

@overload
def uv_to_pixel(uv: torch_.Tensor, size: Union[Tuple[numbers.Number, numbers.Number], torch_.Tensor], pixel_convention: Literal['integer-corner', 'integer-center'] = 'integer-center') -> torch_.Tensor:
    """Convert UV space coordinates to pixel space coordinates.

## Parameters
- `uv` (Tensor): `(..., 2)` uv coordinrates.
- `size` (tuple | Tensor): A tuple `(height, width)` of the image size,
    or an array of shape `(..., 2)` corresponding to the multiple image size(s)
- `pixel_convention`: `str`, optional `'integer-center'` or `'integer-corner'`, whether integer coordinates correspond to pixel centers or corners. Defaults to 'integer-center'.
    - For more definitions, please refer to `pixel_coord_map()`

## Returns
    (Tensor): `(..., 2)` pixel coordinrates"""
    utils3d.torch.transforms.uv_to_pixel

@overload
def depth_linear_to_buffer(depth: torch_.Tensor, near: Union[float, torch_.Tensor], far: Union[float, torch_.Tensor]) -> torch_.Tensor:
    """Project linear depth to depth value in screen space

## Parameters
    depth (Tensor): [...] depth value
    near (float | Tensor): [...] near plane to clip
    far (float | Tensor): [...] far plane to clip

## Returns
    (Tensor): [..., 1] depth value in screen space, value ranging in [0, 1]"""
    utils3d.torch.transforms.depth_linear_to_buffer

@overload
def depth_buffer_to_linear(depth: torch_.Tensor, near: Union[float, torch_.Tensor], far: Union[float, torch_.Tensor]) -> torch_.Tensor:
    """Linearize depth value to linear depth

## Parameters
    depth (Tensor): [...] screen depth value, ranging in [0, 1]
    near (float | Tensor): [...] near plane to clip
    far (float | Tensor): [...] far plane to clip

## Returns
    (Tensor): [...] linear depth"""
    utils3d.torch.transforms.depth_buffer_to_linear

@overload
def project_gl(points: torch_.Tensor, projection: torch_.Tensor, view: torch_.Tensor = None) -> Tuple[torch_.Tensor, torch_.Tensor]:
    """Project 3D points to 2D following the OpenGL convention (except for row major matrices)

## Parameters
    points (Tensor): [..., N, 3] or [..., N, 4] 3D points to project, if the last 
        dimension is 4, the points are assumed to be in homogeneous coordinates
    view (Tensor): [..., 4, 4] view matrix
    projection (Tensor): [..., 4, 4] projection matrix

## Returns
    scr_coord (Tensor): [..., N, 3] screen space coordinates, value ranging in [0, 1].
        The origin (0., 0., 0.) is corresponding to the left & bottom & nearest
    linear_depth (Tensor): [..., N] linear depth"""
    utils3d.torch.transforms.project_gl

@overload
def project_cv(points: torch_.Tensor, intrinsics: torch_.Tensor, extrinsics: Optional[torch_.Tensor] = None) -> Tuple[torch_.Tensor, torch_.Tensor]:
    """Project 3D points to 2D following the OpenCV convention

## Parameters
    points (Tensor): [..., N, 3] 3D points
    intrinsics (Tensor): [..., 3, 3] intrinsics matrix
    extrinsics (Tensor): [..., 4, 4] extrinsics matrix

## Returns
    uv_coord (Tensor): [..., N, 2] uv coordinates, value ranging in [0, 1].
        The origin (0., 0.) is corresponding to the left & top
    linear_depth (Tensor): [..., N] linear depth"""
    utils3d.torch.transforms.project_cv

@overload
def unproject_gl(uv: torch_.Tensor, depth: torch_.Tensor, projection: torch_.Tensor, view: Optional[torch_.Tensor] = None) -> torch_.Tensor:
    """Unproject screen space coordinates to 3D view space following the OpenGL convention (except for row major matrices)

## Parameters
    uv (Tensor): (..., N, 2) screen space XY coordinates, value ranging in [0, 1].
        The origin (0., 0.) is corresponding to the left & bottom
    depth (Tensor): (..., N) linear depth values
    projection (Tensor): (..., 4, 4) projection  matrix
    view (Tensor): (..., 4, 4) view matrix
    
## Returns
    points (Tensor): (..., N, 3) 3d points"""
    utils3d.torch.transforms.unproject_gl

@overload
def unproject_cv(uv: torch_.Tensor, depth: torch_.Tensor, intrinsics: torch_.Tensor, extrinsics: torch_.Tensor = None) -> torch_.Tensor:
    """Unproject uv coordinates to 3D view space following the OpenCV convention

## Parameters
    uv (Tensor): [..., N, 2] uv coordinates, value ranging in [0, 1].
        The origin (0., 0.) is corresponding to the left & top
    depth (Tensor): [..., N] depth value
    extrinsics (Tensor): [..., 4, 4] extrinsics matrix
    intrinsics (Tensor): [..., 3, 3] intrinsics matrix

## Returns
    points (Tensor): [..., N, 3] 3d points"""
    utils3d.torch.transforms.unproject_cv

@overload
def project(points: torch_.Tensor, *, intrinsics: Optional[torch_.Tensor] = None, extrinsics: Optional[torch_.Tensor] = None, view: Optional[torch_.Tensor] = None, projection: Optional[torch_.Tensor] = None) -> Tuple[torch_.Tensor, torch_.Tensor]:
    """Calculate projection. 
- For OpenCV convention, use `intrinsics` and `extrinsics` matrices. 
- For OpenGL convention, use `view` and `projection` matrices.

## Parameters

- `points`: (..., N, 3) 3D world-space points
- `intrinsics`: (..., 3, 3) intrinsics matrix
- `extrinsics`: (..., 4, 4) extrinsics matrix
- `view`: (..., 4, 4) view matrix
- `projection`: (..., 4, 4) projection matrix

## Returns

- `uv`: (..., N, 2) 2D coordinates. 
    - For OpenCV convention, it is the normalized image coordinate where (0, 0) is the top left corner.
    - For OpenGL convention, it is the screen space XY coordinate where (0, 0) is the bottom left corner.
- `depth`: (..., N) linear depth values, where `depth > 0` is visible.
    - For OpenCV convention, it is the Z coordinate in camera space.
    - For OpenGL convention, it is the -Z coordinate in camera space."""
    utils3d.torch.transforms.project

@overload
def unproject(uv: torch_.Tensor, depth: Optional[torch_.Tensor], *, intrinsics: Optional[torch_.Tensor] = None, extrinsics: Optional[torch_.Tensor] = None, projection: Optional[torch_.Tensor] = None, view: Optional[torch_.Tensor] = None) -> torch_.Tensor:
    """Calculate inverse projection. 
- For OpenCV convention, use `intrinsics` and `extrinsics` matrices. 
- For OpenGL convention, use `view` and `projection` matrices.

## Parameters

- `uv`: (..., N, 2) 2D coordinates. 
    - For OpenCV convention, it is the normalized image coordinate where (0, 0) is the top left corner.
    - For OpenGL convention, it is the screen space XY coordinate where (0, 0) is the bottom left corner.
- `depth`: (..., N) linear depth values, where `depth > 0` is visible.
    - For OpenCV convention, it is the Z coordinate in camera space.
    - For OpenGL convention, it is the -Z coordinate in camera space.
- `intrinsics`: (..., 3, 3) intrinsics matrix
- `extrinsics`: (..., 4, 4) extrinsics matrix
- `view`: (..., 4, 4) view matrix
- `projection`: (..., 4, 4) projection matrix

## Returns

- `points`: (..., N, 3) 3D world-space points"""
    utils3d.torch.transforms.unproject

@overload
def skew_symmetric(v: torch_.Tensor):
    """Skew symmetric matrix from a 3D vector"""
    utils3d.torch.transforms.skew_symmetric

@overload
def rotation_matrix_from_vectors(v1: torch_.Tensor, v2: torch_.Tensor):
    """Rotation matrix that rotates v1 to v2"""
    utils3d.torch.transforms.rotation_matrix_from_vectors

@overload
def euler_axis_angle_rotation(axis: str, angle: torch_.Tensor) -> torch_.Tensor:
    """Return the rotation matrices for one of the rotations about an axis
of which Euler angles describe, for each value of the angle given.

## Parameters
    axis: Axis label "X" or "Y or "Z".
    angle: any shape tensor of Euler angles in radians

## Returns
    Rotation matrices as tensor of shape (..., 3, 3)."""
    utils3d.torch.transforms.euler_axis_angle_rotation

@overload
def euler_angles_to_matrix(euler_angles: torch_.Tensor, convention: str = 'XYZ') -> torch_.Tensor:
    """Convert rotations given as Euler angles in radians to rotation matrices.

## Parameters
    euler_angles: Euler angles in radians as tensor of shape (..., 3), XYZ
    convention: permutation of "X", "Y" or "Z", representing the order of Euler rotations to apply.

## Returns
    Rotation matrices as tensor of shape (..., 3, 3)."""
    utils3d.torch.transforms.euler_angles_to_matrix

@overload
def matrix_to_euler_angles(matrix: torch_.Tensor, convention: str) -> torch_.Tensor:
    """Convert rotations given as rotation matrices to Euler angles in radians.
NOTE: The composition order eg. `XYZ` means `Rz * Ry * Rx` (like blender), instead of `Rx * Ry * Rz` (like pytorch3d)

## Parameters
    matrix: Rotation matrices as tensor of shape (..., 3, 3).
    convention: Convention string of three uppercase letters.

## Returns
    Euler angles in radians as tensor of shape (..., 3), in the order of XYZ (like blender), instead of convention (like pytorch3d)"""
    utils3d.torch.transforms.matrix_to_euler_angles

@overload
def matrix_to_quaternion(rot_mat: torch_.Tensor, eps: float = 1e-12) -> torch_.Tensor:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)

## Parameters
    rot_mat (Tensor): shape (..., 3, 3), the rotation matrices to convert

## Returns
    Tensor: shape (..., 4), the quaternions corresponding to the given rotation matrices"""
    utils3d.torch.transforms.matrix_to_quaternion

@overload
def quaternion_to_matrix(quaternion: torch_.Tensor, eps: float = 1e-12) -> torch_.Tensor:
    """Converts a batch of quaternions (w, x, y, z) to rotation matrices

## Parameters
    quaternion (Tensor): shape (..., 4), the quaternions to convert

## Returns
    Tensor: shape (..., 3, 3), the rotation matrices corresponding to the given quaternions"""
    utils3d.torch.transforms.quaternion_to_matrix

@overload
def matrix_to_axis_angle(rot_mat: torch_.Tensor, eps: float = 1e-12) -> torch_.Tensor:
    """Convert a batch of 3x3 rotation matrices to axis-angle representation (rotation vector)

## Parameters
    rot_mat (Tensor): shape (..., 3, 3), the rotation matrices to convert

## Returns
    Tensor: shape (..., 3), the axis-angle vectors corresponding to the given rotation matrices"""
    utils3d.torch.transforms.matrix_to_axis_angle

@overload
def axis_angle_to_matrix(axis_angle: torch_.Tensor, eps: float = 1e-12) -> torch_.Tensor:
    """Convert axis-angle representation (rotation vector) to rotation matrix, whose direction is the axis of rotation and length is the angle of rotation

## Parameters
    axis_angle (Tensor): shape (..., 3), axis-angle vcetors

## Returns
    Tensor: shape (..., 3, 3) The rotation matrices for the given axis-angle parameters"""
    utils3d.torch.transforms.axis_angle_to_matrix

@overload
def axis_angle_to_quaternion(axis_angle: torch_.Tensor, eps: float = 1e-12) -> torch_.Tensor:
    """Convert axis-angle representation (rotation vector) to quaternion (w, x, y, z)

## Parameters
    axis_angle (Tensor): shape (..., 3), axis-angle vcetors

## Returns
    Tensor: shape (..., 4) The quaternions for the given axis-angle parameters"""
    utils3d.torch.transforms.axis_angle_to_quaternion

@overload
def quaternion_to_axis_angle(quaternion: torch_.Tensor, eps: float = 1e-12) -> torch_.Tensor:
    """Convert a batch of quaternions (w, x, y, z) to axis-angle representation (rotation vector)

## Parameters
    quaternion (Tensor): shape (..., 4), the quaternions to convert

## Returns
    Tensor: shape (..., 3), the axis-angle vectors corresponding to the given quaternions"""
    utils3d.torch.transforms.quaternion_to_axis_angle

@overload
def make_affine_matrix(M: torch_.Tensor, t: torch_.Tensor):
    """Make an affine transformation matrix from a linear matrix and a translation vector.

## Parameters
    M (Tensor): [..., D, D] linear matrix (rotation, scaling or general deformation)
    t (Tensor): [..., D] translation vector

## Returns
    Tensor: [..., D + 1, D + 1] affine transformation matrix"""
    utils3d.torch.transforms.make_affine_matrix

@overload
def random_rotation_matrix(*size: int, dtype=torch_.float32, device: torch_.device = None) -> torch_.Tensor:
    """Generate random 3D rotation matrix.

## Parameters
    dtype: The data type of the output rotation matrix.

## Returns
    Tensor: `(*size, 3, 3)` random rotation matrix."""
    utils3d.torch.transforms.random_rotation_matrix

@overload
def lerp(v1: torch_.Tensor, v2: torch_.Tensor, t: torch_.Tensor) -> torch_.Tensor:
    """Linear interpolation between two vectors.

## Parameters
- `v1` (Tensor): `(..., D)` vector 1
- `v2` (Tensor): `(..., D)` vector 2
- `t` (Tensor): `(..., N)` interpolation parameter in [0, 1]

## Returns
    Tensor: `(..., N, D)` interpolated vector"""
    utils3d.torch.transforms.lerp

@overload
def slerp(v1: torch_.Tensor, v2: torch_.Tensor, t: torch_.Tensor, eps: float = 1e-12) -> torch_.Tensor:
    """Spherical linear interpolation between two (unit) vectors. 

## Parameters
    `v1` (Tensor): `(..., D)` (unit) vector 1
    `v2` (Tensor): `(..., D)` (unit) vector 2
    `t` (Tensor): `(..., N)` interpolation parameter in [0, 1]

## Returns
    Tensor: `(..., N, D)` interpolated unit vector"""
    utils3d.torch.transforms.slerp

@overload
def slerp_rotation_matrix(R1: torch_.Tensor, R2: torch_.Tensor, t: Union[numbers.Number, torch_.Tensor]) -> torch_.Tensor:
    """Spherical linear interpolation between two 3D rotation matrices

## Parameters
    R1 (Tensor): shape (..., 3, 3), the first rotation matrix
    R2 (Tensor): shape (..., 3, 3), the second rotation matrix
    t (Tensor): scalar or shape (..., N), the interpolation factor

## Returns
    Tensor: shape (..., N, 3, 3), the interpolated rotation matrix"""
    utils3d.torch.transforms.slerp_rotation_matrix

@overload
def interpolate_se3_matrix(T1: torch_.Tensor, T2: torch_.Tensor, t: torch_.Tensor):
    """Interpolate between two SE(3) transformation matrices.
- Spherical linear interpolation (SLERP) is used for the rotational part.
- Linear interpolation is used for the translational part.

## Parameters
- `T1` (Tensor): (..., 4, 4) SE(3) matrix 1
- `T2` (Tensor): (..., 4, 4) SE(3) matrix 2
- `t` (Tensor): (..., N) interpolation parameter in [0, 1]

## Returns
    Tensor: (..., N, 4, 4) interpolated SE(3) matrix"""
    utils3d.torch.transforms.interpolate_se3_matrix

@overload
def extrinsics_to_essential(extrinsics: torch_.Tensor):
    """extrinsics matrix `[[R, t] [0, 0, 0, 1]]` such that `x' = R (x - t)` to essential matrix such that `x' E x = 0`

## Parameters
    extrinsics (Tensor): [..., 4, 4] extrinsics matrix

## Returns
    (Tensor): [..., 3, 3] essential matrix"""
    utils3d.torch.transforms.extrinsics_to_essential

@overload
def rotation_matrix_2d(theta: Union[float, torch_.Tensor]):
    """2x2 matrix for 2D rotation

## Parameters
    theta (float | Tensor): rotation angle in radians, arbitrary shape (...,)

## Returns
    (Tensor): (..., 2, 2) rotation matrix"""
    utils3d.torch.transforms.rotation_matrix_2d

@overload
def rotate_2d(theta: Union[float, torch_.Tensor], center: torch_.Tensor = None):
    """3x3 matrix for 2D rotation around a center
```
   [[Rxx, Rxy, tx],
    [Ryx, Ryy, ty],
    [0,     0,  1]]
```
## Parameters
    theta (float | Tensor): rotation angle in radians, arbitrary shape (...,)
    center (Tensor): rotation center, arbitrary shape (..., 2). Default to (0, 0)
    
## Returns
    (Tensor): (..., 3, 3) transformation matrix"""
    utils3d.torch.transforms.rotate_2d

@overload
def translate_2d(translation: torch_.Tensor):
    """Translation matrix for 2D translation
```
   [[1, 0, tx],
    [0, 1, ty],
    [0, 0,  1]]
```
## Parameters
    translation (Tensor): translation vector, arbitrary shape (..., 2)

## Returns
    (Tensor): (..., 3, 3) transformation matrix"""
    utils3d.torch.transforms.translate_2d

@overload
def scale_2d(scale: Union[float, torch_.Tensor], center: torch_.Tensor = None):
    """Scale matrix for 2D scaling
```
   [[s, 0, tx],
    [0, s, ty],
    [0, 0,  1]]
```
## Parameters
    scale (float | Tensor): scale factor, arbitrary shape (...,)
    center (Tensor): scale center, arbitrary shape (..., 2). Default to (0, 0)

## Returns
    (Tensor): (..., 3, 3) transformation matrix"""
    utils3d.torch.transforms.scale_2d

@overload
def transform_points(x: torch_.Tensor, *Ts: torch_.Tensor) -> torch_.Tensor:
    """Apply transformation(s) to a point or a set of points.
It is like `(Tn @ ... @ T2 @ T1 @ x[:, None]).squeeze(0)`, but: 
1. Automatically handle the homogeneous coordinate;
        - x will be padded with homogeneous coordinate 1.
        - Each T will be padded by identity matrix to match the dimension. 
2. Using efficient contraction path when array sizes are large, based on `einsum`.

## Parameters
- `x`: Tensor, shape `(..., D)`: the points to be transformed.
- `Ts`: Tensor, shape `(..., D + 1, D + 1)`: the affine transformation matrix (matrices)
    If more than one transformation is given, they will be applied in corresponding order.
## Returns
- `y`: Tensor, shape `(..., D)`: the transformed point or a set of points.

## Example Usage

- Just linear transformation

    ```
    y = transform(x_3, mat_3x3) 
    ```

- Affine transformation

    ```
    y = transform(x_3, mat_3x4)
    ```

- Chain multiple transformations

    ```
    y = transform(x_3, T1_4x4, T2_3x4, T3_3x4)
    ```"""
    utils3d.torch.transforms.transform_points

@overload
def angle_between(v1: torch_.Tensor, v2: torch_.Tensor, eps: float = 1e-08) -> torch_.Tensor:
    """Calculate the angle between two (batches of) vectors.
Better precision than using the arccos dot product directly.

## Parameters
- `v1`: Tensor, shape (..., D): the first vector.
- `v2`: Tensor, shape (..., D): the second vector.
- `eps`: float, optional: prevents zero angle difference (indifferentiable).

## Returns
`angle`: Tensor, shape (...): the angle between the two vectors."""
    utils3d.torch.transforms.angle_between

@overload
def triangulate_mesh(faces: torch_.Tensor, vertices: torch_.Tensor = None, method: Literal['fan', 'strip', 'diagonal'] = 'fan') -> torch_.Tensor:
    """Triangulate a polygonal mesh.

## Parameters
- `faces` (Tensor): [L, P] polygonal faces
- `vertices` (Tensor, optional): [N, 3] 3-dimensional vertices.
    If given, the triangulation is performed according to the distance
    between vertices. Defaults to None.
- `method`

## Returns
    (Tensor): [L * (P - 2), 3] triangular faces"""
    utils3d.torch.mesh.triangulate_mesh

@overload
def compute_face_corner_angles(vertices: torch_.Tensor, faces: Optional[torch_.Tensor] = None) -> torch_.Tensor:
    """Compute face corner angles of a mesh

## Parameters
- `vertices` (Tensor): `(..., N, 3)` vertices if `faces` is provided, or `(..., F, P, 3)` if `faces` is None
- `faces` (Tensor, optional): `(F, P)` face vertex indices, where P is the number of vertices per face

## Returns
- `angles` (Tensor): `(..., F, P)` face corner angles"""
    utils3d.torch.mesh.compute_face_corner_angles

@overload
def compute_face_corner_normals(vertices: torch_.Tensor, faces: Optional[torch_.Tensor] = None, normalize: bool = True) -> torch_.Tensor:
    """Compute the face corner normals of a mesh

## Parameters
- `vertices` (Tensor): `(..., N, 3)` vertices if `faces` is provided, or `(..., F, P, 3)` if `faces` is None
- `faces` (Tensor, optional): `(F, P)` face vertex indices, where P is the number of vertices per face
- `normalize` (bool): whether to normalize the normals to unit vectors. If not, the normals are the raw cross products.

## Returns
- `normals` (Tensor): (..., F, P, 3) face corner normals"""
    utils3d.torch.mesh.compute_face_corner_normals

@overload
def compute_face_corner_tangents(vertices: torch_.Tensor, uv: torch_.Tensor, faces_vertices: Optional[torch_.Tensor] = None, faces_uv: Optional[torch_.Tensor] = None, normalize: bool = True) -> torch_.Tensor:
    """    Compute the face corner tangent (and bitangent) vectors of a mesh

    ## Parameters
    - `vertices` (Tensor): `(..., N, 3)` if `faces` is provided, or `(..., F, P, 3)` if `faces_vertices` is None
    - `uv` (Tensor): `(..., N, 2)` if `faces` is provided, or `(..., F, P, 2)` if `faces_uv` is None
    - `faces_vertices` (Tensor, optional): `(F, P)` face vertex indices
    - `faces_uv` (Tensor, optional): `(F, P)` face UV indices
    - `normalize` (bool): whether to normalize the tangents to unit vectors. If not, the tangents (dX/du, dX/dv) matches the UV parameterized manifold.
s
    ## Returns
    - `tangents` (Tensor): `(..., F, P, 3, 2)` face corner tangents (and bitangents), 
        where the last dimension represents the tangent and bitangent vectors.
    """
    utils3d.torch.mesh.compute_face_corner_tangents

@overload
def compute_face_normals(vertices: torch_.Tensor, faces: Optional[torch_.Tensor] = None) -> torch_.Tensor:
    """Compute face normals of a mesh

## Parameters
- `vertices` (Tensor): `(..., N, 3)` vertices if `faces` is provided, or `(..., F, P, 3)` if `faces` is None
- `faces` (Tensor, optional): `(F, P)` face vertex indices, where P is the number of vertices per face

## Returns
- `normals` (Tensor): `(..., F, 3)` face normals. Always normalized."""
    utils3d.torch.mesh.compute_face_normals

@overload
def compute_face_tangents(vertices: torch_.Tensor, uv: torch_.Tensor, faces_vertices: Optional[torch_.Tensor] = None, faces_uv: Optional[torch_.Tensor] = None, normalize: bool = True) -> torch_.Tensor:
    """Compute the face corner tangent (and bitangent) vectors of a mesh

## Parameters
- `vertices` (Tensor): `(..., N, 3)` if `faces` is provided, or `(..., F, P, 3)` if `faces_vertices` is None
- `uv` (Tensor): `(..., N, 2)` if `faces` is provided, or `(..., F, P, 2)` if `faces_uv` is None
- `faces_vertices` (Tensor, optional): `(F, P)` face vertex indices
- `faces_uv` (Tensor, optional): `(F, P)` face UV indices

## Returns
- `tangents` (Tensor): `(..., F, 3, 2)` face corner tangents (and bitangents), 
    where the last dimension represents the tangent and bitangent vectors."""
    utils3d.torch.mesh.compute_face_tangents

@overload
def mesh_edges(faces: torch_.Tensor, return_face2edge: bool = False, return_edge2face: bool = False, return_counts: bool = False) -> Tuple[torch_.Tensor, torch_.Tensor, torch_.Tensor, torch_.Tensor]:
    """Get undirected edges of a mesh. Optionally return additional mappings.

## Parameters
- `faces` (Tensor): polygon faces
    - `(F, P)` dense array of indices, where each face has `P` vertices.
    - `(F, V)` binary sparse csr array of indices, each row corresponds to the vertices of a face.
- `return_face2edge` (bool): whether to return the face to edge mapping
- `return_edge2face` (bool): whether to return the edge to face mapping
- `return_counts` (bool): whether to return the counts of edges

## Returns
- `edges` (Tensor): `(E, 2)` unique edges' vertex indices

If `return_face2edge`, `return_edge2face`, `return_opposite_edge`, or `return_counts` is True, the corresponding outputs will be appended in order:

- `face2edge` (Tensor): mapping from faces to the indices of edges
    - `(F, P)` if input `faces` is a dense array
    - `(F, E)` if input `faces` is a sparse csr array
- `edge2face` (Tensor): `(E, F)` binary sparse CSR matrix of edge to face.
- `counts` (Tensor): `(E,)` counts of each edge"""
    utils3d.torch.mesh.mesh_edges

@overload
def mesh_half_edges(faces: torch_.Tensor, return_face2edge: bool = False, return_edge2face: bool = False, return_twin: bool = False, return_next: bool = False, return_prev: bool = False, return_counts: bool = False) -> Tuple[torch_.Tensor, torch_.Tensor, torch_.Tensor, torch_.Tensor, torch_.Tensor, torch_.Tensor, torch_.Tensor]:
    """Get half edges of a mesh. Optionally return additional mappings.

## Parameters
- `faces` (Tensor): polygon faces
    - `(F, P)` dense array of indices, where each face has `P` vertices.
    - `(F, V)` binary sparse csr array of indices, each row corresponds to the vertices of a face.
- `return_face2edge` (bool): whether to return the face to edge mapping
- `return_edge2face` (bool): whether to return the edge to face mapping
- `return_twin` (bool): whether to return the mapping from one edge to its opposite/twin edge
- `return_next` (bool): whether to return the mapping from one edge to its next edge in the face loop
- `return_prev` (bool): whether to return the mapping from one edge to its previous edge in the face loop
- `return_counts` (bool): whether to return the counts of edges

## Returns
- `edges` (Tensor): `(E, 2)` unique edges' vertex indices

If `return_face2edge`, `return_edge2face`, `return_opposite_edge`, or `return_counts` is True, the corresponding outputs will be appended in order:

- `face2edge` (Tensor | Tensor): mapping from faces to the indices of edges
    - `(F, P)` if input `faces` is a dense array
    - `(F, E)` if input `faces` is a sparse csr array
- `edge2face` (Tensor): `(E, F)` binary sparse CSR matrix of edge to face.
- `twin` (Tensor): `(E,)` mapping from edges to indices of opposite edges. -1 if not found. 
- `next` (Tensor): `(E,)` mapping from edges to indices of next edges in the face loop.
- `prev` (Tensor): `(E,)` mapping from edges to indices of previous edges in the face loop.
- `counts` (Tensor): `(E,)` counts of each half edge

NOTE: If the mesh is not manifold, `twin`, `next`, and `prev` can point to arbitrary one of the candidates."""
    utils3d.torch.mesh.mesh_half_edges

@overload
def mesh_dual_graph(faces: torch_.Tensor) -> Tuple[torch_.Tensor, torch_.Tensor]:
    """Get dual graph of a mesh. (Mesh face as dual graph's vertex, adjacency by edge sharing)

## Parameters
- `faces`: `Tensor` faces indices
    - `(F, P)` dense tensor 

## Returns
- `dual_graph` (Tensor): `(F, F)` binary sparse CSR matrix. Adjacency matrix of the dual graph."""
    utils3d.torch.mesh.mesh_dual_graph

@overload
def mesh_connected_components(faces: torch_.Tensor, num_vertices: Optional[int] = None) -> List[torch_.Tensor]:
    """Compute connected components of a mesh.

## Parameters
- `faces` (Tensor): polygon faces
    - `(F, P)` dense tensor of indices, where each face has `P` vertices.
    - `(F, V)` binary sparse csr tensor of indices, each row corresponds to the vertices of a face.
- `num_vertices` (int, optional): total number of vertices. If given, the returned components will include all vertices. Defaults to None.

## Returns

If `num_vertices` is given, return:
- `labels` (Tensor): (N,) component labels of each vertex

If `num_vertices` is None, return:
- `vertices_ids` (Tensor): (N,) vertex indices that are in the edges
- `labels` (Tensor): (N,) component labels corresponding to `vertices_ids`"""
    utils3d.torch.mesh.mesh_connected_components

@overload
def graph_connected_components(edges: torch_.Tensor, num_vertices: Optional[int] = None) -> Union[torch_.Tensor, Tuple[torch_.Tensor, torch_.Tensor]]:
    """Compute connected components of an undirected graph.

## Parameters
- `edges` (Tensor): (E, 2) edge indices

## Returns

If `num_vertices` is given, return:
- `labels` (Tensor): (N,) component labels of each vertex

If `num_vertices` is None, return:
- `vertices_ids` (Tensor): (N,) vertex indices that are in the edges
- `labels` (Tensor): (N,) component labels corresponding to `vertices_ids`"""
    utils3d.torch.mesh.graph_connected_components

@overload
def compute_boundaries(faces: torch_.Tensor, edges: torch_.Tensor = None, face2edge: torch_.Tensor = None, edge_degrees: torch_.Tensor = None) -> Tuple[List[torch_.Tensor], List[torch_.Tensor]]:
    """Compute boundary edges of a mesh.

## Parameters
    faces (Tensor): [T, 3] triangular face indices
    edges (Tensor): [E, 2] edge indices.
    face2edge (Tensor): [T, 3] mapping from face to edge.
    edge_degrees (Tensor): [E] degree of each edge.

## Returns
    boundary_edge_indices (List[Tensor]): list of boundary edge indices
    boundary_face_indices (List[Tensor]): list of boundary face indices"""
    utils3d.torch.mesh.compute_boundaries

@overload
def remove_unused_vertices(faces: torch_.Tensor, *vertice_attrs, return_indices: bool = False) -> Tuple[torch_.Tensor, ...]:
    """Remove unreferenced vertices of a mesh. 
Unreferenced vertices are removed, and the face indices are updated accordingly.

## Parameters
    faces (Tensor): [T, P] face indices
    *vertice_attrs: vertex attributes

## Returns
    faces (Tensor): [T, P] face indices
    *vertice_attrs: vertex attributes
    indices (Tensor, optional): [N] indices of vertices that are kept. Defaults to None."""
    utils3d.torch.mesh.remove_unused_vertices

@overload
def remove_corrupted_faces(faces: torch_.Tensor) -> torch_.Tensor:
    """Remove corrupted faces (faces with duplicated vertices)

## Parameters
    faces (Tensor): [F, 3] face indices

## Returns
    Tensor: [F_reduced, 3] face indices"""
    utils3d.torch.mesh.remove_corrupted_faces

@overload
def remove_isolated_pieces(vertices: torch_.Tensor, faces: torch_.Tensor, connected_components: List[torch_.Tensor] = None, thresh_num_faces: int = None, thresh_radius: float = None, thresh_boundary_ratio: float = None, remove_unreferenced: bool = True) -> Tuple[torch_.Tensor, torch_.Tensor]:
    """Remove isolated pieces of a mesh. 
Isolated pieces are removed, and the face indices are updated accordingly.
If no face is left, will return the largest connected component.

## Parameters
    vertices (Tensor): [N, 3] 3-dimensional vertices
    faces (Tensor): [T, 3] triangular face indices
    connected_components (List[Tensor], optional): connected components of the mesh. If None, it will be computed. Defaults to None.
    thresh_num_faces (int, optional): threshold of number of faces for isolated pieces. Defaults to None.
    thresh_radius (float, optional): threshold of radius for isolated pieces. Defaults to None.
    remove_unreferenced (bool, optional): remove unreferenced vertices after removing isolated pieces. Defaults to True.

## Returns
    vertices (Tensor): [N_, 3] 3-dimensional vertices
    faces (Tensor): [T, 3] triangular face indices"""
    utils3d.torch.mesh.remove_isolated_pieces

@overload
def merge_duplicate_vertices(vertices: torch_.Tensor, faces: torch_.Tensor, tol: float = 1e-06) -> Tuple[torch_.Tensor, torch_.Tensor]:
    """Merge duplicate vertices of a triangular mesh. 
Duplicate vertices are merged by selecte one of them, and the face indices are updated accordingly.

## Parameters
    vertices (Tensor): [N, 3] 3-dimensional vertices
    faces (Tensor): [T, 3] triangular face indices
    tol (float, optional): tolerance for merging. Defaults to 1e-6.

## Returns
    vertices (Tensor): [N_, 3] 3-dimensional vertices
    faces (Tensor): [T, 3] triangular face indices"""
    utils3d.torch.mesh.merge_duplicate_vertices

@overload
def subdivide_mesh(vertices: torch_.Tensor, faces: torch_.Tensor, n: int = 1) -> Tuple[torch_.Tensor, torch_.Tensor]:
    """Subdivide a triangular mesh by splitting each triangle into 4 smaller triangles.
NOTE: All original vertices are kept, and new vertices are appended to the end of the vertex list.

## Parameters
    vertices (Tensor): [N, 3] 3-dimensional vertices
    faces (Tensor): [T, 3] triangular face indices
    n (int, optional): number of subdivisions. Defaults to 1.

## Returns
    vertices (Tensor): [N_, 3] subdivided 3-dimensional vertices
    faces (Tensor): [4 * T, 3] subdivided triangular face indices"""
    utils3d.torch.mesh.subdivide_mesh

@overload
def compute_mesh_laplacian(vertices: torch_.Tensor, faces: torch_.Tensor, weight: str = 'uniform') -> torch_.Tensor:
    """Laplacian smooth with cotangent weights

## Parameters
    vertices (Tensor): shape (..., N, 3)
    faces (Tensor): shape (T, 3)
    weight (str): 'uniform' or 'cotangent'"""
    utils3d.torch.mesh.compute_mesh_laplacian

@overload
def laplacian_smooth_mesh(vertices: torch_.Tensor, faces: torch_.Tensor, weight: str = 'uniform', times: int = 5) -> torch_.Tensor:
    """Laplacian smooth with cotangent weights

## Parameters
    vertices (Tensor): shape (..., N, 3)
    faces (Tensor): shape (T, 3)
    weight (str): 'uniform' or 'cotangent'"""
    utils3d.torch.mesh.laplacian_smooth_mesh

@overload
def taubin_smooth_mesh(vertices: torch_.Tensor, faces: torch_.Tensor, lambda_: float = 0.5, mu_: float = -0.51) -> torch_.Tensor:
    """Taubin smooth mesh

## Parameters
    vertices (Tensor): _description_
    faces (Tensor): _description_
    lambda_ (float, optional): _description_. Defaults to 0.5.
    mu_ (float, optional): _description_. Defaults to -0.51.

## Returns
    Tensor: _description_"""
    utils3d.torch.mesh.taubin_smooth_mesh

@overload
def laplacian_hc_smooth_mesh(vertices: torch_.Tensor, faces: torch_.Tensor, times: int = 5, alpha: float = 0.5, beta: float = 0.5, weight: str = 'uniform'):
    """HC algorithm from Improved Laplacian Smoothing of Noisy Surface Meshes by J.Vollmer et al.
    """
    utils3d.torch.mesh.laplacian_hc_smooth_mesh

@overload
def create_cube_mesh(tri: bool = False, device: Optional[torch_.device] = None) -> Tuple[torch_.Tensor, torch_.Tensor]:
    """Create a cube mesh of size 1 centered at origin.

### Parameters
    tri (bool, optional): return triangulated mesh. Defaults to False, which returns quad mesh.

### Returns
    vertices (Tensor): shape (8, 3) float32
    faces (Tensor): shape (12, 3) int32"""
    utils3d.torch.mesh.create_cube_mesh

@overload
def create_camera_frustum_mesh(extrinsics: torch_.Tensor, intrinsics: torch_.Tensor, depth: float = 1.0) -> Tuple[torch_.Tensor, torch_.Tensor, torch_.Tensor]:
    """Create a triangle mesh of camera frustum."""
    utils3d.torch.mesh.create_camera_frustum_mesh

@overload
def create_icosahedron_mesh(device: Optional[torch_.device] = None) -> Tuple[torch_.Tensor, torch_.Tensor]:
    """Create an icosahedron mesh of centered at origin."""
    utils3d.torch.mesh.create_icosahedron_mesh

@overload
def uv_map(*size: Union[int, Tuple[int, int]], top: float = 0.0, left: float = 0.0, bottom: float = 1.0, right: float = 1.0, dtype: torch_.dtype = torch_.float32, device: torch_.device = None) -> torch_.Tensor:
    """Get image UV coordinate map. By default, (0., 0.) is the top-left corner of the image, and (1., 1.) is the bottom-right corner of the image.

## Parameters
- `*size`: `Tuple[int, int]` or two integers of map size `(height, width)`
- `top`: `float` defaults to 0.
- `left`: `float` defaults to 0.
- `bottom`: `float` defaults to 1.
- `right`: `float` defaults to 1.
- `dtype`: `torch.dtype` data type of the output uv map. Defaults to torch.float32.
- `device`: `torch.device`, device of the output uv map. Defaults to None.

## Returns
- `uv (Tensor)`: shape `(height, width, 2)`

## Example Usage

>>> uv_map(10, 10):
[[[0.05, 0.05], [0.15, 0.05], ..., [0.95, 0.05]],
 [[0.05, 0.15], [0.15, 0.15], ..., [0.95, 0.15]],
  ...             ...                  ...
 [[0.05, 0.95], [0.15, 0.95], ..., [0.95, 0.95]]]"""
    utils3d.torch.maps.uv_map

@overload
def pixel_coord_map(*size: Union[int, Tuple[int, int]], top: int = 0, left: int = 0, convention: Literal['integer-center', 'integer-corner'] = 'integer-center', dtype: torch_.dtype = torch_.float32, device: torch_.device = None) -> torch_.Tensor:
    """Get image pixel coordinates map. Support two conventions: `'integer-center'` and `'integer-corner'`.

## Parameters
- `*size`: `Tuple[int, int]` or two integers of map size `(height, width)`
- `top`: `int`, optional top boundary of the pixel coord map. Defaults to 0.
- `left`: `int`, optional left boundary of the pixel coord map. Defaults to 0.
- `convention`: `str`, optional `'integer-center'` or `'integer-corner'`, whether integer coordinates correspond to pixel centers or corners. Defaults to 'integer-center'.
    - `'integer-center'`: `pixel[i][j]` has integer coordinates `(j, i)` as its center, and occupies square area `[j - 0.5, j + 0.5) × [i - 0.5, i + 0.5)`. 
        The top-left corner of the top-left pixel is `(-0.5, -0.5)`, and the bottom-right corner of the bottom-right pixel is `(width - 0.5, height - 0.5)`.
    - `'integer-corner'`: `pixel[i][j]` has coordinates `(j + 0.5, i + 0.5)` as its center, and occupies square area `[j, j + 1) × [i, i + 1)`.
        The top-left corner of the top-left pixel is `(0, 0)`, and the bottom-right corner of the bottom-right pixel is `(width, height)`.
- `dtype`: `torch.dtype`, optional data type of the output pixel coord map. Defaults to torch.float32.

## Returns
    Tensor: shape (height, width, 2)

>>> pixel_coord_map(10, 10, convention='integer-center', dtype=torch.long):
[[[0, 0], [1, 0], ..., [9, 0]],
 [[0, 1], [1, 1], ..., [9, 1]],
    ...      ...         ...
 [[0, 9], [1, 9], ..., [9, 9]]]

>>> pixel_coord_map(10, 10, convention='integer-corner', dtype=torch.float32):
[[[0.5, 0.5], [1.5, 0.5], ..., [9.5, 0.5]],
 [[0.5, 1.5], [1.5, 1.5], ..., [9.5, 1.5]],
  ...             ...                  ...
[[0.5, 9.5], [1.5, 9.5], ..., [9.5, 9.5]]]"""
    utils3d.torch.maps.pixel_coord_map

@overload
def screen_coord_map(*size: Union[int, Tuple[int, int]], top: float = 1.0, left: float = 0.0, bottom: float = 0.0, right: float = 1.0, dtype: torch_.dtype = torch_.float32, device: torch_.device = None) -> torch_.Tensor:
    """Get screen space coordinate map, where (0., 0.) is the bottom-left corner of the image, and (1., 1.) is the top-right corner of the image.
This is commonly used in graphics APIs like OpenGL.

## Parameters
- `*size`: `Tuple[int, int]` or two integers of map size `(height, width)`
- `top`: `float`, optional top boundary in the screen space. Defaults to 1.
- `left`: `float`, optional left boundary in the screen space. Defaults to 0.
- `bottom`: `float`, optional bottom boundary in the screen space. Defaults to 0.
- `right`: `float`, optional right boundary in the screen space. Defaults to 1.
- `dtype`: `torch.dtype`, optional data type of the output map. Defaults to torch.float32.

## Returns
    (Tensor): shape (height, width, 2)"""
    utils3d.torch.maps.screen_coord_map

@overload
def build_mesh_from_map(*maps: torch_.Tensor, mask: Optional[torch_.Tensor] = None, tri: bool = False) -> Tuple[torch_.Tensor, ...]:
    """Get a mesh regarding image pixel uv coordinates as vertices and image grid as faces.

## Parameters
    *maps (Tensor): attribute maps in shape (height, width, [channels])
    mask (Tensor, optional): binary mask of shape (height, width), dtype=bool. Defaults to None.

## Returns
    faces (Tensor): faces connecting neighboring pixels. shape (T, 4) if tri is False, else (T, 3)
    *attributes (Tensor): vertex attributes in corresponding order with itorchut maps
    indices (Tensor, optional): indices of vertices in the original mesh"""
    utils3d.torch.maps.build_mesh_from_map

@overload
def build_mesh_from_depth_map(depth: torch_.Tensor, *other_maps: torch_.Tensor, intrinsics: torch_.Tensor, extrinsics: Optional[torch_.Tensor] = None, atol: Optional[float] = None, rtol: Optional[float] = 0.05, tri: bool = False) -> Tuple[torch_.Tensor, ...]:
    """Get a mesh by lifting depth map to 3D, while removing depths of large depth difference.

## Parameters
    depth (Tensor): [H, W] depth map
    extrinsics (Tensor, optional): [4, 4] extrinsics matrix. Defaults to None.
    intrinsics (Tensor, optional): [3, 3] intrinsics matrix. Defaults to None.
    *other_maps (Tensor): [H, W, C] vertex attributes. Defaults to None.
    atol (float, optional): absolute tolerance. Defaults to None.
    rtol (float, optional): relative tolerance. Defaults to None.
        triangles with vertices having depth difference larger than atol + rtol * depth will be marked.
    remove_by_depth (bool, optional): whether to remove triangles with large depth difference. Defaults to True.
    return_uv (bool, optional): whether to return uv coordinates. Defaults to False.
    return_indices (bool, optional): whether to return indices of vertices in the original mesh. Defaults to False.

## Returns
    faces (Tensor): [T, 3] faces
    vertices (Tensor): [N, 3] vertices
    *other_attrs (Tensor): [N, C] vertex attributes"""
    utils3d.torch.maps.build_mesh_from_depth_map

@overload
def depth_map_edge(depth: torch_.Tensor, atol: float = None, rtol: float = None, kernel_size: int = 3, mask: torch_.Tensor = None) -> torch_.BoolTensor:
    """Compute the edge mask of a depth map. The edge is defined as the pixels whose neighbors have a large difference in depth.

## Parameters
    depth (Tensor): shape (..., height, width), linear depth map
    atol (float): absolute tolerance
    rtol (float): relative tolerance

## Returns
    edge (Tensor): shape (..., height, width) of dtype torch.bool"""
    utils3d.torch.maps.depth_map_edge

@overload
def depth_map_aliasing(depth: torch_.Tensor, atol: float = None, rtol: float = None, kernel_size: int = 3, mask: torch_.Tensor = None) -> torch_.BoolTensor:
    """Compute the map that indicates the aliasing of a depth map. The aliasing is defined as the pixels which neither close to the maximum nor the minimum of its neighbors.
## Parameters
    depth (Tensor): shape (..., height, width), linear depth map
    atol (float): absolute tolerance
    rtol (float): relative tolerance

## Returns
    edge (Tensor): shape (..., height, width) of dtype torch.bool"""
    utils3d.torch.maps.depth_map_aliasing

@overload
def point_map_to_normal_map(point: torch_.Tensor, mask: torch_.Tensor = None) -> torch_.Tensor:
    """Calculate normal map from point map. Value range is [-1, 1].

## Parameters
    point (Tensor): shape (..., height, width, 3), point map
    mask (Tensor): shape (..., height, width), binary mask. Defaults to None.

## Returns
    normal (Tensor): shape (..., height, width, 3), normal map. """
    utils3d.torch.maps.point_map_to_normal_map

@overload
def depth_map_to_point_map(depth: torch_.Tensor, intrinsics: torch_.Tensor, extrinsics: torch_.Tensor = None):
    utils3d.torch.maps.depth_map_to_point_map

@overload
def depth_map_to_normal_map(depth: torch_.Tensor, intrinsics: torch_.Tensor, mask: torch_.Tensor = None) -> torch_.Tensor:
    """Calculate normal map from depth map. Value range is [-1, 1]. Normal direction in OpenCV identity camera's coordinate system.

## Parameters
    depth (Tensor): shape (..., height, width), linear depth map
    intrinsics (Tensor): shape (..., 3, 3), intrinsics matrix
## Returns
    normal (Tensor): shape (..., height, width, 3), normal map. """
    utils3d.torch.maps.depth_map_to_normal_map

@overload
def chessboard(*size: Union[int, Tuple[int, int]], grid_size: int, color_a: torch_.Tensor, color_b: torch_.Tensor) -> torch_.Tensor:
    """Get a chessboard image

## Parameters
- `*size`: `Tuple[int, int]` or two integers of map size `(height, width)`
- `grid_size`: `int`, size of chessboard grid
- `color_a`: `Tensor`, shape (channels,), color of the grid at the top-left corner
- `color_b`: `Tensor`, shape (channels,), color in complementary grids

## Returns
- `image` (Tensor): shape (height, width, channels), chessboard image"""
    utils3d.torch.maps.chessboard

@overload
def bounding_rect_from_mask(mask: torch_.BoolTensor):
    """Get bounding rectangle of a mask

## Parameters
    mask (Tensor): shape (..., height, width), mask

## Returns
    rect (Tensor): shape (..., 4), bounding rectangle (left, top, right, bottom)"""
    utils3d.torch.maps.bounding_rect_from_mask

@overload
def masked_nearest_resize(*image: torch_.Tensor, mask: torch_.Tensor, size: Tuple[int, int], return_index: bool = False) -> Tuple[Unpack[Tuple[torch_.Tensor, ...]], torch_.Tensor, Tuple[torch_.Tensor, ...]]:
    """Resize image(s) by nearest sampling with mask awareness. Suitable for sparse maps. ![masked_nearest_resize.png](doc/masked_nearest_resize.png)
- Downsampling: Assign the nearest valid pixel within the target pixel's receptive field.
- Upsampling: Assign the valid pixel to only the nearest pixel in the resized map.


### Parameters
- `*image`: Itorchut image(s) of shape `(..., H, W, C)` or `(... , H, W)` 
    - You can pass multiple images to be resized at the same time for efficiency.
- `mask`: itorchut mask of shape `(..., H, W)`, dtype=bool
- `size`: target size `(H', W')`
- `return_index`: whether to return the nearest neighbor indices in the original map for each pixel in the resized map.
    Defaults to False.

### Returns
- `*resized_image`: resized image(s) of shape `(..., H', W', C)`. or `(..., H', W')`
- `resized_mask`: mask of the resized map of shape `(..., H', W')`
- `nearest_indices`: tuple of shape `(..., H', W')`. The nearest neighbor indices of the resized map of each dimension."""
    utils3d.torch.maps.masked_nearest_resize

@overload
def masked_area_resize(*image: torch_.Tensor, mask: torch_.Tensor, size: Tuple[int, int]) -> Tuple[Unpack[Tuple[torch_.Tensor, ...]], torch_.Tensor]:
    """Resize 2D map by area sampling with mask awareness.

### Parameters
- `*image`: Itorchut image(s) of shape `(..., H, W, C)` or `(..., H, W)`
    - You can pass multiple images to be resized at the same time for efficiency.
- `mask`: Itorchut mask of shape `(..., H, W)`
- `size`: target image size `(H', W')`

### Returns
- `*resized_image`: resized image(s) of shape `(..., H', W', C)`. or `(..., H', W')`
- `resized_mask`: mask of the resized map of shape `(..., H', W')`"""
    utils3d.torch.maps.masked_area_resize

@overload
def flood_fill(*image: torch_.Tensor, mask: torch_.Tensor, return_index: bool = False) -> torch_.Tensor:
    """Flooding fill the holes in the image(s) according to the mask. ![img](doc/flood_fill.png)

Parameters
----
- `*image` (Tensor): shape (..., height, width, [C]), itorchut image(s)
- `mask` (Tensor): shape (..., height, width), binary mask indicating valid regions

Returns
----
- `*filled_image` (Tensor): shape (..., height, width, [C]), flood filled map
- `filled_indices` (Tuple[Tensor, ...], optional): tuple of shape (..., height, width). The nearest neighbor indices of each pixel in the original map.
    It satisfies `filled_image = image[filled_indices]`"""
    utils3d.torch.maps.flood_fill

@overload
def perlin_noise(x: torch_.Tensor, seed: Optional[int] = None) -> torch_.Tensor:
    """Generate Perlin noise for the given coordinates.

Parameters
----
- `x` (Tensor): shape (*batch_shape, N_1, ..., N_D, D), coordinates to sample Perlin.
    If itorchut ndim is more than D + 1, the leading dimensions are treated as batch dimensions. Instances in the batch have different noise patterns.
- `seed` (int, optional): random seed. The same seed will generate the same noise pattern(s). Defaults to None.

Returns
----
- `y` (Tensor): shape (*batch_shape, N_1, ..., N_D), Perlin noise value at the given coordinates and seed. Value range is approximately [-1, 1]"""
    utils3d.torch.maps.perlin_noise

@overload
def perlin_noise_map(size: Tuple[int, ...], frequency: Union[float, torch_.Tensor], seed: Optional[int] = None, dtype: Optional[torch_.dtype] = None, device: Optional[torch_.device] = None) -> torch_.Tensor:
    """Generate Perlin noise map.

Parameters
----
- `size` (Tuple[int, ...]): size of the noise map (..., H, W)
- `frequency` (float | Tensor): frequency relative to map's larger dimension max(H, W) of the Perlin noise. 
    Represents how many periods of noise fit in the larger dimension of the map.
- `seed` (int, optional): random seed. The same seed will generate the same noise pattern(s). Defaults to None.

Returns
----
- `noise_map` (Tensor): shape (..., H, W), Perlin noise map. Value range is approximately [-1, 1]"""
    utils3d.torch.maps.perlin_noise_map

@overload
def fractal_perlin_noise_map(size: Tuple[int, ...], base_frequency: Union[float, torch_.Tensor], octaves: int = 4, lacunarity: float = 2.0, gain: float = 0.5, seed: Optional[int] = None, dtype: Optional[torch_.dtype] = None, device: Optional[torch_.device] = None) -> torch_.Tensor:
    """Generate fractal Perlin noise map. ![fractal_perlin_base_frequeny2_octaves7_gain0.7.png](doc/fractal_perlin_base_frequeny2_octaves7_gain0.7.png)

Parameters
----
- `size` (Tuple[int, ...]): size of the noise map (..., H, W)
- `base_frequency` (float | Tensor): base frequency (relative to map's larger dimension max(H, W)) of the Perlin noise.
    Represents how many periods of noise fit in the larger dimension of the map.
- `octaves` (int, optional): number of octaves. Defaults to 4.
- `lacunarity` (float, optional): frequency multiplier between octaves. Defaults to 2.0.
- `gain` (float, optional): amplitude multiplier between octaves. Defaults to 0.5.
- `seed` (int, optional): random seed. The same seed will generate the same noise pattern. Defaults to None.

Returns
----
- `noise_map` (Tensor): shape (..., H, W), fractal Perlin noise map. Value range is approximately [-1, 1]"""
    utils3d.torch.maps.fractal_perlin_noise_map

@overload
def RastContext(nvd_ctx: Union[nvdiffrast.torch.ops.RasterizeCudaContext, nvdiffrast.torch.ops.RasterizeGLContext] = None, *, backend: Literal['cuda', 'gl'] = 'cuda', device: Union[str, torch_.device] = None):
    """Create a rasterization context. Nothing but a wrapper of nvdiffrast.torch.RasterizeCudaContext or nvdiffrast.torch.RasterizeGLContext."""
    utils3d.torch.rasterization.RastContext

@overload
def rasterize_triangles(size: Tuple[int, int], *, vertices: torch_.Tensor, attributes: Optional[torch_.Tensor] = None, faces: torch_.Tensor, view: torch_.Tensor = None, projection: torch_.Tensor = None, extrinsics: torch_.Tensor = None, intrinsics: torch_.Tensor = None, near: float = 0.01, far: float = inf, return_image_derivatives: bool = False, return_depth: bool = False, return_interpolation: bool = False, antialiasing: bool = False, ctx: Optional[utils3d.torch.rasterization.RastContext] = None) -> Tuple[torch_.Tensor, torch_.Tensor, Optional[torch_.Tensor]]:
    """Rasterize triangles.

Parameters
----
    size (Tuple[int, int]): (height, width) of the output image
    vertices (np.ndarray): (B, N, 2 or 3 or 4)
    faces (Tensor): (T, 3)
    attributes (Tensor, optional): (B, N, C) vertex attributes. Defaults to None.
    texture (Tensor, optional): (B, C, H, W) texture. Defaults to None.
    view | extrinsics (Tensor, optional): ([B,] 4, 4) view matrix or extrinsics matrix. Provide either one of them. Defaults to identity.
    projection | intrinsics (Tensor, optional): ([B,] 4, 4) projection matrix or ([B,] 3, 3) intrinsics matrix. Provide either one of them. Defaults to identity.
    near (float, optional): near plane. Defaults to 0.01. Only used for intrinsics. Ignored if projection matrix is provided.
    far (float, optional): far plane. Defaults to inf. Only used for intrinsics. Ignored if projection matrix is provided.
    return_image_derivatives (bool, optional): whether to return screen space derivatives of the attributes. Defaults to False.
    return_depth (bool, optional): whether to return depth map. Defaults to False.
    return_interpolation (bool, optional): whether to return triangle interpolation maps. Defaults to False.
    antialiasing (Union[bool, List[int]], optional): whether to perform antialiasing. Defaults to True. If a list of indices is provided, only those channels will be antialiased.
    ctx (RastContext): rasterization context. Defaults to the thread-local default context. If custom context is needed, provide one with utils3d.pt.RastContext().

Returns
----
A dictionary containing:
    - image: (Tensor): (B, H, W, C)
    - image_dr: (Tensor): (B, H, W, C * 2) screen space derivatives of the attributes
    - depth: (Tensor): (B, H, W) Linear depth. Empty pixels have depth inf.
    - mask: (torch.BoolTensor): (B, H, W) mask of valid pixels
    - interpolation_id: (Tensor): (B, H, W) triangle ID map. For empty pixels, the value is -1.
    - interpolation_uv: (Tensor): (B, H, W, 2) triangle UV (first two channels of barycentric coordinates)"""
    utils3d.torch.rasterization.rasterize_triangles

@overload
def rasterize_triangles_peeling(size: Tuple[int, int], *, vertices: torch_.Tensor, attributes: Optional[torch_.Tensor] = None, faces: torch_.Tensor, view: torch_.Tensor = None, projection: torch_.Tensor = None, extrinsics: torch_.Tensor = None, intrinsics: torch_.Tensor = None, near: float = 0.01, far: float = inf, return_image_derivatives: bool = False, return_depth: bool = False, return_interpolation: bool = False, antialiasing: bool = False, ctx: Optional[utils3d.torch.rasterization.RastContext] = None) -> Iterator[Iterator[Dict[str, torch_.Tensor]]]:
    """Rasterize a mesh with vertex attributes using depth peeling.

Parameters
----
    size (Tuple[int, int]): (height, width) of the output image
    vertices (np.ndarray): (B, N, 2 or 3 or 4)
    faces (Tensor): (T, 3)
    attributes (Tensor, optional): (B, N, C) vertex attributes. Defaults to None.
    texture (Tensor, optional): (B, C, H, W) texture. Defaults to None.
    view | extrinsics (Tensor, optional): ([B,] 4, 4) view matrix or extrinsics matrix. Provide either one of them. Defaults to identity.
    projection | intrinsics (Tensor, optional): ([B,] 4, 4) projection matrix or ([B,] 3, 3) intrinsics matrix. Provide either one of them. Defaults to identity.
    near (float, optional): near plane. Defaults to 0.01. Only used for intrinsics. Ignored if projection matrix is provided.
    far (float, optional): far plane. Defaults to inf. Only used for intrinsics. Ignored if projection matrix is provided.
    return_image_derivatives (bool, optional): whether to return screen space derivatives of the attributes. Defaults to False.
    return_depth (bool, optional): whether to return depth map. Defaults to False.
    return_interpolation (bool, optional): whether to return triangle interpolation maps. Defaults to False.
    antialiasing (Union[bool, List[int]], optional): whether to perform antialiasing. Defaults to True. If a list of indices is provided, only those channels will be antialiased.
    ctx (RastContext): rasterization context. Defaults to the thread-local default context. If custom context is needed, provide one with utils3d.pt.RastContext().

Returns
----
A generator of dictionaries for each layer containing:
    - mask: (List[torch.BoolTensor]): (B, H, W) mask of valid pixels in this layer
    - image: (List[Tensor]): (B, C, H, W) rendered images
    - image_dr: (List[Tensor]): (B, *, H, W) screen space derivatives of the attributes
    - depth: (List[Tensor]): (B, H, W) linear depth. Empty pixels have depth inf.
    - interpolation_id: (List[Tensor]): (B, H, W) triangle ID map. For empty pixels, the value is -1.
    - interpolation_uv: (List[Tensor]): (B, H, W, 2) triangle UV (first two channels of barycentric coordinates)

The last layer yielded will be empty, then the generator will stop.

Example
----
```
for i, layer_output in enumerate(rasterize_triangles_peeling(
    (512, 512), 
    vertices=vertices, 
    faces=faces, 
    attributes=attributes,
    view=view,
    projection=projection
)):
    print(f"Layer {i}:")
    for key, value in layer_output.items():
        print(f"  {key}: {value.shape}")
    if i >= 4:  # Stop after 5 layers at most
        break
```"""
    utils3d.torch.rasterization.rasterize_triangles_peeling

@overload
def sample_texture(texture: torch_.Tensor, uv: torch_.Tensor, uv_dr: torch_.Tensor) -> torch_.Tensor:
    """Interpolate texture using uv coordinates.

## Parameters
    texture (Tensor): (B, H, W, C) texture
    uv (Tensor): (B, H, W, 2) uv coordinates
    uv_dr (Tensor): (B, H, W, 4) uv derivatives
    
## Returns
    Tensor: (B, H, W, C) interpolated texture"""
    utils3d.torch.rasterization.sample_texture

@overload
def texture_composite(texture: torch_.Tensor, uv: List[torch_.Tensor], uv_da: List[torch_.Tensor], background: torch_.Tensor = None) -> Tuple[torch_.Tensor, torch_.Tensor]:
    """Composite textures with depth peeling output.

## Parameters
    texture (Tensor): (B, C+1, H, W) texture
        NOTE: the last channel is alpha channel
    uv (List[Tensor]): list of (B, H, W, 2) uv coordinates
    uv_da (List[Tensor]): list of (B, H, W, 4) uv derivatives
    background (Optional[Tensor], optional): (B, C, H, W) background image. Defaults to None (black).
    
## Returns
    image: (Tensor): (B, C, H, W) rendered image
    alpha: (Tensor): (B, H, W) alpha channel"""
    utils3d.torch.rasterization.texture_composite

