"""File containing utils for sharding."""



import jax
import numpy as np
from jax.experimental import mesh_utils
from jax.tree_util import DictKey, GetAttrKey, SequenceKey



def unroll_path(path):
    path_list = list(path)
    path_params = []
    for path_element in path_list:
        if isinstance(path_element, DictKey):
            path_params.append(str(path_element.key))
        elif isinstance(path_element, GetAttrKey):
            path_params.append(path_element.name)
        elif isinstance(path_element, SequenceKey):
            path_params.append(str(path_element.idx))
        else:
            raise ValueError(f"Unknown path element: {path_element}")
    return ".".join(path_params)


def apply_sharding(state_shape, mesh):
    def get_sharding(path, x):
        repl_sharding = (None,) * x.ndim

        return jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec(*repl_sharding)
        )

    shardings = jax.tree_util.tree_map_with_path(get_sharding, state_shape)
    return shardings


def create_device_mesh(
    config_mesh,
    *,
    allow_split_physical_axes=False,
):
    """Returns a JAX device mesh.

    Args:
        config_mesh: A list of tuples of (axis_name, axis_size). It is advised to
        sort the axis in increasing order of network communication intensity.
        allow_split_physical_axes: Whether to allow splitting physical axes.
    """
    devices = jax.devices()
    mesh_axes, mesh_size = tuple(zip(*config_mesh))
    # Because jax.utils do not support `-1` shape size.
    mesh_size = np.array(devices).reshape(mesh_size).shape
    device_mesh = mesh_utils.create_device_mesh(
        mesh_size, devices=devices, allow_split_physical_axes=allow_split_physical_axes
    )
    return jax.sharding.Mesh(device_mesh, mesh_axes)


def build_shardings(
    mesh,
    data_axis,
    allow_split_physical_axes=False,
):
    device_mesh = create_device_mesh(
        mesh, allow_split_physical_axes=allow_split_physical_axes
    )
    data_sharding = jax.sharding.NamedSharding(
        device_mesh, jax.sharding.PartitionSpec(data_axis)
    )
    repl_sharding = jax.sharding.NamedSharding(
        device_mesh, jax.sharding.PartitionSpec()
    )

    return device_mesh, data_sharding, repl_sharding


def make_fsarray_from_local_slice(
    local_slice,
    global_devices,
):
    """Create a fully-sharded global device array from local host arrays.

    Args:
        local_slice: Something convertible to a numpy array (eg also TF tensors)
        that is this host's slice of the global array.
        global_devices: The list of global devices. Needed for consistent ordering.

    Returns:
        The global on-device array which consists of all local slices stacked
        together in the order consistent with the devices.
    """
    mesh = jax.sharding.Mesh(global_devices, ("data",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    local_ds = mesh.local_devices

    x = np.asarray(local_slice)
    xs = jax.device_put(np.split(x, len(local_ds), axis=0), local_ds)

    global_shape = (x.shape[0] * jax.process_count(), *x.shape[1:])
    return jax.make_array_from_single_device_arrays(global_shape, sharding, xs)


def get_local_slice_from_fsarray(global_array):
    """Return numpy array for the host-local slice of fully-sharded array.

    Args:
        global_array: JAX array, globally sharded on devices across hosts (potentially undressable).

    Returns:
        NumPy array that holds the part of `global_array` that is held by the
        devices on the host that calls this function.
    """
    # For now, for simplicity, we only implement slicing along the first axis.
    for shard in global_array.addressable_shards:
        assert all(
            idx == slice(None) for idx in shard.index[1:]
        ), f"global_array is sharded along non-first dimensions:\n{shard.index}"

    # Get the shards back in the same order in which the global array was created
    # in the first place. This makes sure it's consistent with other things in the
    # batch, for example (assuming the whole batch is consistent).
    m = {s.device: s for s in global_array.addressable_shards}
    local_shards = [m[d] for d in global_array.sharding.mesh.local_devices]
    return np.concatenate([jax.device_get(s.data) for s in local_shards], axis=0)
