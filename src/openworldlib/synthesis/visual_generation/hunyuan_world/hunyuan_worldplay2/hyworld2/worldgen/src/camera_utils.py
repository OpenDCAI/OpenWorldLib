# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Visualization utilities using trimesh
# --------------------------------------------------------
from typing import Union

import PIL.Image
import numpy as np
import open3d as o3d
import scipy
import torch
import trimesh
from pytorch3d.renderer.cameras import look_at_rotation
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation as R

OPENGL = np.array([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])

CAM_COLORS = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 255), (255, 204, 0), (0, 204, 204),
              (128, 255, 255), (255, 128, 255), (255, 255, 128), (0, 0, 0), (128, 128, 128)]

TRAJECTORY_COLORS = {
        "regular_0": {"start": (255, 70, 70), "end": (255, 140, 100)},  # Red -> light red
        "regular_1": {"start": (50, 200, 80), "end": (120, 255, 150)},  # Green -> light green
        "regular_2": {"start": (50, 100, 255), "end": (100, 180, 255)},  # Blue -> light blue

        "surround": {"start": (255, 180, 30), "end": (255, 220, 80)},  # Golden yellow
        "reconstruct": {"start": (30, 200, 220), "end": (100, 255, 220)},  # Cyan
        "exploration": {"start": (200, 80, 255), "end": (255, 150, 255)},  # Purple -> pink
        "aerial": {"start": (30, 144, 255), "end": (80, 220, 255)},  # Blue
    }


def geotrf(Trf, pts, ncol=None, norm=False):
    """ Apply a geometric transformation to a list of 3-D points.

    H: 3x3 or 4x4 projection matrix (typically a Homography)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)

    ncol: int. number of columns of the result (2 or 3)
    norm: float. if != 0, the resut is projected on the z=norm plane.

    Returns an array of projected 2d points.
    """
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    # adapt shape if necessary
    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    # optimized code
    if (isinstance(Trf, torch.Tensor) and isinstance(pts, torch.Tensor) and
            Trf.ndim == 3 and pts.ndim == 4):
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d + 1:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts) + Trf[:, None, None, :d, d]
        else:
            raise ValueError(f'bad shape, not ending with 3 or 4, for {pts.shape=}')
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim - 2
            assert Trf.shape[:n] == pts.shape[:n], 'batch size does not match'
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:
                # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:
                # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
                pts = pts[:, None, :]

        if pts.shape[-1] + 1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= 2:
                pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res


def add_scene_cam(scene, pose_c2w, edge_color, image=None, focal=None, imsize=None,
                  screen_width=0.03, marker=None, edge_width=1.0):
    """
    edge_width: Border thickness multiplier. Defaults to 1.0; larger values make thicker borders (recommended 1.0-5.0).
    """
    if image is not None:
        image = np.asarray(image)
        H, W, THREE = image.shape
        assert THREE == 3
        if image.dtype != np.uint8:
            image = np.uint8(255 * image)
    elif imsize is not None:
        W, H = imsize
    elif focal is not None:
        H = W = focal / 1.1
    else:
        H = W = 1

    if isinstance(focal, np.ndarray):
        focal = focal[0]
    if not focal:
        focal = min(H, W) * 1.1

    height = max(screen_width / 10, focal * screen_width / H)
    width = screen_width * 0.5 ** 0.5
    rot45 = np.eye(4)
    rot45[:3, :3] = Rotation.from_euler('z', np.deg2rad(45)).as_matrix()
    rot45[2, 3] = -height
    aspect_ratio = np.eye(4)
    aspect_ratio[0, 0] = W / H
    transform = pose_c2w @ OPENGL @ aspect_ratio @ rot45
    cam = trimesh.creation.cone(width, height, sections=4)

    # this is the image
    if image is not None:
        vertices = geotrf(transform, cam.vertices[[4, 5, 1, 3]])
        faces = np.array([[0, 1, 2], [0, 2, 3], [2, 1, 0], [3, 2, 0]])
        img = trimesh.Trimesh(vertices=vertices, faces=faces)
        uv_coords = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        img.visual = trimesh.visual.TextureVisuals(uv_coords, image=PIL.Image.fromarray(image))
        scene.add_geometry(img)

    base_offset = 0.05        # Original offset = 1 - 0.95
    base_angle  = 2.0         # Original rotation angle in degrees

    scale_factor = 1.0 - base_offset * edge_width   # edge_width=1 → 0.95
    rot_angle    = base_angle * edge_width           # edge_width=1 → 2°

    # Keep scale_factor from becoming too small
    scale_factor = max(scale_factor, 0.5)

    rot2 = np.eye(4)
    rot2[:3, :3] = Rotation.from_euler('z', np.deg2rad(rot_angle)).as_matrix()

    vertices = np.r_[cam.vertices,
                      scale_factor * cam.vertices,
                      geotrf(rot2, cam.vertices)]
    vertices = geotrf(transform, vertices)
    faces = []
    for face in cam.faces:
        if 0 in face:
            continue
        a, b, c = face
        a2, b2, c2 = face + len(cam.vertices)
        a3, b3, c3 = face + 2 * len(cam.vertices)

        faces.append((a, b, b2))
        faces.append((a, a2, c))
        faces.append((c2, b, c))

        faces.append((a, b, b3))
        faces.append((a, a3, c))
        faces.append((c3, b, c))

    # no culling
    faces += [(c, b, a) for a, b, c in faces]

    cam = trimesh.Trimesh(vertices=vertices, faces=faces)
    cam.visual.face_colors[:, :3] = edge_color
    scene.add_geometry(cam)

    if marker == 'o':
        marker = trimesh.creation.icosphere(3, radius=screen_width / 4)
        marker.vertices += pose_c2w[:3, 3]
        marker.visual.face_colors[:, :3] = edge_color
        scene.add_geometry(marker)


def camera_backward_forward(c2w, distance):
    c2w[:3, 3:4] = (c2w @ np.array([0, 0, distance, 1.0], dtype=np.float32).reshape(4, 1))[:3]
    return c2w


def camera_left_right(c2w, distance):
    c2w[:3, 3:4] = (c2w @ np.array([distance, 0, 0, 1.0], dtype=np.float32).reshape(4, 1))[:3]
    return c2w


def native_camera_rotation(c2w, medium_depth, phi, theta):
    R_elevation = np.array([[1, 0, 0, 0],
                            [0, np.cos(theta), -np.sin(theta), 0],
                            [0, np.sin(theta), np.cos(theta), 0],
                            [0, 0, 0, 1]], dtype=np.float32)
    R_azimuth = np.array([[np.cos(phi), 0, np.sin(phi), 0],
                          [0, 1, 0, 0],
                          [-np.sin(phi), 0, np.cos(phi), 0],
                          [0, 0, 0, 1]], dtype=np.float32)

    dummy_c2w = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, -medium_depth],
                          [0, 0, 0, 1]], dtype=np.float32)
    dummy_c2w = R_azimuth @ R_elevation @ dummy_c2w
    dummy_c2w[:3, 3] += np.array([0, 0, medium_depth], dtype=np.float32)
    c2w = c2w @ dummy_c2w

    return c2w


def axis_angle_to_matrix(axis, angle):
    """Rodrigues rotation formula, axis must be unit vector, angle in radians"""
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c

    R = np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C]
    ])
    return R


def make_homogeneous_rotation(axis, angle, point):
    """
    Build a 4x4 homogeneous transform for rotation around an arbitrary axis that does not pass through the origin.
    axis: (3,) rotation-axis direction, already normalized
    angle: scalar, in radians
    point: (3,) a point on the axis
    Returns: 4x4 transform matrix
    """
    # Rotation part
    R = axis_angle_to_matrix(axis, angle)
    # Translate to the origin
    T1 = np.eye(4)
    T1[:3, 3] = -point
    # Rotate
    R_homo = np.eye(4)
    R_homo[:3, :3] = R
    # Translate back
    T2 = np.eye(4)
    T2[:3, 3] = point
    # Combined transform
    M = T2 @ R_homo @ T1
    return M


def rotate_cam2world_around_axis(cam2world, axis, angle, point):
    """
    cam2world: 4x4 numpy array
    axis: 3-element array, rotation axis (a direction in arbitrary space, not necessarily normalized)
    angle: float, rotation angle (radian)
    point: 3-element array, a point on the rotation axis, such as a point the axis passes through
    """
    axis = np.asarray(axis)
    point = np.asarray(point)
    M = make_homogeneous_rotation(axis, angle, point)
    # Use M @ cam2world or cam2world @ M depending on the world/camera frame convention.
    new_cam2world = M @ cam2world
    return new_cam2world


def camera_rotation(c2w, medium_depth, phi, theta):
    # Iterative camera motion needs the initial pitch angle first.
    z0 = c2w[2, 3]
    if z0 != 0 and phi != 0:
        axis_origin = np.array([0, 0, medium_depth, 1], dtype=np.float32)
        axis_origin = c2w @ axis_origin.reshape(4, 1)
        axis_origin = axis_origin[:3, 0]
        axis_origin[2] = 0
        return rotate_cam2world_around_axis(c2w, axis=np.array([0, 0, 1], dtype=np.float32), angle=-phi, point=axis_origin)
    else:
        return native_camera_rotation(c2w, medium_depth, phi, theta)


def interpolate_poses(poses, M):
    """
    poses: (N,4,4) numpy array with N camera extrinsics
    M: Number of cameras after interpolation, M > N

    Returns: (M,4,4) numpy array of interpolated extrinsics
    """
    N = poses.shape[0]
    assert N >= 2, "需要至少两个姿态进行插值"
    assert poses.shape[1:] == (4, 4), "输入Pose格式错误"

    # Time parameter, simply linearly divided over [0, 1]
    t_orig = np.linspace(0, 1, N)
    t_interp = np.linspace(0, 1, M)

    # 1) Extract rotation and translation
    rotations = poses[:, :3, :3]  # (N,3,3)
    translations = poses[:, :3, 3]  # (N,3)

    # 2) Convert rotations to quaternions
    r = scipy.spatial.transform.Rotation.from_matrix(rotations)

    # 3) Create the slerp object
    slerp = scipy.spatial.transform.Slerp(t_orig, r)

    # 4) Interpolate rotations
    interp_rots = slerp(t_interp)
    interp_rot_mats = interp_rots.as_matrix()  # (M,3,3)

    # 5) Linearly interpolate translations
    interp_trans = np.empty((M, 3))
    for i in range(3):
        interp_trans[:, i] = np.interp(t_interp, t_orig, translations[:, i])

    # 6) Assemble final matrices
    interp_poses = np.zeros((M, 4, 4))
    interp_poses[:, :3, :3] = interp_rot_mats
    interp_poses[:, :3, 3] = interp_trans
    interp_poses[:, 3, 3] = 1.0
    return interp_poses


def compute_points_to_mesh_distance(points: np.ndarray, mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """
    Compute the minimum distance from N points to the mesh.

    Args:
        points: (N, 3) numpy array
        mesh: open3d TriangleMesh

    Returns:
        distances: (N,) numpy array with the minimum distance from each point to the mesh
    """
    # Convert the mesh to tensor format
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    # Create RaycastingScene
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)

    # Compute unsigned distances, from each point to the nearest surface
    query_points = o3d.core.Tensor(points.astype(np.float32))
    distances = scene.compute_distance(query_points)

    return distances.numpy()


def get_c2w(c2w_start, move, median_depth, air_bound, n_inter=20, kdtree=None, mesh=None, distance_threshold=0.02, local_rank=0, obs_decay=0.5, obs_limit=4):
    distance_threshold = min(distance_threshold, median_depth * 0.1)
    if type(distance_threshold) == torch.Tensor:
        distance_threshold = distance_threshold.item()

    c2ws = []
    if move["type"] == "normal":
        for j in range(n_inter):
            # Construct camera motion at the intermediate point
            move_inter = dict()
            for key in move:
                if key not in ("type", "name"):
                    if type(move[key]) == list:
                        move_inter[key] = [(j + 1) / n_inter * v for v in move[key]]
                    else:
                        move_inter[key] = (j + 1) / n_inter * move[key]

            c2w = c2w_start.copy()
            for key in move_inter:
                if key == "rotation" and np.sum(np.abs(move_inter[key])) == 0:
                    continue
                if key in ("backward-forward", "left-right") and move_inter[key] == 0:
                    continue

                if key == "backward-forward":
                    c2w = camera_backward_forward(c2w, air_bound * move_inter[key])
                elif key == "left-right":
                    c2w = camera_left_right(c2w, air_bound * move_inter[key])
                elif key == "rotation":
                    phi, theta = move_inter[key]
                    phi = np.deg2rad(phi)
                    theta = np.deg2rad(theta)
                    c2w = camera_rotation(c2w, median_depth, phi, theta)
                else:
                    raise NotImplementedError
            c2ws.append(c2w)
    elif move["type"] == "eloop":
        look_at_point = (0, 0, median_depth)
        angles = np.linspace(0, 2 * np.pi, n_inter + 1)[1:]
        move["radius_x"] *= median_depth
        move["radius_y"] *= median_depth
        for angle in angles:
            cam_pos = np.array([move['radius_x'] * np.sin(angle), move['radius_y'] * np.cos(angle) - move['radius_y'], 0], dtype=np.float32)
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, 3] = cam_pos
            R_new = look_at_rotation(cam_pos, at=(look_at_point,), up=((0, 1, 0),), device="cpu").numpy()[0]
            c2w[:3, :3] = R_new
            c2w = c2w_start @ c2w
            c2ws.append(c2w)
    elif move["type"] == "aerial":
        phi, theta = move["rotation"]
        n_inter_phi = int(round(np.abs(phi) / (np.abs(phi) + np.abs(theta)) * n_inter))
        n_inter_theta = n_inter - n_inter_phi
        for j in range(n_inter_theta):  # Rotate pitch first, then rotate the horizontal look-down angle
            theta_j = np.deg2rad((j + 1) / n_inter_theta * theta)
            c2w = camera_rotation(c2w_start.copy(), median_depth, 0, theta_j)
            c2ws.append(c2w)
        c2w_middle = c2ws[-1].copy()
        for j in range(n_inter_phi):
            phi_j = np.deg2rad((j + 1) / n_inter_phi * phi)
            c2w = camera_rotation(c2w_middle.copy(), median_depth, phi_j, 0)
            c2ws.append(c2w)
    else:
        raise NotImplementedError

    c2ws = np.array(c2ws)
    if c2ws.shape[0] < 80:
        c2ws_ = interpolate_poses(c2ws, 80)
    else:
        c2ws_ = c2ws.copy()
    # Query nearest-neighbor indices and distances
    query_points = c2ws_[:, :3, 3]
    if mesh is not None: # Prefer mesh-based obstacle avoidance
        distances = compute_points_to_mesh_distance(query_points, mesh)
    else:
        distances, indices = kdtree.query(query_points, k=5)
        distances = distances.mean(axis=1)
    min_distance = distances.min()
    obs_iteration = 0
    while min_distance < distance_threshold and obs_iteration < obs_limit + 1:  # Halve the motion distance, up to 2 times
        if local_rank == 0:
            print(f"Obstruction is detected in candidate: {move}", "min distance:", min_distance, "reduce the trajectory by half...")
        if move["type"] == "normal":
            c2ws = c2ws[:int(c2ws.shape[0] * obs_decay)]  # obs_decay: shrink the camera-motion range by this factor each time
            c2ws = interpolate_poses(c2ws, n_inter)
        elif move["type"] == "eloop":
            c2ws = []
            look_at_point = (0, 0, median_depth)
            angles = np.linspace(0, 2 * np.pi, n_inter + 1)[1:]
            move["radius_x"] *= obs_decay
            move["radius_y"] *= obs_decay
            for angle in angles:
                cam_pos = np.array([move['radius_x'] * np.sin(angle), move['radius_y'] * np.cos(angle) - move['radius_y'], 0], dtype=np.float32)
                c2w = np.eye(4, dtype=np.float32)
                c2w[:3, 3] = cam_pos
                R_new = look_at_rotation(cam_pos, at=(look_at_point,), up=((0, 1, 0),), device="cpu").numpy()[0]
                c2w[:3, :3] = R_new
                c2w = c2w_start @ c2w
                c2ws.append(c2w)
            c2ws = np.array(c2ws)
        elif move["type"] == "aerial":
            c2ws = []
            phi, theta = move["rotation"]
            n_dist_phi = int(round(np.abs(phi) / (np.abs(phi) + np.abs(theta)) * distances.shape[0]))
            n_dist_theta = distances.shape[0] - n_dist_phi
            dist_ = distances < distance_threshold
            if np.sum(dist_[:n_dist_theta]) > 0:  # If pitch collides, reduce the pitch angle
                move["rotation"][1] *= obs_decay
            if np.sum(dist_[n_dist_theta:]) > 0:  # If the horizontal look-down angle collides, reduce it
                move["rotation"][0] *= obs_decay
            phi, theta = move["rotation"]
            n_inter_phi = int(round(np.abs(phi) / (np.abs(phi) + np.abs(theta)) * n_inter))
            n_inter_theta = n_inter - n_inter_phi
            for j in range(n_inter_theta):  # Rotate pitch first, then rotate the horizontal look-down angle
                theta_j = np.deg2rad((j + 1) / n_inter_theta * theta)
                c2w = camera_rotation(c2w_start.copy(), median_depth, 0, theta_j)
                c2ws.append(c2w)
            c2w_middle = c2ws[-1].copy()
            for j in range(n_inter_phi):
                phi_j = np.deg2rad((j + 1) / n_inter_phi * phi)
                c2w = camera_rotation(c2w_middle.copy(), median_depth, phi_j, 0)
                c2ws.append(c2w)
            c2ws = np.array(c2ws)
        else:
            raise NotImplementedError
        if c2ws.shape[0] < 80:
            c2ws_ = interpolate_poses(c2ws, 80)
        else:
            c2ws_ = c2ws.copy()
        query_points = c2ws_[:, :3, 3]
        distances, indices = kdtree.query(query_points, k=5)
        distances = distances.mean(axis=1)
        min_distance = distances.min()
        obs_iteration += 1
        # print(f"New distance: {min_distance}")

    # print(f"candidate: {move}", f"min distance: {min_distance}", f"obs iteration: {obs_iteration}")

    return c2ws, obs_iteration


def sample_ones_from_binary_map(
        binary_map: np.ndarray,
        n_samples: int,
        random_seed: int = None,
) -> np.ndarray:
    """
    Randomly sample positions whose value is 1 from a binary numpy array (nonzero is 1), with replacement.
    """
    # 1. Input validation
    if binary_map.ndim != 2:
        raise ValueError(f"输入必须是2D数组，当前维度：{binary_map.ndim}")
    if not np.all(np.isin(binary_map, [0, 1])):
        raise ValueError("输入数组必须仅包含0和1")

    # 2. Extract coordinates of all positions with value 1 (rows: row indices, cols: column indices)
    rows, cols = np.where(binary_map == 1)
    n_ones = len(rows)
    if n_ones == 0:
        raise ValueError("输入数组中无值为1的点位，无法采样")

    # 3. Set random seed, if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    # 4. Sample indices with replacement; replace=True is the key behavior.
    # np.random.choice defaults to replace=True, but specifying it is clearer.
    sample_indices = np.random.choice(n_ones, size=n_samples, replace=True)

    # 5. Get coordinates from sampled indices and combine them into an (n_samples, 2) array
    sampled_points = np.stack([rows[sample_indices], cols[sample_indices]], axis=1)

    return sampled_points


def get_random_rotation_matrix(min_deg=45, max_deg=135, upright=False):
    """
    Generate a constrained World-to-Camera rotation matrix in the OpenCV coordinate system.

    Args:
        min_deg: Minimum included angle in degrees
        max_deg: Maximum included angle in degrees
        upright:
            False (default) -> produce a fully random rotation, including random roll
            True -> force the camera to stay level, with the X axis parallel to the ground plane

    Returns:
        R (np.ndarray): 3x3 rotation matrix
    """

    # 1. Construct the camera Z axis (Forward)
    min_rad = np.deg2rad(min_deg)
    max_rad = np.deg2rad(max_deg)

    # Uniformly sample spherical coordinates
    cos_theta = np.random.uniform(np.cos(max_rad), np.cos(min_rad))
    theta = np.arccos(cos_theta)
    phi = np.random.uniform(0, 2 * np.pi)

    z_c = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        cos_theta
    ])

    # 2. Construct the camera X axis (Right)
    if upright:
        # Mode A: keep level
        world_up = np.array([0, 0, 1])
        x_c = np.cross(world_up, z_c)
        # The direction does not need special handling here; step 4 corrects it automatically.
        if np.linalg.norm(x_c) < 1e-6:
            # Rarely, z_c is parallel to world_up (0 or 180 degrees); handle it for robustness despite the 45-135 constraint.
            x_c = np.array([1, 0, 0])
        else:
            x_c = x_c / np.linalg.norm(x_c)
    else:
        # Mode B: fully random roll
        random_vec = np.random.randn(3)
        x_c = random_vec - np.dot(random_vec, z_c) * z_c
        x_c = x_c / np.linalg.norm(x_c)

    # 3. Construct the camera Y axis (Down)
    # Right-hand rule: Y = Z cross X
    y_c = np.cross(z_c, x_c)
    y_c = y_c / np.linalg.norm(y_c)

    # =================================================================
    # 4. Constraint check and correction
    # Requirement: the angle between the camera Y axis (Down) and negative world Z (World Down) is < 90 degrees
    # Equivalent to: Dot(y_c, [0,0,-1]) > 0  =>  -y_c[2] > 0  =>  y_c[2] < 0
    # =================================================================
    if y_c[2] >= 0:
        # If the Y-axis Z component is nonnegative, the camera is upside down or the Y axis points upward.
        # Strategy: flip X and Y together.
        # Principle: (-X) cross (-Y) = Z, preserving the Z axis and the right-handed system,
        # while rotating the camera 180 degrees around Z so the Y axis points downward.
        x_c = -x_c
        y_c = -y_c

    # 5. Assemble the rotation matrix
    # The rows of R_cw are the camera axes
    R_cw = np.stack([x_c, y_c, z_c])

    return R_cw


def get_origin_height(bottom_mesh):
    """
    Get the bottom height of the mesh.

    Args:
        bottom_mesh: open3d.geometry.TriangleMesh (Legacy mesh)

    Returns:
        origin_height (float): Bottom height of the mesh.
    """
    bottom_mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(bottom_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(bottom_mesh_t)

    ray_origin = np.zeros((1, 3), dtype=np.float32)
    # Direction: (0, 0, -1), vertically downward
    ray_dir = np.zeros((1, 3), dtype=np.float32)
    ray_dir[:, 2] = -1.0

    # Construct the ray tensor: shape (N, 6) -> [ox, oy, oz, dx, dy, dz]
    ray = np.concatenate([ray_origin, ray_dir], axis=1).astype(np.float32)
    rays = o3d.core.Tensor(ray, dtype=o3d.core.Dtype.Float32)

    # 4. Cast rays
    # cast_rays returns a dictionary containing t_hit (distance), geometry_ids, etc.
    ans = scene.cast_rays(rays)

    # 5. Parse results
    t_hit = ans['t_hit'].numpy()  # Get distance

    z_values_bottom = ray_origin[:, 2] - t_hit

    return z_values_bottom


def get_z_from_xy(bottom_mesh, upper_mesh, x, y, z_max):
    """
    Given (x, y), compute the z value on the mesh surface by casting rays from top to bottom.

    Args:
        bottom_mesh: open3d.geometry.TriangleMesh (Legacy mesh)
        upper_mesh: open3d.geometry.TriangleMesh (Legacy mesh)
        x, y: [N,]*2

    Returns:
        z (float): z coordinate of the intersection point. Returns None if there is no intersection.
    """

    # 1. Convert meshes to tensor format because RaycastingScene requires tensor meshes
    # If your mesh is already an o3d.t.geometry.TriangleMesh, this step can be skipped.
    bottom_mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(bottom_mesh)
    upper_mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(upper_mesh)

    # 2. Create the raycasting scene and add the mesh
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(bottom_mesh_t)

    # 3. Determine ray origin and direction
    # Get the mesh bounding box to ensure the ray origin is above the mesh

    # Origin: (x, y, z_max + 1.0), slightly higher to ensure it is outside the object
    ray_origin = np.zeros((x.shape[0], 3), dtype=np.float32)
    ray_origin[:, 0] = x
    ray_origin[:, 1] = y
    ray_origin[:, 2] = z_max + 10.0
    # Direction: (0, 0, -1), vertically downward
    ray_dir = np.zeros((x.shape[0], 3))
    ray_dir[:, 2] = -1.0

    # Construct the ray tensor: shape (N, 6) -> [ox, oy, oz, dx, dy, dz]
    ray = np.concatenate([ray_origin, ray_dir], axis=1).astype(np.float32)
    rays = o3d.core.Tensor(ray, dtype=o3d.core.Dtype.Float32)

    # 4. Cast rays
    # cast_rays returns a dictionary containing t_hit (distance), geometry_ids, etc.
    ans = scene.cast_rays(rays)

    # 5. Parse results
    t_hit = ans['t_hit'].numpy()  # Get distance

    # Compute the actual Z value
    # Z_intersection = Z_origin + (direction_z * t_hit)
    # Since direction_z is -1, this is Z_origin - t_hit
    z_values_bottom = ray_origin[:, 2] - t_hit
    # Set infinity values (misses) to NaN
    z_values_bottom[np.isinf(t_hit)] = np.nan

    # Repeat the process above to get Z values for the upper part
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(upper_mesh_t)

    ray_origin = np.zeros((x.shape[0], 3), dtype=np.float32)
    ray_origin[:, 0] = x
    ray_origin[:, 1] = y
    ray_origin[:, 2] = z_values_bottom + 1e-3  # Slightly above the bottom Z value
    # Direction: (0, 0, 1), vertically upward
    ray_dir = np.zeros((x.shape[0], 3))
    ray_dir[:, 2] = 1.0

    # Construct the ray tensor: shape (N, 6) -> [ox, oy, oz, dx, dy, dz]
    ray = np.concatenate([ray_origin, ray_dir], axis=1).astype(np.float32)
    rays = o3d.core.Tensor(ray, dtype=o3d.core.Dtype.Float32)

    ans = scene.cast_rays(rays)
    t_hit = ans['t_hit'].numpy()
    z_values_upper = ray_origin[:, 2] + t_hit
    z_values_upper[np.isinf(t_hit)] = np.nan

    return z_values_bottom, z_values_upper


def add_camera_pose_noise(c2w, trans_noise_range, rot_noise_degree_range):
    """
    Add random perturbations to N*4*4 c2w matrices in the local coordinate system.

    Args:
        c2w: (N, 4, 4) numpy array, original camera extrinsics
        trans_noise_range: list/tuple of 3 floats [x_max, y_max, z_max]
                           Translation perturbation range for the xyz axes, in meters/scene units.
                           Perturbations are uniformly sampled from [-max, max].
        rot_noise_degree_range: list/tuple of 3 floats [x_deg, y_deg, z_deg]
                                Rotation perturbation range for the xyz axes, in degrees.
                                Perturbations are uniformly sampled from [-deg, deg].
    Returns:
        perturbed_c2w: (N, 4, 4) extrinsics after adding perturbations
    """
    N = c2w.shape[0]

    # 1. Generate translation noise -> (N, 3)
    # Uniformly sample from [-limit, limit]
    tx_noise = np.random.uniform(-trans_noise_range[0], trans_noise_range[0], N)
    ty_noise = np.random.uniform(-trans_noise_range[1], trans_noise_range[1], N)
    tz_noise = np.random.uniform(-trans_noise_range[2], trans_noise_range[2], N)
    t_noise = np.stack([tx_noise, ty_noise, tz_noise], axis=1)

    # 2. Generate rotation noise -> (N, 3, 3)
    # Euler angles -> rotation matrix
    rx_noise = np.random.uniform(-rot_noise_degree_range[0], rot_noise_degree_range[0], N)
    ry_noise = np.random.uniform(-rot_noise_degree_range[1], rot_noise_degree_range[1], N)
    rz_noise = np.random.uniform(-rot_noise_degree_range[2], rot_noise_degree_range[2], N)
    euler_noise = np.stack([rx_noise, ry_noise, rz_noise], axis=1)

    # Use scipy for efficient conversion; 'xyz' is the rotation order and degrees=True means the inputs are degrees.
    rot_mat_noise = R.from_euler('xyz', euler_noise, degrees=True).as_matrix()

    # 3. Build the noise transform matrix T_noise (N, 4, 4)
    noise_mat = np.eye(4)[None, ...].repeat(N, axis=0)  # Initialize as identity matrices
    noise_mat[:, :3, :3] = rot_mat_noise
    noise_mat[:, :3, 3] = t_noise

    # 4. Apply perturbations
    # c2w @ noise_mat applies perturbations in the camera local coordinate system (recommended)
    # noise_mat @ c2w applies perturbations in the global world coordinate system
    perturbed_c2w = c2w @ noise_mat

    return perturbed_c2w


def compute_lookat_xy_angle(c2ws_R: Union[torch.Tensor, np.ndarray]):
    """
    Compute the angle between the camera look-at direction and the xy plane in the OpenCV coordinate system.

    OpenCV coordinate system:
        x -> right
        y -> down
        z -> forward, the direction the camera looks

    Args:
        c2ws_R: (N, 3, 3) camera-to-world rotation matrices

    Returns:
        angles: (N,) angle between each camera and the xy plane, in degrees, range [0, 90]
    """
    is_numpy = isinstance(c2ws_R, np.ndarray)
    if is_numpy:
        c2ws_R = torch.from_numpy(c2ws_R).float()

    # OpenCV: the camera looks along +z, so R @ [0, 0, 1]^T is the 3rd column of R
    look_at = c2ws_R[:, :, 2]  # (N, 3)

    # Normalize
    look_at = look_at / torch.norm(look_at, dim=-1, keepdim=True)

    # Angle to the xy plane = arcsin(|z component|)
    z_component = look_at[:, 2]
    angles_rad = torch.arcsin(torch.clamp(torch.abs(z_component), -1.0, 1.0))
    angles_deg = torch.rad2deg(angles_rad)

    if is_numpy:
        angles_deg = angles_deg.numpy()

    return angles_deg


def create_arrow_mesh(start, end, color, shaft_radius=0.003, head_radius=0.008, head_length_ratio=0.2):
    """
    Create an arrow mesh from start to end, composed of a cylinder and cone head.

    Args:
        start:             (3,) start coordinates
        end:               (3,) end coordinates
        color:             (3,) or (4,) color
        shaft_radius:      Arrow shaft radius
        head_radius:       Arrow cone base radius
        head_length_ratio: Ratio of arrow head length to total length

    Returns:
        trimesh.Trimesh: Arrow mesh
    """
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    direction = end - start
    length = np.linalg.norm(direction)

    if length < 1e-8:
        return None

    direction_normalized = direction / length

    # Lengths of the arrow parts
    head_length = length * head_length_ratio
    shaft_length = length - head_length

    # ---- Shaft (cylinder) ---- #
    shaft = trimesh.creation.cylinder(
        radius=shaft_radius,
        height=shaft_length,
        sections=8,
    )
    # The default cylinder is along the Z axis and centered at the origin; move it toward the start direction.
    shaft.apply_translation([0, 0, shaft_length / 2])

    # ---- Arrow head (cone) ---- #
    head = trimesh.creation.cone(
        radius=head_radius,
        height=head_length,
        sections=8,
    )
    # The default cone is along the Z axis with its base at the origin; move it to the shaft tip.
    head.apply_translation([0, 0, shaft_length + head_length / 2])

    # ---- Merge ---- #
    arrow = trimesh.util.concatenate([shaft, head])

    # ---- Rotation: align the Z axis to direction ---- #
    z_axis = np.array([0, 0, 1], dtype=np.float64)
    cross = np.cross(z_axis, direction_normalized)
    dot = np.dot(z_axis, direction_normalized)

    if np.linalg.norm(cross) < 1e-8:
        if dot > 0:
            rotation_matrix = np.eye(3)
        else:
            # 180-degree rotation
            rotation_matrix = np.diag([1, -1, -1]).astype(np.float64)
    else:
        cross_normalized = cross / np.linalg.norm(cross)
        angle = np.arccos(np.clip(dot, -1, 1))
        rotation_matrix = Rotation.from_rotvec(cross_normalized * angle).as_matrix()

    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = start
    arrow.apply_transform(transform)

    # ---- Color ---- #
    color = np.asarray(color, dtype=np.uint8)
    if len(color) == 3:
        color = np.append(color, 255)
    arrow.visual.face_colors = color

    return arrow


def add_trajectory_arrows(
    scene,
    poses_c2w,
    edge_color,
    arrow_interval=1,
    shaft_radius=0.003,
    head_radius=0.008,
    head_length_ratio=0.25,
    arrow_scale=1.0,
    show_trajectory_line=True,
    line_color=None,
):
    """
    Visualize camera trajectory motion directions as arrows in a trimesh.Scene.

    Args:
        scene:               trimesh.Scene object
        poses_c2w:           list of (4,4) np.ndarray c2w matrices in temporal order
        edge_color:          (3,) color, matching the camera color
        arrow_interval:      Draw one arrow every N frames; 1 means every frame
        shaft_radius:        Arrow shaft thickness
        head_radius:         Arrow head size
        head_length_ratio:   Ratio of arrow head length to total length
        arrow_scale:         Overall arrow scale; > 1 lengthens it, < 1 shortens it
        show_trajectory_line: Whether to additionally draw the trajectory line
        line_color:          Trajectory line color; None uses edge_color
    """
    if len(poses_c2w) < 2:
        return

    edge_color = np.asarray(edge_color, dtype=np.uint8)
    if line_color is None:
        line_color = edge_color

    # Extract all camera centers
    centers = np.array([pose[:3, 3] for pose in poses_c2w])  # (T, 3)

    # ---- 1. Trajectory line, simulated with a thin mesh ---- #
    if show_trajectory_line:
        line_radius = shaft_radius * 0.4
        for i in range(len(centers) - 1):
            seg_start = centers[i]
            seg_end = centers[i + 1]
            seg_len = np.linalg.norm(seg_end - seg_start)
            if seg_len < 1e-8:
                continue

            seg_dir = (seg_end - seg_start) / seg_len
            line_seg = trimesh.creation.cylinder(
                radius=line_radius,
                height=seg_len,
                sections=6,
            )

            # Align direction
            z_axis = np.array([0, 0, 1.0])
            cross = np.cross(z_axis, seg_dir)
            dot = np.dot(z_axis, seg_dir)
            if np.linalg.norm(cross) < 1e-8:
                rot = np.eye(3) if dot > 0 else np.diag([1, -1, -1.0])
            else:
                angle = np.arccos(np.clip(dot, -1, 1))
                rot = Rotation.from_rotvec(cross / np.linalg.norm(cross) * angle).as_matrix()

            T = np.eye(4)
            T[:3, :3] = rot
            T[:3, 3] = (seg_start + seg_end) / 2
            line_seg.apply_transform(T)

            lc = np.append(line_color, 180) if len(line_color) == 3 else line_color
            line_seg.visual.face_colors = lc.astype(np.uint8)
            scene.add_geometry(line_seg)

    # ---- 2. Direction arrows ---- #
    for i in range(0, len(centers) - 1, arrow_interval):
        start = centers[i]
        end = centers[i + 1]

        # Optionally scale the arrow length by arrow_scale
        if arrow_scale != 1.0:
            direction = end - start
            mid = (start + end) / 2
            start = mid - direction * arrow_scale / 2
            end = mid + direction * arrow_scale / 2

        arrow = create_arrow_mesh(
            start, end, edge_color,
            shaft_radius=shaft_radius,
            head_radius=head_radius,
            head_length_ratio=head_length_ratio,
        )
        if arrow is not None:
            scene.add_geometry(arrow)
