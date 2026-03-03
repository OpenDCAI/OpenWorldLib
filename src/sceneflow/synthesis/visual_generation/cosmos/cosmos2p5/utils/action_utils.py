import math

import numpy as np


def alpha2rotm(a):
    """Converts an alpha Euler angle to a 3x3 rotation matrix.

    This corresponds to a rotation around the x-axis.

    Args:
        a (float): The alpha Euler angle in radians.

    Returns:
        np.ndarray: The 3x3 rotation matrix.
    """
    rotm = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    return rotm


def beta2rotm(b):
    """Converts a beta Euler angle to a 3x3 rotation matrix.

    This corresponds to a rotation around the y-axis.

    Args:
        b (float): The beta Euler angle in radians.

    Returns:
        np.ndarray: The 3x3 rotation matrix.
    """
    rotm = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    return rotm


def gamma2rotm(c):
    """Converts a gamma Euler angle to a 3x3 rotation matrix.

    This corresponds to a rotation around the z-axis.

    Args:
        c (float): The gamma Euler angle in radians.

    Returns:
        np.ndarray: The 3x3 rotation matrix.
    """
    rotm = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return rotm


def euler2rotm(euler_angles):
    """Converts a set of Euler angles (ZYX convention) to a 3x3 rotation
    matrix.

    Args:
        euler_angles (np.ndarray): A 1D array of three Euler angles [alpha, beta, gamma] in radians.

    Returns:
        np.ndarray: The corresponding 3x3 rotation matrix.
    """
    alpha = euler_angles[0]
    beta = euler_angles[1]
    gamma = euler_angles[2]

    rotm_a = alpha2rotm(alpha)
    rotm_b = beta2rotm(beta)
    rotm_c = gamma2rotm(gamma)

    # The final rotation matrix is the product of the individual rotations.
    # The order is R = Rz * Ry * Rx
    rotm = rotm_c @ rotm_b @ rotm_a

    return rotm


def isRotm(R):
    """Checks if a given 3x3 matrix is a valid rotation matrix.

    A matrix is a valid rotation matrix if its transpose is its inverse and its determinant is 1.
    This function checks the first condition by verifying if R.T * R is close to the identity matrix.

    Args:
        R (np.ndarray): The 3x3 matrix to check.

    Returns:
        bool: True if the matrix is a valid rotation matrix, False otherwise.
    """
    # Forked from Andy Zeng
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I_mat = np.identity(3, dtype=R.dtype)
    # Check the Frobenius norm of the difference between R.T * R and the identity matrix.
    n = np.linalg.norm(I_mat - shouldBeIdentity)
    return n < 1e-6


def rotm2euler(R):
    """Converts a 3x3 rotation matrix to a set of Euler angles (ZYX
    convention).

    Args:
        R (np.ndarray): The 3x3 rotation matrix.

    Returns:
        np.ndarray: A 1D array of three Euler angles [x, y, z] in radians.
    """
    # Forked from: https://learnopencv.com/rotation-matrix-to-euler-angles/
    # The rotation matrix is assumed to be in the order R = Rz * Ry * Rx.
    assert isRotm(R)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    # Normalize angles to the range (-pi, pi]
    while x > np.pi:
        x -= 2 * np.pi
    while x <= -np.pi:
        x += 2 * np.pi
    while y > np.pi:
        y -= 2 * np.pi
    while y <= -np.pi:
        y += 2 * np.pi
    while z > np.pi:
        z -= 2 * np.pi
    while z <= -np.pi:
        z += 2 * np.pi
    return np.array([x, y, z])


def get_actions_from_states(states, action_scale=20.0, gripper_scale=1.0):
    """Calculates a sequence of actions from a sequence of states.

    An action is defined as the relative change in position and orientation between two consecutive states,
    plus the gripper state.

    Args:
        states (np.ndarray): A sequence of states, where each state is a 7-element array:
                             [x, y, z, roll, pitch, yaw, gripper_state].
        action_scale (float): A scaling factor for the position and rotation actions.
        gripper_scale (float): A scaling factor for the gripper action.

    Returns:
        np.ndarray: A sequence of actions, where each action is a 7-element array:
                    [dx, dy, dz, droll, dpitch, dyaw, gripper_state].
    """
    sequence_length = len(states)
    actions = np.zeros((sequence_length - 1, 7))
    for k in range(1, sequence_length):
        # Get previous and current states
        prev_xyz = states[k - 1, 0:3]
        prev_rpy = states[k - 1, 3:6]
        prev_rotm = euler2rotm(prev_rpy)
        curr_xyz = states[k, 0:3]
        curr_rpy = states[k, 3:6]
        curr_gripper = states[k, -1]
        curr_rotm = euler2rotm(curr_rpy)

        # Calculate relative position in the previous frame's coordinate system
        rel_xyz = np.dot(prev_rotm.T, curr_xyz - prev_xyz)
        # Calculate relative rotation matrix
        rel_rotm = prev_rotm.T @ curr_rotm
        # Convert relative rotation matrix to Euler angles
        rel_rot = rotm2euler(rel_rotm)

        # Store the action
        actions[k - 1, 0:3] = rel_xyz
        actions[k - 1, 3:6] = rel_rot
        actions[k - 1, 6] = curr_gripper

    # Scale the actions
    scales = np.array([action_scale, action_scale, action_scale, action_scale, action_scale, action_scale, gripper_scale])
    actions *= scales
    return actions
