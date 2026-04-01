"""
Initially written by Ming Hsiao in MATLAB
Rewritten in Python by Wei Dong (weidong@andrew.cmu.edu), 2021
"""

import numpy as np
import scipy.linalg
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import tyro
from dataclasses import dataclass, field
from typing import Literal
import matplotlib.pyplot as plt
from solvers import *
from utils import *


def warp2pi(angle_rad):
    r"""
    Warps an angle in [-pi, pi]. Used in the update step.
    \param angle_rad Input angle in radius
    \return angle_rad_warped Warped angle to [-\pi, \pi].
    """
    angle_rad = angle_rad - 2 * np.pi * np.floor((angle_rad + np.pi) / (2 * np.pi))
    return angle_rad


def init_states(odoms, observations, n_poses, n_landmarks):
    """
    Initialize the state vector given odometry and observations.
    """
    traj = np.zeros((n_poses, 2))
    landmarks = np.zeros((n_landmarks, 2))
    landmarks_mask = np.zeros((n_landmarks), dtype=np.bool)

    for i in range(len(odoms)):
        traj[i + 1, :] = traj[i, :] + odoms[i, :]

    for i in range(len(observations)):
        pose_idx = int(observations[i, 0])
        landmark_idx = int(observations[i, 1])

        if not landmarks_mask[landmark_idx]:
            landmarks_mask[landmark_idx] = True

            pose = traj[pose_idx, :]
            theta, d = observations[i, 2:]

            landmarks[landmark_idx, 0] = pose[0] + d * np.cos(theta)
            landmarks[landmark_idx, 1] = pose[1] + d * np.sin(theta)

    return traj, landmarks


def odometry_estimation(x, i):
    r"""
    \param x State vector containing both the pose and landmarks
    \param i Index of the pose to start from (odometry between pose i and i+1)
    \return odom Odometry (\Delta x, \Delta) in the shape (2, )
    """
    # TODO: return odometry estimation
    pose_i = x[2*i : 2*i+2]
    pose_next = x[2*(i+1) : 2*(i+1)+2]
    odom = pose_next - pose_i

    return odom


def bearing_range_estimation(x, i, j, n_poses):
    r"""
    \param x State vector containing both the pose and landmarks
    \param i Index of the pose to start from
    \param j Index of the landmark to be measured
    \param n_poses Number of poses
    \return obs Observation from pose i to landmark j (theta, d) in the shape (2, )
    """
    # TODO: return bearing range estimations
    pose_i = x[2*i : 2*i+2]
    lm_j = x[2*n_poses + 2*j : 2*n_poses + 2*j+2]
    
    dx = lm_j[0] - pose_i[0]
    dy = lm_j[1] - pose_i[1]
    
    theta = np.arctan2(dy, dx)
    d = np.sqrt(dx**2 + dy**2)
    obs = np.array([theta, d])

    return obs


def compute_meas_obs_jacobian(x, i, j, n_poses):
    r"""
    \param x State vector containing both the pose and landmarks
    \param i Index of the pose to start from
    \param j Index of the landmark to be measured
    \param n_poses Number of poses
    \return jacobian Derived Jacobian matrix in the shape (2, 4)
    """
    # TODO: return jacobian matrix
    pose_i = x[2*i : 2*i+2]
    lm_j = x[2*n_poses + 2*j : 2*n_poses + 2*j+2]
    
    dx = lm_j[0] - pose_i[0]
    dy = lm_j[1] - pose_i[1]
    
    q = dx**2 + dy**2
    sqrt_q = np.sqrt(q)
    
    jacobian = np.array([[ dy / q     , -dx / q     , -dy / q     , dx / q   ],
                       [-dx / sqrt_q, -dy / sqrt_q,  dx / sqrt_q, dy / sqrt_q]])
    return jacobian

def create_linear_system(
    x, odoms, observations, sigma_odom, sigma_observation, n_poses, n_landmarks
):
    r"""
    \param x State vector x at which we linearize the system.
    \param odoms Odometry measurements between i and i+1 in the global coordinate system. Shape: (n_odom, 2).
    \param observations Landmark measurements between pose i and landmark j in the global coordinate system. Shape: (n_obs, 4).
    \param sigma_odom Shared covariance matrix of odometry measurements. Shape: (2, 2).
    \param sigma_observation Shared covariance matrix of landmark measurements. Shape: (2, 2).

    \return A (M, N) Jacobian matrix.
    \return b (M, ) Residual vector.
    where M = (n_odom + 1) * 2 + n_obs * 2, total rows of measurements.
          N = n_poses * 2 + n_landmarks * 2, length of the state vector.
    """

    n_odom = len(odoms)
    n_obs = len(observations)

    M = (n_odom + 1) * 2 + n_obs * 2
    N = n_poses * 2 + n_landmarks * 2

    A = np.zeros((M, N))
    b = np.zeros((M,))

    # Prepare Sigma^{-1/2}.
    sqrt_inv_odom = np.linalg.inv(scipy.linalg.sqrtm(sigma_odom))
    sqrt_inv_obs = np.linalg.inv(scipy.linalg.sqrtm(sigma_observation))

    # First fill in the prior to anchor the 1st pose at (0, 0)
    
    # A[0:2, 0:2] = sqrt_inv_obs
    A[0:2, 0:2] = np.eye(2)

    # Then fill in odometry measurements
    for i in range(n_odom):
        r = 2 + 2 * i
        c_curr = 2 * i
        c_next = 2 * (i + 1)
        
        A[r:r+2, c_curr:c_curr+2] = -sqrt_inv_odom
        A[r:r+2, c_next:c_next+2] =  sqrt_inv_odom
        
        z = odoms[i]
        z_hat = odometry_estimation(x, i)
        residual = z - z_hat
        
        b[r:r+2] = sqrt_inv_odom @ residual

    # Then fill in landmark measurements
    obs_offset = 2 + 2 * n_odom
    for j in range(n_obs):
        pose_idx = int(observations[j, 0])
        lm_idx = int(observations[j, 1])
        
        r = obs_offset + 2 * j
        c_pose = 2 * pose_idx
        c_lm = 2 * n_poses + 2 * lm_idx
        
        H = compute_meas_obs_jacobian(x, pose_idx, lm_idx, n_poses)
        
        A_block = sqrt_inv_obs @ H
        
        A[r:r+2, c_pose:c_pose+2] = A_block[:, 0:2]
        A[r:r+2, c_lm:c_lm+2]   = A_block[:, 2:4]
        
        z = observations[j, 2:4]
        z_hat = bearing_range_estimation(x, pose_idx, lm_idx, n_poses)
        
        residual = z - z_hat
        residual[0] = warp2pi(residual[0])
        
        b[r:r+2] = sqrt_inv_obs @ residual

    return csr_matrix(A), b


@dataclass
class Args:
    data: str = "../data/2d_nonlinear.npz"
    method: list[Literal["default", "pinv", "qr", "lu", "qr_colamd", "lu_colamd"]] = (
        field(default_factory=lambda: ["default"])
    )


if __name__ == "__main__":
    args = tyro.cli(Args)

    data = np.load(args.data)

    # Plot gt trajectory and landmarks for a sanity check.
    gt_traj = data["gt_traj"]
    gt_landmarks = data["gt_landmarks"]
    plt.plot(gt_traj[:, 0], gt_traj[:, 1], "b-")
    plt.scatter(gt_landmarks[:, 0], gt_landmarks[:, 1], c="b", marker="+")
    plt.show()

    n_poses = len(gt_traj)
    n_landmarks = len(gt_landmarks)

    odom = data["odom"]
    observations = data["observations"]
    sigma_odom = data["sigma_odom"]
    sigma_landmark = data["sigma_landmark"]

    # Initialize: non-linear optimization requires a good init.
    for method in args.method:
        print(f"Applying {method}")
        traj, landmarks = init_states(odom, observations, n_poses, n_landmarks)
        print("Before optimization")
        plot_traj_and_landmarks(traj, landmarks, gt_traj, gt_landmarks)

        # Iterative optimization
        x = vectorize_state(traj, landmarks)
        for i in range(10):
            A, b = create_linear_system(
                x, odom, observations, sigma_odom, sigma_landmark, n_poses, n_landmarks
            )
            dx, _ = solve(A, b, method)
            x = x + dx
        traj, landmarks = devectorize_state(x, n_poses)
        print("After optimization")
        plot_traj_and_landmarks(traj, landmarks, gt_traj, gt_landmarks)
