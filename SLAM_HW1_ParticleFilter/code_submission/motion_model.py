'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import sys
import numpy as np
import math


class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """
    def __init__(self):
        """
        TODO : Tune Motion Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._alpha1 = 0.0001
        self._alpha2 = 0.0001
        self._alpha3 = 0.01
        self._alpha4 = 0.01


    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
                          All particles [num_particles, 3]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
                          All particles [num_particles, 3]
        """
        """
        TODO : Add your code here
        """
        # Odometry Motion Model (Table 5.6 in Probabilistic Robotics)

        x_t0 = np.asarray(x_t0)
        num_particles = x_t0.shape[0]

        # Extract odometry readings at t-1 and t
        x_bar0, y_bar0, theta_bar0 = u_t0[0], u_t0[1], u_t0[2]
        x_bar1, y_bar1, theta_bar1 = u_t1[0], u_t1[1], u_t1[2]

        # Compute relative motion from odometry
        delta_trans = np.sqrt((x_bar1 - x_bar0)**2 + (y_bar1 - y_bar0)**2)

        # Handle near-stationary case to avoid atan2 instability
        delta_rot1 = np.where(
            delta_trans < 0.01,
            0.0,
            np.arctan2(y_bar1 - y_bar0, x_bar1 - x_bar0) - theta_bar0
        )
        delta_rot2 = theta_bar1 - theta_bar0 - delta_rot1

        # Add noise
        delta_rot1_var = self._alpha1 * delta_rot1**2 + self._alpha2 * delta_trans**2
        delta_trans_var = self._alpha3 * delta_trans**2 + self._alpha4 * (delta_rot1**2 + delta_rot2**2)
        delta_rot2_var = self._alpha1 * delta_rot2**2 + self._alpha2 * delta_trans**2

        # noise sampling for all particles
        delta_rot1_hat = delta_rot1 + np.random.normal(0, np.sqrt(np.maximum(delta_rot1_var, 1e-10)), num_particles)
        delta_trans_hat = delta_trans + np.random.normal(0, np.sqrt(np.maximum(delta_trans_var, 1e-10)), num_particles)
        delta_rot2_hat = delta_rot2 + np.random.normal(0, np.sqrt(np.maximum(delta_rot2_var, 1e-10)), num_particles)

        # Apply noisy motion to particle state
        x = x_t0[:, 0]
        y = x_t0[:, 1]
        theta = x_t0[:, 2]
        
        x_new = x + delta_trans_hat * np.cos(theta + delta_rot1_hat)
        y_new = y + delta_trans_hat * np.sin(theta + delta_rot1_hat)
        theta_new = theta + delta_rot1_hat + delta_rot2_hat
        theta_new = (theta_new + np.pi) % (2*np.pi) - np.pi


        x_t1 = np.column_stack((x_new, y_new, theta_new))
        
        return x_t1
