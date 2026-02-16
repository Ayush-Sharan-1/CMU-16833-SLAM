'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm

from map_reader import MapReader


class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """
    def __init__(self, map_obj):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._z_hit = 0.4
        self._z_short = 0.01
        self._z_max = 0.02
        self._z_rand = 0.57

        self._sigma_hit = 400
        self._lambda_short = 0.005

        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = 1000

        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.35

        # Used in sampling angles in ray casting
        self._subsampling = 2

        self.occupancy_map = map_obj.get_map()
        self.map_size_x = map_obj.get_map_size_x()
        self.map_size_y = map_obj.get_map_size_y()
        self.map_resolution = self.map_size_x / self.occupancy_map.shape[0]

        self.laser_offset = 25 #in cm

        self.precompute_num_directions = 180

        self._hit_gaussian_norm = 1.0 / (np.sqrt(2.0 * np.pi) * self._sigma_hit)
        self._hit_inv_sigma = 1.0 / (2.0 * self._sigma_hit * self._sigma_hit)


    def ray_casting(self, x0, y0, theta):
        x1 = x0 + self._max_range * np.cos(theta)
        y1 = y0 + self._max_range * np.sin(theta)

        x0_grid = int(x0 / self.map_resolution)
        y0_grid = int(y0 / self.map_resolution)
        x1_grid = int(x1 / self.map_resolution)
        y1_grid = int(y1 / self.map_resolution)

        # Get map dimensions
        H, W = self.occupancy_map.shape

        # Check if starting point is out of bounds
        if x0_grid < 0 or x0_grid >= W or y0_grid < 0 or y0_grid >= H:
            return self._max_range

        dx = abs(x1_grid - x0_grid)
        dy = abs(y1_grid - y0_grid)

        x, y = x0_grid, y0_grid
        sx = 1 if x0_grid < x1_grid else -1
        sy = 1 if y0_grid < y1_grid else -1

        if dx > dy:
            err = dx / 2.0
            while x != x1_grid:
                if x < 0 or x >= W or y < 0 or y >= H:
                    return self._max_range
                if self.occupancy_map[y, x] > self._min_probability:
                    return np.sqrt((x - x0_grid)**2 + (y - y0_grid)**2) * self.map_resolution
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1_grid:
                if x < 0 or x >= W or y < 0 or y >= H:
                    return self._max_range
                if self.occupancy_map[y, x] > self._min_probability:
                    return np.sqrt((x - x0_grid)**2 + (y - y0_grid)**2) * self.map_resolution
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy

        return self._max_range

    def precompute_directional_ray_table(self, save_path=None):
        """
        Precompute ray casting for each map cell and global direction.
        """

        H, W = self.occupancy_map.shape

        self.directional_ray_table = np.full(
            (H, W, self.precompute_num_directions),
            self._max_range,
            dtype=np.float32
        )

        print("Precomputing directional ray table...")

        for y in range(H):
            for x in range(W):

                # skip occupied or unknown cells
                if (self.occupancy_map[y, x] > self._min_probability) or (self.occupancy_map[y, x] < 0.0):
                    continue

                x_world = x * self.map_resolution
                y_world = y * self.map_resolution

                for d in range(self.precompute_num_directions):
                    theta = 2.0 * np.pi * d / self.precompute_num_directions
                    dist = self.ray_casting(x_world, y_world, theta)
                    self.directional_ray_table[y, x, d] = dist

            if y % 20 == 0:
                print(f"Row {y}/{H}")

        if save_path is not None:
            np.save(save_path, self.directional_ray_table)

    def get_predicted_range(self, x_laser, y_laser, theta_beam):
        """
        Lookup predicted beam range with interpolation.
        """

        # convert to grid
        x_grid = int(x_laser / self.map_resolution)
        y_grid = int(y_laser / self.map_resolution)

        H, W, _ = self.directional_ray_table.shape

        # out of bounds
        if x_grid < 0 or x_grid >= W or y_grid < 0 or y_grid >= H:
            return self._max_range

        # normalize angle
        theta = theta_beam % (2.0 * np.pi)

        # convert to bin
        angle_float = theta / (2.0 * np.pi) * self.precompute_num_directions

        low = int(np.floor(angle_float)) % self.precompute_num_directions
        high = (low + 1) % self.precompute_num_directions

        alpha = angle_float - np.floor(angle_float)

        # interpolate
        d_low = self.directional_ray_table[y_grid, x_grid, low]
        d_high = self.directional_ray_table[y_grid, x_grid, high]

        return (1.0 - alpha) * d_low + alpha * d_high

    def compute_hit_eta(self, z_t_k_star, sigma_hit):
        upper = (self._max_range - z_t_k_star)/sigma_hit
        lower = (0 - z_t_k_star)/sigma_hit

        denom = norm.cdf(upper) - norm.cdf(lower)
        denom = max(denom, 1e-12)
        eta = 1.0/denom

        return eta
    
    def compute_hit_likelihood(self, z_t1, z_t_k_star, hit_inv_sigma, hit_gaussian_norm, sigma_hit  ):
        diff = z_t1 - z_t_k_star
        p_hit = hit_gaussian_norm * np.exp(-hit_inv_sigma * diff * diff)
        eta = self.compute_hit_eta(z_t_k_star, sigma_hit)
        return eta * p_hit
    
    def compute_short_likelihood(self, z_t1, z_t_k_star, lambda_short):

        denom = 1 - np.exp(-lambda_short * z_t_k_star)
        denom = max(denom, 1e-12)
        eta = 1.0/denom
        
        if(z_t1 >= 0 and z_t1 <= z_t_k_star):
            # p_short = eta * self._lambda_short * np.exp(-self._lambda_short * z_t1)
            p_short = eta * lambda_short * np.exp(-lambda_short * z_t1)
        else:
            p_short = 0

        return p_short
    
    def compute_max_likelihood(self, z_t1):
        if (z_t1 == self._max_range):
            p_max = 1
        else:
            p_max = 0

        return p_max

    def compute_rand_likelihood(self, z_t1):
        
        if(z_t1 >= 0 and z_t1 < self._max_range):
            p_rand = 1.0/self._max_range
        else:
            p_rand = 0

        return p_rand

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        """
        TODO : Add your code here
        """

        x_laser = x_t1[0] + self.laser_offset * np.cos(x_t1[2])
        y_laser = x_t1[1] + self.laser_offset * np.sin(x_t1[2])

        k_arr = np.arange(0, 180, self._subsampling)
        angles = -np.pi/2 + k_arr * (np.pi/180)
        theta_beams = x_t1[2] + angles
        
        z_t1_values = z_t1_arr[k_arr]
        
        get_predicted_range_vec = np.vectorize(self.get_predicted_range)
        z_t_k_star = get_predicted_range_vec(x_laser, y_laser, theta_beams)
        
        # hit likelihood
        diff = z_t1_values - z_t_k_star
        p_hit = self._hit_gaussian_norm * np.exp(-self._hit_inv_sigma * diff * diff)
        # hit_eta
        upper = (self._max_range - z_t_k_star) / self._sigma_hit
        lower = (0 - z_t_k_star) / self._sigma_hit
        denom = norm.cdf(upper) - norm.cdf(lower)
        denom = np.maximum(denom, 1e-12)
        eta_hit = 1.0 / denom
        p_hit = eta_hit * p_hit
        
        #short likelihood
        denom_short = 1 - np.exp(-self._lambda_short * z_t_k_star)
        denom_short = np.maximum(denom_short, 1e-12)
        eta_short = 1.0 / denom_short
        p_short = np.where((z_t1_values >= 0) & (z_t1_values <= z_t_k_star),
                          eta_short * self._lambda_short * np.exp(-self._lambda_short * z_t1_values),0.0)
        
        # max likelihood
        p_max = np.where(z_t1_values == self._max_range, 1.0, 0.0)
        
        # rand likelihood 
        p_rand = np.where((z_t1_values >= 0) & (z_t1_values < self._max_range),
                         1.0 / self._max_range, 0.0)
        
        p = self._z_hit * p_hit + self._z_short * p_short + self._z_max * p_max + self._z_rand * p_rand
        
        p = np.maximum(p, 1e-12)
        prob_zt1 = np.sum(np.log(p))
        
        return np.exp(prob_zt1)
