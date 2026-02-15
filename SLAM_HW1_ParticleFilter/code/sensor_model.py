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
        self._z_hit = 0.7
        self._z_short = 0.1
        self._z_max = 0.1
        self._z_rand = 0.1

        self._sigma_hit = 50
        self._lambda_short = 0.1

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
        
        # TIMING_START: ray_casting_instance_var
        self.ray_casting_time = 0.0  # Instance variable to track ray casting time
        # TIMING_END: ray_casting_instance_var

    def ray_casting(self, x0, y0, theta_beam):
        step_size = self.map_resolution/2

        d = step_size

        while d < self._max_range:

            x = x0 + d * np.cos(theta_beam)
            y = y0 + d * np.sin(theta_beam)

            x_grid = int(np.floor(x/self.map_resolution))
            y_grid = int(np.floor(y/self.map_resolution))

            if(x_grid >= self.occupancy_map.shape[1] or x_grid < 0 or \
               y_grid >= self.occupancy_map.shape[0] or y_grid <0):
                return self._max_range
            
            if(self.occupancy_map[y_grid, x_grid] > self._min_probability):
                return d
            
            d += step_size

        return self._max_range

    def compute_hit_eta(self, z_t_k_star):
        upper = (self._max_range - z_t_k_star)/self._sigma_hit
        lower = (0 - z_t_k_star)/self._sigma_hit

        denom = norm.cdf(upper) - norm.cdf(lower)
        denom = max(denom, 1e-12)
        eta = 1.0/denom

        return eta
    
    def compute_hit_likelihood(self, z_t1, z_t_k_star):
        p_hit = norm.pdf(z_t1, loc=z_t_k_star, scale=self._sigma_hit)
        eta = self.compute_hit_eta(z_t_k_star)
        
        return eta * p_hit
    
    def compute_short_likelihood(self, z_t1, z_t_k_star):

        denom = 1 - np.exp(-self._lambda_short * z_t_k_star)
        denom = max(denom, 1e-12)
        eta = 1.0/denom
        
        if(z_t1 >= 0 and z_t1 <= z_t_k_star):
            p_short = eta * self._lambda_short * np.exp(-self._lambda_short * z_t1)
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

        prob_zt1 = 0.0 # changed 1.0 -> 0.0 as we are summing log likelihoods

        x_laser = x_t1[0] + self.laser_offset * np.cos(x_t1[2])
        y_laser = x_t1[1] + self.laser_offset * np.sin(x_t1[2])

        # TIMING_START: ray_casting_total
        ray_casting_total_time = 0.0
        # TIMING_END: ray_casting_total

        for k in range(0, 180, self._subsampling):
            angle = -np.pi/2 + k * (np.pi/180)
            theta_beam = x_t1[2] + angle

            # TIMING_START: ray_casting
            ray_casting_start = time.time()
            # TIMING_END: ray_casting
            z_t_k_star = self.ray_casting(x_laser, y_laser, theta_beam)
            # TIMING_START: ray_casting
            ray_casting_time = time.time() - ray_casting_start
            ray_casting_total_time += ray_casting_time
            # TIMING_END: ray_casting

            p_hit = self.compute_hit_likelihood(z_t1_arr[k], z_t_k_star)
            p_short = self.compute_short_likelihood(z_t1_arr[k], z_t_k_star)
            p_max = self.compute_max_likelihood(z_t1_arr[k])
            p_rand = self.compute_rand_likelihood(z_t1_arr[k])

            p = self._z_hit * p_hit + self._z_short * p_short + \
                self._z_max * p_max + self._z_rand * p_rand
            
            p = max(p, 1e-12)
            prob_zt1 += np.log(p)

        # TIMING_START: ray_casting_total
        # Store ray casting time for this sensor model call
        self.ray_casting_time = ray_casting_total_time
        # TIMING_END: ray_casting_total

        return prob_zt1
