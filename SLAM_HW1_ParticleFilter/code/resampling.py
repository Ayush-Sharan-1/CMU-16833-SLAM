'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np


class Resampling:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    """
    def __init__(self):
        """
        TODO : Initialize resampling process parameters here
        """

    def multinomial_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        X_bar_resampled =  np.zeros_like(X_bar)
        return X_bar_resampled

    def low_variance_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        X_bar_resampled =  np.zeros_like(X_bar)

        wt_sum = np.sum(X_bar[:, 3])
        
        if wt_sum == 0:
            wt_norm = np.ones(X_bar.shape[0]) / X_bar.shape[0]
        else:
            wt_norm = X_bar[:, 3] / wt_sum

        X_bar_cumsum = np.cumsum(wt_norm)
        num_particles = X_bar.shape[0]
        r = np.random.uniform(0, 1/num_particles)
        U = r + np.arange(num_particles) / num_particles
        idx = np.searchsorted(X_bar_cumsum, U)
        X_bar_resampled = X_bar[idx]

        return X_bar_resampled
