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

    def augmented_mcl_sampler(self, X_bar, w_fast, w_slow, occupancy_map, map_resolution=10.0):
        """
        Augmented MCL resampling with random particle injection for kidnapped robot problem.
        Based on Table 8.3 from Probabilistic Robotics.
        
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[in] w_fast : short-term average of measurement likelihood
        param[in] w_slow : long-term average of measurement likelihood
        param[in] occupancy_map : occupancy grid map
        param[in] map_resolution : resolution of the map (default 10.0 cm per cell)
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        X_bar_resampled = np.zeros_like(X_bar)
        num_particles = X_bar.shape[0]
        
        # Normalize weights for resampling
        wt_sum = np.sum(X_bar[:, 3])
        if wt_sum == 0:
            wt_norm = np.ones(num_particles) / num_particles
        else:
            wt_norm = X_bar[:, 3] / wt_sum
        
        # Precompute cumulative sum for efficient sampling
        X_bar_cumsum = np.cumsum(wt_norm)
        
        # Calculate probability of adding random particles
        if w_slow > 0:
            random_prob = max(0.0, 1.0 - w_fast / w_slow)
        else:
            random_prob = 0.0
        
        # Prepare free space cells for random sampling
        obstacle_mask = (occupancy_map > 0.2) | (occupancy_map < 0.0)
        free_mask = ~obstacle_mask
        free_cells = np.argwhere(free_mask)
        num_free_cells = free_cells.shape[0]
        
        # Resampling with random particle injection
        r = np.random.uniform(0, 1/num_particles)
        U = r + np.arange(num_particles) / num_particles
        
        for m in range(num_particles):
            # With probability random_prob, add a random pose
            if num_free_cells > 0 and np.random.rand() < random_prob:
                # Add random pose from free space
                random_idx = np.random.randint(0, num_free_cells)
                random_cell = free_cells[random_idx]
                X_bar_resampled[m, 0] = random_cell[1] * map_resolution
                X_bar_resampled[m, 1] = random_cell[0] * map_resolution
                X_bar_resampled[m, 2] = np.random.uniform(-np.pi, np.pi)
                X_bar_resampled[m, 3] = 1.0 / num_particles
            else:
                # Normal resampling: draw from weighted distribution
                idx = np.searchsorted(X_bar_cumsum, U[m])
                idx = min(idx, num_particles - 1)  # Ensure valid index
                X_bar_resampled[m] = X_bar[idx]
                X_bar_resampled[m, 3] = 1.0 / num_particles  # Reset weight after resampling
        
        return X_bar_resampled
