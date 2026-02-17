'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import argparse
import numpy as np
import sys, os

from map_reader import MapReader
from motion_model import MotionModel
from sensor_model import SensorModel
from resampling import Resampling

from matplotlib import pyplot as plt
from matplotlib import figure as fig


def visualize_map(occupancy_map):
    fig = plt.figure()
    mng = plt.get_current_fig_manager()
    plt.ion()
    plt.imshow(occupancy_map, cmap='Greys')
    plt.axis([0, 800, 0, 800])


def visualize_timestep(X_bar, tstep, output_path):
    x_locs = X_bar[:, 0] / 10.0
    y_locs = X_bar[:, 1] / 10.0
    scat = plt.scatter(x_locs, y_locs, c='r', marker='o')
    plt.savefig('{}/{:04d}.png'.format(output_path, tstep))
    plt.pause(0.00001)
    scat.remove()


def init_particles_random(num_particles, occupancy_map):
    y0_vals = np.random.uniform(0, 7000, (num_particles, 1))
    x0_vals = np.random.uniform(3000, 7000, (num_particles, 1))
    theta0_vals = np.random.uniform(-3.14, 3.14, (num_particles, 1))

    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles

    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))

    return X_bar_init


def init_particles_freespace(num_particles, occupancy_map):
    """
    TODO : Add your code here
    This version converges faster than init_particles_random
    """
    X_bar_init = np.zeros((num_particles, 4))

    obstacle_mask = (occupancy_map > 0.2) | (occupancy_map < 0.0)
    free_mask = ~obstacle_mask
    free_cells = np.argwhere(free_mask)
    num_free_cells = free_cells.shape[0]
    sampled_indices = np.random.choice(num_free_cells, num_particles, replace=True)
    sampled_cells = free_cells[sampled_indices]
    X_bar_init[:, 0] = sampled_cells[:, 1] * 10
    X_bar_init[:, 1] = sampled_cells[:, 0] * 10
    X_bar_init[:, 2] = np.random.uniform(-np.pi, np.pi, num_particles)
    X_bar_init[:, 3] = 1.0 / num_particles

    return X_bar_init


def calculate_particle_variance(X_bar):

    x_vals = X_bar[:, 0]
    y_vals = X_bar[:, 1]
    theta_vals = X_bar[:, 2]
    
    var_x = np.var(x_vals)
    var_y = np.var(y_vals)
    
    cos_theta = np.cos(theta_vals)
    sin_theta = np.sin(theta_vals)
    mean_cos = np.mean(cos_theta)
    mean_sin = np.mean(sin_theta)
    mean_angle = np.arctan2(mean_sin, mean_cos)
    
    R = np.sqrt(mean_cos**2 + mean_sin**2)
    var_theta = 1.0 - R

    scale_x = np.ptp(x_vals) if np.ptp(x_vals) > 0 else 1.0
    scale_y = np.ptp(y_vals) if np.ptp(y_vals) > 0 else 1.0
    scale_theta = 2.0 * np.pi
    
    norm_var_x = var_x / (scale_x**2) if scale_x > 0 else 0.0
    norm_var_y = var_y / (scale_y**2) if scale_y > 0 else 0.0
    norm_var_theta = var_theta
    
    total_variance = norm_var_x + norm_var_y + norm_var_theta
    
    return total_variance


def adaptive_sampling(X_bar, current_variance, initial_variance, initial_particles, min_particles=500):
    """
    Adaptively reduce the number of particles based on variance decrease.
    Removes particles with lowest weights.
    """
    num_particles = X_bar.shape[0]
    
    # Calculate reduction factor based on variance ratio
    if initial_variance > 0:
        reduction_factor = current_variance / initial_variance
    else:
        reduction_factor = 1.0
    
    target_particles = max(min_particles, int(initial_particles * reduction_factor))
    
    if target_particles >= num_particles:
        return X_bar
    
    sorted_indices = np.argsort(X_bar[:, 3])
    
    keep_indices = sorted_indices[-target_particles:]
    X_bar_reduced = X_bar[keep_indices]
    
    weight_sum = np.sum(X_bar_reduced[:, 3])
    if weight_sum > 0:
        X_bar_reduced[:, 3] = X_bar_reduced[:, 3] / weight_sum
    else:
        X_bar_reduced[:, 3] = 1.0 / target_particles
    
    return X_bar_reduced


if __name__ == '__main__':
    """
    Description of variables used
    u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
    u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
    x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
    x_t1 : particle state belief [x, y, theta] at time t [world_frame]
    X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
    z_t : array of 180 range measurements for each laser scan
    """
    """
    Initialize Parameters
    """
    np.random.seed(52)
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_map', default='../data/map/wean.dat')
    parser.add_argument('--path_to_log', default='../data/log/robotdata6.log')
    parser.add_argument('--output', default='../results')
    parser.add_argument('--num_particles', default=20000, type=int)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    src_path_map = args.path_to_map
    src_path_log = args.path_to_log
    os.makedirs(args.output, exist_ok=True)

    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()
    logfile = open(src_path_log, 'r')

    motion_model = MotionModel()
    sensor_model = SensorModel(map_obj)
    resampler = Resampling()

    num_particles = args.num_particles
    initial_num_particles = num_particles
    # X_bar = init_particles_random(num_particles, occupancy_map)
    X_bar = init_particles_freespace(num_particles, occupancy_map)
    
    # Variables for adaptive sampling
    initial_variance = None
    min_particles = 500
    adaptive_sampling_interval = 1
    adaptive_sampling_interval = 1
    adaptive_sampling_counter = 0
    min_steps_before_reduction = 75
    w_fast_alpha = 0.9 
    w_slow_alpha = 0.01
    w_fast_average = 0.0
    w_slow_average = 0.0
    step_count = 0
    w_t_log = np.zeros((num_particles, 1))

    if not os.path.exists('../data/directional_ray_table.npy'):
        sensor_model.precompute_directional_ray_table(save_path='../data/directional_ray_table.npy')
    else:
        sensor_model.directional_ray_table = np.load('../data/directional_ray_table.npy')
        sensor_model.precompute_num_directions = sensor_model.directional_ray_table.shape[2]
        
    """
    Monte Carlo Localization Algorithm : Main Loop
    """
    if args.visualize:
        visualize_map(occupancy_map)

    first_time_idx = True
    for time_idx, line in enumerate(logfile):

        step_count += 1
        # Read a single 'line' from the log file (can be either odometry or laser measurement)
        # L : laser scan measurement, O : odometry measurement
        meas_type = line[0]

        # convert measurement values from string to double
        meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')

        # odometry reading [x, y, theta] in odometry frame
        odometry_robot = meas_vals[0:3]
        time_stamp = meas_vals[-1]

        # ignore pure odometry measurements for (faster debugging)
        # if ((time_stamp <= 0.0) | (meas_type == "O")):
        #     continue

        if (meas_type == "L"):
            # [x, y, theta] coordinates of laser in odometry frame
            odometry_laser = meas_vals[3:6]
            # 180 range measurement values from single laser scan
            ranges = meas_vals[6:-1]

        print("Processing time step {} at time {}s".format(
            time_idx, time_stamp))

        if first_time_idx:
            u_t0 = odometry_robot
            first_time_idx = False
            continue

        u_t1 = odometry_robot

        # MOTION MODEL
        X_t0 = X_bar[:, 0:3]
        X_t1 = motion_model.update(u_t0, u_t1, X_t0)

        # SENSOR MODEL
        if (meas_type == "L"):
            z_t = ranges
            w_t, w_t_log = sensor_model.beam_range_finder_model(z_t, X_t1)
            w_t, w_t_log = sensor_model.beam_range_finder_model(z_t, X_t1)
            X_bar_new = np.hstack((X_t1, w_t[:, np.newaxis]))
        else:
            X_bar_new = np.hstack((X_t1, X_bar[:, 3:4]))

        X_bar = X_bar_new
        u_t0 = u_t1

        """
        KIDNAPPED ROBOT
        """
        initial_variance = calculate_particle_variance(X_bar)

        w_avg = np.mean(w_t_log)
        w_fast_average = w_fast_average + w_fast_alpha * (w_avg - w_fast_average)
        w_slow_average = w_slow_average + w_slow_alpha * (w_avg - w_slow_average)
        if w_slow_average > 0 and w_fast_average/w_slow_average < 0.2:
            print("KIDNAPPED ROBOT")
            X_bar = init_particles_freespace(num_particles, occupancy_map)
            w_fast_average = 0.0
            w_slow_average = 0.0
            initial_variance = calculate_particle_variance(X_bar)
            step_count = 0
        
        """
        RESAMPLING
        """
        X_bar[:, 3] = X_bar[:, 3] / (np.sum(X_bar[:, 3]) + 1e-12)
        X_bar[:, 3] = X_bar[:, 3] / (np.sum(X_bar[:, 3]) + 1e-12)
        X_bar = resampler.low_variance_sampler(X_bar)
        
        """
        ADAPTIVE SAMPLING
        """
        adaptive_sampling_counter += 1
        if adaptive_sampling_counter >= adaptive_sampling_interval and step_count >= min_steps_before_reduction:
            adaptive_sampling_counter = 0
    
            current_variance = calculate_particle_variance(X_bar)
            
            if initial_variance > 0 and current_variance < initial_variance:
                X_bar = adaptive_sampling(X_bar, current_variance, initial_variance, initial_num_particles, min_particles)
                num_particles = X_bar.shape[0]
                # if num_particles < initial_num_particles:
                #     print("Adaptive sampling: Reduced to {} particles (variance: {:.6f})".format(
                #         num_particles, current_variance))
                # if num_particles < initial_num_particles:
                #     print("Adaptive sampling: Reduced to {} particles (variance: {:.6f})".format(
                #         num_particles, current_variance))

        if args.visualize:
            visualize_timestep(X_bar, time_idx, args.output)
