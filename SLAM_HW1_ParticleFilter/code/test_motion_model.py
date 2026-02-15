'''
    Test script for motion model
    Tests the motion model by moving a single particle using only odometry data
'''

import argparse
import numpy as np
import sys, os

from map_reader import MapReader
from motion_model import MotionModel
from main import init_particles_freespace, visualize_map

from matplotlib import pyplot as plt


def update_visualization(occupancy_map, particle_positions, trajectory_line, current_point):
    """
    Update the visualization with current particle position and trajectory
    Returns updated plot objects
    """
    # Convert particle positions from cm to pixels
    x_locs = [pos[0] / 10.0 for pos in particle_positions]
    y_locs = [pos[1] / 10.0 for pos in particle_positions]
    
    # Remove old trajectory line if it exists
    if trajectory_line is not None:
        try:
            trajectory_line.remove()
        except:
            pass
    
    # Remove old current point if it exists
    if current_point is not None:
        try:
            current_point.remove()
        except:
            pass
    
    # Plot trajectory line
    if len(particle_positions) > 1:
        line_objects = plt.plot(x_locs, y_locs, 'b-', alpha=0.5, linewidth=1)
        trajectory_line = line_objects[0] if line_objects else None
    else:
        trajectory_line = None
    
    # Plot current position
    if len(particle_positions) > 0:
        current_point = plt.scatter(x_locs[-1], y_locs[-1], c='r', marker='o', s=50)
    else:
        current_point = None
    
    return trajectory_line, current_point


if __name__ == '__main__':
    """
    Test the motion model with a single particle
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_map', default='../data/map/wean.dat')
    parser.add_argument('--path_to_log', default='../data/log/robotdata1.log')
    parser.add_argument('--speed', default=1.0, type=float, 
                        help='Speed multiplier (1.0 = baseline, 2.0 = 2x faster, 0.5 = 2x slower)')
    args = parser.parse_args()

    src_path_map = args.path_to_map
    src_path_log = args.path_to_log

    # Load the map
    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()

    # Initialize motion model
    motion_model = MotionModel()

    # Initialize a single particle randomly in free space
    X_bar = init_particles_freespace(1, occupancy_map)
    particle_state = X_bar[0, 0:3]  # [x, y, theta] in world frame
    
    print(f"Initial particle position: x={particle_state[0]:.2f}, y={particle_state[1]:.2f}, theta={particle_state[2]:.2f}")

    # Store trajectory for visualization
    particle_positions = [particle_state.copy()]

    # Set up visualization
    visualize_map(occupancy_map)
    plt.title('Motion Model Test - Particle Trajectory')
    trajectory_line = None
    current_point = None

    # Open log file and read odometry
    logfile = open(src_path_log, 'r')
    
    first_time_idx = True
    time_idx = 0
    
    # Timing: baseline is 100 timesteps per second = 0.01 seconds per timestep
    # Speed multiplier: divide by speed (2.0 = 2x faster = 0.005s, 0.5 = 2x slower = 0.02s)
    baseline_timestep_duration = 0.01
    timestep_duration = baseline_timestep_duration / args.speed
    print(f"Visualization speed: {args.speed}x (timestep duration: {timestep_duration:.4f}s)")
    
    for line in logfile:
        # Read measurement type (L for laser, O for odometry)
        meas_type = line[0]
        
        # Convert measurement values from string to double
        meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')
        
        # Odometry reading [x, y, theta] in odometry frame (first 3 values)
        odometry_robot = meas_vals[0:3]
        time_stamp = meas_vals[-1]
        
        if first_time_idx:
            u_t0 = odometry_robot
            first_time_idx = False
            continue
        
        u_t1 = odometry_robot
        
        # Apply motion model to move the particle
        x_t0 = particle_state
        x_t1 = motion_model.update(u_t0, u_t1, x_t0)
        
        # Update particle state
        particle_state = x_t1
        particle_positions.append(particle_state.copy())
        
        # Update visualization
        trajectory_line, current_point = update_visualization(occupancy_map, particle_positions, trajectory_line, current_point)
        plt.pause(timestep_duration)
        
        # Update odometry for next iteration
        u_t0 = u_t1
        time_idx += 1
        
        if time_idx % 100 == 0:
            print(f"Time step {time_idx}: particle at x={particle_state[0]:.2f}, y={particle_state[1]:.2f}, theta={particle_state[2]:.2f}")
    
    logfile.close()
    
    print(f"\nTotal timesteps processed: {time_idx}")
    print(f"Final particle position: x={particle_state[0]:.2f}, y={particle_state[1]:.2f}, theta={particle_state[2]:.2f}")
    
    # Keep the plot open
    plt.show(block=True)

