'''
    Test script for ray casting model
    Loads pre-computed ray casts and visualizes them on the occupancy map
'''

import argparse
import numpy as np
import sys
import os

from map_reader import MapReader
from sensor_model import SensorModel
from main import visualize_map

from matplotlib import pyplot as plt


def sample_free_space_positions(num_positions, occupancy_map, map_resolution):
    """
    Sample random positions in free space.
    Returns array of [x, y] positions in world coordinates (cm).
    """
    # Create obstacle mask (matches init_particles_freespace logic)
    obstacle_mask = (occupancy_map > 0.35) | (occupancy_map < 0.0)
    
    # Get free space cells
    free_mask = ~obstacle_mask
    free_cells = np.argwhere(free_mask)
    num_free_cells = free_cells.shape[0]
    
    if num_free_cells == 0:
        raise ValueError("No free space found in the map!")
    
    # Sample random free space cells
    sampled_indices = np.random.choice(num_free_cells, num_positions, replace=True)
    sampled_cells = free_cells[sampled_indices]
    
    # Convert to world coordinates (cm)
    positions = np.zeros((num_positions, 2))
    positions[:, 0] = sampled_cells[:, 1] * map_resolution  # x coordinate
    positions[:, 1] = sampled_cells[:, 0] * map_resolution  # y coordinate
    
    return positions


def lookup_ray_distance(x_world, y_world, theta, directional_ray_table, map_resolution, 
                        num_directions, max_range):
    """
    Lookup ray distance from pre-computed ray table.
    
    Args:
        x_world, y_world: Position in world coordinates (cm)
        theta: Direction angle in radians
        directional_ray_table: Pre-computed ray table (H, W, num_directions)
        map_resolution: Map resolution in cm per grid cell
        num_directions: Number of pre-computed directions
        max_range: Maximum ray range
    
    Returns:
        Ray distance in cm
    """
    # Convert to grid coordinates
    x_grid = int(x_world / map_resolution)
    y_grid = int(y_world / map_resolution)
    
    H, W, _ = directional_ray_table.shape
    
    # Check bounds
    if x_grid < 0 or x_grid >= W or y_grid < 0 or y_grid >= H:
        return max_range
    
    # Normalize angle to [0, 2*pi)
    theta_normalized = theta % (2.0 * np.pi)
    
    # Convert to direction index
    direction_index = int(theta_normalized / (2.0 * np.pi) * num_directions) % num_directions
    
    # Lookup distance
    distance = directional_ray_table[y_grid, x_grid, direction_index]
    
    return distance


def plot_rays_on_map(occupancy_map, positions, rays, map_resolution):
    """
    Plot rays on the occupancy map.
    
    Args:
        occupancy_map: Occupancy map array
        positions: Array of [x, y] positions in world coordinates (cm)
        rays: List of ray data, each containing [x, y, theta, distance]
        map_resolution: Map resolution in cm per grid cell
    """
    # Display the occupancy map
    visualize_map(occupancy_map)
    plt.title('Ray Casting Visualization')
    
    # Plot each ray
    for i, (x_start, y_start, theta, distance) in enumerate(rays):
        # Convert start position to grid coordinates (for plotting)
        x_start_grid = x_start / map_resolution
        y_start_grid = y_start / map_resolution
        
        # Calculate end position
        x_end = x_start + distance * np.cos(theta)
        y_end = y_start + distance * np.sin(theta)
        
        # Convert end position to grid coordinates
        x_end_grid = x_end / map_resolution
        y_end_grid = y_end / map_resolution
        
        # Plot the ray
        plt.plot([x_start_grid, x_end_grid], [y_start_grid, y_end_grid], 
                'r-', linewidth=1.5, alpha=0.7)
        
        # Plot start point
        plt.plot(x_start_grid, y_start_grid, 'bo', markersize=4)
    
    plt.xlabel('X (grid cells)')
    plt.ylabel('Y (grid cells)')
    plt.show(block=True)


if __name__ == '__main__':
    """
    Test the ray casting model by loading pre-computed rays and visualizing them.
    """
    parser = argparse.ArgumentParser(description='Test ray casting model visualization')
    parser.add_argument('--path_to_map', default='../data/map/wean.dat',
                       help='Path to map file')
    parser.add_argument('--path_to_ray_table', default='../data/directional_ray_table.npy',
                       help='Path to pre-computed ray table')
    parser.add_argument('--num_rays', default=10, type=int,
                       help='Number of rays to sample per position')
    parser.add_argument('--num_positions', default=1, type=int,
                       help='Number of random positions to sample')
    args = parser.parse_args()
    
    src_path_map = args.path_to_map
    src_path_ray_table = args.path_to_ray_table
    
    # Check if ray table exists
    if not os.path.exists(src_path_ray_table):
        print(f"Error: Ray table file not found at {src_path_ray_table}")
        print("Please run main.py first to generate the ray table, or specify the correct path.")
        sys.exit(1)
    
    # Load the map
    print(f"Loading map from {src_path_map}...")
    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()
    map_resolution = 10  # cm per grid cell (from MapReader)
    
    # Initialize sensor model to get map properties
    sensor_model = SensorModel(map_obj)
    
    # Load pre-computed ray table
    print(f"Loading ray table from {src_path_ray_table}...")
    directional_ray_table = np.load(src_path_ray_table)
    num_directions = directional_ray_table.shape[2]
    print(f"Ray table shape: {directional_ray_table.shape}")
    print(f"Number of pre-computed directions: {num_directions}")
    
    # Sample random positions in free space
    print(f"Sampling {args.num_positions} random position(s) in free space...")
    positions = sample_free_space_positions(args.num_positions, occupancy_map, map_resolution)
    
    # Sample rays for each position
    print(f"Sampling {args.num_rays} ray(s) per position...")
    rays = []
    
    for pos_idx, (x_pos, y_pos) in enumerate(positions):
        print(f"Position {pos_idx + 1}: x={x_pos:.2f} cm, y={y_pos:.2f} cm")
        
        # Sample random directions
        for ray_idx in range(args.num_rays):
            # Random direction in [0, 2*pi)
            theta = np.random.uniform(0, 2 * np.pi)
            
            # Lookup ray distance
            distance = lookup_ray_distance(
                x_pos, y_pos, theta,
                directional_ray_table,
                map_resolution,
                num_directions,
                sensor_model._max_range
            )
            
            rays.append([x_pos, y_pos, theta, distance])
            print(f"  Ray {ray_idx + 1}: theta={np.degrees(theta):.1f}°, distance={distance:.2f} cm")
    
    # Plot rays on map
    print("Plotting rays on occupancy map...")
    plot_rays_on_map(occupancy_map, positions, rays, map_resolution)
    
    print("Visualization complete!")

