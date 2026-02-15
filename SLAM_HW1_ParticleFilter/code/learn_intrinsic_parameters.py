from sensor_model import SensorModel
import numpy as np
import argparse
from map_reader import MapReader

def learn_intrinsic_parameters(Z_samples, odometry_samples, beam_sample_rate, laser_offset, tol=1e-3, max_iterations=100):
    """
    Learn the intrinsic parameters of the sensor model.
    """
    z_hit = 0.5
    z_short = 0.1
    z_max = 0.3
    z_rand = 0.1
    sigma_hit = 50
    lambda_short = 0.1

    N = len(Z_samples.flatten())

    e_hit = np.zeros(N)
    e_short = np.zeros(N)
    e_max = np.zeros(N)
    e_rand = np.zeros(N)

    
    for iter in range(max_iterations):
        Z_star_samples = []
        Z_flat = []
        idx = 0

        e_hit.fill(0)
        e_short.fill(0)
        e_max.fill(0)
        e_rand.fill(0)

        for Z_sample, odometry_sample in zip(Z_samples, odometry_samples):
            x_laser = odometry_sample[0] + laser_offset * np.cos(odometry_sample[2])
            y_laser = odometry_sample[1] + laser_offset * np.sin(odometry_sample[2])
            for i in range(len(Z_sample)):
                theta_beam = odometry_sample[2] + i * beam_sample_rate * np.pi / 180 - np.pi/2
                Z_star = sensor_model.get_predicted_range(x_laser, y_laser, theta_beam)
                Z_star_samples.append(Z_star)
                Z_flat.append(Z_sample[i])
        
                hit_inv_sigma = 1.0 / (2.0 * sigma_hit * sigma_hit)
                hit_gaussian_norm = 1.0 / (np.sqrt(2.0 * np.pi) * sigma_hit)
                p_hit = sensor_model.compute_hit_likelihood(Z_sample[i], Z_star, hit_inv_sigma, hit_gaussian_norm, sigma_hit)
                p_short = sensor_model.compute_short_likelihood(Z_sample[i], Z_star, lambda_short)
                p_max = sensor_model.compute_max_likelihood(Z_sample[i])
                p_rand = sensor_model.compute_rand_likelihood(Z_sample[i])
                
                eta = 1.0/(z_hit * p_hit + z_short * p_short + z_max * p_max + z_rand * p_rand + 1e-12)

                e_hit[idx] = eta * z_hit * p_hit
                e_short[idx] = eta * z_short * p_short
                e_max[idx] = eta * z_max * p_max
                e_rand[idx] = eta * z_rand * p_rand 

                idx += 1

        z_hit_new = np.mean(e_hit)
        z_short_new = np.mean(e_short)
        z_max_new = np.mean(e_max)
        z_rand_new = np.mean(e_rand)

        Z_star_samples = np.array(Z_star_samples)
        Z_flat = np.array(Z_flat)

        sigma_hit_new = np.sqrt(
            np.sum(e_hit * (Z_flat - Z_star_samples) ** 2) / (np.sum(e_hit) + 1e-12)
        )
        sigma_hit_new = max(sigma_hit_new, 1e-3)

        lambda_short_new = (
            np.sum(e_short) /
            (np.sum(e_short * Z_flat) + 1e-12)
        )

        diff = (
                abs(z_hit - z_hit_new) +
                abs(z_short - z_short_new) +
                abs(z_max - z_max_new) +
                abs(z_rand - z_rand_new) +
                abs(sigma_hit - sigma_hit_new) +
                abs(lambda_short - lambda_short_new)
            )
        if diff < tol:
            break

        z_hit, z_short, z_max, z_rand = (
            z_hit_new, z_short_new, z_max_new, z_rand_new
        )
        sigma_hit = sigma_hit_new
        lambda_short = lambda_short_new

    print(f"Iterations: {iter}")
    print(f"diff: {diff}")

    return {
        "z_hit": z_hit,
        "z_short": z_short,
        "z_max": z_max,
        "z_rand": z_rand,
        "sigma_hit": sigma_hit,
        "lambda_short": lambda_short,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_map', default='../data/map/wean.dat')
    parser.add_argument('--path_to_logs', nargs='+', default=['../data/log/robotdata1.log', '../data/log/robotdata2.log',\
         '../data/log/robotdata3.log', '../data/log/robotdata4.log', '../data/log/robotdata5.log'])
    parser.add_argument('--output', default='../data')
    parser.add_argument('--laser_sample_rate',  default=2, type=int,)
    parser.add_argument('--beam_sample_rate', default=5, type=int)
    args = parser.parse_args()
    
    src_path_map = args.path_to_map
    src_path_logs = args.path_to_logs
    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()
    sensor_model = SensorModel(map_obj)

    sensor_model.directional_ray_table = np.load('../data/directional_ray_table.npy')
    sensor_model.precompute_num_directions = sensor_model.directional_ray_table.shape[2]

    Z_samples = []
    odometry_samples = []

    for src_path_log in src_path_logs:
        logfile = open(src_path_log, 'r')

        laser_scan_count = 0
        for line in logfile:
            meas_type = line[0]
            meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')
            if meas_type == 'L':
                laser_scan_count += 1
                if laser_scan_count % args.laser_sample_rate == 0:
                    odometry_laser = meas_vals[3:6]
                    ranges = meas_vals[6:-1: args.beam_sample_rate]
                    Z_samples.append(ranges)
                    odometry_samples.append(odometry_laser)

    Z_samples = np.array(Z_samples)
    odometry_samples = np.array(odometry_samples)

    print(Z_samples.shape)
    print(odometry_samples.shape)

    intrinsic_parameters = learn_intrinsic_parameters(Z_samples, odometry_samples, args.beam_sample_rate, laser_offset=sensor_model.laser_offset)
    print(intrinsic_parameters)


