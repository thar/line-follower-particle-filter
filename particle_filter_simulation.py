from math import sqrt, pi, degrees, cos, sin, atan2
from PIL import Image
from time import time, sleep
import numpy as np
import matplotlib.pyplot as plt
import threading
import argparse

particles_plot = None
position_plot = None
trajectory_plot = None
program_end = False
draw_loop_time = 0.01


def plot_function(background_image):
    global particles_plot
    global position_plot
    global trajectory_plot

    plt.ion()

    plt.imshow(background_image)

    particles_plot, = plt.plot(-100, -100, 'r.')
    position_plot, = plt.plot(-100, -100, 'b')
    trajectory_plot, = plt.plot(-100, -100, 'y')
    plt.draw()
    while not program_end:
        plt.draw()
        plt.pause(0.001)


class ParticleFilter:
    def __init__(self, lines_map, N=1000, forward_noise=0.2,
                 turn_noise=0.05, sense_noise=100.,
                 distance_to_sensors=10.,
                 number_of_sensors=8, sensors_length=8.,
                 x=0., y=0., o=0.):
        self.lines_map = lines_map
        self.N = N  # number of particles

        self.forward_noise = forward_noise
        self.turn_noise = turn_noise
        self.sense_noise = sense_noise

        self.distance_to_sensors = distance_to_sensors
        self.number_of_sensors = number_of_sensors
        self.sensors_length = sensors_length

        self.sensors_positions = np.reshape(np.linspace(
            -self.sensors_length/2, self.sensors_length/2,
            self.number_of_sensors),
            (1, self.number_of_sensors))

        # particles variables
        # they are N length arrays
        self.x_positions = np.zeros((self.N, 1))
        self.y_positions = np.zeros((self.N, 1))
        self.orientations = np.zeros((self.N, 1))
        self.x_positions.fill(x)
        self.y_positions.fill(y)
        self.orientations.fill(o)
        self.probabilities = np.ones((self.N, 1))

        # Precomputation of values for probability calulation
        self.a = -2. * (self.sense_noise ** 2)
        self.b = self.sense_noise * sqrt(2.0 * pi)

    def set_position(self, px, py, orientacion):
        self.x_positions.fill(px)
        self.y_positions.fill(py)
        self.orientations.fill(orientacion)

    def set_noise(self, forward_noise, turn_noise, sense_noise):
        self.forward_noise = forward_noise
        self.turn_noise = turn_noise
        self.sense_noise = sense_noise

    # move function
    # Takes 2 scalars, forward_movement and turn_movement
    def move(self, forward_movement, turn_movement):
        forward_move_with_noise = np.reshape(
            np.random.normal(0, self.forward_noise, self.N),
            (self.N, 1)) + forward_movement
        turn_move_with_nose = np.reshape(
            np.random.normal(0, self.turn_noise, self.N),
            (self.N, 1)) + turn_movement
        temporal_orientation = self.orientations + turn_move_with_nose/2

        # x_positions, forward_move_with_noise and temporal_orientation
        # are arrays, forward_movement and turn_movement are scalars
        self.x_positions = self.x_positions + \
            (forward_move_with_noise * np.cos(temporal_orientation))
        self.y_positions = self.y_positions + \
            (forward_move_with_noise * np.sin(temporal_orientation))
        self.orientations = (self.orientations + turn_move_with_nose) % \
            (2 * pi)

    # sense function returns an array with each particle weight
    def sense(self, measured_distance=None):
        [x, y] = self._get_sensors_coordinates()

        measurements = \
            self._get_particles_distances_from_sensors_coordinates(x, y)

        # it is possible for the robot not to see a line
        if measured_distance is not None:
            # Update particles probabilities
            self.probabilities = np.exp(np.power(
                (measurements - measured_distance), 2) / self.a) / self.b

        return measurements

    def resample(self):
        sorting_index = np.argsort(self.probabilities, kind='mergesort')
        prob_distribution = np.cumsum(self.probabilities[sorting_index])
        prob_distribution /= prob_distribution[-1]

        probabilities_random = np.random.random(self.N)
        indexes = np.searchsorted(
            prob_distribution, probabilities_random, side='left')

        self.x_positions = self.x_positions[sorting_index[indexes]]
        self.y_positions = self.y_positions[sorting_index[indexes]]
        self.orientations = self.orientations[sorting_index[indexes]]

    # disperse simply adds some noise to particles position
    def disperse(self):
        self.move(0, 0)

    def loop(self, lineal_movement, angular_movement, measured_distance):
        self.move(lineal_movement, angular_movement)
        self.sense(measured_distance)
        self.resample()

    def get_position(self, return_variance=False):
        # orientation is tricky because it is cyclic. By normalizing
        # around the first particle we are somewhat more robust to
        # the 0=2pi problem
        normalized_orientations = (((self.orientations -
                                     self.orientations[0] + pi)
                                    % (2. * pi)) + self.orientations[0] - pi)

        x = self.x_positions.mean()
        y = self.y_positions.mean()
        orientation = normalized_orientations.mean()

        if not return_variance:
            return [x, y, orientation]

        intermediate_variance_matrix = self.x_positions - x
        x_variance = np.dot(intermediate_variance_matrix.ravel(),
                            intermediate_variance_matrix.ravel()) / self.N

        intermediate_variance_matrix = self.y_positions - y
        y_variance = np.dot(intermediate_variance_matrix.ravel(),
                            intermediate_variance_matrix.ravel()) / self.N

        intermediate_variance_matrix = normalized_orientations - orientation
        orientation_variance = np.dot(intermediate_variance_matrix.ravel(),
                                      intermediate_variance_matrix.ravel()) / \
            self.N

        return [x, y, degrees(orientation)], \
            [x_variance, y_variance, orientation_variance]

    def _get_sensors_coordinates(self):
        sensor_bar_orientation = self.orientations + pi/2

        x_pixels = self.x_positions + \
            self.distance_to_sensors * np.cos(self.orientations) + \
            np.cos(sensor_bar_orientation) * self.sensors_positions + \
            0.5
        y_pixels = self.y_positions + \
            self.distance_to_sensors * np.sin(self.orientations) + \
            np.sin(sensor_bar_orientation) * self.sensors_positions + \
            0.5

        # We need these values to be map indexes
        x = x_pixels.ravel().astype(int)
        y = y_pixels.ravel().astype(int)

        # We dont want to get out of map positions
        x[np.where(x < 0)] = 0
        y[np.where(y < 0)] = 0

        lines_map_shape = self.lines_map.shape
        x[np.where(x >= lines_map_shape[1])] = lines_map_shape[1] - 1
        y[np.where(y >= lines_map_shape[0])] = lines_map_shape[0] - 1

        return [x, y]

    def _get_particles_distances_from_sensors_coordinates(self, x, y):
        # N x number_of_sensors array with sensor readings in each position
        map_vaules = np.reshape(self.lines_map[y, x],
                                (self.N, self.number_of_sensors))

        weights = np.arange(1, self.number_of_sensors + 1) * 2000 / \
            self.number_of_sensors
        sensors_bar_value = np.sum(map_vaules, axis=1)
        # It is known that this triggers the next message:
        # RuntimeWarning: invalid value encountered in divide
        measurements = np.nan_to_num(
            np.dot(map_vaules, weights) / sensors_bar_value) - \
            (1000 + 1000./self.number_of_sensors)

        return measurements


class Robot(ParticleFilter):
    def __init__(self, lines_map, x=0., y=0., o=0., distance_to_sensors=10.,
                 number_of_sensors=8, sensors_length=8.,
                 fw_n=0.000001, tr_n=0.000001, ss_n=0.000001):
        ParticleFilter.__init__(self, lines_map, 1, fw_n, tr_n, ss_n,
                                distance_to_sensors,
                                number_of_sensors, sensors_length, x, y, o)

    def plot(self, plot):
        sensors_orientation = self.orientations + pi/2

        x_pixels_s_1 = self.x_positions + \
            self.distance_to_sensors * cos(self.orientations) + \
            cos(sensors_orientation) * np.array([-self.sensors_length / 2, 0])\
            + 0.5
        y_pixels_s_1 = self.y_positions + \
            self.distance_to_sensors * sin(self.orientations) + \
            sin(sensors_orientation) * np.array([-self.sensors_length / 2, 0])\
            + 0.5

        x_pixels_c = self.x_positions + cos(sensors_orientation) * \
            np.array([-self.sensors_length / 2, self.sensors_length / 2]) + \
            0.5
        y_pixels_c = self.y_positions + sin(sensors_orientation) * \
            np.array([-self.sensors_length / 2, self.sensors_length / 2]) + \
            0.5

        x_pixels_s_2 = self.x_positions + \
            self.distance_to_sensors * cos(self.orientations) + \
            cos(sensors_orientation) * np.array([0, self.sensors_length / 2])\
            + 0.5
        y_pixels_s_2 = self.y_positions + \
            self.distance_to_sensors * sin(self.orientations) + \
            sin(sensors_orientation) * np.array([0, self.sensors_length / 2])\
            + 0.5

        x_pixels = np.append(x_pixels_s_1, x_pixels_c)
        x_pixels = np.append(x_pixels, x_pixels_s_2)
        y_pixels = np.append(y_pixels_s_1, y_pixels_c)
        y_pixels = np.append(y_pixels, y_pixels_s_2)

        plot.set_ydata(y_pixels.ravel())
        plot.set_xdata(x_pixels.ravel())

    def __repr__(self):
        [x, y, o] = self.get_position()
        return '[x=%.6s y=%.6s orient=%.6s]' % (x, y, str(degrees(o)))


def simulate_filter(robot_start_point_coordinates,
                    number_of_particles, lines_map, trajectory_map,
                    forward_noise, turn_noise, sense_noise,
                    draw=False):
    particle_filter = ParticleFilter(lines_map, number_of_particles,
                                     forward_noise, turn_noise, sense_noise,
                                     x=robot_start_point_coordinates[0],
                                     y=robot_start_point_coordinates[1],
                                     o=robot_start_point_coordinates[2])
    # Spread the particles around the robot start point
    for i in range(50):
        particle_filter.disperse()

    real_robot = Robot(lines_map, *robot_start_point_coordinates)
    simulated_robot = Robot(trajectory_map, *robot_start_point_coordinates)

    if draw:
        # Wait til the plot is ready
        while particles_plot is None or position_plot is None:
            pass
        particles_plot.set_ydata(particle_filter.y_positions.ravel())
        particles_plot.set_xdata(particle_filter.x_positions.ravel())
        real_robot.plot(position_plot)
        sleep(0.51)
        raw_input('Press Enter to continue...')

    print 'press Ctrl+c to stop execution'

    input_time = time()
    loops_count = 0
    try:
        forward_movement = 4.  # 1 unit is 1cm per scan
        turn_movement = 0.0
        robot_sensors_measurement = 0.

        # Loop until 'Ctrl + c'
        while True:
            particle_filter.loop(forward_movement, turn_movement,
                                 robot_sensors_measurement)

            if draw:
                particles_plot.set_ydata(particle_filter.y_positions.ravel())
                particles_plot.set_xdata(particle_filter.x_positions.ravel())
                real_robot.plot(position_plot)
                sleep(draw_loop_time)

            [x, y, z] = particle_filter.get_position()
            simulated_robot.set_position(x, y, z)
            distance_to_trajectory = simulated_robot.sense()
            if distance_to_trajectory <= -1000:
                distance_to_trajectory = 0

            # TODO: fix the turn_movement calculation
            # The magic numbers (500 and 7) are take from try and error
            # The need here is to compute a simulation of the PID
            # algorithm that makes the robot to follow the trajectory
            turn_movement = atan2((real_robot.sensors_length *
                                   distance_to_trajectory / 500),
                                  real_robot.distance_to_sensors) / 7

            real_robot.move(forward_movement, turn_movement)
            robot_sensors_measurement = real_robot.sense()
            loops_count += 1

    except KeyboardInterrupt:
        pass

    output_time = time()
    if not draw:
        print 'loop time in ms (mean): %f\ntotal execution time: %f' % \
            (1000*(output_time-input_time)/loops_count, output_time-input_time)
    return particle_filter.get_position()


def coords(s):
    try:
        x, y, o = map(float, s.split(','))
        return x, y, o
    except:
        raise argparse.ArgumentTypeError('Coordinates must be x,y,orientation')


def main():
    global program_end
    global draw_loop_time

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--particles', default=1000,
                        help='Number of particles in the simulation. \
                        Default is 1000')
    parser.add_argument('--no-draw', action='store_true',
                        help='Disables the graphical representation of the \
                        filter')
    parser.add_argument('--lines-image', default='track.bmp',
                        help='Path to the lines map image')
    parser.add_argument('--trajectory-image', default='desired_trayectory.bmp',
                        help='Path to the desired trajectory map image')
    parser.add_argument('--start-point', default=(450., 338., 0.),
                        help='Start point in the lines map in the form \
                        x,y,orientation', type=coords, nargs=3)
    parser.add_argument('--forward-speed', default=4.,
                        help='Number of cm the robot will move forward in each \
                        filter loop')
    parser.add_argument('--forward-noise', default=0.5,
                        help='Error to the forward movement. Default is 0.5')
    parser.add_argument('--turn-noise', default=0.1,
                        help='Error to the turn movement. Default is 0.1')
    parser.add_argument('--sense-noise', default=100.,
                        help='Error in the sensors read. Default is 100.0')
    parser.add_argument('--draw-loop-time', default=10.0, type=float,
                        help='Time in ms between filter loops when in draw mode. \
                        Default is 10.0')

    args = parser.parse_args()

    draw_loop_time = args.draw_loop_time / 1000.

    lines_bitmap_path = args.lines_image
    trajectory_bitmap_path = args.trajectory_image

    lines_bitmap = Image.open(lines_bitmap_path)
    lines_map = np.array(lines_bitmap.convert('L'))
    lines_map[np.where(lines_map == 0)] = 1
    lines_map[np.where(lines_map == 255)] = 0

    trajectory_bitmap = Image.open(trajectory_bitmap_path)
    trajectory_map = np.array(trajectory_bitmap.convert('L'))
    trajectory_map[np.where(trajectory_map == 0)] = 1
    trajectory_map[np.where(trajectory_map == 255)] = 0

    background_lines = lines_bitmap.convert('RGBA')
    data = np.array(background_lines)
    red, green, blue, alpha = data.T
    black_areas = (red == 0) & (blue == 0) & (green == 0)
    data[..., :-1][black_areas.T] = (0, 0, 0)  # Transpose back needed

    background_trajectory = trajectory_bitmap.convert('RGBA')
    data_traj = np.array(background_trajectory)
    red, green, blue, alpha = data_traj.T

    # Replace black trajectory with green... (leaves alpha values alone...)
    black_areas = (red == 0) & (blue == 0) & (green == 0)
    data[..., :-1][black_areas.T] = (0, 255, 0)  # Transpose back needed

    background_image = Image.fromarray(data)

    draw = not args.no_draw

    if draw:
        plot_thread = threading.Thread(target=plot_function,
                                       args=(background_image,))
        plot_thread.start()

    simulate_filter(args.start_point, args.particles, lines_map,
                    trajectory_map, args.forward_noise, args.turn_noise,
                    args.sense_noise, draw)

    if draw:
        program_end = True  # This flag stops the plot thread
        plot_thread.join()

    # TODO: Fix the simulation exit when in draw mode that shows
    # RuntimeError: main thread is not in main loop


if __name__ == '__main__':
    main()
