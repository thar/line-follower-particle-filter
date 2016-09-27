line-follower-particle-filter
===================

This is a particle filter implementation to solve the localization problem in a line following robot.
It is based on the particle filter described in the course "Udacity course Artificial Intelligence for Robotics".
This implementation makes use of numpy and represents the particles as numpy matrices, so the operations can take advantage of the good performance of the numpy matrix operations.

----------

Maps
-------------
Two files are used as a map for the simulation. Both of them must be of same pixels shape

- **track.bmp**: This is the circuit map. Pixels in black represent the lines of the circuit. Other pixels must be white
- **desired_trajectory.bmp**: This is the desired trajectory for the robot in the circuit. It can cross from one line to other. The pixels representing the trajectory must be black. Other pixels must be white

The trajectory will be superposed to the circuit map represented in **track.bmp**, so it is important that both images are aligned.

> **Note:**
> Each pixel in the map represent a surface of 1cm^2

Execution
-------------
The simulator is written in Python, so Python is needed to run it.
There are also some dependencies:

- numpy
- Python Image Library (PIL or Pillow)
- matplotlib

To run the simulation execute the following command from the repository folder:
```
python particle_filter_simulation.py
```
A screen with the map, the robot and the particles will popup.
In order to start the simulation it is needed to input an 'Enter' in the console from were the simulation was executed.
To stop the simulation a 'Ctrl-c' is required in the console from were the simulation was executed.

Simulation tuning
-------------
####  map change
The map itself can be changed overwriting the two files noted in the Map section or changing the files paths in the file **particle_filter_simulation.py**
> **map files path location in the code:**
>  **13** lines_bitmap_path = 'track.bmp'
>  **14** trayectory_bitmap_path = 'desired_trayectory.bmp'

Together with the map, the robot start point must be changed. The start point must be the coordinates of the pixel the robot will start from. It is needed to also set the orientation in radians.
Its values is (x_coordinate, y_coordinate, orientation)
> **robot start position in the code:**
>  **16** robot_start_point_coordinates = (450, 338, 0.)

####  Performance check
The simulation can be executed without a graphical representation of the movement. If executed this way an output with the time it takes the PC to execute each simulation loop is printed when exiting
To run in this mode change the **draw** variable to **False** in the **particle_filter_simulation.py** file
> **draw variable in the code:**
>  **356** draw = True  # Change this to False to get a loop time calculation

####  Number of particles in the filter
It is possible to set the number of particles used in the simulation.
> **Number of particles in the code:**
>  **362** position = simulate_filter(number_of_particles=1000, draw=draw)

Future improvements
-------------
Enable the tuning of parameters from the command line
