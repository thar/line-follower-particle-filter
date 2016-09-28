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
The program has a help that shows the tuneables for the simulation with a bref description of its meaning
```
python particle_filter_simulation.py -h
usage: particle_filter_simulation.py [-h] [-p PARTICLES] [--no-draw]
                                     [--lines-image LINES_IMAGE]
                                     [--trajectory-image TRAJECTORY_IMAGE]
                                     [--start-point START_POINT START_POINT START_POINT]
                                     [--forward-speed FORWARD_SPEED]
                                     [--forward-noise FORWARD_NOISE]
                                     [--turn-noise TURN_NOISE]
                                     [--sense-noise SENSE_NOISE]
                                     [--draw-loop-time DRAW_LOOP_TIME]

optional arguments:
  -h, --help            show this help message and exit
  -p PARTICLES, --particles PARTICLES
                        Number of particles in the simulation. Default is 1000
  --no-draw             Disables the graphical representation of the filter
  --lines-image LINES_IMAGE
                        Path to the lines map image
  --trajectory-image TRAJECTORY_IMAGE
                        Path to the desired trajectory map image
  --start-point START_POINT START_POINT START_POINT
                        Start point in the lines map in the form
                        x,y,orientation
  --forward-speed FORWARD_SPEED
                        Number of cm the robot will move forward in each
                        filter loop
  --forward-noise FORWARD_NOISE
                        Error to the forward movement. Default is 0.5
  --turn-noise TURN_NOISE
                        Error to the turn movement. Default is 0.1
  --sense-noise SENSE_NOISE
                        Error in the sensors read. Default is 100.0
  --draw-loop-time DRAW_LOOP_TIME
                        Time in ms between filter loops when in draw mode.
                        Default is 10.0
```

####  map change
The map can be changed from the command line by setting the path to the new map files.
Use the parameters **--lines-image**, **--trajectory-image** and **--start-point**
The start point must be the coordinates of the pixel the robot will start from. It is needed to also set the orientation in radians.
The orientation must be set in radians.

> **Note:**
> See the Map section to know the maps files requirements.

####  Performance check
The simulation can be executed without a graphical representation of the movement. If executed this way an output with the time it takes the PC to execute each simulation loop is printed when exiting. To do so run with the parameter **--no-draw**.


####  Number of particles in the filter
It is possible to set the number of particles used in the simulation with the parameter **-p**.
More particles usualy tend to make the algorithm more robust. If too few particles are set the robot will get lost and the particles will diverge from the robot position.

#### Robot speed
The simulation moves the robot position in each loop. The ammount of forward movement (in cm) that is applied can be set with the parameter **--forward-speed**. A too high value will make the robot simulation not to be able to follow the wanted trajectory.

#### Noise parameters
All the noise parameters refere to the error the measurements of the robot always have. Higher values will imply that the robot readings are worse, so the particles will separete more from the actual robot position.
If the error is too high the robot will get lost when running the simulation.
An increase in the number of particles is supposed to help make the system more robust against high error values.
Play with the parameters **--forward-noise**, **--turn-noise** and **--sense-noise** to view how the particles are affected in the simulation

#### Simulation speed
The time between two loops can be modified by setting the parameter **--draw-loop-time**. Higher values will make the simulation to run slower, so it will be possible follow easily the particles in the simulation screen.
