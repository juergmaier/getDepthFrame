# getDepthFrame
create top view of obstacles

A python task using a kinect camera mounted on a robot to create a black and white image of the limiting borders.

The task uses the primesense library to capture the depth frame from the Kinect.

Based on the mounting height of the kinect and a mount angle (pitch) the floor and objects above a maximum height are filtered out

The resulting image is a top view of the closest objects by looking in each column for the shortest distance in the filtered view.

In my instance the obstacle borders are enlarged to account for the footprint of my robot.

The task uses rpyc to accept commands and return the top view image.
