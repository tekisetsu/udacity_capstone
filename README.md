## Goals


This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. In this project, we will implement Carla's different ROS nodes that you can see in the following system diagram

![architecture](report_resources/final-project-ros-graph-v2.png)

## Set up

Please use **one** of the two installation options, either native **or** docker installation.

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
* [Install Docker](https://docs.docker.com/engine/installation/)

* Build the docker container
```bash
docker build . -t capstone
```

* Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Port Forwarding

The system integration project uses its own simulator which will interface with your ROS code and has traffic light detection. You can download the simulator [here](https://github.com/udacity/CarND-Capstone/releases). To improve performance while using a VM, we recommend downloading the simulator for your host operating system and using this outside of the VM. You will be able to run project code within the VM while running the simulator natively in the host using port forwarding on port 4567

### Usage

1. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
2. Make and run styx
```bash
cd ros
catkin_make &&
source devel/setup.sh &&
roslaunch launch/styx.launch
```
3. Run the simulator

<!--catkin_make && source devel/setup.sh && roslaunch launch/styx.launch -->

## Red light detector

### In the simulator

Because there is nothing red but the red lights in the simulator, we can use open cv methods and count the number of red pixels in the camera's image to detect red lights. Indeed training a NN for red light detection in a simulator is really overkill. Deeplearning comes handy if we the red lights come under different light conditions, positions, form and also also types (e.g red light for going left or right)

### Real world testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images
