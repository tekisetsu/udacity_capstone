import rospy

from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, brake_deadband, decel_limit, accel_limit, wheel_radius,
                 wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
        # instanciate the yaw controller
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)
        self.acceleration_controller = PID(kp=0.3, ki=0.1, kd=0.1, mn=0., mx=0.2)

        # Because the current velocity is noisy we create a low pass filter
        tau = 0.5
        ts = 0.02
        self.velocity_lpf = LowPassFilter(tau, ts)

        # binding some variables to the controller
        self.vehicle_mass = vehicle_mass
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius

        # state variable
        self.last_time = rospy.get_time()

    def control(self, bdw_enabled, linear_velocity, angular_velocity, current_velocity):

        # if not enabled reset
        if not bdw_enabled:
            self.acceleration_controller.reset()
            return 0., 0., 0.

        # filter current velocity
        current_velocity = self.velocity_lpf.filt(current_velocity)

        # get current steering
        steering = self.yaw_controller.get_steering(linear_velocity, angular_velocity, current_velocity)

        # getting current throttle
        velocity_error = linear_velocity - current_velocity
        current_time = rospy.get_time()
        sample_time = current_time - self.last_time

        acceleration = self.acceleration_controller.step(velocity_error, sample_time)

        # throttle case
        throttle = acceleration
        brake = 0

        # hold the car at the same place (a = 1m.s-2)
        if linear_velocity == 0 and current_velocity < 0.1:
            throttle = 0
            brake = 700  # value adjusted to Carla

        # break case
        elif acceleration < 0.1 and velocity_error < 0:
            throttle = 0
            decel = max(acceleration, self.decel_limit)
            brake = self.vehicle_mass*abs(decel)*self.wheel_radius  # in Torques

        self.last_time = current_time
        return throttle, brake, steering
