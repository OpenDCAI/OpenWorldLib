"""Vehicle agent module for simulating vehicles in traffic."""
from enum import Enum, auto

import numpy as np

from .base_agent import BaseAgent
from ..traffic.ad_algorithm.pid_controller import PIDController
from ..utils.traffic_utils import cal_waypoints
from ..utils.vector import Vector


class VehicleState(Enum):
    """Enumeration of possible vehicle states."""
    WAITING = auto()  # waiting for making u-turn
    MAKING_U_TURN = auto()  # making u-turn
    MOVING = auto()  # moving
    STOPPED = auto()  # stopped for red light or avoiding collision


class Vehicle(BaseAgent):
    """Vehicle agent for traffic simulation."""

    _id_counter = 0

    def __init__(self, position: Vector, direction: Vector, current_lane, vehicle_reference: str, config, length: float = 500, width: float = 200):
        """Initialize a vehicle agent.

        Args:
            position: Initial position vector.
            direction: Initial direction vector.
            current_lane: Lane where the vehicle starts.
            vehicle_reference: Reference identifier for the vehicle.
            config: Configuration dictionary.
            length: Vehicle length in units.
            width: Vehicle width in units.
        """
        super().__init__(position, direction)
        self.id = Vehicle._id_counter
        Vehicle._id_counter += 1

        self.config = config

        # movement attributes
        self.current_lane = current_lane
        self.waypoints = []

        self.state = VehicleState.STOPPED

        self.steering_pid = PIDController(k_p=self.config['traffic.vehicle.steering_pid.kp'], k_i=self.config['traffic.vehicle.steering_pid.ki'], k_d=self.config['traffic.vehicle.steering_pid.kd'])

        # vehicle attributes
        self.vehicle_reference = vehicle_reference
        self.length = length
        self.width = width
        self.max_steering = self.config['traffic.vehicle.max_steering']

        self.throttle = 0
        self.brake = 0
        self.steering = 0

    @classmethod
    def reset_id_counter(cls):
        """Reset the vehicle ID counter to zero."""
        cls._id_counter = 0

    def __str__(self):
        """Return a string representation of the vehicle."""
        return f'Vehicle(id={self.id}, position={self.position}, direction={self.direction}, yaw={self.yaw})'

    def __repr__(self):
        """Return a detailed string representation of the vehicle."""
        return f'Vehicle(id={self.id}, current_lane={self.current_lane.id}, position={self.position}, direction={self.direction}, yaw={self.yaw}, waypoints={self.waypoints})'

    def compute_control(self, waypoint, dt):
        """Compute throttle, brake, steering.

        Args:
            waypoint: Target waypoint to calculate control values.
            dt: Time delta for PID controller.

        Returns:
            tuple: Throttle, brake, steering values, and a boolean indicating control change.
        """
        target_x, target_y = waypoint.x, waypoint.y

        # Compute target yaw
        delta_x = target_x - self.position.x
        delta_y = target_y - self.position.y
        target_yaw = np.degrees(np.arctan2(delta_y, delta_x))

        # Compute yaw error
        yaw_error = target_yaw - self.yaw
        if yaw_error > 180:
            yaw_error -= 360
        elif yaw_error < -180:
            yaw_error += 360

        # Calculate normal distance to lane
        lane_direction = self.current_lane.direction
        vehicle_direction = self.direction
        lane_start = self.current_lane.start
        vehicle_to_lane = self.position - lane_start
        normal_distance = abs(vehicle_to_lane.cross(lane_direction))

        angle_diff = abs(np.degrees(np.arccos(lane_direction.dot(vehicle_direction))))

        # Consider both angle difference and normal distance
        if angle_diff < 3 and normal_distance < self.config['traffic.vehicle.lane_deviation']:
            steering = 0
            self.steering_pid.reset()
        else:
            if abs(yaw_error) < 1:
                yaw_error = 0
            steering = self.steering_pid.update(yaw_error, dt)
            steering = np.clip(steering, -self.max_steering, self.max_steering)

        if steering != 0:
            throttle = 0.4
        else:
            throttle = 1
        brake = 0

        # if no changes, don't update
        if throttle == self.throttle and brake == self.brake and steering == self.steering:
            return 0, 0, 0, False

        self.throttle = throttle
        self.brake = brake
        self.steering = steering

        return throttle, brake, steering, True

    def is_close_to_end(self):
        """Check if the vehicle is close to the end of the current lane.

        Returns:
            bool: True if vehicle is close to the end of the lane.
        """
        return self.position.distance(self.current_lane.end) < self.config['traffic.vehicle.distance_to_end'] + self.length / 2

    def is_close_to_object(self, vehicles, pedestrians):
        """Detect objects in the vehicle's path.

        Args:
            vehicles: List of vehicles to check for proximity.
            pedestrians: List of pedestrians to check for proximity.

        Returns:
            bool: True if the vehicle is close to any object.
        """
        # Define detection area (cone-shaped area in front of vehicle)
        DETECTION_ANGLE = self.config['traffic.detection_angle']
        PEDESTRIAN_DETECTION_DISTANCE = self.config['traffic.distance_between_objects'] + self.length / 2

        # Check vehicles
        for vehicle in vehicles:
            VEHICLE_DETECTION_DISTANCE = 1.5 * self.config['traffic.distance_between_objects'] + self.length / 2 + vehicle.length / 2
            if vehicle.id == self.id:
                continue

            # Calculate relative position
            position_diff = vehicle.position - self.position
            distance = self.position.distance(vehicle.position)

            # Check if object is in detection cone
            if distance > VEHICLE_DETECTION_DISTANCE:
                continue

            # Filter out vehicles from opposite lanes (different directions)
            if vehicle.direction.dot(self.direction) <= 0:  # opposite or perpendicular lanes
                continue

            # Calculate angle between vehicle direction and position difference
            angle = abs(np.degrees(np.arccos(np.clip(position_diff.normalize().dot(self.direction), -1, 1))))

            if angle <= DETECTION_ANGLE:
                return True

        # Check pedestrians
        for pedestrian in pedestrians:
            position_diff = pedestrian.position - self.position
            distance = self.position.distance(pedestrian.position)

            if distance > PEDESTRIAN_DETECTION_DISTANCE:
                continue

            angle = abs(np.degrees(np.arccos(np.clip(position_diff.normalize().dot(self.direction), -1, 1))))

            if angle <= DETECTION_ANGLE:
                return True

        return False

    def change_to_next_lane(self, next_lane):
        """Change the vehicle's current lane to the next lane.

        Args:
            next_lane: The lane to change to.
        """
        self.current_lane.vehicles.remove(self)
        self.current_lane = next_lane
        self.current_lane.vehicles.append(self)

        waypoints = cal_waypoints(next_lane.start, next_lane.end, self.config['traffic.gap_between_waypoints'])
        self.add_waypoint(waypoints)

    def add_waypoint(self, waypoint: list[Vector]):
        """Add waypoints to the vehicle's path.

        Args:
            waypoint: List of waypoint vectors to add.
        """
        self.waypoints.extend(waypoint)

    def pop_waypoint(self):
        """Remove and return the first waypoint.

        Returns:
            Vector: The first waypoint.
        """
        return self.waypoints.pop(0)

    def set_attributes(self, throttle: float, brake: float, steering: float):
        """Set the vehicle's control attributes.

        Args:
            throttle: Throttle value.
            brake: Brake value.
            steering: Steering value.
        """
        self.throttle = throttle
        self.brake = brake
        self.steering = steering

    def get_attributes(self):
        """Get the vehicle's current control attributes.

        Returns:
            tuple: Current throttle, brake, and steering values.
        """
        return self.throttle, self.brake, self.steering

    def completed_u_turn(self):
        """Check if the vehicle has completed a U-turn.

        Returns:
            bool: True if the U-turn is completed.
        """
        # Calculate angle between vehicle direction and lane direction
        angle_diff = abs(np.degrees(np.arccos(self.direction.dot(self.current_lane.direction))))
        return angle_diff < 15
