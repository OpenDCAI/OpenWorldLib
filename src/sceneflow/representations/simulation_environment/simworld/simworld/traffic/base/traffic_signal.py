"""Traffic signal module for representing traffic lights in the simulation.

This module defines the traffic signal state enumeration and the traffic signal class
used to represent and control traffic lights and pedestrian crossing signals.
"""
from enum import Enum

from ...utils.vector import Vector


class TrafficSignalState(Enum):
    """Enumeration of possible traffic signal states.

    Defines the possible states for both vehicle traffic lights and pedestrian signals.
    """
    VEHICLE_RED = 'vehicle_red'
    VEHICLE_GREEN = 'vehicle_green'
    PEDESTRIAN_RED = 'pedestrian_red'
    PEDESTRIAN_GREEN = 'pedestrian_green'


class TrafficSignal:
    """Represents a traffic signal in the simulation.

    This class models both vehicle traffic lights and pedestrian crossing signals,
    maintaining their state and position in the simulation.
    """
    _id_counter = 0

    def __init__(self, position: Vector, direction: Vector, lane_id: int = None, crosswalk_id: int = None, type: str = 'both'):
        """Initialize a traffic signal.

        Args:
            position: The position of the traffic signal in the simulation.
            direction: The direction the traffic signal is facing.
            lane_id: The ID of the lane the signal controls.
            crosswalk_id: The ID of the crosswalk the signal controls.
            type: The type of signal, either 'both' or 'pedestrian'.
        """
        self.id = TrafficSignal._id_counter
        TrafficSignal._id_counter += 1
        self.lane_id = lane_id
        self.crosswalk_id = crosswalk_id
        self.state = (TrafficSignalState.VEHICLE_RED, TrafficSignalState.PEDESTRIAN_RED)

        self.type = type
        assert self.type in ['both', 'pedestrian'], 'Invalid traffic light type'

        self.position = position
        self.direction = direction

        self.left_time = 0

    @classmethod
    def reset_id_counter(cls):
        """Reset the ID counter for traffic signals.

        Used when resetting the simulation to ensure IDs start from 0.
        """
        cls._id_counter = 0

    def __repr__(self):
        """Return a string representation of the traffic signal.

        Returns:
            A string containing the traffic signal's attributes.
        """
        return f'TrafficSignal(id={self.id}, lane_id={self.lane_id}, crosswalk_id={self.crosswalk_id}, position={self.position}, direction={self.direction}, state={self.state})'

    def set_state(self, state):
        """Set the state of the traffic signal.

        Args:
            state: A tuple of (vehicle_state, pedestrian_state).

        Raises:
            ValueError: If the state is not a valid tuple of TrafficSignalState enums.
        """
        if isinstance(state, tuple) and len(state) == 2:
            self.state = state
        else:
            raise ValueError(f'Invalid state: {state}. Must be a TrafficSignalState enum value')

    def get_state(self):
        """Get the current state of the traffic signal.

        Returns:
            A tuple of (vehicle_state, pedestrian_state).
        """
        return self.state

    def set_left_time(self, left_time):
        """Set the remaining time for the current signal state.

        Args:
            left_time: The time in seconds until the signal changes.
        """
        self.left_time = left_time

    def get_left_time(self):
        """Get the remaining time for the current signal state.

        Returns:
            The time in seconds until the signal changes.
        """
        return self.left_time
