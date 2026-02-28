"""Traffic controller module for coordinating all traffic-related components.

This module provides the main controller for the traffic simulation, handling initialization,
spawning, and coordination of all traffic elements including vehicles, pedestrians, and traffic signals.
"""
import random
import time
import traceback
from collections import defaultdict
from threading import Event
from typing import Callable

from ...agent.pedestrian import Pedestrian
from ...agent.vehicle import Vehicle
from ...communicator.communicator import Communicator
from ...communicator.unrealcv import UnrealCV
from ...config import Config
from ..base.crosswalk import Crosswalk
from ..base.intersection import Intersection
from ..base.road import Road
from ..base.sidewalk import Sidewalk
from ..base.traffic_lane import TrafficLane
from ..base.traffic_signal import (TrafficSignal,
                                                  TrafficSignalState)
from ..manager.intersection_manager import IntersectionManager
from ..manager.pedestrian_manager import PedestrianManager
from ..manager.vehicle_manager import VehicleManager
from ...utils.load_json import load_json
from ...utils.logger import Logger
from ...utils.vector import Vector


class TrafficController:
    """Main controller class for the traffic simulation system.

    Coordinates all aspects of the traffic simulation including road network, vehicles,
    pedestrians, and traffic signals.
    """
    def __init__(self, config: Config, num_vehicles: int = None, num_pedestrians: int = None, map: str = None, seed: int = None, dt: float = None):
        """Initialize the traffic controller with configuration.

        Args:
            config: Configuration object containing all simulation parameters.
            num_vehicles: Number of vehicles to spawn.
            num_pedestrians: Number of pedestrians to spawn.
            map: Path to the map file.
            seed: Seed for the random number generator.
            dt: Time step for the simulation.
        """
        self.config = config
        self.num_vehicles = num_vehicles if num_vehicles is not None else config['traffic.num_vehicles']
        self.num_pedestrians = num_pedestrians if num_pedestrians is not None else config['traffic.num_pedestrians']
        self.map = map if map is not None else config['traffic.map_path']
        self.seed = seed if seed is not None else config['simworld.seed']
        self.dt = dt if dt is not None else config['simworld.dt']
        self.communicator = None

        # logger
        self.logger = Logger.get_logger('TrafficController')
        self.logger.info(f'TrafficController initialized with {self.num_vehicles} vehicles and {self.num_pedestrians} pedestrians')

        # set seed
        random.seed(self.seed)

        # read map from file
        self.init_roadnet_from_file(self.map)

        # initialize controllers
        self.intersection_manager = IntersectionManager(self.intersections, self.config)
        self.vehicle_manager = VehicleManager(self.roads, self.num_vehicles, self.config)
        self.pedestrian_manager = PedestrianManager(self.roads, self.num_pedestrians, self.config)

    # Initialization
    def init_communicator(self, communicator=None):
        """Initialize the communication interface with the simulation environment.

        Args:
            communicator: Optional communicator instance. If None, a new one is created.
        """
        if communicator is None:
            self.communicator = Communicator(UnrealCV(port=9000, ip='127.0.0.1', resolution=(720, 600)))
        else:
            self.communicator = communicator

    def disconnect_communicator(self):
        """Disconnect the communication interface from the simulation environment."""
        self.communicator.disconnect()

    def init_roadnet_from_file(self, file_path):
        """Initialize the road network from a JSON file.

        Args:
            file_path: Path to the JSON file containing road network data.

        Raises:
            ValueError: If the file path is not absolute.
        """
        # if not os.path.isabs(file_path):
        #     raise ValueError(f'Expected an absolute path, got: {file_path}')

        self.logger.info(f'Loading road network from file: {file_path}')

        data = load_json(file_path)

        roads = []
        _intersections = defaultdict(list)

        # Log roads information
        self.logger.info(f'Found {len(data["roads"])} roads in map file')
        for _, road_data in enumerate(data['roads']):
            start_point = Vector(round(road_data['start']['x'])*100, round(road_data['start']['y'])*100)
            end_point = Vector(round(road_data['end']['x'])*100, round(road_data['end']['y'])*100)

            road = Road(start=start_point, end=end_point, num_lanes=self.config['traffic.num_lanes'],
                        lane_offset=self.config['traffic.lane_offset'], intersection_offset=self.config['traffic.intersection_offset'],
                        sidewalk_offset=self.config['traffic.sidewalk_offset'], crosswalk_offset=self.config['traffic.crosswalk_offset'])
            roads.append(road)

            self.logger.info(
                f'Road {road.id}: Start=({start_point.x}, {start_point.y}), '
                f'End=({end_point.x}, {end_point.y})'
            )

            _intersections[Vector(start_point.x, start_point.y)].append(road)
            _intersections[Vector(end_point.x, end_point.y)].append(road)

        for road in roads:
            road.init_road()

        self.roads = roads

        self.logger.info(f'Creating {len(roads)} lanes')
        for lane in self.lanes:
            self.logger.info(f'Lane {lane.id}: Start=({lane.start.x}, {lane.start.y}), End=({lane.end.x}, {lane.end.y}), Road={lane.road_id}')

        self.logger.info(f'Creating {len(self.sidewalks)} sidewalks')
        for sidewalk in self.sidewalks:
            self.logger.info(f'Sidewalk {sidewalk.id}: Start=({sidewalk.start.x}, {sidewalk.start.y}), End=({sidewalk.end.x}, {sidewalk.end.y}), Road={sidewalk.road_id}')

        self.logger.info(f'Creating {len(self.crosswalks)} crosswalks')
        for crosswalk in self.crosswalks:
            self.logger.info(f'Crosswalk {crosswalk.id}: Start=({crosswalk.start.x}, {crosswalk.start.y}), End=({crosswalk.end.x}, {crosswalk.end.y}), Road={crosswalk.road_id}')

        intersections = []
        self.logger.info(f'Creating {len(_intersections)} intersections')
        for i, (intersection_point, connected_roads) in enumerate(_intersections.items()):
            intersection = Intersection(center=intersection_point, roads=connected_roads)
            intersections.append(intersection)
            self.logger.info(
                f'Intersection {intersection.id}: Position=({intersection_point.x}, {intersection_point.y}), '
                f'Connected roads={[road.id for road in connected_roads]}'
            )

        self.intersections = intersections

        self.logger.info('Road network initialization completed')

    def spawn_objects_in_unreal_engine(self):
        """Spawn all traffic-related objects in the Unreal Engine simulation."""
        try:
            self.spawn_vehicles()
            self.spawn_pedestrians()
            self.spawn_traffic_signals()
        except Exception as e:
            self.logger.error(f'Error occurred in {__file__}:{e.__traceback__.tb_lineno}')
            self.logger.error(f'Error type: {type(e).__name__}')
            self.logger.error(f'Error message: {str(e)}')
            self.logger.error('Error traceback:')
            traceback.print_exc()

    def spawn_vehicles(self):
        """Spawn vehicles in the simulation environment."""
        self.vehicle_manager.spawn_vehicles(self.communicator)
        self.logger.info('Vehicles spawned')

    def spawn_pedestrians(self):
        """Spawn pedestrians in the simulation environment."""
        self.pedestrian_manager.spawn_pedestrians(self.communicator)
        self.logger.info('Pedestrians spawned')

    def spawn_traffic_signals(self):
        """Spawn traffic signals in the simulation environment."""
        self.intersection_manager.spawn_traffic_signals(self.communicator)
        self.logger.info('Traffic signals spawned')

    # Reset
    def reset(self, num_vehicles: int, num_pedestrians: int, map: str):
        """Reset the simulation with new parameters.

        Args:
            num_vehicles: New number of vehicles to spawn.
            num_pedestrians: New number of pedestrians to spawn.
            map: Path to the new map file to use.
        """
        self.num_vehicles = num_vehicles
        self.num_pedestrians = num_pedestrians
        self.map = map

        self.communicator.clean_traffic_only(self.vehicles, self.pedestrians, self.traffic_signals)

        Vehicle.reset_id_counter()
        Pedestrian.reset_id_counter()
        TrafficSignal.reset_id_counter()
        Road.reset_id_counter()
        Intersection.reset_id_counter()
        TrafficLane.reset_id_counter()
        Sidewalk.reset_id_counter()
        Crosswalk.reset_id_counter()

        # set seed
        random.seed(self.seed)

        # read map from file
        self.init_roadnet_from_file(self.map)

        # initialize controllers
        self.intersection_manager = IntersectionManager(self.intersections, self.config)
        self.vehicle_manager = VehicleManager(self.roads, self.num_vehicles, self.config)
        self.pedestrian_manager = PedestrianManager(self.roads, self.num_pedestrians, self.config)

    def stop_simulation(self):
        """Stop the traffic simulation."""
        self.logger.info('Stopping simulation')
        self.vehicle_manager.stop_vehicles(self.communicator)
        self.pedestrian_manager.stop_pedestrians(self.communicator)

    # Simulation
    def simulation(self, physical_update_function: Callable, exit_event: Event = None, signal_event: Event = None):
        """Run the traffic simulation continuously.

        Continuously updates the state of all simulation components at fixed time intervals.

        Args:
            physical_update_function: Function to update the physical state of the simulation.
            exit_event: Event to signal the end of the simulation.
            signal_event: Event to signal the end of the simulation.
        """
        try:
            self.logger.info('Starting simulation')
            self.pedestrian_manager.set_pedestrians_max_speed(self.communicator)
            self.intersection_manager.set_traffic_signal_duration(self.communicator)

            while not (exit_event and exit_event.is_set()):
                physical_update_function()
                self.vehicle_manager.update_vehicles(self.communicator, self.intersection_manager, self.pedestrians)
                self.pedestrian_manager.update_pedestrians(self.communicator, self.intersection_manager)
                self.intersection_manager.update_intersections(self.communicator)

                if signal_event is not None:
                    while not signal_event.is_set():
                        time.sleep(self.dt)
                        signal_event.clear()  # Reset the event for next iteration
                else:
                    time.sleep(self.dt)

            self.stop_simulation()
            self.logger.info('Simulation ended')
        except KeyboardInterrupt:
            self.logger.info('Simulation interrupted')
            self.stop_simulation()
        except Exception as e:
            self.logger.error(f'Error occurred in {__file__}:{e.__traceback__.tb_lineno}')
            self.logger.error(f'Error type: {type(e).__name__}')
            self.logger.error(f'Error message: {str(e)}')
            self.logger.error('Error traceback:')
            traceback.print_exc()
            self.stop_simulation()

    def update_states(self):
        """Update the states of all traffic components from the simulation."""
        vehicle_ids = [vehicle.id for vehicle in self.vehicles]
        pedestrian_ids = [pedestrian.id for pedestrian in self.pedestrians]
        traffic_signal_ids = [signal.id for signal in self.traffic_signals]
        result = self.communicator.get_position_and_direction(vehicle_ids, pedestrian_ids, traffic_signal_ids)
        for (type, object_id), values in result.items():
            if type == 'vehicle':
                position, direction = values
                self.vehicles[object_id].position = position
                self.vehicles[object_id].direction = direction
            elif type == 'pedestrian':
                position, direction = values
                self.pedestrians[object_id].position = position
                self.pedestrians[object_id].direction = direction
            elif type == 'traffic_signal':
                is_vehicle_green, is_pedestrian_walk, left_time = values
                for signal in self.traffic_signals:
                    if signal.id == object_id:
                        if is_vehicle_green:
                            signal.set_state((TrafficSignalState.VEHICLE_GREEN, TrafficSignalState.PEDESTRIAN_RED))
                        elif is_pedestrian_walk:
                            signal.set_state((TrafficSignalState.VEHICLE_RED, TrafficSignalState.PEDESTRIAN_GREEN))
                        else:
                            signal.set_state((TrafficSignalState.VEHICLE_RED, TrafficSignalState.PEDESTRIAN_RED))
                        signal.set_left_time(left_time)
                        break

    @property
    def vehicles(self):
        """Get all vehicles in the simulation.

        Returns:
            List of all vehicle objects.
        """
        return self.vehicle_manager.vehicles

    @property
    def pedestrians(self):
        """Get all pedestrians in the simulation.

        Returns:
            List of all pedestrian objects.
        """
        return self.pedestrian_manager.pedestrians

    @property
    def lanes(self):
        """Get all traffic lanes in the simulation.

        Returns:
            List of all traffic lane objects.
        """
        lanes = []
        for road in self.roads:
            lanes.extend(road.lanes.values())
        return lanes

    @property
    def sidewalks(self):
        """Get all sidewalks in the simulation.

        Returns:
            List of all sidewalk objects.
        """
        sidewalks = []
        for road in self.roads:
            sidewalks.extend(road.sidewalks.values())
        return sidewalks

    @property
    def crosswalks(self):
        """Get all crosswalks in the simulation.

        Returns:
            List of all crosswalk objects.
        """
        crosswalks = []
        for road in self.roads:
            crosswalks.extend(road.crosswalks)
        return crosswalks

    @property
    def traffic_signals(self):
        """Get all traffic signals in the simulation.

        Returns:
            List of all traffic signal objects.
        """
        traffic_signals = []
        for intersection in self.intersections:
            traffic_signals.extend(intersection.traffic_lights)
            traffic_signals.extend(intersection.pedestrian_lights)
        return traffic_signals
