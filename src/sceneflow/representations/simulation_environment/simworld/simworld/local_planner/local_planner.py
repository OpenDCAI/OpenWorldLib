"""Local Planner module: translates high-level plans into simulator actions."""

import math
import time
from threading import Event
from typing import Optional

import numpy as np

from ..llm.a2a_llm import A2ALLM
from ..local_planner.action_space import (HighLevelAction,
                                                 HighLevelActionSpace,
                                                 LowLevelAction,
                                                 LowLevelActionSpace)
from ..local_planner.prompt.prompt import (NAVIGATION_SYSTEM_PROMPT,
                                                  NAVIGATION_USER_PROMPT,
                                                  PARSER_SYSTEM_PROMPT,
                                                  PARSER_USER_PROMPT)
from ..map.map import Map
from ..traffic.base.traffic_signal import TrafficSignalState
from ..utils.logger import Logger
from ..utils.vector import Vector


class LocalPlanner:
    """Converts a high-level plan into low-level navigation actions."""

    def __init__(
        self,
        agent,
        model: A2ALLM,
        max_history_step: int = 3,
        dt: float = 0.1,
        observation_viewmode: str = 'lit',
        rule_based: bool = True,
        exit_event: Event = None
    ):
        """Initialize the Local Planner.

        Args:
            agent: Simulator agent interface.
            model: Language model for instruction parsing.
            max_history_step: Max steps of history to retain.
            camera_id: Camera ID for observations.
            dt: Simulation time step.
            observation_viewmode: Rendering mode for observations.
            rule_based: Whether to use rule-based navigation.
            exit_event: Event to signal when the agent should stop.
        """
        self.model = model
        self.agent = agent
        self.communicator = agent.communicator
        self.max_history_step = max_history_step
        self.camera_id = self.agent.camera_id
        self.observation_viewmode = observation_viewmode
        self.map: Map = self.agent.map
        self.rule_based = rule_based
        self.dt = dt
        self.exit_event = exit_event
        self.logger = Logger.get_logger('LocalPlanner')

        self.action_history = []
        self.last_image = None

    def parse(self, plan: str) -> HighLevelActionSpace:
        """Parse a plan string and execute the resulting actions."""
        user_prompt = PARSER_USER_PROMPT.format(
            position=self.agent.position,
            map=self.map,
            action_list=HighLevelAction.get_action_list(),
            plan=plan
        )

        response, call_time = self.model.generate_instructions(
            system_prompt=PARSER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_format=HighLevelActionSpace,
        )
        self.logger.info(f'Agent {self.agent.id} Response: {response}, call time: {call_time}')

        if response is None:
            self.logger.error('Parse failed, response is None')
            return None

        actions = HighLevelActionSpace.from_json(response)
        self.logger.info(f'Agent {self.agent.id} Actions: {actions}')

        return actions

    def execute(self, actions: HighLevelActionSpace) -> None:
        """Execute a list of actions."""
        action_queue = actions.action_queue
        destination = actions.destination
        object_name = actions.object_name

        # Check if destination is valid
        all_possible_destinations = [node.position for node in self.map.nodes]
        if destination not in all_possible_destinations:
            self.logger.error(f'Agent {self.agent.id} Destination {destination} is not valid')
            return

        for action in action_queue:
            if action == HighLevelAction.NAVIGATE.value:
                self.navigate_to(destination)
            elif action == HighLevelAction.PICK_UP.value:
                self.pick_up(object_name)

    def navigate_to(self, destination: Vector) -> None:
        """Navigate from current position to a given destination."""
        self.logger.info(f'Agent {self.agent.id} Target destination: {destination}')

        if self.rule_based:
            # Get the shortest path from current position to the target destination
            path = self.map.get_shortest_path(self.map.get_closest_node(self.agent.position), self.map.get_closest_node(destination))
            path = [n.position for n in path]
            self.logger.info(f'Agent {self.agent.id} Shortest Path: {path}')
            for point in path:
                self.navigate_rule_based(point)
        else:
            self.navigate_vision_based(destination)

    def navigate_rule_based(self, point: Vector) -> None:
        """Navigate using traffic rules and conditions."""
        self.logger.info(f'Agent {self.agent.id} is navigating to {point}, current position: {self.agent.position}, rule based mode')
        if self.map.traffic_signals:
            current_node = self.map.get_closest_node(self.agent.position)
            if current_node.type == 'intersection':
                traffic_light = None
                min_distance = self.agent.config['traffic.sidewalk_offset'] * 2
                for signal in self.map.traffic_signals:
                    distance = self.agent.position.distance(signal.position)
                    if distance < min_distance:
                        min_distance = distance
                        traffic_light = signal

                if traffic_light is not None:
                    while self.exit_event is None or not self.exit_event.is_set():
                        state = traffic_light.get_state()
                        left_time = traffic_light.get_left_time()
                        if state[1] == TrafficSignalState.PEDESTRIAN_GREEN and left_time > min(15, self.agent.config['traffic.traffic_signal.pedestrian_green_light_duration']):
                            break
                        time.sleep(self.dt)

        time.sleep(2)
        self.communicator.humanoid_move_forward(self.agent.id)
        while not self._walk_arrive_at_waypoint(point) and (self.exit_event is None or not self.exit_event.is_set()):
            while not self._align_direction(point) and (self.exit_event is None or not self.exit_event.is_set()):
                self.communicator.humanoid_stop(self.agent.id)
                angle, turn = self._get_angle_and_direction(point)
                self.communicator.humanoid_rotate(self.agent.id, angle, turn)
                time.sleep(self.dt)
            self.communicator.humanoid_move_forward(self.agent.id)
            time.sleep(self.dt)
        self.communicator.humanoid_stop(self.agent.id)

    def navigate_vision_based(self, point: Vector) -> None:
        """Placeholder for vision-based navigation logic."""
        self.logger.info(f'Agent {self.agent.id} is navigating to {point}, current position: {self.agent.position}, vision based mode')
        while not self._walk_arrive_at_waypoint(point) and (self.exit_event is None or not self.exit_event.is_set()):
            time.sleep(self.dt)

            images = []
            image = self.communicator.get_camera_observation(self.camera_id, self.observation_viewmode, mode='direct')
            if self.last_image is not None:
                images.append(self.last_image)
            else:
                images.append(image)
            images.append(image)
            self.last_image = image

            # Distance calculation remains the same as it uses magnitude
            relative_distance = self.agent.position.distance(point)

            # Calculate relative direction considering UE coordinate system
            # Convert yaw to radians - UE yaw is clockwise from X axis
            current_yaw_rad = math.radians(self.agent.yaw)

            # Calculate target angle in UE coordinates
            dx = point.x - self.agent.position.x
            dy = point.y - self.agent.position.y
            target_yaw_rad = math.atan2(dy, dx)

            # Calculate relative angle
            # Normalize the difference to [-π, π]
            angle = math.degrees(target_yaw_rad - current_yaw_rad)
            if angle > 180:
                angle -= 360
            elif angle < -180:
                angle += 360

            action_str = f'I was at {self.agent.position} and I want to go to {point}. The relative distance is {relative_distance} cm and the relative angle is {angle} degrees. After I made the decision, '

            user_prompt = NAVIGATION_USER_PROMPT.format(
                current_position=self.agent.position,
                current_direction=self.agent.direction,
                target_position=point,
                relative_distance=relative_distance,
                relative_angle=angle,
                action_history=self.action_history
            )

            response, _ = self.model.generate_instructions(
                system_prompt=NAVIGATION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                images=images,
                response_format=LowLevelActionSpace,
            )

            if response is None:
                self.logger.error('Parse failed, response is None')
                continue

            vlm_action = LowLevelActionSpace.from_json(response)
            self.logger.info(f'Agent {self.agent.id} is taking action {vlm_action}')

            if vlm_action.choice == LowLevelAction.STEP_FORWARD:
                self.communicator.humanoid_step_forward(self.agent.id, vlm_action.duration, vlm_action.direction)
                if vlm_action.direction == 0:
                    action_str += f'I chose to step forward for {vlm_action.duration} seconds.'
                else:
                    action_str += f'I chose to step backward for {vlm_action.duration} seconds.'

                _human_collision, _object_collision, _building_collision = self.communicator.get_collision_number(self.agent.id)
                if _human_collision > 0 or _object_collision > 0 or _building_collision > 0:
                    action_str += 'But I have collided with something.'

                self.last_position = Vector(self.agent.position.x, self.agent.position.y)  # Create new Vector instance

            elif vlm_action.choice == LowLevelAction.TURN_AROUND:
                clockwise = 'right' if vlm_action.clockwise else 'left'
                action_str += f'I chose to turn {clockwise} {vlm_action.angle} degrees.'
                self.communicator.humanoid_rotate(self.agent.id, vlm_action.angle, clockwise)

            elif vlm_action.choice == LowLevelAction.DO_NOTHING:
                action_str += 'I chose to do nothing.'

            self.action_history.append(action_str)
            if len(self.action_history) > self.max_history_step:
                self.action_history.pop(0)

    def _walk_arrive_at_waypoint(self, waypoint: Vector) -> bool:
        """Return True if humanoid is within threshold of waypoint."""
        threshold = self.agent.config['user.waypoint_distance_threshold']
        if self.agent.position.distance(waypoint) < threshold:
            self.logger.info(f'Agent {self.agent.id} Arrived at {waypoint}')
            return True
        return False

    def _get_angle_and_direction(
        self,
        waypoint: Vector,
    ) -> tuple[float, Optional[str]]:
        """Compute angle and turn direction to face the waypoint."""
        to_wp = waypoint - self.agent.position
        angle = math.degrees(
            math.acos(np.clip(self.agent.direction.dot(to_wp.normalize()), -1, 1))
        )
        cross = self.agent.direction.cross(to_wp)
        turn_direction = 'left' if cross < 0 else 'right'
        if angle < 2:
            return 0.0, None
        return angle, turn_direction

    def _align_direction(self, waypoint: Vector) -> bool:
        """Return True if facing the waypoint within a small angle."""
        to_wp = waypoint - self.agent.position
        angle = math.degrees(
            math.acos(np.clip(self.agent.direction.dot(to_wp.normalize()), -1, 1))
        )
        return angle < 5

    def pick_up(self, object_name: str) -> None:
        """Pick up an object."""
        ret = self.communicator.humanoid_pick_up_object(self.agent.id, object_name)
        self.logger.info(f'Agent {self.agent.id} picked up {object_name}: {ret}')