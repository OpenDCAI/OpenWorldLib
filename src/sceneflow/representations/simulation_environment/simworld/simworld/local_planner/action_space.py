"""ActionSpace module: defines Action enum, Waypoint model, and ActionSpace container."""

import json
from enum import Enum
from typing import Optional

from pydantic import BaseModel

from ..utils.logger import Logger
from ..utils.vector import Vector

logger = Logger.get_logger('ActionSpace')


class HighLevelAction(Enum):
    """High-level actions that an agent can perform."""
    DO_NOTHING = 0
    NAVIGATE = 1
    PICK_UP = 2

    def __str__(self):
        """Return the string representation of the action."""
        return self.name

    def __repr__(self):
        """Return the string representation of the action."""
        return self.__str__()

    @classmethod
    def get_all_actions(cls):
        """Return a formatted string describing all available actions."""
        return f'The action can be one of the following: {", ".join([f"{action.value}: {action.name}" for action in cls])}.'

    @classmethod
    def get_action_list(cls):
        """Return a list of all available actions."""
        return '0: DO_NOTHING. Do nothing.\n1: NAVIGATE. Navigate to the destination. Must specify the destination.\n2: PICK_UP. Pick up the object. Must specify the object.'


class LowLevelAction(Enum):
    """Low-level actions that an agent can perform."""
    DO_NOTHING = 0
    STEP_FORWARD = 1
    TURN_AROUND = 2

    def __str__(self):
        """Return the string representation of the action."""
        return self.name

    def __repr__(self):
        """Return the string representation of the action."""
        return self.__str__()

    @classmethod
    def get_all_actions(cls):
        """Return a formatted string describing all available actions.

        Returns:
            str: A string describing all available actions in the format:
                 "The action can be one of the following: 0: DO_NOTHING, 1: STEP_FORWARD, 2: TURN_AROUND"
        """
        actions = [f'{action.value}: {action.name}' for action in cls]
        return f'The action can be one of the following: {", ".join(actions)}.'

    @classmethod
    def get_action_list(cls):
        """Return a list of all available actions."""
        return [action.value for action in cls]


class LowLevelActionSpace(BaseModel):
    """Low-level action space that an agent can perform."""
    choice: LowLevelAction = LowLevelAction.DO_NOTHING
    duration: Optional[float] = None
    direction: Optional[int] = None
    angle: Optional[float] = None
    clockwise: Optional[bool] = None
    reasoning: Optional[str] = None

    def __str__(self):
        """Return the string representation of the action."""
        return f'LowLevelActionSpace(choice={self.choice}, duration={self.duration}, direction={self.direction}, angle={self.angle}, clockwise={self.clockwise}, reasoning={self.reasoning})'

    def __repr__(self):
        """Return the string representation of the action."""
        return self.__str__()

    @classmethod
    def from_json(cls, json_str):
        """Parse the action space from a json string."""
        if isinstance(json_str, str):
            try:
                json_str = json.loads(json_str)
            except Exception as e:
                logger.warning(f'Parse action space from json failed: {e}, using default action: DO_NOTHING')
                return cls()
        try:
            choice = json_str.get('choice', 0)
            duration = json_str.get('duration', 0) if json_str.get('duration', 0) is not None else None
            direction = json_str.get('direction', 0) if json_str.get('direction', 0) is not None else None
            angle = json_str.get('angle', 0) if json_str.get('angle', 0) is not None else None
            clockwise = json_str.get('clockwise', True) if json_str.get('clockwise', True) is not None else None
            reasoning = json_str.get('reasoning', None) if json_str.get('reasoning', None) is not None else None
            return cls(choice=choice, duration=duration, direction=direction, angle=angle, clockwise=clockwise, reasoning=reasoning)
        except Exception as e:
            logger.warning(f'Parse action space from json failed: {e}, using default action: DO_NOTHING')
            return cls()

    @classmethod
    def to_json_schema(cls):
        """Return the json schema of the action space."""
        return {
            'name': 'LowLevelActionSpace',
            'strict': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'choice': {'type': 'integer', 'description': 'The choice of the action. ' + LowLevelAction.get_all_actions()},
                    'duration': {'type': 'number', 'description': 'The duration of step forward. The duration is the time of the action.'},
                    'direction': {'type': 'number', 'description': 'The direction of step forward. The direction is the direction of the action. 0 means forward, 1 means backward.'},
                    'angle': {'type': 'number', 'description': 'The angle of turn around. The angle is the angle of the action.'},
                    'clockwise': {'type': 'boolean', 'description': 'The clockwise of turn around. The clockwise is the direction of the action.'},
                    'reasoning': {'type': 'string', 'description': 'The reasoning of the action. The reasoning is the reasoning of the action.'},
                },
                'required': ['choice']
            }
        }


class HighLevelActionSpace(BaseModel):
    """High-level action space that an agent can perform."""
    action_queue: Optional[list[int]] = None
    destination: Optional[Vector] = None
    object_name: Optional[str] = None
    reasoning: Optional[str] = None

    def __str__(self):
        """Return the string representation of the action."""
        return f'HighLevelActionSpace(destination={self.destination}, object_name={self.object_name}, action_queue={self.action_queue}, reasoning={self.reasoning})'

    def __repr__(self):
        """Return the string representation of the action."""
        return self.__str__()

    @classmethod
    def from_json(cls, json_str):
        """Parse the action space from a json string."""
        if isinstance(json_str, str):
            try:
                json_str = json.loads(json_str)
            except Exception as e:
                logger.warning(f'Parse action space from json failed: {e}, using default action: DO_NOTHING')
                return cls()
        try:
            destination = Vector(json_str.get('destination', [0, 0])) if json_str.get('destination', [0, 0]) is not None else None
            reasoning = json_str.get('reasoning', None) if json_str.get('reasoning', None) is not None else None
            object_name = json_str.get('object_name', None) if json_str.get('object_name', None) is not None else None
            action_queue = json_str.get('action_queue', None) if json_str.get('action_queue', None) is not None else None
            return cls(destination=destination, object_name=object_name, action_queue=action_queue, reasoning=reasoning)
        except Exception as e:
            logger.warning(f'Parse action space from json failed: {e}, using default action: DO_NOTHING')
            return cls()

    @classmethod
    def to_json_schema(cls):
        """Return the json schema of the action space."""
        return {
            'name': 'HighLevelActionSpace',
            'strict': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'destination': {'type': 'string', 'description': 'The destination of the navigate action. You should specify the destination of the navigate action.'},
                    'object_name': {'type': 'string', 'description': 'The name of the object to interact with.'},
                    'action_queue': {'type': 'array', 'items': {'type': 'integer'}, 'description': 'A list of actions (index of the action) to be performed.'},
                    'reasoning': {'type': 'string', 'description': 'The reasoning of your choice.'},
                },
                'required': ['destination', 'action_queue']
            }
        }
