"""Prompt module: defines system and user prompt templates for SimWorld agents."""

PARSER_SYSTEM_PROMPT = """You are a planner for a embodied agent. You are given a plan from the user, you should parse the plan into a list of actions.
"""

PARSER_USER_PROMPT = """You are now at {position} in a city, where the unit is cm. And you have a map of the city structured as a graph with nodes and edges:
{map}

You have the following actions:
{action_list}

You are given a plan in natural language:
{plan}

You should parse the plan into a list of actions. Output your decision in JSON format with only the following keys: action_queue, destination, object_name, reasoning.

Example outputs:

{{"action_queue": [0, 1], "destination": [200, 0], "reasoning": "I need to go to the destination."}}
{{"action_queue": [0, 1, 2], "destination": [100, 100], "object_name": "bottle", "reasoning": "I need to go to the destination and pick up the bottle."}}
"""


NAVIGATION_SYSTEM_PROMPT = """
You are an embodied agent in a 3D simulation environment, where the unit is centimeters (cm). You are good at navigating in a city environment. Your task is to navigate from your current position to a specified destination safely and efficiently.
"""

NAVIGATION_USER_PROMPT = """
You are currently at {current_position} and your direction is {current_direction}. Your final destination is {target_position}. The destination is approximately {relative_distance:.2f} cm away, and the relative angle to it is {relative_angle:.2f} degrees (negative = to your left, positive = to your right). Your walking speed is 200 cm/s.

You are given two images:
- Previous view (1 step ago): what you saw before your last decision.
- Current view: what you see now.
Use these images to understand how your environment is changing and make smart decisions.

You have the following action history (most recent at the bottom):
{action_history}
Before making your decision, you should consider the history of actions.

The relative distance and angle only give you a rough idea of where the destination is. **You must not walk directly toward the destination without checking the surroundings.** Carefully plan your path instead.

Always walk on the **sidewalk**. Do **not** step into the **roadway**, **crash into buildings**, or ** hit obstacles** on your way.

You are required to make a decision for your next action. You have the following options:
- 0: Do nothing.
- 1: Step. Must specify duration (max 5 sec) and direction (0 = forward, 1 = backward).
- 2: Turn. Specify angle (0–180 degrees) and direction (clockwise = true/false).

Output only the action JSON with the following keys: choice, duration, direction, angle, clockwise.

Example outputs:
{{"choice": 0}}
{{"choice": 1, "duration": 3, "direction": 0}}
{{"choice": 2, "angle": 90, "clockwise": true}}
"""
