"""Module for exporting city data to various file formats.

This module provides functionality to export city data including roads, buildings, and
elements to JSON files for visualization and integration with external tools.
"""
import json
import math
import os
import random
from typing import Dict

from ..citygen.dataclass.dataclass import Building, Segment


class DataExporter:
    """Manages the export of city data to various file formats.

    This class provides methods to export roads, buildings, and other city elements
    to structured data formats for external use.
    """
    def __init__(self, city_generator):
        """Initialize the data exporter with a city generator.

        Args:
            city_generator: The city generator object containing city data.
        """
        self.city_generator = city_generator

    def export_road_data(self) -> Dict:
        """Export all road data.

        Returns:
            Dictionary containing road segment data.
        """
        roads_data = []

        for segment in self.city_generator.road_manager.roads:
            road = {
                'start': {'x': round(segment.start.x, 4), 'y': round(segment.start.y, 4)},
                'end': {'x': round(segment.end.x, 4), 'y': round(segment.end.y, 4)},
                'is_highway': segment.q.highway
            }
            roads_data.append(road)

        return {'roads': roads_data}

    def export_building_data(self) -> Dict:
        """Export all building data.

        Returns:
            Dictionary containing building data.
        """
        buildings_data = []

        for building in self.city_generator.building_manager.buildings:
            building_data = {
                'center': {'x': round(building.center.x, 4), 'y': round(building.center.y, 4)},
                'rotation': round(building.rotation, 4),
                'type': building.building_type.name,
                'bounds': {
                    'x': round(building.bounds.x, 4),
                    'y': round(building.bounds.y, 4),
                    'width': round(building.bounds.width, 4),
                    'height': round(building.bounds.height, 4),
                    'rotation': round(building.bounds.rotation, 4)
                }
            }
            segment = self.city_generator.building_generator.building_to_segment.get(building)
            if segment:
                building_data['segment_assignment'] = {
                    'start': {'x': round(segment.start.x, 4), 'y': round(segment.start.y, 4)},
                    'end': {'x': round(segment.end.x, 4), 'y': round(segment.end.y, 4)},
                    'angle': round(segment.get_angle(), 4)
                }
            buildings_data.append(building_data)

        return {'buildings': buildings_data}

    def export_element_data(self) -> Dict:
        """Export all element data.

        Returns:
            Dictionary containing element data.
        """
        elements_data = []

        for element in self.city_generator.element_manager.elements:
            element_data = {
                'center': {'x': round(element.center.x, 4), 'y': round(element.center.y, 4)},
                'rotation': round(element.rotation, 4),
                'type': element.element_type.name,
                'bounds': {
                    'x': round(element.bounds.x, 4),
                    'y': round(element.bounds.y, 4),
                    'width': round(element.bounds.width, 4),
                    'height': round(element.bounds.height, 4),
                    'rotation': round(element.bounds.rotation, 4)
                }
            }
            owner = self.city_generator.element_generator.element_to_owner.get(element)
            if isinstance(owner, Building):
                element_data['building_assignment'] = {
                    'building_type': owner.building_type.name,
                    'center': {'x': round(owner.center.x, 4), 'y': round(owner.center.y, 4)},
                    'bounds': {
                        'x': round(owner.bounds.x, 4),
                        'y': round(owner.bounds.y, 4),
                        'width': round(owner.bounds.width, 4),
                        'height': round(owner.bounds.height, 4),
                        'rotation': round(owner.bounds.rotation, 4)
                    }
                }
            elif isinstance(owner, Segment):
                element_data['segment_assignment'] = {
                    'start': {'x': round(owner.start.x, 4), 'y': round(owner.start.y, 4)},
                    'end': {'x': round(owner.end.x, 4), 'y': round(owner.end.y, 4)},
                    'angle': round(owner.get_angle(), 4)
                }
            elements_data.append(element_data)
        return {'elements': elements_data}

    def export_routes_data(self) -> Dict:
        """Export all route data.

        Returns:
            Dictionary containing route data.
        """
        routes_data = []
        for route in self.city_generator.route_manager.routes:
            route_data = {
                'start': {'x': round(route.start.x, 4), 'y': round(route.start.y, 4)},
                'end': {'x': round(route.end.x, 4), 'y': round(route.end.y, 4)},
                'points': [{'x': round(point.x, 4), 'y': round(point.y, 4)} for point in route.points]
            }
            routes_data.append(route_data)

        return {'routes': routes_data}

    def export_to_json(self, output_dir: str):
        """Export all data to JSON files.

        Args:
            output_dir: Directory path where the JSON files will be written.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        road_data = self.export_road_data()
        with open(f'{output_dir}/roads.json', 'w') as f:
            json.dump(road_data, f, indent=2)

        # Export building data
        building_data = self.export_building_data()
        with open(f'{output_dir}/buildings.json', 'w') as f:
            json.dump(building_data, f, indent=2)

        # Export element data
        element_data = self.export_element_data()
        with open(f'{output_dir}/elements.json', 'w') as f:
            json.dump(element_data, f, indent=2)

        self.convert_layout_to_world(output_dir, building_data, road_data, element_data)

        # Export routes data
        routes_data = self.export_routes_data()
        with open(f'{output_dir}/routes.json', 'w') as f:
            json.dump(routes_data, f, indent=2)

    def _calculate_rotation(self, start, end):
        """Calculate rotation angle between two points.

        Args:
            start: Start point coordinates (dict with x and y).
            end: End point coordinates (dict with x and y).

        Returns:
            Rotation angle in degrees.
        """
        dx = end['x'] - start['x']
        dy = end['y'] - start['y']

        angle = math.atan2(dy, dx)

        degrees = math.degrees(angle)
        if degrees < 0:
            degrees += 360
        return degrees

    def _calculate_center(self, start, end):
        """Calculate the center point between two points.

        Args:
            start: Start point coordinates (dict with x and y).
            end: End point coordinates (dict with x and y).

        Returns:
            Center point coordinates.
        """
        return {
            'x': round((start['x'] + end['x']) / 2, 4),
            'y': round((start['y'] + end['y']) / 2, 4)
        }

    @staticmethod
    def _export_point_label(output_dir: str, data: Dict):
        """Export point label data to JSON file.

        Args:
            output_dir: Directory path where the JSON file will be written.
            data: Dictionary containing label and point data.
        """
        with open(f'{output_dir}/point_label.json', 'w') as f:
            json.dump(data, f, indent=2)

    def convert_layout_to_world(self, output_dir: str, buildings_data, roads_data, elements_data):
        """Convert the city layout to a world format compatible with the simulation engine.

        Args:
            output_dir: Directory path where the world JSON file will be written.
            buildings_data: Dictionary containing building data.
            roads_data: Dictionary containing road data.
            elements_data: Dictionary containing element data.
        """
        world_data = {
            'base_map': {
                'name': 'map_1',
                'env_bin': 'gym_citynav\\Binaries\\Win64\\gym_citynav.exe',
                'width': 1000,
                'height': 1000
            },
            'nodes': []
        }

        for road in roads_data['roads']:
            start = road['start']
            end = road['end']

            center = self._calculate_center(start, end)
            rotation = self._calculate_rotation(start, end)

            scale_x = math.sqrt((end['x'] - start['x']) ** 2 + (end['y'] - start['y']) ** 2) * 100 / 20000
            scale_y = 1.0
            obj = {
                'id': f"GEN_Road_{len(world_data['nodes'])}",
                'instance_name': 'BP_Road1_C',
                'properties': {
                    'location': {
                        'x': round(center['x'] * 100, 4),
                        'y': round(center['y'] * 100, 4),
                        'z': 0
                    },
                    'orientation': {
                        'pitch': 0,
                        'yaw': round(rotation, 4),
                        'roll': 0
                    },
                    'scale': {
                        'x': round(scale_x * 0.95, 4),
                        'y': round(scale_y * 0.9, 4),
                        'z': 1.0
                    }
                    # 'scale': {
                    #     'x': 0.88,
                    #     'y': 1.0,
                    #     'z': 1.0
                    # }
                }
            }
            world_data['nodes'].append(obj)

        for building in buildings_data['buildings']:
            obj = {
                'id': f"GEN_{building['type']}_{len(world_data['nodes'])}",
                'instance_name': f"{building['type']}",
                'properties': {
                    'location': {
                        'x': round(building['center']['x'] * 100, 4),
                        'y': round(building['center']['y'] * 100, 4),
                        'z': 0
                    },
                    'orientation': {
                        'pitch': 0,
                        'yaw': round(building['rotation'], 4),
                        'roll': 0
                    },
                    'scale': {
                        'x': 1.0,
                        'y': 1.0,
                        'z': 1.0
                    }
                }
            }
            obj['segment_assignment'] = building['segment_assignment']
            world_data['nodes'].append(obj)

        for element in elements_data['elements']:
            num = random.randint(1, 4)
            obj = {
                'id': f"GEN_{element['type'][:-1]}{num}_{len(world_data['nodes'])}",
                'instance_name': f"{element['type']}",
                'properties': {
                    'location': {
                        'x': round(element['center']['x'] * 100, 4),
                        'y': round(element['center']['y'] * 100, 4),
                        'z': 20
                    },
                    'orientation': {
                        'pitch': 0,
                        'yaw': round(element['rotation'], 4),
                        'roll': 0
                    },
                    'scale': {
                        'x': 1.0,
                        'y': 1.0,
                        'z': 1.0
                    }
                }
            }
            if 'building_assignment' in element:
                obj['building_assignment'] = element['building_assignment']
            if 'segment_assignment' in element:
                obj['segment_assignment'] = element['segment_assignment']
            world_data['nodes'].append(obj)

        output_file = os.path.join(output_dir, 'progen_world.json')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_file, 'w') as f:
            json.dump(world_data, f, indent=2)
