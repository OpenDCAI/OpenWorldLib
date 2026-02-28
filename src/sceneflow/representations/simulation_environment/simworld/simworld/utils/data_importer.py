"""This module imports the city data and reconstruct the city generator."""
import math
import os
from typing import Dict, List

from ..citygen.city.city_generator import CityGenerator
from ..citygen.dataclass.dataclass import (Bounds, Building,
                                                  BuildingType, Element,
                                                  MetaInfo, Point, Segment)
from ..config import Config
from ..utils.load_json import load_json
from ..utils.logger import Logger


class DataImporter:
    """The DataImporter class imports city data based on the configuration."""
    def __init__(self, config: Config):
        """Initialize the data importer with configuration.

        Args:
            config: The configuration set up in config.yaml.
        """
        self.config = config
        self.city = CityGenerator(config)
        self.logger = Logger.get_logger('DataImporter')

    def import_city_data(self, input_dir: str = None):
        """Rebuild the city_generator with the provided json file and print the result.

        Returns:
            city_generator: The city generator object containing city data.
        """
        progen_world_filepath = os.path.join(input_dir, 'progen_world.json')
        self.import_from_file(progen_world_filepath)
        self.logger.info(f'Imported road segments: {len(self.city.road_manager.roads)}')
        self.logger.info(f'Imported buildings: {len(self.city.building_manager.buildings)}')
        self.logger.info(f'Imported elements: {len(self.city.element_manager.elements)}')
        return self.city

    def import_from_file(self, filepath: str):
        """Rebuild the city_generator with the provided json file.

        Args:
            filepath: The json file that contain the city data.
        """
        self.logger.info(f'Importing city data from {filepath}')
        data = load_json(filepath)
        # import all nodes
        BUILDING_TYPES, _, ELEMENT_TYPES, _, _, _ = self.city._load_bounding_boxes()
        if 'nodes' in data:
            roads = []
            buildings = []
            elements = []
            # handle different classes of nodes
            for node in data['nodes']:
                instance_name = node['instance_name']
                if 'BP_Road1' in instance_name:
                    roads.append(node)
                elif any(keyword in instance_name for keyword in ['BP_Building', 'BP_School', 'BP_Hospital']):
                    buildings.append(node)
                else:
                    elements.append(node)
            if roads:
                self._import_roads(roads)
            if buildings:
                self._import_buildings(buildings, BUILDING_TYPES)
            if elements:
                self._import_elements(elements, ELEMENT_TYPES)
        self.logger.info('Import completed successfully')
        return True

    def _import_roads(self, roads_data: List[Dict]):
        """Import road data based on the city data nodes.

        Args:
            roads_data: The list of nodes of roads in city data.
        """
        for road in roads_data:
            props = road['properties']
            location = props['location']
            orientation = props['orientation']
            scale = props['scale']
            center = Point(
                location['x'] / 100,        # convert to the original coordinate
                location['y'] / 100
            )
            # use orientation['yaw'] and scale to compute the end point
            angle_rad = math.radians(orientation['yaw'])
            length = scale['x'] * 20000 / 100
            length = length / 0.95          # temp scale
            start = Point(
                center.x - length * math.cos(angle_rad) / 2,
                center.y - length * math.sin(angle_rad) / 2
            )
            end = Point(
                start.x + length * math.cos(angle_rad),
                start.y + length * math.sin(angle_rad)
            )
            meta = MetaInfo(
                highway=True if 'Highway' in road['instance_name'] else False,
                t=0.0
            )
            segment = Segment(start, end, meta)
            self.city.road_manager.add_segment(segment)
        self.city.road_manager.rebuild_quadtree()

    def _import_buildings(self, buildings_data: List[Dict], BUILDING_TYPES):
        """Import building data based on the city data nodes.

        Args:
            buildings_data: The list of nodes of buildings in city data.
            BUILDING_TYPES: the set of buildings
        """
        for building in buildings_data:
            props = building['properties']
            location = props['location']
            orientation = props['orientation']
            building_type = next(
                (bt for bt in BUILDING_TYPES if bt.name in building['instance_name']),
                BUILDING_TYPES[0]
            )
            center = Point(
                location['x'] / 100,
                location['y'] / 100
            )
            scale = props['scale']
            # angle_rad = math.radians(orientation['yaw'])
            width = building_type.width * scale['x']
            height = building_type.height * scale['y']
            bounds = Bounds(
                center.x - width / 2,
                center.y - height / 2,
                width,
                height,
                0
            )
            building_obj = Building(
                building_type=building_type,
                bounds=bounds,
                rotation=orientation['yaw']
            )

            segment_data = building['segment_assignment']
            segment = Segment(
                start=Point(x=segment_data['start']['x'], y=segment_data['start']['y']),
                end=Point(x=segment_data['end']['x'], y=segment_data['end']['y'])
            )
            self.city.building_manager.add_building(building_obj)
            self.city.building_generator.building_to_segment[building_obj] = segment

        self.city.building_manager.rebuild_quadtree()

    def _import_elements(self, elements_data: List[Dict], ELEMENT_TYPES):
        """Import element data based on the city data nodes.

        Args:
            elements_data: The list of nodes of elements in city data.
            ELEMENT_TYPES: the set of elements
        """
        for element in elements_data:
            props = element['properties']
            location = props['location']
            orientation = props['orientation']
            # extract instance_name from element type
            instance_name = element['instance_name']
            base_name = instance_name
            element_type = next(
                (dt for dt in ELEMENT_TYPES if dt.name.startswith(base_name)),
                ELEMENT_TYPES[0]        # default type
            )
            center = Point(
                location['x'] / 100,
                location['y'] / 100
            )
            element_obj = Element(
                element_type=element_type,
                bounds=Bounds(
                    center.x - element_type.width / 2,
                    center.y - element_type.height / 2,
                    element_type.width,
                    element_type.height,
                    0
                ),
                rotation=orientation['yaw'],
                building=None
            )
            if 'building_assignment' in element:
                building_data = element['building_assignment']
                building_type = BuildingType(
                    name=building_data['building_type'],
                    width=building_data['bounds']['width'],
                    height=building_data['bounds']['height']
                )

                building = Building(
                    building_type=building_type,
                    bounds=Bounds(
                        x=building_data['bounds']['x'],
                        y=building_data['bounds']['y'],
                        width=building_data['bounds']['width'],
                        height=building_data['bounds']['height'],
                        rotation=building_data['bounds']['rotation']
                    )
                )
                self.city.element_generator.element_to_owner[element_obj] = building

            if 'segment_assignment' in element:
                segment_data = element['segment_assignment']
                segment = Segment(
                    start=Point(x=segment_data['start']['x'], y=segment_data['start']['y']),
                    end=Point(x=segment_data['end']['x'], y=segment_data['end']['y'])
                )
                self.city.element_generator.element_to_owner[element_obj] = segment

            self.city.element_manager.add_element(element_obj)

        self.city.element_manager.rebuild_quadtree()
