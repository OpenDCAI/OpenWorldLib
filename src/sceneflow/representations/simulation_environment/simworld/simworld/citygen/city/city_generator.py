"""City generator module for generating cities with roads, buildings, and elements."""
import random
from enum import Enum, auto

from ...citygen.building.building_generator import BuildingGenerator
from ...citygen.dataclass.dataclass import BuildingType, ElementType
from ...citygen.element.element_generator import ElementGenerator
from ...citygen.road.road_generator import RoadGenerator
from ...citygen.route.route_generator import RouteGenerator
from ...utils.load_json import load_json
from ...utils.logger import Logger


class GenerationState(Enum):
    """Enum to track the generation state."""
    GENERATING_ROADS = auto()
    GENERATING_BUILDINGS = auto()
    GENERATING_ELEMENTS = auto()
    GENERATING_ROUTES = auto()
    COMPLETED = auto()


class CityGenerator:
    """Manages the complete city generation process including roads, buildings, and elements."""

    def __init__(self, config, seed: int = None, num_segments: int = None, generate_element: bool = None, generate_route: bool = None):
        """Initialize the city generator with configuration.

        Args:
            config: Configuration dictionary for the city generation.
            seed: Seed for the random number generator.
            num_segments: Number of road segments to generate.
            generate_element: Whether to generate elements.
            generate_route: Whether to generate routes.
        """
        self.config = config
        random.seed(self.config['simworld.seed'] if seed is None else seed)

        self.num_segments = num_segments if num_segments is not None else self.config['citygen.road.segment_count_limit']

        self.building_types, self.building_colors, self.element_types, \
            self.element_colors, self.element_offsets, self.map_element_offsets = self._load_bounding_boxes()

        # Initialize generator
        self.road_generator = RoadGenerator(self.config, self.num_segments)
        self.building_generator = BuildingGenerator(self.config, self.building_types)
        self.element_generator = ElementGenerator(self.config, self.element_types, self.map_element_offsets)
        self.route_generator = RouteGenerator(self.config)

        # Initialize generation state
        self.generation_state = GenerationState.GENERATING_ROADS

        # for element generation
        self.current_segment_index = 0
        self.current_building_index = 0
        self.current_element_segment_index = 0

        self.input_path = self.config['citygen.input_roads']
        self.input = self.config['citygen.input_layout']

        self.generate_element = generate_element if generate_element is not None else self.config['citygen.element.generation']
        self.generate_route = generate_route if generate_route is not None else self.config['citygen.route.generation']

        self.logger = Logger.get_logger('CityGenerator')

    def generate(self):
        """Generate the city."""
        while not self.is_generation_complete():
            self.generate_step()

    def generate_step(self) -> bool:
        """Generate one step of the city.

        Returns:
            bool: True if generation is complete.
        """
        # generate roads
        if self.generation_state == GenerationState.GENERATING_ROADS:
            # generate roads randomly
            if not self.input:
                if len(self.roads) == 0:
                    self.logger.info('Generating roads randomly')
                    # Initialize road generation
                    self.road_generator.generate_initial_segments()
                # check if the number of roads has reached the limit
                if len(self.roads) >= self.num_segments:
                    self.road_generator.find_intersections()
                    self.generation_state = GenerationState.GENERATING_BUILDINGS
                    return False
                # continue generating roads
                self.road_generator.generate_step()
                return False
            # generate roads from existing file
            else:
                self.logger.info(f'Generating roads from existing file {self.input_path}')
                self.generation_state = GenerationState.GENERATING_BUILDINGS
                return self.road_generator.generate_roads_from_file(self.input_path)

        # generate buildings
        elif self.generation_state == GenerationState.GENERATING_BUILDINGS:
            if self.current_segment_index < len(self.roads):
                segment = self.roads[self.current_segment_index]
                self.building_generator.generate_buildings_along_segment(segment, self.road_quadtree)
                self.current_segment_index += 1
                return False
            else:
                # self.building_generator.filter_overlapping_buildings(self.road_quadtree)  not use anymore
                self.generation_state = GenerationState.GENERATING_ELEMENTS
                self.current_segment_index = 0
                return False

        # generate elements
        elif self.generation_state == GenerationState.GENERATING_ELEMENTS:
            if not self.generate_element:
                self.generation_state = GenerationState.GENERATING_ROUTES
                return False

            if self.current_building_index < len(self.buildings):
                # Generate elements around buildings
                if self.config['citygen.element.generation_thread_number']:
                    thread_num = self.config['citygen.element.generation_thread_number']
                    # generate elements around multiple buildings in one time
                    buildings = self.buildings[self.current_building_index:min(self.current_building_index + thread_num, len(self.buildings))]
                    self.element_generator.generate_elements_around_buildings_multithread(buildings)
                    self.current_building_index += min(thread_num, len(self.buildings) - self.current_building_index)
                else:
                    building = self.buildings[self.current_building_index]
                    self.element_generator.generate_elements_around_building(building)
                    self.current_building_index += 1
                self.element_generator.filter_elements_by_buildings(self.building_quadtree)
                return False
            # Generate elements on roads
            if self.current_element_segment_index < len(self.roads):
                if self.config['citygen.element.generation_thread_number']:
                    thread_num = self.config['citygen.element.generation_thread_number']
                    segments = self.roads[self.current_element_segment_index:min(self.current_element_segment_index + thread_num, len(self.roads))]
                    self.element_generator.generate_elements_on_road_multithread(segments)
                    self.current_element_segment_index += min(thread_num, len(self.roads) - self.current_element_segment_index)
                else:
                    pass
                return False
            else:
                self.generation_state = GenerationState.GENERATING_ROUTES
                return False

        # generate routes
        elif self.generation_state == GenerationState.GENERATING_ROUTES:
            if not self.generate_route:
                self.generation_state = GenerationState.COMPLETED
                return True
            # Generate routes
            self.logger.info('Generating routes')
            target_data_list = []
            for _ in range(self.config['citygen.route.number']):
                target_point = self.route_generator.generate_target_point_randomly()
                target_label = self.route_generator.get_point_around_label(target_point, self.city_quadtrees)
                target_data = {
                    'label': target_label,
                    'point': target_point.to_dict()
                }
                target_data_list.append(target_data)
            self.generation_state = GenerationState.COMPLETED
        return True

    def is_generation_complete(self) -> bool:
        """Check if city generation is complete.

        Returns:
            bool: True if generation is complete.
        """
        return self.generation_state == GenerationState.COMPLETED

    def _load_bounding_boxes(self):
        """Load bounding boxes from configuration file.

        Returns:
            tuple: Building types, building colors, element types, element colors,
                  element offsets, and mapped element offsets.
        """
        data = load_json(self.config['citygen.input_bounding_boxes'])
        buildings = data['buildings']
        elements = data['elements']

        # Define different types of buildings
        BUILDING_TYPES = []
        BUILDING_COLORS = {}

        # Define different types of elements
        ELEMENT_TYPES = []
        ELEMENT_COLORS = {}

        for name, building in buildings.items():
            x = building['bbox']['x'] / 100
            y = building['bbox']['y'] / 100
            num_limit = building.get('num_limit', -1)

            BUILDING_TYPES.append(BuildingType(name, x, y, num_limit))
            BUILDING_COLORS[name] = '#{:06x}'.format(random.randint(0, 0xFFFFFF))

        for name, element in elements.items():
            x = element['bbox']['x'] / 100
            y = element['bbox']['y'] / 100
            ELEMENT_TYPES.append(ElementType(name, x, y))
            ELEMENT_COLORS[name] = '#{:06x}'.format(random.randint(0, 0xFFFFFF))

        ELEMENT_OFFSETS = {}
        for element_type in ELEMENT_TYPES:
            # TODO: add parking offset
            if element_type.name.lower().startswith('bp_tree'):
                ELEMENT_OFFSETS[element_type.name] = self.config['citygen.element.tree_offset']
            else:
                ELEMENT_OFFSETS[element_type.name] = self.config['citygen.element.furniture_offset']

        MAP_ELEMENT_OFFSETS = {}
        for name, offset in ELEMENT_OFFSETS.items():
            MAP_ELEMENT_OFFSETS[offset] = MAP_ELEMENT_OFFSETS.get(offset, []) + [name]

        return BUILDING_TYPES, BUILDING_COLORS, ELEMENT_TYPES, ELEMENT_COLORS, ELEMENT_OFFSETS, MAP_ELEMENT_OFFSETS

    @property
    def city_quadtrees(self):
        """Get all city quadtrees.

        Returns:
            list: List of road, building, and element quadtrees.
        """
        return [self.road_quadtree, self.building_quadtree, self.element_quadtree]

    @property
    def roads(self):
        """Get all roads.

        Returns:
            list: List of road segments.
        """
        return self.road_generator.road_manager.roads

    @property
    def routes(self):
        """Get all routes.

        Returns:
            list: List of routes.
        """
        return self.route_generator.route_manager.routes

    @property
    def road_quadtree(self):
        """Get the road quadtree.

        Returns:
            Quadtree: Quadtree for efficient road queries.
        """
        return self.road_generator.road_manager.road_quadtree

    @property
    def building_quadtree(self):
        """Get the building quadtree.

        Returns:
            Quadtree: Quadtree for efficient building queries.
        """
        return self.building_generator.building_manager.building_quadtree

    @property
    def element_quadtree(self):
        """Get the element quadtree.

        Returns:
            Quadtree: Quadtree for efficient element queries.
        """
        return self.element_generator.element_manager.element_quadtree

    @property
    def intersections(self):
        """Get all intersections.

        Returns:
            list: List of road intersections.
        """
        return self.road_generator.road_manager.intersections

    @property
    def buildings(self):
        """Get all buildings.

        Returns:
            list: List of buildings.
        """
        return self.building_generator.building_manager.buildings

    @property
    def elements(self):
        """Get all elements.

        Returns:
            list: List of elements.
        """
        return self.element_generator.element_manager.elements

    @property
    def road_manager(self):
        """Get the road manager.

        Returns:
            RoadManager: Manager for road operations.
        """
        return self.road_generator.road_manager

    @property
    def building_manager(self):
        """Get the building manager.

        Returns:
            BuildingManager: Manager for building operations.
        """
        return self.building_generator.building_manager

    @property
    def element_manager(self):
        """Get the element manager.

        Returns:
            ElementManager: Manager for element operations.
        """
        return self.element_generator.element_manager

    @property
    def route_manager(self):
        """Get the route manager.

        Returns:
            RouteManager: Manager for route operations.
        """
        return self.route_generator.route_manager
