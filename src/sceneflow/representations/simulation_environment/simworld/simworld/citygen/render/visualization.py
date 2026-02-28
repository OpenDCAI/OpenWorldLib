"""City visualization module for rendering and visualizing generated city layouts.

This module provides functionality for loading city data from JSON files and
rendering a visualization of the city layout including roads, buildings, and elements.
"""
import random
from typing import List

import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QLabel, QMainWindow, QVBoxLayout,
                             QWidget)

from ...citygen.dataclass.dataclass import (Bounds, Building,
                                                  BuildingType, Element,
                                                  ElementType)
from ...config import Config
from ...utils.load_json import load_json


class CityData:
    """Container for city visualization data."""

    def __init__(self):
        """Initialize empty city data structures."""
        self.roads: List[dict] = []  # Using dict for roads since no Road class in custom_types
        self.buildings: List[Building] = []
        self.elements: List[Element] = []

    def load_from_files(self, output_dir='output'):
        """Load city data from JSON files.

        Args:
            output_dir: Directory containing the city data JSON files.
        """
        roads_path = f'{output_dir}/roads.json'
        buildings_path = f'{output_dir}/buildings.json'
        elements_path = f'{output_dir}/elements.json'
        try:
            # Load roads
            roads_data = load_json(roads_path)
            self.roads = roads_data['roads']  # Store as raw dict

            # Load buildings
            buildings_data = load_json(buildings_path)
            self.buildings = []
            for b in buildings_data['buildings']:
                building_type = BuildingType(
                    name=b['type'],
                    width=b['bounds']['width'],
                    height=b['bounds']['height']
                )
                bounds = Bounds(
                    x=b['bounds']['x'],
                    y=b['bounds']['y'],
                    width=b['bounds']['width'],
                    height=b['bounds']['height'],
                    rotation=b['bounds']['rotation']
                )
                self.buildings.append(Building(
                    building_type=building_type,
                    bounds=bounds,
                    rotation=b['rotation']
                ))

            # Load elements
            elements_data = load_json(elements_path)
            self.elements = []
            for e in elements_data['elements']:
                element_type = ElementType(
                    name=e['type'],
                    width=e['bounds']['width'],
                    height=e['bounds']['height']
                )
                bounds = Bounds(
                    x=e['bounds']['x'],
                    y=e['bounds']['y'],
                    width=e['bounds']['width'],
                    height=e['bounds']['height'],
                    rotation=e['bounds']['rotation']
                )
                self.elements.append(Element(
                    element_type=element_type,
                    bounds=bounds,
                    rotation=e['rotation']
                ))

            print('Successfully loaded city data')

        except Exception as e:
            print(f'Error loading city data: {e}')


class CityVisualizer(QMainWindow):
    """Visualization renderer for city data."""

    def __init__(self, config: Config, input_dir: str = None):
        """Initialize visualizer.

        Args:
            config: Configuration settings for the visualization.
            input_dir: Directory containing the city data JSON files.
        """
        super().__init__()
        self.setWindowTitle('City Visualization')

        # Create main window widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        main_layout = QVBoxLayout(self.main_widget)

        # Create title label
        self.title_label = QLabel()
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet(
            'QLabel {color: #34495E;font-size: 14px;font-weight: bold;padding: 5px;}'
        )
        main_layout.addWidget(self.title_label)

        # Create plot window
        self.plot_widget = pg.PlotWidget()
        main_layout.addWidget(self.plot_widget)

        # Set plot style
        self.plot_widget.setBackground('#F0F8FF')
        self.plot_widget.showGrid(True, True, alpha=0.3)
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.setXRange(config['citygen.quadtree.bounds.x'], config['citygen.quadtree.bounds.x'] + config['citygen.quadtree.bounds.width'])
        self.plot_widget.setYRange(config['citygen.quadtree.bounds.y'], config['citygen.quadtree.bounds.y'] + config['citygen.quadtree.bounds.height'])

        # 添加这一行来翻转 y 轴
        self.plot_widget.getViewBox().invertY(True)

        # Enable mouse interaction
        self.plot_widget.setMouseEnabled(x=True, y=True)
        self.plot_widget.setMenuEnabled(False)

        # Initialize city data
        self.city = CityData()
        self.city.load_from_files(output_dir=config['citygen.output_dir'] if input_dir is None else input_dir)

        # Set window size
        self.resize(1280, 960)

    def draw_frame(self):
        """Draw current state of the city."""
        self.plot_widget.clear()

        # Draw roads
        highways = []
        normal_roads = []
        for road in self.city.roads:
            points = [
                (road['start']['x'], road['start']['y']),
                (road['end']['x'], road['end']['y']),
            ]
            if road['is_highway']:
                highways.extend(points)
            else:
                normal_roads.extend(points)

        # Draw normal roads
        if normal_roads:
            for i in range(0, len(normal_roads), 2):
                line = pg.PlotDataItem(
                    x=[normal_roads[i][0], normal_roads[i + 1][0]],
                    y=[normal_roads[i][1], normal_roads[i + 1][1]],
                    pen=pg.mkPen('#2E5984', width=1.8),
                    antialias=True,
                )
                self.plot_widget.addItem(line)

        # Draw highways
        if highways:
            for i in range(0, len(highways), 2):
                line = pg.PlotDataItem(
                    x=[highways[i][0], highways[i + 1][0]],
                    y=[highways[i][1], highways[i + 1][1]],
                    pen=pg.mkPen('#1E3F66', width=3.0),
                    antialias=True,
                )
                self.plot_widget.addItem(line)

        # Generate random colors for each building type
        building_colors = {}
        for building in self.city.buildings:
            building_type = building.building_type.name
            if building_type not in building_colors:
                # Generate random RGB values with good visibility
                r = random.randint(100, 255)
                g = random.randint(100, 255)
                b = random.randint(100, 255)
                building_colors[building_type] = f'#{r:02x}{g:02x}{b:02x}'

        # Draw buildings
        for building in self.city.buildings:
            rect = pg.QtWidgets.QGraphicsRectItem(
                building.bounds.x,
                building.bounds.y,
                building.bounds.width,
                building.bounds.height,
            )

            rect.setTransformOriginPoint(
                building.bounds.x + building.bounds.width / 2,
                building.bounds.y + building.bounds.height / 2,
            )
            rect.setRotation(building.rotation)

            building_type = building.building_type.name
            color = building_colors[building_type]
            rect.setPen(pg.mkPen(color, width=2))
            rect.setBrush(pg.mkBrush(color))
            self.plot_widget.addItem(rect)

        # Generate random colors for each element type
        element_colors = {}
        for element in self.city.elements:
            element_type = element.element_type.name
            if element_type not in element_colors:
                # Generate random RGB values with good visibility
                r = random.randint(100, 255)
                g = random.randint(100, 255)
                b = random.randint(100, 255)
                element_colors[element_type] = f'#{r:02x}{g:02x}{b:02x}'

        # Draw elements
        for element in self.city.elements:
            size = min(element.bounds.width, element.bounds.height) * 1
            element_type = element.element_type.name
            color = element_colors[element_type]

            circle = pg.ScatterPlotItem(
                pos=[(element.center.x, element.center.y)],
                size=size,
                pen=pg.mkPen('w'),  # White border
                brush=pg.mkBrush(color),
                symbol='o',
                antialias=True,
                pxMode=False
            )
            self.plot_widget.addItem(circle)

        # Update status bar information
        stats_text = f'ROADS: {len(self.city.roads)} | BUILDINGS: {len(self.city.buildings)} | ELEMENTS: {len(self.city.elements)}'
        self.title_label.setText(stats_text)


def visualize(config: Config, input_dir: str = None):
    """Main function.

    Args:
        config: Configuration settings for the visualization.
        input_dir: Directory containing the city data JSON files.
    """
    app = QApplication([])
    app.setStyle('Fusion')

    visualizer = CityVisualizer(config=config, input_dir=input_dir)
    visualizer.show()
    visualizer.draw_frame()

    app.exec_()


if __name__ == '__main__':
    visualize(config=Config())
