"""Map module: defines Road, Node, Edge, and Map graph structures for navigation."""

import heapq
import random
import sys
from collections import defaultdict
from typing import List

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import QApplication, QWidget

from ..config import Config
from ..utils.load_json import load_json
from ..utils.vector import Vector


class Road:
    """Represents a road segment between two points."""

    def __init__(self, start: Vector, end: Vector):
        """Initialize a Road.

        Args:
            start: Starting position vector.
            end: Ending position vector.
        """
        self.start = start
        self.end = end
        self.direction = (end - start).normalize()
        self.length = start.distance(end)
        self.center = (start + end) / 2

    def __str__(self) -> str:
        """Return a readable string representation of the road."""
        return f'Road(start={self.start}, end={self.end})'

    def __repr__(self) -> str:
        """Alias for __str__."""
        return self.__str__()


class Node:
    """Graph node with a position and type ('sidewalk', 'crosswalk', or 'intersection')."""

    def __init__(self, position: Vector, direction: Vector, type: str = 'sidewalk'):
        """Initialize a Node.

        Args:
            position: Position vector of the node.
            direction: Direction vector of the node.
            type: Node type; 'sidewalk', 'crosswalk', or 'intersection'.
        """
        self.position = position
        self.direction = direction
        self.type = type

    def __str__(self) -> str:
        """Return a readable string representation of the node."""
        return f'Node(position={self.position}, type={self.type})'

    def __repr__(self) -> str:
        """Alias for __str__."""
        return self.__str__()

    def __lt__(self, other):
        """For the purpose of calculating shortest path."""
        return self.position.x < other.position.x

    def __eq__(self, other) -> bool:
        """Compare nodes by position."""
        if not isinstance(other, Node):
            return False
        return self.position == other.position

    def __hash__(self) -> int:
        """Hash node by its position."""
        return hash(self.position)


class Edge:
    """Undirected weighted edge between two nodes."""

    def __init__(self, node1: Node, node2: Node, type: str = 'sidewalk'):
        """Initialize an Edge.

        Args:
            node1: First endpoint.
            node2: Second endpoint.
            type: Edge type; 'sidewalk' or 'crosswalk'.
        """
        self.node1 = node1
        self.node2 = node2
        self.weight = node1.position.distance(node2.position)
        self.type = type

    def __str__(self) -> str:
        """Return a readable string of the edge."""
        return f'Edge(node1={self.node1}, node2={self.node2}, distance={self.weight}, type={self.type})'

    def __repr__(self) -> str:
        """Alias for __str__."""
        return self.__str__()

    def __eq__(self, other) -> bool:
        """Edges are equal if they connect the same positions (unordered)."""
        if not isinstance(other, Edge):
            return False
        p1, p2 = self.node1.position, self.node2.position
        q1, q2 = other.node1.position, other.node2.position
        return (p1 == q1 and p2 == q2) or (p1 == q2 and p2 == q1)

    def __hash__(self) -> int:
        """Hash edge by its sorted endpoint positions."""
        if self.node1.position.x < self.node2.position.x or (
            self.node1.position.x == self.node2.position.x and
            self.node1.position.y <= self.node2.position.y
        ):
            pos1, pos2 = self.node1.position, self.node2.position
        else:
            pos1, pos2 = self.node2.position, self.node1.position
        return hash((pos1, pos2))


class Map:
    """Graph of nodes and edges supporting path queries and random access."""

    def __init__(self, config: Config, traffic_signals: list = None):
        """Initialize an empty Map."""
        self.nodes = set()
        self.edges = set()
        self.roads = []
        self.adjacency_list = defaultdict(list)
        self.config = config
        self.traffic_signals = traffic_signals

    def __str__(self) -> str:
        """Return a summary of nodes and edges."""
        return f'Nodes: {self.nodes}\nEdges: {self.edges}\n'

    def __repr__(self) -> str:
        """Alias for __str__."""
        return self.__str__()

    def initialize_map_from_file(self, roads_file: str = None, sidewalk_offset: float = None, fine_grained: bool = False, num_waypoints_normal: int = 3, waypoints_distance: float = 900, waypoints_normal_distance: float = 150):
        """Initialize the map from the input roads file.

        Args:
            roads_file: Path to the roads file.
            sidewalk_offset: Sidewalk offset.
            fine_grained: Whether to use fine-grained map.
            num_waypoints_normal: Number of waypoints in the normal direction.
            waypoints_distance: Waypoints distance.
            waypoints_normal_distance: Waypoints normal distance.
        """
        file_path = roads_file if roads_file else self.config['map.input_roads']
        side_offset = sidewalk_offset if sidewalk_offset else self.config['traffic.sidewalk_offset']
        roads_data = load_json(file_path)

        road_items = roads_data.get('roads', [])
        road_objects = []
        for road in road_items:
            start = Vector(road['start']['x'] * 100, road['start']['y'] * 100)
            end = Vector(road['end']['x'] * 100, road['end']['y'] * 100)
            road_objects.append(Road(start, end))

        self.roads = road_objects

        for road in road_objects:
            normal = Vector(road.direction.y, -road.direction.x)
            offset = side_offset
            p1 = road.start - normal * offset + road.direction * offset
            p2 = road.end - normal * offset - road.direction * offset
            p3 = road.end + normal * offset - road.direction * offset
            p4 = road.start + normal * offset + road.direction * offset

            nodes = [Node(point, road.direction, 'intersection') for point in (p1, p2, p3, p4)]
            for node in nodes:
                self.add_node(node)

            self.add_edge(Edge(nodes[0], nodes[1], 'sidewalk'))
            self.add_edge(Edge(nodes[2], nodes[3], 'sidewalk'))
            self.add_edge(Edge(nodes[0], nodes[3], 'crosswalk'))
            self.add_edge(Edge(nodes[1], nodes[2], 'crosswalk'))

        self._connect_adjacent_roads(side_offset * 2 + 100)

        if fine_grained:
            self._interpolate_nodes(num_waypoints_normal, waypoints_distance, waypoints_normal_distance, side_offset)

    def get_shortest_path(self, start: Node, end: Node):
        """Get the shortest path between two nodes using A* algorithm. Include the start node and end node in the path.

        Args:
            start: Start node.
            end: End node.

        Returns:
            List of nodes in the shortest path.
        """
        open_heap = []
        heapq.heappush(open_heap, (start.position.distance(end.position), 0, start))
        came_from = {}
        g_score = {start: 0}
        closed_set = set()

        while open_heap:
            _, current_g, current = heapq.heappop(open_heap)
            if current == end:
                return self._reconstruct_path(start, end, came_from)
            if current in closed_set:
                continue
            closed_set.add(current)

            for neighbor in self.adjacency_list.get(current, []):
                tentative_g = g_score[current] + current.position.distance(neighbor.position)
                if neighbor in closed_set and tentative_g >= g_score.get(neighbor, float('inf')):
                    continue
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + neighbor.position.distance(end.position)
                    heapq.heappush(open_heap, (f_score, tentative_g, neighbor))
        return None

    def add_node(self, node: Node) -> None:
        """Add a node to the map.

        Args:
            node: Node to add.
        """
        self.nodes.add(node)
        self.adjacency_list[node] = []

    def add_edge(self, edge: Edge) -> None:
        """Add an edge and update adjacency.

        Args:
            edge: Edge to add.
        """
        self.edges.add(edge)
        self.adjacency_list[edge.node1].append(edge.node2)
        self.adjacency_list[edge.node2].append(edge.node1)

    def get_adjacency_list(self) -> dict:
        """Get the adjacency list mapping each node to its neighbors."""
        return self.adjacency_list

    def get_adjacent_points(self, node: Node) -> List[Vector]:
        """Get neighboring node positions for a given node.

        Args:
            node: Node to get neighbors for.

        Returns:
            List of neighboring node positions.
        """
        return [nbr.position for nbr in self.adjacency_list[node]]

    def get_closest_node(self, position: Vector) -> Node:
        """Find the node nearest to a given position.

        Args:
            position: Position to find nearest node to.

        Returns:
            Nearest node.
        """
        min_distance = float('inf')
        closest_node = None
        for node in self.nodes:
            distance = position.distance(node.position)
            if distance < min_distance:
                min_distance = distance
                closest_node = node
        return closest_node

    def get_random_node(self, type: str = None, exclude: List[Node] = None) -> Node:
        """Get a random node from the map.

        Args:
            type: Node type.
            exclude: List of nodes to exclude.

        Returns:
            A random node from the map.
        """
        if type:
            nodes = [node for node in self.nodes if getattr(node, 'type', None) == type]
        else:
            nodes = list(self.nodes)
        if exclude:
            nodes = [node for node in nodes if node not in exclude]
        return random.choice(nodes)

    def get_nodes_by_type(self, type: str) -> List[Node]:
        """Get all nodes of a given type.

        Args:
            type: Node type.

        Returns:
            List of nodes of the given type.
        """
        return [node for node in self.nodes if getattr(node, 'type', None) == type]

    def has_edge(self, edge: Edge) -> bool:
        """Check if an edge exists in the map.

        Args:
            edge: Edge to check.

        Returns:
            True if the edge exists, False otherwise.
        """
        return edge in self.edges

    def _reconstruct_path(self, start, end, came_from):
        """Reconstruct the path from the end node to the start node using the came_from dictionary.

        Args:
            start: Start node.
            end: End node.
            came_from: Dictionary of nodes and their predecessors.

        Returns:
            List of nodes in the shortest path. If no path is found, return None.
        """
        current = end
        path = [current]
        while current != start:
            # Use the came_from dictionary instead of g_score
            if current not in came_from:
                # If no path is found, return None
                return None
            current = came_from[current]
            path.append(current)
        # Reverse the path, from start to end
        return path[::-1]

    def _connect_adjacent_roads(self, threshold: float) -> None:
        """Link nodes from nearby roads within a threshold."""
        nodes = [node for node in self.nodes if getattr(node, 'type', None) == 'intersection']
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                n1, n2 = nodes[i], nodes[j]
                if (n1.position.distance(n2.position) < threshold and
                        not self.has_edge(Edge(n1, n2))):
                    self.add_edge(Edge(n1, n2, 'crosswalk'))

    def _interpolate_nodes(self, num_waypoints_normal: int, waypoints_distance: float, waypoints_normal_distance: float, sidewalk_offset: float):
        """Interpolate normal nodes between existing nodes along each edge. For each edge, insert a node every waypoints_distance. Each interpolation point is classified as crosswalk or sidewalk based on the length of the edge. Sidewalk points are further classified as near_road, middle, or far_road.

        Args:
            num_waypoints_normal: Number of waypoints in the normal direction.
            waypoints_distance: Waypoints distance.
            waypoints_normal_distance: Waypoints normal distance.
            sidewalk_offset: Sidewalk offset.
        """
        # Copy the current edges to avoid modifying the set during iteration
        original_edges = list(self.edges)

        for edge in original_edges:
            start = edge.node1.position
            end = edge.node2.position
            direction = (end - start).normalize()
            length = start.distance(end)
            num_points = int(length // waypoints_distance)

            is_crosswalk = abs(length - 2 * sidewalk_offset) < 1e-3
            node_type = 'crosswalk' if is_crosswalk else 'sidewalk'

            # The first and last nodes are intersections (only one node)
            intersection_start = edge.node1
            intersection_end = edge.node2

            # Store layers of nodes for connection
            layers = []

            # Add the first intersection node as the first layer (single node)
            layers.append([intersection_start])

            # Insert nodes along the edge
            if num_points < 2:
                pos = start + direction * (length / 2)
                normal = Vector(-direction.y, direction.x)
                # Generate offsets based on num_waypoints_normal
                offsets = []
                if num_waypoints_normal % 2 == 1:  # odd number of points
                    half_points = num_waypoints_normal // 2
                    for i in range(-half_points, half_points + 1):
                        offsets.append(i * waypoints_normal_distance)
                else:  # even number of points
                    half_points = num_waypoints_normal // 2
                    for i in range(-half_points, half_points):
                        offsets.append((i + 0.5) * waypoints_normal_distance)
                nodes = [Node(pos + normal * offset, direction, node_type) for offset in offsets]
                for node in nodes:
                    self.add_node(node)
                layers.append(nodes)
            else:
                for i in range(1, num_points):
                    pos = start + direction * (i * waypoints_distance)
                    normal = Vector(-direction.y, direction.x)
                    # Generate offsets based on num_waypoints_normal
                    offsets = []
                    if num_waypoints_normal % 2 == 1:  # odd number of points
                        half_points = num_waypoints_normal // 2
                        for i in range(-half_points, half_points + 1):
                            offsets.append(i * waypoints_normal_distance)
                    else:  # even number of points
                        half_points = num_waypoints_normal // 2
                        for i in range(-half_points, half_points):
                            offsets.append((i + 0.5) * waypoints_normal_distance)
                    nodes = [Node(pos + normal * offset, direction, node_type) for offset in offsets]
                    for node in nodes:
                        self.add_node(node)
                    layers.append(nodes)

            # Add the last intersection node as the last layer (single node)
            layers.append([intersection_end])

            # Connect nodes between consecutive layers
            for i in range(len(layers) - 1):
                current_layer = layers[i]
                next_layer = layers[i + 1]
                for node_a in current_layer:
                    for node_b in next_layer:
                        # Avoid duplicate edges
                        if not self.has_edge(Edge(node_a, node_b)):
                            if node_a.type == 'sidewalk' or node_b.type == 'sidewalk':
                                self.add_edge(Edge(node_a, node_b, 'sidewalk'))
                            else:
                                self.add_edge(Edge(node_a, node_b, 'crosswalk'))

            # Remove the original long edge
            if edge in self.edges:
                self.edges.remove(edge)
                # Optionally, also remove from adjacency_list if needed
                if edge.node2 in self.adjacency_list.get(edge.node1, []):
                    self.adjacency_list[edge.node1].remove(edge.node2)
                if edge.node1 in self.adjacency_list.get(edge.node2, []):
                    self.adjacency_list[edge.node2].remove(edge.node1)

    def _point_to_segment_distance(self, p, a, b):
        ab = b - a
        ap = p - a
        ab_len2 = ab.x ** 2 + ab.y ** 2
        if ab_len2 == 0:
            return ap.length()
        t = max(0, min(1, (ap.x * ab.x + ap.y * ab.y) / ab_len2))
        proj = a + ab * t
        return (p - proj).length()

    def _nearest_road_distance(self, node):
        return min(self._point_to_segment_distance(node.position, road.start, road.end) for road in self.roads)

    def visualize_by_type(self):
        """Visualize the map by node type."""
        class TypeViewer(QWidget):
            def __init__(self, nodes, edges):
                super().__init__()
                self.nodes = nodes
                self.edges = edges
                self.setMinimumSize(800, 800)
                self.setWindowTitle('Map Visualization by Type')
                self._set_bounds()

                self.scale = 1.0
                self.offset_x = 0
                self.offset_y = 0
                self.last_mouse_pos = None

            def _set_bounds(self):
                self.min_x = min(node.position.x for node in self.nodes)
                self.max_x = max(node.position.x for node in self.nodes)
                self.min_y = min(node.position.y for node in self.nodes)
                self.max_y = max(node.position.y for node in self.nodes)

            def paintEvent(self, event):
                painter = QPainter(self)
                painter.setRenderHint(QPainter.Antialiasing)
                width, height, margin = self.width(), self.height(), 50
                scale_x = (width - 2 * margin) / (self.max_x - self.min_x) if self.max_x > self.min_x else 1
                scale_y = (height - 2 * margin) / (self.max_y - self.min_y) if self.max_y > self.min_y else 1
                base_scale = min(scale_x, scale_y) * self.scale

                painter.translate(self.offset_x, self.offset_y)

                # Draw legend
                self._draw_legend(painter, margin)

                # Draw edges by type
                for edge in self.edges:
                    edge_type = getattr(edge, 'type', 'sidewalk')
                    if edge_type == 'sidewalk':
                        color = QColor(100, 200, 100)
                        width = 2
                    elif edge_type == 'crosswalk':
                        color = QColor(100, 100, 255)
                        width = 3
                    else:
                        color = QColor(200, 200, 200)
                        width = 1

                    painter.setPen(QPen(color, width))
                    x1 = margin + (edge.node1.position.x - self.min_x) * base_scale
                    y1 = margin + (edge.node1.position.y - self.min_y) * base_scale
                    x2 = margin + (edge.node2.position.x - self.min_x) * base_scale
                    y2 = margin + (edge.node2.position.y - self.min_y) * base_scale
                    painter.drawLine(int(x1), int(y1), int(x2), int(y2))

                # Draw nodes by type
                for node in self.nodes:
                    node_type = getattr(node, 'type', None)
                    color = Qt.gray
                    if node_type == 'sidewalk':
                        color = Qt.green
                    elif node_type == 'crosswalk':
                        color = Qt.blue
                    elif node_type == 'intersection':
                        color = Qt.red
                    painter.setPen(QPen(color, 6))
                    x = margin + (node.position.x - self.min_x) * base_scale
                    y = margin + (node.position.y - self.min_y) * base_scale
                    painter.drawPoint(int(x), int(y))

            def wheelEvent(self, event):
                angle = event.angleDelta().y()
                factor = 1.15 if angle > 0 else 0.85
                old_scale = self.scale
                self.scale *= factor
                mouse_pos = event.pos()
                dx = mouse_pos.x() - self.offset_x
                dy = mouse_pos.y() - self.offset_y
                self.offset_x -= dx * (self.scale - old_scale) / old_scale
                self.offset_y -= dy * (self.scale - old_scale) / old_scale
                self.update()

            def mousePressEvent(self, event):
                if event.button() == Qt.LeftButton:
                    self.last_mouse_pos = event.pos()

            def mouseMoveEvent(self, event):
                if self.last_mouse_pos is not None:
                    delta = event.pos() - self.last_mouse_pos
                    self.offset_x += delta.x()
                    self.offset_y += delta.y()
                    self.last_mouse_pos = event.pos()
                    self.update()

            def mouseReleaseEvent(self, event):
                if event.button() == Qt.LeftButton:
                    self.last_mouse_pos = None

            def _draw_legend(self, painter, margin):
                """Draw a legend showing the different types of edges and nodes."""
                # Set font for legend text
                font = painter.font()
                font.setPointSize(10)
                painter.setFont(font)

                # Legend position (top-right corner)
                legend_x = self.width() - 200
                legend_y = margin + 20

                # Draw legend background
                painter.setPen(QPen(QColor(255, 255, 255), 1))
                painter.setBrush(QColor(255, 255, 255, 200))
                painter.drawRect(legend_x - 10, legend_y - 10, 190, 120)

                # Draw legend title
                painter.setPen(QPen(QColor(0, 0, 0), 1))
                painter.drawText(legend_x, legend_y, 'Legend')

                # Draw edge types
                legend_y += 25

                # Sidewalk edge
                painter.setPen(QPen(QColor(100, 200, 100), 2))
                painter.drawLine(legend_x, legend_y, legend_x + 20, legend_y)
                painter.setPen(QPen(QColor(0, 0, 0), 1))
                painter.drawText(legend_x + 25, legend_y + 5, 'Sidewalk')

                # Crosswalk edge
                legend_y += 20
                painter.setPen(QPen(QColor(100, 100, 255), 3))
                painter.drawLine(legend_x, legend_y, legend_x + 20, legend_y)
                painter.setPen(QPen(QColor(0, 0, 0), 1))
                painter.drawText(legend_x + 25, legend_y + 5, 'Crosswalk')

                # Draw node types
                legend_y += 25

                # Sidewalk node
                painter.setPen(QPen(Qt.green, 6))
                painter.drawPoint(legend_x + 10, legend_y)
                painter.setPen(QPen(QColor(0, 0, 0), 1))
                painter.drawText(legend_x + 25, legend_y + 5, 'Sidewalk Node')

                # Crosswalk node
                legend_y += 20
                painter.setPen(QPen(Qt.blue, 6))
                painter.drawPoint(legend_x + 10, legend_y)
                painter.setPen(QPen(QColor(0, 0, 0), 1))
                painter.drawText(legend_x + 25, legend_y + 5, 'Crosswalk Node')

                # Intersection node
                legend_y += 20
                painter.setPen(QPen(Qt.red, 6))
                painter.drawPoint(legend_x + 10, legend_y)
                painter.setPen(QPen(QColor(0, 0, 0), 1))
                painter.drawText(legend_x + 25, legend_y + 5, 'Intersection Node')

        app = QApplication.instance() or QApplication(sys.argv)
        viewer = TypeViewer(list(self.nodes), list(self.edges))
        viewer.show()
        app.exec_()

    def visualize_path(self, path):
        """Visualize the path.

        Args:
            path: Path to visualize. A list of Node instances.
        """
        class PathViewer(QWidget):
            def __init__(self, nodes, edges, path):
                super().__init__()
                self.nodes = nodes
                self.edges = edges
                self.path = path or []
                self.setMinimumSize(800, 800)
                self.setWindowTitle('Map Visualization by Path')
                self._set_bounds()
                self.scale = 1.0
                self.offset_x = 0
                self.offset_y = 0
                self.last_mouse_pos = None

            def _set_bounds(self):
                self.min_x = min(node.position.x for node in self.nodes)
                self.max_x = max(node.position.x for node in self.nodes)
                self.min_y = min(node.position.y for node in self.nodes)
                self.max_y = max(node.position.y for node in self.nodes)

            def paintEvent(self, event):
                painter = QPainter(self)
                painter.setRenderHint(QPainter.Antialiasing)
                width, height, margin = self.width(), self.height(), 50
                scale_x = (width - 2 * margin) / (self.max_x - self.min_x) if self.max_x > self.min_x else 1
                scale_y = (height - 2 * margin) / (self.max_y - self.min_y) if self.max_y > self.min_y else 1
                base_scale = min(scale_x, scale_y) * self.scale

                painter.translate(self.offset_x, self.offset_y)

                # Draw all edges (gray)
                painter.setPen(QPen(QColor(200, 200, 200), 1))
                for edge in self.edges:
                    x1 = margin + (edge.node1.position.x - self.min_x) * base_scale
                    y1 = margin + (edge.node1.position.y - self.min_y) * base_scale
                    x2 = margin + (edge.node2.position.x - self.min_x) * base_scale
                    y2 = margin + (edge.node2.position.y - self.min_y) * base_scale
                    painter.drawLine(int(x1), int(y1), int(x2), int(y2))

                # Draw route edges (highlighted color, e.g. orange-red)
                if self.path and len(self.path) > 1:
                    painter.setPen(QPen(QColor(255, 69, 0), 4))  # OrangeRed
                    for i in range(len(self.path) - 1):
                        n1, n2 = self.path[i], self.path[i+1]
                        x1 = margin + (n1.position.x - self.min_x) * base_scale
                        y1 = margin + (n1.position.y - self.min_y) * base_scale
                        x2 = margin + (n2.position.x - self.min_x) * base_scale
                        y2 = margin + (n2.position.y - self.min_y) * base_scale
                        painter.drawLine(int(x1), int(y1), int(x2), int(y2))

                # Draw all non-route, non-obstacle nodes (gray)
                for node in self.nodes:
                    if node not in self.path:
                        painter.setPen(QPen(QColor(180, 180, 180), 4))
                        x = margin + (node.position.x - self.min_x) * base_scale
                        y = margin + (node.position.y - self.min_y) * base_scale
                        painter.drawPoint(int(x), int(y))

                # Draw route nodes: start (red), end (blue), middle (green)
                if self.path:
                    # Set the middle nodes to green
                    painter.setPen(QPen(QColor(0, 200, 0), 8))
                    for node in self.path[1:-1]:
                        x = margin + (node.position.x - self.min_x) * base_scale
                        y = margin + (node.position.y - self.min_y) * base_scale
                        painter.drawPoint(int(x), int(y))
                    # Set the start node to red
                    painter.setPen(QPen(QColor(255, 0, 0), 10))
                    x = margin + (self.path[0].position.x - self.min_x) * base_scale
                    y = margin + (self.path[0].position.y - self.min_y) * base_scale
                    painter.drawPoint(int(x), int(y))
                    # Set the end node to blue
                    painter.setPen(QPen(QColor(0, 0, 255), 10))
                    x = margin + (self.path[-1].position.x - self.min_x) * base_scale
                    y = margin + (self.path[-1].position.y - self.min_y) * base_scale
                    painter.drawPoint(int(x), int(y))

                # Draw coordinate axes legend
                self._draw_coordinate_axes(painter, margin, base_scale)

            def _draw_coordinate_axes(self, painter, margin, base_scale):
                """Draw coordinate axes with arrows and labels."""
                # Calculate the center of the visible area
                center_x = margin + (self.max_x - self.min_x) * base_scale / 2
                center_y = margin + (self.max_y - self.min_y) * base_scale / 2

                # Draw X-axis (red arrow)
                painter.setPen(QPen(QColor(255, 0, 0), 3))  # Red for X-axis
                axis_length = 100
                painter.drawLine(int(center_x), int(center_y), int(center_x + axis_length), int(center_y))

                # Draw X-axis arrowhead
                arrow_size = 10
                painter.drawLine(int(center_x + axis_length), int(center_y), int(center_x + axis_length - arrow_size), int(center_y - arrow_size))
                painter.drawLine(int(center_x + axis_length), int(center_y), int(center_x + axis_length - arrow_size), int(center_y + arrow_size))

                # Draw Y-axis (green arrow)
                painter.setPen(QPen(QColor(0, 255, 0), 3))  # Green for Y-axis
                painter.drawLine(int(center_x), int(center_y), int(center_x), int(center_y - axis_length))

                # Draw Y-axis arrowhead
                painter.drawLine(int(center_x), int(center_y - axis_length), int(center_x - arrow_size), int(center_y - axis_length + arrow_size))
                painter.drawLine(int(center_x), int(center_y - axis_length), int(center_x + arrow_size), int(center_y - axis_length + arrow_size))

                # Draw labels
                painter.setPen(QPen(QColor(0, 0, 0), 1))
                font = painter.font()
                font.setPointSize(12)
                font.setBold(True)
                painter.setFont(font)

                # X-axis label
                painter.drawText(int(center_x + axis_length + 5), int(center_y + 5), 'X')
                # Y-axis label
                painter.drawText(int(center_x - 15), int(center_y - axis_length - 5), 'Y')

                # Draw legend box
                legend_x = 20
                legend_y = 20
                legend_width = 200
                legend_height = 80

                # Draw legend background
                painter.setPen(QPen(QColor(0, 0, 0), 1))
                painter.setBrush(QColor(255, 255, 255, 200))  # Semi-transparent white
                painter.drawRect(legend_x, legend_y, legend_width, legend_height)

                # Draw legend title
                painter.setPen(QPen(QColor(0, 0, 0), 1))
                font.setPointSize(10)
                font.setBold(True)
                painter.setFont(font)
                painter.drawText(legend_x + 10, legend_y + 20, 'Axis direction')

                # Draw legend items
                font.setPointSize(8)
                font.setBold(False)
                painter.setFont(font)

                # X-axis legend
                painter.setPen(QPen(QColor(255, 0, 0), 2))
                painter.drawLine(legend_x + 10, legend_y + 35, legend_x + 30, legend_y + 35)
                painter.setPen(QPen(QColor(0, 0, 0), 1))
                painter.drawText(legend_x + 35, legend_y + 40, 'X axis')

                # Y-axis legend
                painter.setPen(QPen(QColor(0, 255, 0), 2))
                painter.drawLine(legend_x + 10, legend_y + 55, legend_x + 30, legend_y + 55)
                painter.setPen(QPen(QColor(0, 0, 0), 1))
                painter.drawText(legend_x + 35, legend_y + 60, 'Y axis')

            def wheelEvent(self, event):
                angle = event.angleDelta().y()
                factor = 1.15 if angle > 0 else 0.85
                old_scale = self.scale
                self.scale *= factor
                mouse_pos = event.pos()
                dx = mouse_pos.x() - self.offset_x
                dy = mouse_pos.y() - self.offset_y
                self.offset_x -= dx * (self.scale - old_scale) / old_scale
                self.offset_y -= dy * (self.scale - old_scale) / old_scale
                self.update()

            def mousePressEvent(self, event):
                if event.button() == Qt.LeftButton:
                    self.last_mouse_pos = event.pos()

            def mouseMoveEvent(self, event):
                if self.last_mouse_pos is not None:
                    delta = event.pos() - self.last_mouse_pos
                    self.offset_x += delta.x()
                    self.offset_y += delta.y()
                    self.last_mouse_pos = event.pos()
                    self.update()

            def mouseReleaseEvent(self, event):
                if event.button() == Qt.LeftButton:
                    self.last_mouse_pos = None

        app = QApplication.instance() or QApplication(sys.argv)
        viewer = PathViewer(list(self.nodes), list(self.edges), path)
        viewer.show()
        app.exec_()

    def sample_cycle(self, start_node, min_len=3, max_len=6, max_attempts=1000):
        """Randomly sample a cycle starting and ending at `start_node` within [min_len, max_len] length.

        Args:
            start_node: The starting node.
            min_len: The minimum length of the cycle.
            max_len: The maximum length of the cycle.
            max_attempts: The maximum number of attempts to sample a cycle.

        Returns:
            A list of Node instances representing the cycle, or None if not found.
        """

        def dfs(current, path, visited):
            if len(path) > max_len:
                return None
            neighbors = list(self.adjacency_list[current])
            random.shuffle(neighbors)  # randomize exploration
            for neighbor in neighbors:
                if neighbor == start_node and len(path) >= min_len:
                    return path + [start_node]
                if neighbor not in visited:
                    result = dfs(neighbor, path + [neighbor], visited | {neighbor})
                    if result:
                        return result
            return None

        for _ in range(max_attempts):
            result = dfs(start_node, [start_node], {start_node})
            if result:
                return result
        return None

    def get_road_at_position(self, position: Vector) -> Road:
        """Find the road that contains the given position.

        Args:
            position: Position vector to check.

        Returns:
            The Road object if position is on any road.
        """
        if not self.roads:
            return None

        min_distance = float('inf')
        closest_road = None

        for road in self.roads:
            distance = self._point_to_segment_distance(position, road.start, road.end)
            if distance < min_distance:
                min_distance = distance
                closest_road = road

        return closest_road

    def get_road_info_at_position(self, position: Vector) -> dict:
        """Get detailed information about the road at the given position.

        Args:
            position: Position vector to check.

        Returns:
            Dictionary containing road information and distance, or None if no road found.
        """
        road = self.get_road_at_position(position)
        if road is None:
            return None

        distance = self._point_to_segment_distance(position, road.start, road.end)

        # Calculate relative position along the road (0.0 = start, 1.0 = end)
        road_vector = road.end - road.start
        road_length = road_vector.length()
        if road_length == 0:
            relative_position = 0.0
        else:
            to_position = position - road.start
            relative_position = (to_position.x * road_vector.x + to_position.y * road_vector.y) / (road_length ** 2)
            relative_position = max(0.0, min(1.0, relative_position))  # Clamp to [0, 1]

        return {
            'road': road,
            'distance_to_road': distance,
            'relative_position': relative_position,
            'is_at_start': relative_position < 0.1,
            'is_at_end': relative_position > 0.9,
            'is_at_middle': 0.1 <= relative_position <= 0.9
        }


if __name__ == '__main__':
    random.seed(42)
    map = Map(Config())
    map.initialize_map_from_file()
    node = map.get_random_node()
    print(node)
    cycle = map.sample_cycle(node, min_len=5, max_len=10)
    print(cycle)
    # map.visualize_path(cycle)
