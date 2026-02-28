"""This module includes the utilities needed for assets retrieval and placement."""
import json
import math
import os
import random
from functools import lru_cache
from typing import List

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.metrics.pairwise import cosine_similarity

from .clip_embedder import CLIPEmbedder
from .input_parser import InputParser
from ..citygen.dataclass.dataclass import Bounds, Building, Point
from ..utils.load_json import load_json


def get_parsed_input(natural_language_input):
    """Use LLMs to parse the natural language input into 4 parts for post-handling.

    Args:
        natural_language_input: the input text prompt.

    Returns:
        asset_to_place: the asset that user wants to place
        reference_asset: the reference asset of placing
        relation: which direction/relation should be placed relative to reference_asset
        surrounding_assets: the surrounding assets of the user
    """
    inputParser = InputParser()
    parsed_input = inputParser.parse_input(natural_language_input)
    asset_to_place = parsed_input['asset_to_place']
    reference_asset_query = parsed_input['reference_asset']
    relation = parsed_input['relation']
    surroundings_query = parsed_input['surrounding_assets']
    return parsed_input, asset_to_place, reference_asset_query, relation, surroundings_query


@lru_cache(maxsize=1)
def load_instance_desc_map(description_map_path: str):
    """Load the instance description mapping from a JSON file.

    Args:
        description_map_path: the file path to the JSON mapping file.

    Returns:
        A dictionary mapping instance names to their descriptions.
    """
    return load_json(description_map_path)


def get_surroundings(data, description_map_path: str):
    """Generate a surrounding description string based on element and building statistics.

    Args:
        data: the layout statistics containing 'element_stats' and 'building_stats'.
        description_map_path: path to the instance description map JSON file.

    Returns:
        A concatenated string describing nearby elements and buildings.
    """
    details = []
    # print(data)
    for asset, count in data['element_stats'].items():
        details.extend([asset] * count)
    buildings = list(data['building_stats'].keys())
    instance_desc_map = load_instance_desc_map(description_map_path)
    buildings_descriptions = [instance_desc_map.get(name, '') for name in buildings]
    candidate_surroundings_str = ', '.join(details + buildings_descriptions)
    return candidate_surroundings_str


def vector_cosine_similarity(vec1, vec2):
    """Calculate the cosine similarity between two vectors.

    Args:
        vec1: the first vector.
        vec2: the second vector.

    Returns:
        Cosine similarity value between vec1 and vec2.
    """
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot / (norm1 * norm2)


def construct_building_from_candidate(candidate: dict, building_file_path: str) -> Building:
    """Construct a Building object based on candidate asset information and building data.

    Args:
        candidate: dictionary containing instance name and location.
        building_file_path: path to the building file.

    Returns:
        A Building object matching the candidate, or None if no match is found.
    """
    with open(building_file_path, 'r', encoding='utf-8') as f:
        buildings_data = json.load(f)

    candidate_type = candidate.get('instance_name', '')
    candidate_location = candidate.get('properties', {}).get('location', {})
    candidate_x = candidate_location.get('x', 0) / 100.0
    candidate_y = candidate_location.get('y', 0) / 100.0

    threshold: float = 1.0

    for b in buildings_data.get('buildings', []):
        if b.get('type', '') != candidate_type:
            continue
        center = b.get('center', {})
        center_x = center.get('x', 0)
        center_y = center.get('y', 0)
        distance = math.sqrt((center_x - candidate_x) ** 2 + (center_y - candidate_y) ** 2)
        if distance <= threshold:
            b_bounds = b.get('bounds', {})
            bounds = Bounds(
                x=b_bounds.get('x', 0),
                y=b_bounds.get('y', 0),
                width=b_bounds.get('width', 0),
                height=b_bounds.get('height', 0),
                rotation=b_bounds.get('rotation', 0)
            )
            return Building(building_type=candidate_type, bounds=bounds)

    # print('No matching building found.')
    return None


def get_coordinates_around_building(conf, building: Building, relation: str, num_points: int = 1) -> List[Point]:
    """Generate coordinates around a building based on a spatial relation.

    Args:
        conf: configuration dictionary containing offset values.
        building: the reference Building object.
        relation: spatial relation such as 'front', 'back', 'left', or 'right'.
        num_points: number of coordinate points to generate.

    Returns:
        A list of Point objects representing the generated coordinates.
    """
    r = math.radians(building.rotation)
    cx, cy = building.center.x, building.center.y

    offset = conf['citygen.element.element_building_distance']
    variation_ratio = 0.3
    random_variation = lambda: random.uniform(-variation_ratio, variation_ratio) * offset

    dirs = ['front', 'back', 'left', 'right']
    if relation.lower() not in dirs:
        relation = random.choice(dirs)

    if relation.lower() == 'front':
        distance = building.height / 2 + offset
        dvec = (math.cos(r), math.sin(r))
        pvec = (-math.sin(r), math.cos(r))
        span = building.width
    elif relation.lower() == 'back':
        distance = building.height / 2 + offset
        dvec = (-math.cos(r), -math.sin(r))
        pvec = (-math.sin(r), math.cos(r))
        span = building.width
    elif relation.lower() == 'left':
        distance = building.width / 2 + offset
        dvec = (-math.sin(r), math.cos(r))
        pvec = (-math.cos(r), -math.sin(r))
        span = building.height
    elif relation.lower() == 'right':
        distance = building.width / 2 + offset
        dvec = (math.sin(r), -math.cos(r))
        pvec = (math.cos(r), math.sin(r))
        span = building.height

    base_x = cx + distance * dvec[0]
    base_y = cy + distance * dvec[1]

    points = []
    if num_points == 1:
        x = base_x + random_variation()
        y = base_y + random_variation()
        points.append(Point(x, y))
    else:
        effective_span = span * 0.5
        for i in range(num_points):
            if num_points == 1:
                offset_along = 0
            else:
                offset_along = -effective_span / 2 + (effective_span * i / (num_points - 1))
            x = base_x + offset_along * pvec[0] + random_variation()
            y = base_y + offset_along * pvec[1] + random_variation()
            points.append(Point(x, y))
    return points


def retrieve_target_asset(assets_description, folder_path: str, model_ID: str):
    """Retrieve relevant assets from a local folder using CLIP-based similarity to the given description.

    Args:
        assets_description: a list of textual keywords describing the desired asset.
        folder_path: path to the folder containing asset images.
        model_ID: identifier of the CLIP model to use.

    Returns:
        A list of asset names that best match the description.
    """
    embedder = CLIPEmbedder(model_ID)
    image_data_df = create_image_dataframe(folder_path)
    image_data_df['img_embeddings'] = image_data_df['image'].apply(embedder.get_image_embedding)

    related_assets = get_related_assets(assets_description, model_ID, image_data_df)
    return related_assets


def place_target_asset(related_assets, positions, output_dir: str):
    """Save the selected assets and their positions to a JSON file for placement in the scene.

    Args:
        related_assets: list of asset names to be placed.
        positions: list of Point objects representing where to place each asset.
        output_dir: directory to save the output JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)

    json_output = generate_json(related_assets, positions)

    output_file = os.path.join(output_dir, 'simple_world_asset.json')
    with open(output_file, 'w') as f:
        f.write(json_output)


def get_local_images(folder_path: str) -> list:
    """Retrieve all image file paths from the given folder and its subdirectories.

    Args:
        folder_path: path to the folder containing images.

    Returns:
        A list of image file paths with supported extensions.
    """
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff')):
                image_paths.append(os.path.join(root, file))
    return image_paths


def load_image(image_path: str):
    """Load an image from the specified path and convert it to RGB or RGBA.

    Args:
        image_path: path to the image file.

    Returns:
        A PIL Image object, or None if loading fails.
    """
    try:
        image = Image.open(image_path)
        if image.mode == 'P' and 'transparency' in image.info:
            image = image.convert('RGBA')
        else:
            image = image.convert('RGB')
        return image
    except (FileNotFoundError, UnidentifiedImageError):
        return None


def create_image_dataframe(folder_path: str) -> pd.DataFrame:
    """Create a DataFrame containing image paths and PIL Image objects.

    Args:
        folder_path: folder containing the images.

    Returns:
        A pandas DataFrame with valid images and their paths.
    """
    image_paths = get_local_images(folder_path)
    data = {
        'image_path': image_paths,
        'image': [load_image(path) for path in image_paths]
    }
    df = pd.DataFrame(data)
    df = df[df['image'].notnull()]
    return df


def generate_random_properties(x, y):
    """Generate random asset properties for placement, including location, orientation, and scale.

    Args:
        x: x-coordinate of the asset.
        y: y-coordinate of the asset.

    Returns:
        A dictionary representing asset properties.
    """
    return {
        'location': {
            'x': x,
            'y': y,
            'z': 0
        },
        'orientation': {
            'pitch': 0,
            'yaw': round(random.uniform(0, 360), 2),
            'roll': 0
        },
        'scale': {
            'x': round(random.uniform(0.8, 1.2), 2),
            'y': round(random.uniform(0.8, 1.2), 2),
            'z': 1.0
        }
    }


def generate_json(assets, positions):
    """Generate a JSON structure for placing assets in the scene.

    Args:
        assets: list of asset names.
        positions: list of Point objects with asset positions.

    Returns:
        A formatted JSON string representing the scene layout.
    """
    data = {'nodes': []}

    for idx, asset in enumerate(assets):
        node = {
            'id': f'BP_GEN_{asset}_{idx}',
            'instance_name': asset + '_C',
            'properties': generate_random_properties(positions[idx].x, positions[idx].y)
        }
        data['nodes'].append(node)

    return json.dumps(data, indent=2)


def get_top_assets(keyword: str, model_ID: str, image_data_df: pd.DataFrame, top_K: int = 1) -> list:
    """Retrieve top-k most relevant assets for a keyword using CLIP similarity.

    Args:
        keyword: the textual description to match.
        model_ID: identifier of the CLIP model to use.
        image_data_df: DataFrame with image embeddings.
        top_K: number of top results to return.

    Returns:
        A list of top matching asset names.
    """
    embedder = CLIPEmbedder(model_ID)
    query_vect = embedder.get_text_embedding(keyword)
    image_data_df['cos_sim'] = image_data_df['img_embeddings'].apply(lambda x: cosine_similarity(query_vect, x)[0][0])
    top_results = image_data_df.sort_values(by='cos_sim', ascending=False)[0:top_K+1].head(top_K)
    return [os.path.splitext(os.path.basename(path))[0] for path in top_results['image_path'].tolist()]


def get_related_assets(keywords: list, model_ID, image_data_df: pd.DataFrame, top_K: int = 1) -> list:
    """Get a list of relevant assets for multiple keywords using CLIP similarity.

    Args:
        keywords: a list of textual asset descriptions.
        model_ID: identifier of the CLIP model to use.
        image_data_df: DataFrame with image embeddings.
        top_K: number of results to return for each keyword.

    Returns:
        A list of top-matching asset names across all keywords.
    """
    return sum([get_top_assets(keyword, model_ID, image_data_df, top_K) for keyword in keywords], [])
