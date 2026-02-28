"""This module provides functionality for retrieving and placing assets in the city simulation."""
import importlib.resources as pkg_resources
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer

from .utils.assets_rp_utils import (
    construct_building_from_candidate, get_coordinates_around_building,
    get_parsed_input, get_surroundings, place_target_asset,
    retrieve_target_asset, vector_cosine_similarity)
from .utils.reference_assets_retriever import ReferenceAssetsRetriever
from ..citygen.dataclass.dataclass import Point
from ..config import Config
from ..utils.data_importer import DataImporter
from ..utils.logger import Logger

class AssetsRetrieverPlacer:
    """Assets Retrieval and Placement ability.

    This class provides methods to retrieve the assets and place them somewhere
    based on the natural language prompts.
    """

    def __init__(self, config: Config, input_dir: str, env_description_retrieval_model_name: str = None):
        """Initialize function call.

        Args:
            config: the configuration user provide.
            input_dir: the directory to load the input data.
            env_description_retrieval_model_name: the name of the environment description retrieval model.
        """
        self.config = config
        self.env_description_retrieval_model_name = env_description_retrieval_model_name if env_description_retrieval_model_name else config['assets_rp.env_description_retrieval_model']
        self.model = SentenceTransformer(self.env_description_retrieval_model_name)
        self.data_importer = DataImporter(config)
        self.input_dir = input_dir
        self.city_generator = self.data_importer.import_city_data(input_dir)

        self.logger = Logger.get_logger('AssetsRP')

    def generate_assets_manually(self, natural_language_input, sample_dataset_dir: str = None, output_dir: str = None,  description_map_path: str = None, assets_retrieval_model: str = None):
        """This function is used to retrieve and place the assets based on user's prompt.

        Args:
            natural_language_input: the text prompt provided by the users.
            sample_dataset_dir: the directory to load the images of the assets.
            output_dir: the directory to save the output.
            description_map_path: the path to the description map file.
            assets_retrieval_model: the name of the assets retrieval model.
        """
        # 1. Parse the input
        parsed_input, asset_to_place, reference_asset_query, relation, surroundings_query = get_parsed_input(natural_language_input)

        self.logger.info('LLM parse result: %s', parsed_input)

        # 2. Load the file that store all the assets. Find the candidates that match "reference_asset_query"
        _progen_world_path = os.path.join(self.input_dir, 'progen_world.json')
        _description_map_path = description_map_path if description_map_path else self.config['assets_rp.input_description_map']
        referenceAssetRetriever = ReferenceAssetsRetriever(_progen_world_path, _description_map_path, self.env_description_retrieval_model_name)
        candidate_nodes = referenceAssetRetriever.retrieve_reference_assets(reference_asset_query)

        # 3. For each candidate, use "_get_point_around_label" to obtain its surrounding asset
        candidate_similarity_scores = []
        for candidate, base_score in candidate_nodes:
            x = candidate['properties']['location']['x'] / 100
            y = candidate['properties']['location']['y'] / 100
            node_position = Point(x, y)
            candidate_surroundings = self.city_generator.route_generator.get_point_around_label(node_position, self.city_generator.city_quadtrees, 200, 20)

            # 4. Integrate the surrounding information to a string and do embedding, and calculate the similarity score.
            candidate_surroundings_str = get_surroundings(candidate_surroundings, _description_map_path)
            candidate_embedding = self.model.encode(candidate_surroundings_str)
            query_embedding = self.model.encode(surroundings_query)

            similarity = vector_cosine_similarity(candidate_embedding, query_embedding)
            candidate_similarity_scores.append((candidate, similarity))
            # self.logger.info(f"candidate nodes: {candidate['id']} similarity score: {similarity:.4f}")

        # 5. Choose the highest score as final reference asset and construct the instance
        best_candidate, best_similarity = max(candidate_similarity_scores, key=lambda x: x[1])
        self.logger.info('best candidate: %s similarity score: %s', best_candidate['id'], best_similarity)

        reference_asset = construct_building_from_candidate(best_candidate, os.path.join(self.input_dir, 'buildings.json'))
        if reference_asset is None:
            self.logger.error('No reference asset found for %s', best_candidate['id'])
            return

        # 6. Use CLIP to obtain the asset and place around the best candidate
        if sample_dataset_dir is None:
            self.logger.info('No sample dataset directory provided, using default')

            _sample_dataset_dir = (
                Path(__file__).resolve()
                .parents[2]        # 根据你 simworld_operator.py 的层级调整
                / 'data'
                / self.config['assets_rp.input_sample_dataset']
            )
        else:
            _sample_dataset_dir = Path(sample_dataset_dir).resolve()

        if assets_retrieval_model is None:
            self.logger.info('No assets retrieval model provided, using default')
            _assets_retrieval_model = self.config['assets_rp.assets_retrieval_model']
        else:
            _assets_retrieval_model = assets_retrieval_model

        self.logger.info('Using %s to retrieve target assets', _assets_retrieval_model)
        target_assets = retrieve_target_asset(asset_to_place, _sample_dataset_dir, _assets_retrieval_model)
        self.logger.info('target assets: %s', target_assets)
        target_positions = get_coordinates_around_building(self.city_generator.config, reference_asset, relation, len(target_assets))
        self.logger.info('target positions: %s', target_positions)
        if len(target_assets) == 0 or len(target_positions) == 0:
            self.logger.error('No target assets or target positions found')
            return
        place_target_asset(target_assets, target_positions, output_dir or self.config['assets_rp.output_dir'])
