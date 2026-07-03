import json
import logging
import random
from pathlib import Path

import numpy as np
from decord import VideoReader, cpu
from torch.utils.data import Dataset as TorchDataset

from . import minecraft
from .batch import Batch
from .segment import Segment


class InputConverter:
    def convert(self, act):
        return act


class CameraLinearConverterMatrixGame2(InputConverter):
    def convert(self, act):
        act[:, 23] = minecraft.compress_mouse_linear(act[:, 23])
        act[:, 24] = minecraft.compress_mouse_linear(act[:, 24])
        return act


input_converters = {
    "CameraLinearConverterMatrixGame2": CameraLinearConverterMatrixGame2(),
}


class VideoReadError(Exception):
    pass


class Dataset(TorchDataset):

    def __init__(
        self,
        data_dir,
        dataset_name,
        converters,
        obs_resize=None,
    ):
        super().__init__()

        self.dataset_name = dataset_name
        self.directory = Path(data_dir).expanduser()

        with open(self.directory / "episodes_info.json", "r") as json_file:
            self.episodes_info = json.load(json_file)
        self._num_episodes = self.episodes_info["num_episodes"]
        self._lengths = np.array(
            [ep["length"] for ep in self.episodes_info["episodes"]]
        )
        self._obs_resize = obs_resize
        self._converters = [input_converters[c] for c in converters]

    @property
    def num_episodes(self):
        return self._num_episodes

    def action_dim(self):
        return len(minecraft.ACTION_KEYS)

    @property
    def lengths(self):
        return self._lengths

    def __getitem__(self, segment_id):
        episode_info = self.episodes_info["episodes"][segment_id.episode_id]
        video_path = self.directory / episode_info["video_path"]
        actions_path = self.directory / episode_info["actions_path"]
        try:
            decord_video = VideoReader(str(video_path), ctx=cpu(0))
        except Exception as e:
            raise ValueError(f"Error reading video {video_path}: {e}") from e

        try:
            act = self.read_act_slice(actions_path, segment_id.start, segment_id.stop)
        except Exception as e:
            raise ValueError(
                f"Error reading episode actions {segment_id.episode_id}: {e}"
            ) from e

        try:
            obs_decord = minecraft.read_obs_slice_decord(
                decord_video,
                segment_id.start,
                segment_id.stop,
                self._obs_resize,
            )

        except Exception as e:
            raise VideoReadError(f"Error reading segment slice: {e}") from e

        for converter in self._converters:
            act = converter.convert(act)
        segment = Segment(obs_decord, act)

        return segment

    def read_act_slice(
        self,
        path,
        start,
        stop,
    ):
        full_actions_json = read_actions_json(path)
        return minecraft.read_act_slice_vpt(
            full_actions_json,
            start,
            stop,
        )


class DatasetMultiplayer(TorchDataset):
    def __init__(
        self,
        data_dir,
        dataset_name,
        bot1_name,
        bot2_name,
        converters,
        obs_resize=None,
        shuffle_bots=True,
        shuffle_bot_seed=None,
    ):
        super().__init__()

        self.dataset_name = dataset_name
        self.directory = Path(data_dir).expanduser()
        self.bot1_name = bot1_name
        self.bot2_name = bot2_name
        self._obs_resize = obs_resize
        self._converters = [input_converters[c] for c in converters]
        self._shuffle_bots = shuffle_bots
        self._shuffle_bot_seed = shuffle_bot_seed
        # Create a separate Random instance for shuffling
        self._rng = random.Random(shuffle_bot_seed)
        self._episodes = {}

        mp4_files = sorted(
            [p for p in self.directory.glob(f"*_{self.bot1_name}_*.mp4")],
            key=lambda p: p.name,
        )
        episode_id = 0

        for video_path_bot1 in mp4_files:
            name = video_path_bot1.name

            # Form the expected counterpart filename for bot2 by swapping bot name
            counterpart_name = name.replace(
                f"_{self.bot1_name}_", f"_{self.bot2_name}_", 1
            )
            video_path_bot2 = self.directory / counterpart_name

            # Corresponding action files must exist and share the same stem
            actions_path_bot1 = video_path_bot1.with_suffix(".json")
            actions_path_bot2 = video_path_bot2.with_suffix(".json")
            # If the action file ends with '_camera.json', replace it with '.json'
            if actions_path_bot1.name.endswith("_camera.json"):
                actions_path_bot1 = actions_path_bot1.with_name(
                    actions_path_bot1.name.replace("_camera.json", ".json")
                )
            if actions_path_bot2.name.endswith("_camera.json"):
                actions_path_bot2 = actions_path_bot2.with_name(
                    actions_path_bot2.name.replace("_camera.json", ".json")
                )

            # Validate existence of all four files
            if (
                video_path_bot2.exists()
                and actions_path_bot1.exists()
                and actions_path_bot2.exists()
            ):
                self._episodes[episode_id] = {
                    "bot1_video_path": video_path_bot1.name,
                    "bot1_actions_path": actions_path_bot1.name,
                    "bot2_video_path": video_path_bot2.name,
                    "bot2_actions_path": actions_path_bot2.name,
                }
                episode_id += 1

        self._num_episodes = len(self._episodes)
        logging.info(
            f"Dataset {self.dataset_name} loaded {self._num_episodes} episodes. Data directory: {self.directory}"
        )

    def action_dim(self):
        return len(minecraft.ACTION_KEYS)

    @property
    def num_episodes(self):
        return self._num_episodes

    def get_episode_paths(self, episode_id):
        return self._episodes[episode_id]

    def __getitem__(self, segment_id):
        episode_paths = self.get_episode_paths(segment_id.episode_id)
        video_path_bot1 = self.directory / episode_paths["bot1_video_path"]
        video_path_bot2 = self.directory / episode_paths["bot2_video_path"]
        actions_path_bot1 = self.directory / episode_paths["bot1_actions_path"]
        actions_path_bot2 = self.directory / episode_paths["bot2_actions_path"]
        try:
            decord_video_bot1 = VideoReader(str(video_path_bot1), ctx=cpu(0))
        except Exception as e:
            raise ValueError(f"Error reading video {video_path_bot1}: {e}") from e
        try:
            decord_video_bot2 = VideoReader(str(video_path_bot2), ctx=cpu(0))
        except Exception as e:
            raise ValueError(f"Error reading video {video_path_bot2}: {e}") from e

        try:
            with open(actions_path_bot1, "r") as f:
                bot1_actions = json.load(f)
            with open(actions_path_bot2, "r") as f:
                bot2_actions = json.load(f)
            bot1_actions = bot1_actions[segment_id.bot1_start : segment_id.bot1_stop]
            bot2_actions = bot2_actions[segment_id.bot2_start : segment_id.bot2_stop]
        except Exception as e:
            raise ValueError(
                f"Error reading episode actions {segment_id.episode_id}: {e}"
            ) from e
        bot1_actions = [{**a, "bot": self.bot1_name} for a in bot1_actions]
        bot2_actions = [{**a, "bot": self.bot2_name} for a in bot2_actions]

        try:
            obs_bot1 = minecraft.read_obs_slice_decord(
                decord_video_bot1,
                segment_id.bot1_start,
                segment_id.bot1_stop,
                self._obs_resize,
            )
            obs_bot2 = minecraft.read_obs_slice_decord(
                decord_video_bot2,
                segment_id.bot2_start,
                segment_id.bot2_stop,
                self._obs_resize,
            )
        except Exception as e:
            raise VideoReadError(f"Error reading segment slice: {e}") from e

        bot1_actions_one_hot = minecraft.convert_act_slice_mineflayer(bot1_actions)
        bot2_actions_one_hot = minecraft.convert_act_slice_mineflayer(bot2_actions)
        for converter in self._converters:
            bot1_actions_one_hot = converter.convert(bot1_actions_one_hot)
            bot2_actions_one_hot = converter.convert(bot2_actions_one_hot)

        # Randomly shuffle bot order if enabled
        bot_obs = [obs_bot1, obs_bot2]
        bot_act = [bot1_actions_one_hot, bot2_actions_one_hot]
        if self._shuffle_bots:
            indices = [0, 1]
            self._rng.shuffle(indices)
            bot_obs = [bot_obs[i] for i in indices]
            bot_act = [bot_act[i] for i in indices]

        obs = np.stack(bot_obs, axis=1)
        act = np.stack(bot_act, axis=1)
        segment = Segment(obs, act)

        return segment


def read_actions_json(actions_path):
    try:
        with open(actions_path) as json_file:
            json_lines = json_file.readlines()
    except Exception as e_utf:
        try:
            with open(actions_path, encoding="windows-1252") as json_file:
                json_lines = json_file.readlines()
        except Exception as e_win:
            raise ValueError(
                f"Error reading file {actions_path} in utf-8 and windows-1252 encodings: {e_utf} / {e_win}"
            ) from e_win

    json_data = "[" + ",".join(json_lines) + "]"
    json_data = json.loads(json_data)
    return json_data


def collate_segments_to_batch(sequence_length, pad_batch_to, segments):
    # Filter out None segments
    valid_segments = [s for s in segments if s]
    if not valid_segments:
        raise ValueError("No valid segments to collate.")

    # Find max length for obs and act
    # Pad obs and act to max length
    obs_padded = []
    act_padded = []
    for s in valid_segments:
        obs = s.obs
        act = s.act

        # Calculate pad widths to pad with zeros on the right along the first dimension
        # Also collect the real (unpadded) lengths for each segment
        # real_lengths will be used to indicate where padding starts for each segment
        # (i.e., the original length of each segment)
        # This array will be of size B (number of valid segments)
        obs_pad_width = [(0, sequence_length - len(s))] + [(0, 0)] * (obs.ndim - 1)
        act_pad_width = [(0, sequence_length - len(s))] + [(0, 0)] * (act.ndim - 1)
        obs_padded.append(np.pad(obs, obs_pad_width, mode="constant"))
        act_padded.append(np.pad(act, act_pad_width, mode="constant"))

    real_lengths = np.array([len(s) for s in valid_segments], dtype=np.int32)
    obs_batch = np.stack(obs_padded)
    act_batch = np.stack(act_padded)
    if pad_batch_to is not None and pad_batch_to > len(valid_segments):
        obs_batch = np.pad(
            obs_batch,
            [(0, pad_batch_to - len(valid_segments))] + [(0, 0)] * (obs_batch.ndim - 1),
            mode="constant",
        )
        act_batch = np.pad(
            act_batch,
            [(0, pad_batch_to - len(valid_segments))] + [(0, 0)] * (act_batch.ndim - 1),
            mode="constant",
        )
        real_lengths = np.pad(
            real_lengths, (0, pad_batch_to - len(valid_segments)), mode="constant"
        )

    return Batch(obs_batch, act_batch, real_lengths)
