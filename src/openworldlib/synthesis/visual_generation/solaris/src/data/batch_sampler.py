import bisect
import json
import logging
import math
import os

import numpy as np
import torch

from .dataset import DatasetMultiplayer
from .segment import SegmentId, SegmentIdMultiplayer


class BatchSampler(torch.utils.data.Sampler):

    def __init__(
        self,
        dataset,
        rank,
        batch_size,
        num_replicas,
        num_frames,
        seed=[0],
    ):
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.world_size = num_replicas
        self.batch_size = batch_size
        self.num_frames = num_frames
        self._seed = seed
        self.reset_rng()

    def __len__(self):
        raise NotImplementedError(
            "BatchSampler does not have a fixed length. Use __iter__ instead."
        )

    def __iter__(self):
        while True:
            yield self.sample()

    def reset_rng(self):
        # Reset the random number generator to the initial seed
        self.rng = np.random.default_rng(self._seed)

    def sample(self):
        num_episodes = self.dataset.num_episodes

        episodes_partition = np.arange(self.rank, num_episodes, self.world_size)
        short_episode_ids = np.where(self.dataset.lengths < self.num_frames)[0]
        episodes_partition = episodes_partition[
            ~np.isin(episodes_partition, short_episode_ids)
        ]

        episode_ids = self.rng.choice(
            episodes_partition, size=self.batch_size, replace=True
        )
        starts = self.rng.integers(
            low=0,
            high=self.dataset.lengths[episode_ids] - self.num_frames + 1,
        )
        stops = starts + self.num_frames

        return [SegmentId(*x) for x in zip(episode_ids, starts, stops)]


class BatchSamplerMultiplayer(torch.utils.data.Sampler):

    def __init__(
        self,
        dataset,
        rank,
        batch_size,
        num_replicas,
        num_frames,
        seed=[0],
    ):
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.world_size = num_replicas
        self.batch_size = batch_size
        self.num_frames = num_frames
        self._seed = seed
        self.reset_rng()
        self._episode_infos = {}

    def __len__(self):
        raise NotImplementedError(
            "BatchSampler does not have a fixed length. Use __iter__ instead."
        )

    def __iter__(self):
        while True:
            yield self.sample()

    def reset_rng(self):
        # Reset the random number generator to the initial seed
        self.rng = np.random.default_rng(self._seed)

    def get_episode_info(self, episode_id):
        if episode_id in self._episode_infos:
            return self._episode_infos[episode_id]
        episode_paths = self.dataset.get_episode_paths(episode_id)
        with open(
            self.dataset.directory / episode_paths["bot1_actions_path"], "r"
        ) as f:
            bot1_actions = json.load(f)
        with open(
            self.dataset.directory / episode_paths["bot2_actions_path"], "r"
        ) as f:
            bot2_actions = json.load(f)
        if len(bot1_actions) < 1 or len(bot2_actions) < 1:
            self._episode_infos[episode_id] = None
            logging.warning(
                f"Episode {episode_id} {episode_paths} has less than 1 action, resampling"
            )
            return self._episode_infos[episode_id]
        start_time = max(
            bot1_actions[0]["renderTime"],
            bot2_actions[0]["renderTime"],
        )
        end_time = min(
            bot1_actions[-1]["renderTime"],
            bot2_actions[-1]["renderTime"],
        )

        # Binary search for index ranges within [start_time, end_time]
        bot1_times = [a["renderTime"] for a in bot1_actions]
        bot2_times = [a["renderTime"] for a in bot2_actions]

        # Ensure times are strictly increasing for both bots
        bot1_times_increasing = (
            all(t_next > t_prev for t_prev, t_next in zip(bot1_times, bot1_times[1:])),
            "bot1_times must be strictly increasing",
        )
        bot2_times_increasing = (
            all(t_next > t_prev for t_prev, t_next in zip(bot2_times, bot2_times[1:])),
            "bot2_times must be strictly increasing",
        )
        if not bot1_times_increasing or not bot2_times_increasing:
            self._episode_infos[episode_id] = None
            logging.warning(
                f"Episode {episode_id} {episode_paths} has non-increasing times, resampling"
            )
            return self._episode_infos[episode_id]

        # Leftmost index >= start_time
        bot1_start_idx = bisect.bisect_left(bot1_times, start_time)
        bot2_start_idx = bisect.bisect_left(bot2_times, start_time)
        # Rightmost index <= end_time
        bot1_end_idx = bisect.bisect_right(bot1_times, end_time) - 1
        bot2_end_idx = bisect.bisect_right(bot2_times, end_time) - 1
        bot1_end_idx -= self.num_frames
        bot2_end_idx -= self.num_frames
        if (
            bot1_end_idx - bot1_start_idx < 1
            or bot2_end_idx - bot2_start_idx < 1
            or bot1_end_idx < 0
            or bot2_end_idx < 0
            or bot1_start_idx >= len(bot1_actions)
            or bot2_start_idx >= len(bot2_actions)
        ):
            self._episode_infos[episode_id] = None
            logging.warning(
                f"Episode {episode_id} {episode_paths} doesn't have enough actions. "
                f"bot1_start_idx: {bot1_start_idx}, bot1_end_idx: {bot1_end_idx}, bot2_start_idx: {bot2_start_idx}, bot2_end_idx: {bot2_end_idx}, "
                f"bot1_actions: {len(bot1_actions)}, bot2_actions: {len(bot2_actions)}, "
                f"resampling"
            )
            return self._episode_infos[episode_id]
        start_time = min(
            bot1_actions[bot1_start_idx]["renderTime"],
            bot2_actions[bot2_start_idx]["renderTime"],
        )
        end_time = min(
            bot1_actions[bot1_end_idx]["renderTime"],
            bot2_actions[bot2_end_idx]["renderTime"],
        )

        self._episode_infos[episode_id] = {
            "start_time": start_time,
            "end_time": end_time,
            "bot1_times": bot1_times,
            "bot2_times": bot2_times,
        }
        return self._episode_infos[episode_id]

    def sample(self):
        num_episodes = self.dataset.num_episodes

        episodes_partition = np.arange(self.rank, num_episodes, self.world_size)

        segment_ids = []
        while len(segment_ids) < self.batch_size:
            episode_ids = self.rng.choice(
                episodes_partition,
                size=self.batch_size - len(segment_ids),
                replace=True,
            )
            for episode_id in episode_ids:
                episode_info = self.get_episode_info(episode_id)
                if episode_info is None:
                    continue
                start_time = self.rng.uniform(
                    episode_info["start_time"], episode_info["end_time"]
                )

                # Find the corresponding frame indices for both bots
                bot1_times = episode_info["bot1_times"]
                bot2_times = episode_info["bot2_times"]

                bot1_start_idx = bisect.bisect_left(bot1_times, start_time)
                bot2_start_idx = bisect.bisect_left(bot2_times, start_time)
                bot1_stop_idx = bot1_start_idx + self.num_frames
                bot2_stop_idx = bot2_start_idx + self.num_frames
                segment_ids.append(
                    SegmentIdMultiplayer(
                        episode_id,
                        bot1_start_idx,
                        bot1_stop_idx,
                        bot2_start_idx,
                        bot2_stop_idx,
                    )
                )
                if len(segment_ids) >= self.batch_size:
                    break

        return segment_ids


class EvalBatchSampler(torch.utils.data.Sampler):

    def __init__(
        self,
        dataset,
        rank,
        batch_size,
        num_replicas,
        num_frames,
        num_global_samples=None,  # global number of samples
        seed=[0],
    ):
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.world_size = num_replicas
        self.batch_size = batch_size
        self.num_frames = num_frames
        logging.info(f"Eval num_frames: {num_frames}")
        self._seed = seed
        self.reset_rng()

        base_dir = os.path.dirname(os.path.abspath(__file__))
        # hardcoded full episodes for worldmem_demo
        if "worldmem_demo" in str(dataset.directory):
            self.ids = [
                [0, 0, 1024],
                [1, 0, 1024],
                [2, 0, 1024],
                [3, 0, 1024],
                [4, 0, 1024],
                [5, 0, 1024],
            ]
        else:
            with open(
                os.path.join(
                    base_dir, "eval_ids", f"eval_ids_{dataset.dataset_name}.json"
                ),
                "r",
            ) as f:
                self.ids = json.load(f)

        assert num_frames <= 1024, "num_frames must be at most 1024"
        num_global_samples = (
            min(num_global_samples, len(self.ids))
            if num_global_samples is not None
            else len(self.ids)
        )
        if isinstance(dataset, DatasetMultiplayer):
            self.examples = [
                SegmentIdMultiplayer(
                    episode_id,
                    bot1_start,
                    bot1_start + self.num_frames,
                    bot2_start,
                    bot2_start + self.num_frames,
                )
                for (episode_id, bot1_start, _, bot2_start, _) in self.ids
            ]
        else:
            # For other datasets, replace the endpoint with start + self.num_frames
            # Note this could potentially lead to an error later on when calling Dataset.__getitem__(...) if out of bounds
            self.examples = [
                SegmentId(episode_id, start, start + self.num_frames)
                for (episode_id, start, _) in self.ids
            ]
        self.examples = self.examples[:num_global_samples]
        self.examples = self.examples[self.rank :: self.world_size]
        self.num_batches = math.ceil(len(self.examples) / self.batch_size)

    def reset_rng(self):
        # Reset the random number generator to the initial seed
        self.rng = np.random.default_rng(self._seed)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for i in range(self.num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, len(self.examples))
            yield self.examples[start:end]
