# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

import random
from functools import partial
from typing import Callable

import omegaconf
import webdataset as wds
from webdataset import filters

from gamma_world._src.imaginaire.datasets.webdataset.config.schema import DatasetConfig
from gamma_world._src.imaginaire.datasets.webdataset.utils.iterators import WebDataset
from gamma_world._src.imaginaire.datasets.webdataset.utils.misc import (
    remove_extensions_from_keys,
    skip_keys,
    update_url,
)
from gamma_world._src.imaginaire.datasets.webdataset.webdataset_ext import Dataset as BaseDataset
from gamma_world._src.imaginaire.utils import log
from gamma_world._src.gamma_world.datasets.selective_tar_reader import (
    selective_tar_samples,
)


class SILDataset(BaseDataset):


    def __init__(
        self,
        config: DatasetConfig,
        handler: Callable = wds.warn_and_continue,
        decoder_handler: Callable | None = None,
        detshuffle: bool = False,
        tar_drop_set: set[str] | None = None,
        accepted_clip_ids: set[str] | None = None,
        precomputed_offsets: dict | None = None,
        prob_map: dict[str, float] | None = None,
        default_prob: float = 1.0,
        rejection_seed: int | None = None,
        clip_to_tar: dict[str, str] | None = None,
    ):
        super().__init__(
            config=config,
            handler=handler,
            decoder_handler=decoder_handler,
            detshuffle=detshuffle,
        )
        self.tar_drop_set = tar_drop_set
        self.accepted_clip_ids = accepted_clip_ids
        self.precomputed_offsets = precomputed_offsets
        self.prob_map = prob_map
        self.default_prob = default_prob
        self.rejection_seed = rejection_seed
        self.clip_to_tar = clip_to_tar

    def _use_selective_reader(self) -> bool:
        return self.accepted_clip_ids is not None or self.prob_map is not None

    def build_dataset(self, **kwargs) -> WebDataset:



        if self.tar_drop_set:
            original = self.wdinfo.tar_files
            before = len(original)
            filtered = [u for u in original if f"{u.root}/{u.path}" not in self.tar_drop_set]
            self.wdinfo.tar_files = filtered
            after = len(filtered)
            pct = 100 * (before - after) / max(before, 1)
            log.info(f"SILDataset: tar pre-filter dropped {before - after}/{before} tars ({pct:.1f}% filtered)")

        if not self._use_selective_reader():
            return super().build_dataset(**kwargs)




        from webdataset import DataPipeline

        if self.prob_map is not None:
            log.info(
                f"SILDataset: smooth selective mode ({len(self.prob_map)} clip probs, "
                f"default_prob={self.default_prob}). "
                f"Stochastic tar pre-filtering with single roll."
            )
        else:
            log.info(
                f"SILDataset: selective mode ({len(self.accepted_clip_ids)} accepted clips). "
                f"Byte-range reads will skip rejected clips."
            )

        tar_list = self.wdinfo.tar_files
        num_tars = len(tar_list)
        assert num_tars > 0, "Did not find any data."

        shuffle_buffer_size = getattr(self.config, "buffer_size", self.wdinfo.chunk_size)

        distributor_fn = self.config.distributor
        distributor_fn.set_urls(tar_list)
        distributor_fn.set_chunk_size(self.wdinfo.chunk_size)


        dataset = DataPipeline()

        if self.prob_map is not None:
            _smooth_tars = list(tar_list)
            log.info(f"SILDataset: smooth mode — custom distributor, all workers see all {len(_smooth_tars)} tars")

            def _smooth_distributor():

                import torch.distributed as _dist
                import torch.utils.data as _tud

                _wi = _tud.get_worker_info()
                _worker_id = _wi.id if _wi else 0
                try:
                    _rank = _dist.get_rank() if _dist.is_initialized() else 0
                except Exception:
                    _rank = 0
                _rng = random.Random(_rank * 1000 + _worker_id)
                while True:
                    urls = list(_smooth_tars)
                    _rng.shuffle(urls)
                    for url in urls:
                        yield {"url": url}

            dataset.append(_smooth_distributor)
        else:
            dataset.append(distributor_fn)

        _selective_stage = partial(
            selective_tar_samples,
            accepted_clip_ids=self.accepted_clip_ids,
            s3_clients=self.s3_client,
            s3_bucket_name=self.bucket,
            handler=self.handler,
            precomputed_offsets=self.precomputed_offsets,
            prob_map=self.prob_map,
            default_prob=self.default_prob,
            rejection_seed=self.rejection_seed,
            clip_to_tar=self.clip_to_tar,
        )
        dataset.append(_selective_stage)

        if self.detshuffle:
            dataset.append(filters.detshuffle(shuffle_buffer_size))
        else:
            dataset.append(wds.shuffle(shuffle_buffer_size))

        decoder_list = getattr(self.config, "decoders", [])
        decoder_functions = list(decoder_list)
        dataset.append(wds.decode(*decoder_functions, handler=self.decoder_handler))

        if self.config.remove_extension_from_keys:
            dataset.append(remove_extensions_from_keys)
        dataset.append(skip_keys)

        augmentor_cfg = getattr(self.config, "augmentation", None)
        assert isinstance(augmentor_cfg, (dict, omegaconf.dictconfig.DictConfig))
        augmentation_fn = self.build_data_augmentor(augmentor_cfg)
        dataset.append(augmentation_fn)

        dataset.append(update_url)

        dataset.total_images = self.wdinfo.total_key_count
        log.info(f"Total number of training shards: {num_tars}")
        log.info(f"Total training key count: {dataset.total_images}")

        return dataset
