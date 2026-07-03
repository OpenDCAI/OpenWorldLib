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



import json
import os
from typing import Literal, Mapping

from gamma_world._src.imaginaire import config
from gamma_world._src.imaginaire.datasets.webdataset.config.schema import DatasetInfo
from gamma_world._src.imaginaire.utils import log

DEFAULT_CATALOG: Mapping = {
}


def get_video_dataset_info(
    source_name: str,
    *,
    dataset_keys: list[str] | None = None,
    object_store: Literal["gcs", "s3", "pdx"] = "s3",
    dataset_catalog: dict = DEFAULT_CATALOG,
    max_shards: int = 0,
) -> list[DatasetInfo]:

    if source_name not in dataset_catalog:
        raise KeyError(
            f"Source {source_name} not found in dataset catalog. Available keys are {dataset_catalog.keys()}"
        )


    dataset_infos = []
    for sensitive_type, wdinfos in dataset_catalog[source_name].items():
        if object_store == "gcs":
            bucket = "bucket"
            credentials_path = "credentials/gcp_sil_data.secret"
        elif object_store == "pdx":
            bucket = "webdataset"
            credentials_path = "credentials/s3_data.secret"
        elif object_store == "s3":
            bucket = "bucket" if sensitive_type == "nonsensitive" else "bucket-sensitive"
            credentials_path = "credentials/s3_training.secret"
        else:
            raise ValueError("Cosmos data: only support gcs or s3 for object store")

        if not wdinfos:
            continue


        if max_shards > 0:
            wdinfos = wdinfos[:1]

        dataset_infos.append(
            DatasetInfo(
                object_store_config=config.ObjectStoreConfig(
                    enabled=True,
                    credentials=credentials_path,
                    bucket=bucket,
                ),
                wdinfo=wdinfos,
                per_dataset_keys=dataset_keys,
                source=source_name,
                opts={
                    "aspect_ratio": "16,9",
                },
            )
        )
    return dataset_infos


def load_tar_filter_from_probs(probs_path: str, _parsed: dict | None = None) -> set[str]:

    if _parsed is not None:
        data = _parsed
    elif not probs_path or not os.path.exists(probs_path):
        return set()
    else:
        with open(probs_path) as f:
            data = json.load(f)

    tar_filter = data.get("_tar_filter")
    if not tar_filter:
        return set()

    drop_tars = tar_filter.get("drop_tars", [])
    if drop_tars:
        n_total = tar_filter.get("n_total_tars", "?")
        pct = tar_filter.get("pct_dropped", "?")
        log.info(f"Auto tar filter from rejection probs: {len(drop_tars)}/{n_total} tars to drop ({pct}%)")
    return set(drop_tars)
