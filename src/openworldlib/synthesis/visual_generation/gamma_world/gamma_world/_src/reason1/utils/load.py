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
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import boto3

from gamma_world._src.imaginaire.utils import distributed, log
from gamma_world._src.imaginaire.utils.easy_io import easy_io


def sync_s3_dir_to_local(
    s3_dir: str,
    s3_credential_path: str,
    cache_dir: Optional[str] = None,
    rank_sync: bool = True,
) -> str:
    """
    Download an entire directory from S3 to the local cache directory.

    Args:
        s3_dir (str): The AWS S3 directory to download.
        s3_credential_path (str): The path to the AWS S3 credentials file.
        rank_sync (bool, optional): Whether to synchronize download across
            distributed workers using `distributed.barrier()`. Defaults to True.
        cache_dir (str, optional): The cache folder to sync the S3 directory to.
            If None, the environment variable `IMAGINAIRE_CACHE_DIR` or the user
            cache directory will be used.

    Returns:
        local_dir (str): The path to the local directory.
    """
    if not s3_dir.startswith("s3://"):
        # If the directory exists locally, return the local path
        assert os.path.exists(s3_dir), f"{s3_dir} is not a S3 path or a local path."
        return s3_dir

    # Load AWS credentials from the file
    with open(s3_credential_path, "r") as f:
        credentials = json.load(f)

    # Create an S3 client
    s3 = boto3.client(
        "s3",
        **credentials,
    )

    # Parse the S3 URL
    parsed_url = urlparse(s3_dir)
    source_bucket = parsed_url.netloc
    source_prefix = parsed_url.path.lstrip("/")

    # If the local directory is not specified, use the default cache directory
    cache_dir = (
        os.environ.get("IMAGINAIRE_CACHE_DIR", os.path.expanduser("~/.cache/imaginaire"))
        if cache_dir is None
        else cache_dir
    )
    cache_dir = os.path.expanduser(cache_dir)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # List objects in the bucket with the given prefix
    response = s3.list_objects_v2(Bucket=source_bucket, Prefix=source_prefix)
    # Download each matching object
    for obj in response.get("Contents", []):
        if obj["Key"].startswith(source_prefix):
            # Create the full path for the destination file, preserving the directory structure
            rel_path = os.path.relpath(obj["Key"], source_prefix)
            dest_path = os.path.join(cache_dir, source_prefix, rel_path)

            # Ensure the directory exists
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            # Check if the file already exists
            if os.path.exists(dest_path):
                continue
            else:
                log.info(f"Downloading {obj['Key']} to {dest_path}")
                # Download the file
                if not rank_sync or distributed.get_rank() == 0:
                    s3.download_file(source_bucket, obj["Key"], dest_path)
    if rank_sync:
        distributed.barrier()
    local_dir = os.path.join(cache_dir, source_prefix)
    return local_dir


def download_from_s3_with_cache(
    s3_path: str,
    cache_fp: Optional[str] = None,
    cache_dir: Optional[str] = None,
    rank_sync: bool = True,
    backend_args: Optional[dict] = None,
    backend_key: Optional[str] = None,
) -> str:
    """download data from S3 with optional caching.

    This function first attempts to load the data from a local cache file. If
    the cache file doesn't exist, it downloads the data from S3 to the cache
    location. Caching is performed in a rank-aware manner
    using `distributed.barrier()` to ensure only one download occurs across
    distributed workers (if `rank_sync` is True).

    Args:
        s3_path (str): The S3 path of the data to load.
        cache_fp (str, optional): The path to the local cache file. If None,
            a filename will be generated based on `s3_path` within `cache_dir`.
        cache_dir (str, optional): The directory to store the cache file. If
            None, the environment variable `IMAGINAIRE_CACHE_DIR` or the user
            cache directory will be used.
        rank_sync (bool, optional): Whether to synchronize download across
            distributed workers using `distributed.barrier()`. Defaults to True.
        backend_args (dict, optional): The backend arguments passed to easy_io to construct the backend.
        backend_key (str, optional): The backend key passed to easy_io to registry the backend or retrieve the backend if it is already registered.

    Returns:
        cache_fp (str): The path to the local cache file.

    Raises:
        FileNotFoundError: If the data cannot be found in S3 or the cache.
    """
    if not s3_path.startswith("s3://"):
        # If the file exists locally, return the local path
        assert os.path.exists(s3_path), f"{s3_path} is not a S3 path nor a local path."
        return s3_path

    cache_dir = (
        os.environ.get("IMAGINAIRE_CACHE_DIR", os.path.expanduser("~/.cache/imaginaire"))
        if cache_dir is None
        else cache_dir
    )
    cache_dir = os.path.expanduser(cache_dir)
    if cache_fp is None:
        cache_fp = os.path.join(cache_dir, s3_path.replace("s3://", ""))
    if not cache_fp.startswith("/"):
        cache_fp = os.path.join(cache_dir, cache_fp)

    if rank_sync:
        if distributed.get_rank() == 0:
            if os.path.exists(cache_fp):
                # check the size of cache_fp
                if os.path.getsize(cache_fp) < 1:
                    os.remove(cache_fp)
                    log.warning(f"Removed empty cache file {cache_fp}.")

            if not os.path.exists(cache_fp):
                easy_io.copyfile_to_local(
                    s3_path, cache_fp, dst_type="file", backend_args=backend_args, backend_key=backend_key
                )
                log.info(f"Downloaded {s3_path} to {cache_fp}.")
            else:
                log.info(f"The cache file {cache_fp} already exists.")
        distributed.barrier()
    else:
        if os.path.exists(cache_fp):
            # check the size of cache_fp
            if os.path.getsize(cache_fp) < 1:
                os.remove(cache_fp)
                log.warning(f"Removed empty cache file {cache_fp}.")
        if not os.path.exists(cache_fp):
            easy_io.copyfile_to_local(
                s3_path, cache_fp, dst_type="file", backend_args=backend_args, backend_key=backend_key
            )
            log.info(f"Downloaded {s3_path} to {cache_fp}.")
        else:
            log.info(f"The cache file {cache_fp} already exists")
    return cache_fp
