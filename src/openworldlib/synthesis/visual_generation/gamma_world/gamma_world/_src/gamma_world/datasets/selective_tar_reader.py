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

import io
import random
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Callable, Iterator

if TYPE_CHECKING:
    import boto3


from gamma_world._src.imaginaire.utils import log




_offset_cache: dict[str, list[dict]] = {}
_OFFSET_CACHE_MAX = 200
_small_tar_cache: dict[str, dict[int, tuple[str, bytes]]] = {}
_SMALL_TAR_CACHE_MAX = 500







def _parse_tar_header(header_bytes: bytes) -> tuple[str, int] | None:
    if len(header_bytes) < 512 or header_bytes == b"\0" * 512:
        return None
    name = header_bytes[:100].split(b"\0", 1)[0].decode("utf-8", errors="replace").strip()
    if not name:
        return None
    size_raw = header_bytes[124:136].split(b"\0", 1)[0].decode("ascii", errors="replace").strip()
    try:
        size = int(size_raw, 8)
    except (ValueError, TypeError):
        return None
    return name, size


def _scan_all_headers(s3_client, bucket: str, key: str, max_entries: int = 600) -> list[dict]:

    if key in _offset_cache:
        return _offset_cache[key]

    entries = []
    offset = 0
    consecutive_empty = 0

    for _ in range(max_entries):
        try:
            resp = s3_client.get_object(Bucket=bucket, Key=key, Range=f"bytes={offset}-{offset + 2047}")
            buf = resp["Body"].read()
        except Exception:
            break

        parsed = _parse_tar_header(buf[:512])
        if parsed is None:
            consecutive_empty += 1
            if consecutive_empty >= 2:
                break
            offset += 512
            continue
        consecutive_empty = 0
        name, size = parsed
        data_blocks = (size + 511) // 512

        if name.startswith("././@"):
            next_hdr = 512 + data_blocks * 512
            if next_hdr + 512 <= len(buf):
                parsed2 = _parse_tar_header(buf[next_hdr : next_hdr + 512])
                if parsed2 is not None:
                    name2, size2 = parsed2
                    db2 = (size2 + 511) // 512
                    if not name2.startswith("././@"):
                        entries.append({"name": name2, "data_offset": offset + next_hdr + 512, "size": size2})
                    offset += next_hdr + 512 + db2 * 512
                    continue
            offset += 512 + data_blocks * 512
        else:
            entries.append({"name": name, "data_offset": offset + 512, "size": size})
            offset += 512 + data_blocks * 512

    if entries:
        _offset_cache[key] = entries
        if len(_offset_cache) > _OFFSET_CACHE_MAX:
            oldest = next(iter(_offset_cache))
            del _offset_cache[oldest]
    return entries


def _fetch_range(s3_client, bucket: str, key: str, offset: int, size: int) -> bytes:
    resp = s3_client.get_object(Bucket=bucket, Key=key, Range=f"bytes={offset}-{offset + size - 1}")
    return resp["Body"].read()


def _fetch_small_tar_entries(
    s3_client,
    bucket: str,
    tar_path: str,
) -> dict[int, tuple[str, bytes]]:

    if tar_path in _small_tar_cache:
        return _small_tar_cache[tar_path]

    entries: dict[int, tuple[str, bytes]] = {}
    try:
        resp = s3_client.get_object(Bucket=bucket, Key=tar_path)
        raw = resp["Body"].read()
        with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
            for pos, member in enumerate(tf):
                if member.isfile() and member.size > 0:
                    data = tf.extractfile(member)
                    if data is not None:
                        entries[pos] = (member.name, data.read())
    except Exception as exc:
        if entries:
            log.warning(f"SelectiveTarReader: partial tar extraction for {tar_path}: {exc}")
        return entries
    _small_tar_cache[tar_path] = entries
    if len(_small_tar_cache) > _SMALL_TAR_CACHE_MAX:
        oldest = next(iter(_small_tar_cache))
        del _small_tar_cache[oldest]
    return entries







def _get_clip_positions_from_metas(
    s3_client,
    bucket: str,
    root: str,
    tar_rel: str,
    keys: list[str],
) -> list[tuple[str, int]]:

    METAS_PREFIXES = ("metas_", "captions_")
    metas_key = None
    for k in keys:
        if any(k.startswith(p) for p in METAS_PREFIXES):
            metas_key = k
            break
    if metas_key is None:
        metas_key = keys[0]

    tar_path = f"{root}/{metas_key}/{tar_rel}"
    try:
        resp = s3_client.get_object(Bucket=bucket, Key=tar_path)
        data = resp["Body"].read()
    except Exception as e:
        log.warning(f"SelectiveTarReader: failed to read metas tar {tar_path}: {e}")
        return []

    clips = []
    try:
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:


            for pos, member in enumerate(tf):
                clip_id = member.name.rsplit(".", 1)[0] if "." in member.name else member.name
                if clip_id and clip_id not in (".", ""):
                    clips.append((clip_id, pos))
    except Exception as e:
        log.warning(f"SelectiveTarReader: failed to parse metas tar: {e}")
    return clips







def _fetch_clip_precomputed_parallel(
    s3_client,
    bucket: str,
    root: str,
    tar_rel: str,
    clip_id: str,
    keys: list[str],
    precomputed_offsets: dict,
    pool: "ThreadPoolExecutor | None" = None,
) -> dict[str, tuple[str, bytes]]:

    clip_offsets = precomputed_offsets.get(clip_id, {})
    results = {}

    def fetch_one(cam):
        cam_offset = clip_offsets.get(cam)
        if cam_offset is None:
            return cam, None
        if len(cam_offset) >= 3:
            data_offset, size, ext = cam_offset[0], cam_offset[1], cam_offset[2]
        else:
            data_offset, size = cam_offset
            ext = "mp4" if cam.startswith("video_") else "json"
        tar_path = f"{root}/{cam}/{tar_rel}"
        try:
            data = _fetch_range(s3_client, bucket, tar_path, data_offset, size)
            return cam, (f"{clip_id}.{ext}", data)
        except Exception:
            return cam, None

    _pool = pool or ThreadPoolExecutor(max_workers=min(len(keys), 21))
    try:
        for cam, result in _pool.map(fetch_one, keys):
            if result is not None:
                results[cam] = result
    finally:
        if pool is None:
            _pool.shutdown(wait=False)

    return results







def _read_accepted_from_camera(
    s3_client,
    bucket: str,
    root: str,
    key_name: str,
    tar_rel: str,
    target_positions: set[int],
) -> dict[int, tuple[str, bytes]]:

    tar_path = f"{root}/{key_name}/{tar_rel}"

    if key_name.startswith("video_"):
        all_entries = _scan_all_headers(s3_client, bucket, tar_path)
        results = {}
        for pos in target_positions:
            if pos < len(all_entries):
                info = all_entries[pos]
                data = _fetch_range(s3_client, bucket, tar_path, info["data_offset"], info["size"])
                results[pos] = (info["name"], data)
        return results
    else:
        entries = _fetch_small_tar_entries(s3_client, bucket, tar_path)
        return {pos: entries[pos] for pos in target_positions if pos in entries}







def selective_tar_samples(
    data: Iterator[dict],
    accepted_clip_ids: set[str] | None = None,
    s3_clients: dict[str, "boto3.client"] | None = None,
    s3_bucket_name: dict[str, str] | None = None,
    handler: Callable = lambda e: True,
    precomputed_offsets: dict | None = None,
    prob_map: dict[str, float] | None = None,
    default_prob: float = 1.0,
    rejection_seed: int | None = None,
    clip_to_tar: dict[str, str] | None = None,
) -> Iterator[dict]:

    use_precomputed = precomputed_offsets is not None and len(precomputed_offsets) > 0
    if use_precomputed:
        log.info(f"SelectiveTarReader: using precomputed offsets ({len(precomputed_offsets)} clips)")

    use_smooth = prob_map is not None
    _epoch_accepted: set[str] = set()
    _epoch_tars: set[str] = set()

    if use_smooth:
        import torch.utils.data as _tud

        _wi = _tud.get_worker_info()
        _worker_id = _wi.id if _wi else 0
        try:
            import torch.distributed as _dist

            _rank = _dist.get_rank() if _dist.is_initialized() else 0
        except Exception:
            _rank = 0
        _worker_seed = (rejection_seed or 0) + _rank * 1000 + _worker_id
        _smooth_rng = random.Random(_worker_seed)
        _clip_to_tar = clip_to_tar or {}
        _num_tars = len(set((_clip_to_tar or {}).values())) or len(prob_map)
        _tar_visit_count = 0
        _epoch_count = 0

        def _roll_epoch():

            acc = {k for k, v in prob_map.items() if _smooth_rng.random() < v}
            if not acc:
                acc = {max(prob_map, key=prob_map.get)}
            tars = {_clip_to_tar[c] for c in acc if c in _clip_to_tar}
            return acc, tars

        _epoch_accepted, _epoch_tars = _roll_epoch()
        log.info(
            f"SelectiveTarReader: smooth mode — epoch pre-roll: "
            f"{len(_epoch_accepted)}/{len(prob_map)} clips accepted, "
            f"{len(_epoch_tars)} tars targeted "
            f"(worker_seed={_worker_seed}, rank={_rank}, worker={_worker_id})"
        )

    _tars_seen = 0
    _tars_skipped = 0
    _precomputed_pool = ThreadPoolExecutor(max_workers=14) if use_precomputed else None
    if _precomputed_pool is not None:
        import atexit

        atexit.register(_precomputed_pool.shutdown, wait=False)

    for sample in data:
        url = sample["url"]
        dset_id = url.dset_id
        s3 = s3_clients[dset_id]
        bucket = s3_bucket_name[dset_id]
        keys = url.keys
        root = url.root
        tar_rel = url.path

        if not keys:
            continue

        tar_path = f"{root}/{tar_rel}"

        if use_smooth:
            _tar_visit_count += 1
            if _tar_visit_count >= _num_tars:
                _epoch_accepted, _epoch_tars = _roll_epoch()
                _tar_visit_count = 0
                _epoch_count += 1
                if _epoch_count <= 3 or _epoch_count % 100 == 0:
                    log.info(
                        f"SelectiveTarReader: re-rolled epoch {_epoch_count} — "
                        f"{len(_epoch_accepted)} clips, {len(_epoch_tars)} tars"
                    )
            if _epoch_tars and tar_path not in _epoch_tars:
                continue

        try:
            all_clips = _get_clip_positions_from_metas(s3, bucket, root, tar_rel, keys)
            if not all_clips:
                continue

            _tars_seen += 1
            if use_smooth:
                accepted = [(cid, pos) for cid, pos in all_clips if cid in _epoch_accepted]
            else:
                accepted = [(cid, pos) for cid, pos in all_clips if cid in accepted_clip_ids]

            if not accepted:
                _tars_skipped += 1
                if _tars_seen <= 5 or _tars_seen % 500 == 0:
                    pct = 100 * _tars_skipped / max(_tars_seen, 1)
                    log.info(
                        f"SelectiveTarReader: {_tars_skipped}/{_tars_seen} tars skipped ({pct:.1f}%) — tar {tar_rel}"
                    )
                continue

            if use_precomputed:


                for clip_id, pos in accepted:
                    clip_off = precomputed_offsets.get(clip_id, {})
                    has_offsets = [k for k in keys if k in clip_off]
                    no_offsets = [k for k in keys if k not in clip_off]

                    cam_results = _fetch_clip_precomputed_parallel(
                        s3,
                        bucket,
                        root,
                        tar_rel,
                        clip_id,
                        has_offsets,
                        precomputed_offsets,
                        pool=_precomputed_pool,
                    )
                    if no_offsets:

                        def _fetch_nv(nv_key, _pos=pos):
                            nv_path = f"{root}/{nv_key}/{tar_rel}"
                            entries = _fetch_small_tar_entries(s3, bucket, nv_path)
                            return nv_key, entries.get(_pos)

                        for nv_key, result in _precomputed_pool.map(_fetch_nv, no_offsets):
                            if result is not None:
                                cam_results[nv_key] = result

                    grouped = {"__key__": clip_id, "__url__": url}
                    for cam, (fname, file_bytes) in cam_results.items():
                        ext = fname.rsplit(".", 1)[-1] if "." in fname else ""
                        data_key = cam.replace("/", "_")
                        grouped[f"{data_key}.{ext}"] = file_bytes
                    yield grouped
            else:

                target_positions = {pos for _, pos in accepted}
                camera_results: dict[str, dict[int, tuple[str, bytes]]] = {}
                with ThreadPoolExecutor(max_workers=min(len(keys), 21)) as pool:
                    futures = {
                        pool.submit(
                            _read_accepted_from_camera,
                            s3,
                            bucket,
                            root,
                            key_name,
                            tar_rel,
                            target_positions,
                        ): key_name
                        for key_name in keys
                    }
                    for fut in as_completed(futures):
                        key_name = futures[fut]
                        try:
                            camera_results[key_name] = fut.result()
                        except Exception as e:
                            log.warning(f"SelectiveTarReader: camera {key_name} failed: {e}")
                            camera_results[key_name] = {}

                for clip_id, pos in accepted:
                    grouped = {"__key__": clip_id, "__url__": url}
                    for key_name in keys:
                        cam_data = camera_results.get(key_name, {}).get(pos)
                        if cam_data is None:
                            continue
                        fname, file_bytes = cam_data
                        ext = fname.rsplit(".", 1)[-1] if "." in fname else ""
                        data_key = key_name.replace("/", "_")
                        grouped[f"{data_key}.{ext}"] = file_bytes
                    yield grouped

        except Exception as e:
            log.warning(f"SelectiveTarReader error on {tar_rel}: {e}")
            if handler(e):
                continue
            else:
                break
