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

import io
import re
from random import randint
from typing import List, Tuple

import decord
import numpy as np
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = 933120000
_VIDEO_EXTENSIONS = "mp4 avi webm mov".split()

VIDEO_DECODER_OPTIONS = {}


def video_decoder_register(key):
    def decorator(func):
        VIDEO_DECODER_OPTIONS[key] = func
        return func

    return decorator


def basic_check_on_inputs(
    n_video_frames: int, n_target_frames: int, video_fps: float, min_fps_thres: int, max_fps_thres: int
) -> str:
    if n_video_frames <= 0:
        return "n_video_frames must be positive"
    if min_fps_thres <= 0:
        return "min_fps_thres must be positive"
    if video_fps < 1:
        return "Video fps lower than 1, skipping"
    if max_fps_thres < min_fps_thres:
        return "max_fps_thres must be greater than or equal to min_fps_thres"
    if n_target_frames <= 1:
        return "sequence_length must be greater than 1"
    if n_target_frames > n_video_frames:
        return f"Specified sequence_length {n_target_frames} exceeds num frames in video {n_video_frames}."

    return "success"


def sample_chunk_index_from_chunked_video(
    n_video_frames: int,
    n_target_frames: int,
    chunk_size: int,
) -> Tuple[int, int, str]:
    """
    Sample a chunk from the chunked videos. Our videos are stored as regular mp4 files but with chunked captions. There is one caption per [chunk_size] frames.
    If the last chunk has frames >= chunk_size / 2, it will be treated as a separate chunk and has its own caption. Otherwise, it will be treated as part of the previous chunk.
    e.g. if chunk_size = 256, possible number of frames in a chunk: [1, 383]

    Args:
      n_video_frames: total number of frames in the video
      n_target_frames: number of requested frames
      chunk_size: number of frames in each chunk. The last chunk will be treated differently

    Returns:
      sampled_chunk_indexs
      n_frames_in_chunk
      message
    """
    n_chunks = max(n_video_frames // chunk_size, 1)

    # Check if the last chunk has separate window
    # This happens only if remainder frames >= chunk_size / 2 [data annotation was done this way]
    # Else this is used as a part of previous window.
    n_frames_in_last_chunk = n_video_frames - n_chunks * chunk_size
    if n_frames_in_last_chunk >= int(0.5 * chunk_size):
        if n_frames_in_last_chunk > n_target_frames:
            n_chunks += 1

    sampled_chunk_index = randint(0, n_chunks - 1)
    if sampled_chunk_index == n_chunks - 1:
        # For the last chunk, use all of the remaining frames
        n_frames_in_chunk = n_video_frames - sampled_chunk_index * chunk_size
    else:
        # Else use only the chunk size
        n_frames_in_chunk = chunk_size

    if n_target_frames > n_frames_in_chunk:
        error_message = f"Requested sequence_length {n_target_frames} exceeds curr_chunk_size {n_frames_in_chunk}, n_video_frames={n_video_frames}, chunk_size={chunk_size}, sampled_chunk_index={sampled_chunk_index}."
        return -1, 0, error_message

    return sampled_chunk_index, n_frames_in_chunk, "success"


@video_decoder_register("video_naive_bytes")
def video_naive_bytes(*args, **kwargs):
    """
    do nothing, just return the video bytes
    """
    del args, kwargs

    def video_decoder(
        key: str,
        data: bytes,
    ):
        extension = re.sub(r".*[.]", "", key)
        if extension.lower() not in _VIDEO_EXTENSIONS:
            return None

        return data

    return video_decoder


@video_decoder_register("chunked_video_decoder")
def chunked_video_decoder(
    chunk_size: int = 0,
    sequence_length: int = 34,
    min_fps_thres: int = 1,
    max_fps_thres: int = 9999,
    num_threads=4,
):
    """
    Video decoder for videos with chunked captions.
    It first sample a chunk from the video then sample the start frame within the chunk.
    It has a basic check to make sure the video fps falls within the range [min_fps_thres, max_fps_thres]. Otherwise, it will skip the video sample.

    Args:
    - chunk_size (int): How the video is divided into chunks. Only return frames within a chunk. chunk_size=0 means we use full video length. Defaults to 0.
    - sequence_length (int) : Number of frames returned by the function
    - min_fps_thres (int): Minimum fps threshold allowed.
    - max_fps_thres (int): Maximum fps threshold allowed.
    - num_thread (int): Number of threads for decord.

    Returns:
        dict with video frames tensor and additional attributes including
        - fps
        - orig_fps
        - num_frames
        - chunk_index
        - frame_start
        - frame_end
        - n_orig_video_frames
    """

    def video_decoder(
        key: str,
        data: bytes,
    ):
        extension = re.sub(r".*[.]", "", key)
        if extension.lower() not in _VIDEO_EXTENSIONS:
            return None

        video_buffer = io.BytesIO(data)
        video_reader = decord.VideoReader(video_buffer, num_threads=num_threads)

        n_target_frames = sequence_length if sequence_length > 0 else len(video_reader)
        n_video_frames = len(video_reader)
        video_fps = int(np.round(video_reader.get_avg_fps()))
        cur_chunk_size = n_video_frames if chunk_size == 0 else chunk_size

        # basic check
        message = basic_check_on_inputs(
            n_video_frames=n_video_frames,
            n_target_frames=n_target_frames,
            video_fps=video_fps,
            min_fps_thres=min_fps_thres,
            max_fps_thres=max_fps_thres,
        )
        if message != "success":
            raise ValueError(message)

        # check if video fps is within the specified range
        if video_fps < min_fps_thres:
            raise ValueError(f"Video fps {video_fps} lower than {min_fps_thres}, skipping")
        if video_fps > max_fps_thres:
            raise ValueError(f"Video fps {video_fps} larger than {max_fps_thres}, skipping")

        sampled_chunk_index, n_frames_in_chunk, message = sample_chunk_index_from_chunked_video(
            n_video_frames=n_video_frames,
            n_target_frames=n_target_frames,
            chunk_size=cur_chunk_size,
        )
        if sampled_chunk_index == -1:
            raise ValueError(message)
        else:
            assert message == "success"

        # Select the frame start index and frame end index
        chunk_frame_start = sampled_chunk_index * chunk_size
        # Start index is randomly selected in the chunk
        frame_start = chunk_frame_start + int(np.random.choice(n_frames_in_chunk - n_target_frames, 1))
        frame_end = frame_start + n_target_frames

        # Subsample the frames
        video_frames = video_reader.get_batch(np.arange(frame_start, frame_end).tolist()).asnumpy()
        video_frames = torch.from_numpy(video_frames).permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)

        # Clean up
        video_reader.seek(0)  # set video reader point back to 0 to clean up cache
        del video_reader  # delete the reader to avoid memory leak

        return {
            "video": video_frames,
            "fps": float(video_fps),
            "orig_fps": float(video_fps),
            "num_frames": video_frames.shape[1],
            "chunk_index": sampled_chunk_index,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "n_orig_video_frames": n_video_frames,
        }

    return video_decoder


def get_frame_indices_w_lowered_fps(
    n_video_frames: int,
    video_fps: int,
    min_fps_thres: int,
    max_fps_thres: int,
    n_target_frames: int,
) -> Tuple[List[int], float]:
    """Generates frame indices for video sampling with FPS control.

    This function determines valid stride lengths for sampling frames from a video,
    preferring lower FPS (larger strides) when multiple options are available.
    It returns both the selected frame indices and the resulting FPS.

    Args:
        n_video_frames: Total number of frames in the original video.
        video_fps: Original video frames per second.
        min_fps_thres: Minimum allowed frames per second.
        max_fps_thres: Maximum allowed frames per second.
        n_target_frames: Number of frames to sample.

    Returns:
        A tuple containing:
            - list[int]: Frame indices to sample from the original video.
            - float: The resulting frames per second after sampling.

    Raises:
        ValueError: If no valid stride options are available given the constraints.
        ValueError: If input parameters are invalid (e.g., negative values).
    """
    # Calculate stride range
    min_stride = 1
    max_stride = (n_video_frames - 1) // (n_target_frames - 1)

    valid_strides = []
    for stride in range(min_stride, max_stride + 1):
        # Check if we can get n_target_frames frames with this stride
        if (n_video_frames - stride * (n_target_frames - 1)) > 0:
            new_fps = video_fps / stride
            if min_fps_thres <= new_fps <= max_fps_thres:
                valid_strides.append(stride)

    if not valid_strides:
        raise ValueError(
            f"No valid stride options available for the given constraints. "
            f"stride range = [{min_stride}, {max_stride}]; "
            f"original FPS = {video_fps}; "
            f"n_target_frames = {n_target_frames}; "
            f"min_fps_thres = {min_fps_thres}; "
            f"max_fps_thres = {max_fps_thres}; "
            f"original num_frames = {n_video_frames}"
        )

    # Select stride with weighted probability
    if len(valid_strides) >= 2:
        stride_choices = valid_strides[-2:]  # Taking last two as they're the largest
        weights = [0.01, 0.99]  # [smaller_stride, larger_stride]
        selected_stride = np.random.choice(stride_choices, p=weights)
    else:
        selected_stride = valid_strides[0]

    # Calculate the maximum valid start index and random start frame
    max_start_idx = n_video_frames - (n_target_frames - 1) * selected_stride
    frame_start = np.random.randint(0, max_start_idx)

    # Generate frame indices
    frame_indices = [frame_start + i * selected_stride for i in range(n_target_frames)]
    return frame_indices, video_fps / selected_stride


@video_decoder_register("chunked_video_decoder_w_lower_fps")
def chunked_video_decoder_w_lower_fps(
    chunk_size: int = 0,
    sequence_length: int = 34,
    min_fps_thres: int = 4,
    max_fps_thres: int = 30,
    num_threads: int = 4,
) -> dict:
    """
    Video decoder for videos with chunked captions.
    It first sample a chunk from the video then sample the start frame within the chunk.
    It has high probability (>99%) to lower the fps with frame sampling whenever allowed.

    Args:
        - chunk_size (int): How the video is divided into chunks. Only return frames within a chunk. chunk_size=0 means we use full video length. Defaults to 0.
        - sequence_length (int) : Number of frames returned by the function
        - min_fps_thres: Minimum FPS threshold
        - max_fps_thres: Maximum FPS threshold
        - num_threads: Number of threads for decord

    Returns:
        dict with video frames tensor and additional attributes including
        - fps
        - orig_fps
        - num_frames
        - chunk_index
        - frame_start
        - frame_end
        - n_orig_video_frames
    """

    def video_decoder(
        key: str,
        data: bytes,
    ) -> dict[str, torch.Tensor | int]:
        # Check video extension
        extension = re.sub(r".*[.]", "", key)
        if extension.lower() not in _VIDEO_EXTENSIONS:
            return None

        # Read video
        video_buffer = io.BytesIO(data)
        video_reader = decord.VideoReader(video_buffer, num_threads=num_threads)

        n_target_frames = sequence_length if sequence_length > 0 else len(video_reader)
        n_video_frames = len(video_reader)
        video_fps = int(np.round(video_reader.get_avg_fps()))
        cur_chunk_size = n_video_frames if chunk_size == 0 else chunk_size

        # basic check
        message = basic_check_on_inputs(
            n_video_frames=n_video_frames,
            n_target_frames=n_target_frames,
            video_fps=video_fps,
            min_fps_thres=min_fps_thres,
            max_fps_thres=max_fps_thres,
        )
        if message != "success":
            raise ValueError(message)

        sampled_chunk_index, n_frames_in_chunk, message = sample_chunk_index_from_chunked_video(
            n_video_frames=n_video_frames,
            n_target_frames=n_target_frames,
            chunk_size=cur_chunk_size,
        )
        if sampled_chunk_index == -1:
            raise ValueError(message)
        else:
            assert message == "success"

        chunk_frame_start = sampled_chunk_index * cur_chunk_size

        frame_indices, adjusted_fps = get_frame_indices_w_lowered_fps(
            n_video_frames=n_frames_in_chunk,
            video_fps=video_fps,
            min_fps_thres=min_fps_thres,
            max_fps_thres=max_fps_thres,
            n_target_frames=n_target_frames,
        )
        frame_indices = [chunk_frame_start + idx for idx in frame_indices]

        # Sample frames
        video_frames = video_reader.get_batch(frame_indices).asnumpy()
        video_frames = torch.from_numpy(video_frames).permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)

        # Clean up
        video_reader.seek(0)
        del video_reader

        output = {
            "video": video_frames,
            "fps": float(adjusted_fps),
            "orig_fps": video_fps,
            "num_frames": video_frames.shape[1],
            "chunk_index": sampled_chunk_index,
            "frame_start": frame_indices[0],
            "frame_end": frame_indices[-1],
            "n_orig_video_frames": n_video_frames,
        }
        return output

    return video_decoder


@video_decoder_register("chunked_video_decoder_with_fixed_fps")
def chunked_video_decoder_with_fixed_fps(
    chunk_size: int = 0,
    sequence_length: int = 34,
    min_fps_thres: int = 4,
    max_fps_thres: int = 30,
    num_threads: int = 4,
) -> dict:
    """
    Video decoder optimized for processing videos with chunked captions.

    unlike other video decoders which return video frames of requested sequence_length.
    The function returns a randomly sampled chunk with duration between 4 seconds and 8 seconds whenever possible.
    The chunk will be provided to modeling code and the frame subsampling happens on the modeling side.
    !!! IMPORTANT: it can only work with batch size 1 otherwise, different length video can not be concatenated.

    This decoder first samples a chunk from the video, then selects frames within that chunk.
    It dynamically adjusts the frame rate with a high probability (>99%) to lower the FPS
    through frame sampling when conditions allow, ensuring efficient processing while
    maintaining video quality.

    The decoder handles variable chunk durations with special processing for chunks that are
    either too short or too long, ensuring consistent output regardless of input video properties.

    Args:
        chunk_size (int): Size of video chunks in frames. If set to 0, the entire video length
                          is used as a single chunk. Defaults to 0.
        sequence_length (int): Number of frames to extract from the video. Defaults to 34.
        min_fps_thres (int): Minimum acceptable frames per second. Videos with lower FPS will
                            raise errors. Defaults to 4.
        max_fps_thres (int): Maximum acceptable frames per second. Higher FPS videos will be
                            downsampled. Defaults to 30.
        num_threads (int): Number of threads to use for video decoding with decord. Defaults to 4.

    Returns:
        dict: A dictionary containing video frames tensor and metadata including:
            - video: Tensor of shape (C, T, H, W) containing the sampled video frames
            - fps: Actual frames per second (float)
            - orig_fps: Original video frame rate (int)
            - num_frames: Number of frames extracted
            - chunk_index: Index of the sampled chunk
            - frame_start: Starting frame index in original video
            - frame_end: Ending frame index in original video
            - n_orig_video_frames: Total number of frames in original video

    Raises:
        ValueError: If video duration is too short, if FPS is outside acceptable range,
                   or if selected chunk has insufficient frames.

    Note:
        - Chunks with duration < 4.0 seconds are skipped with an error.
        - Chunks with duration > 8.0 seconds are capped to 8.0 seconds worth of frames.
        - For best results, ensure videos have FPS between min_fps_thres and max_fps_thres.
    """

    def video_decoder(
        key: str,
        data: bytes,
    ) -> dict[str, torch.Tensor | int]:
        # Check video extension
        extension = re.sub(r".*[.]", "", key)
        if extension.lower() not in _VIDEO_EXTENSIONS:
            return None

        # Read video
        video_buffer = io.BytesIO(data)
        video_reader = decord.VideoReader(video_buffer, num_threads=num_threads)

        n_target_frames = sequence_length if sequence_length > 0 else len(video_reader)
        n_video_frames = len(video_reader)
        video_fps_float = video_reader.get_avg_fps()
        video_fps = int(np.round(video_fps_float))
        cur_chunk_size = n_video_frames if chunk_size == 0 else chunk_size

        # basic check
        message = basic_check_on_inputs(
            n_video_frames=n_video_frames,
            n_target_frames=n_target_frames,
            video_fps=video_fps,
            min_fps_thres=min_fps_thres,
            max_fps_thres=max_fps_thres,
        )
        if message != "success":
            raise ValueError(message)

        sampled_chunk_index, n_frames_in_chunk, message = sample_chunk_index_from_chunked_video(
            n_video_frames=n_video_frames,
            n_target_frames=n_target_frames,
            chunk_size=cur_chunk_size,
        )
        if sampled_chunk_index == -1:
            raise ValueError(message)
        else:
            assert message == "success"

        chunk_frame_start = sampled_chunk_index * cur_chunk_size

        chunk_duration = n_frames_in_chunk / video_fps_float
        if chunk_duration < 4.0:
            raise ValueError(f"Chunk duration {chunk_duration} is less than 4.0 seconds, skipping")

        if chunk_duration > 8.0:
            n_frames_needed = int(np.ceil(8.0 * video_fps))
        else:
            n_frames_needed = n_frames_in_chunk

        chunk_frame_end = chunk_frame_start + n_frames_needed

        frame_indices = np.arange(chunk_frame_start, chunk_frame_end).tolist()

        # Sample frames
        video_frames = video_reader.get_batch(frame_indices).asnumpy()
        video_frames = torch.from_numpy(video_frames).permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)

        # Clean up
        video_reader.seek(0)
        del video_reader

        output = {
            "video": video_frames,
            "fps": video_fps_float,
            "orig_fps": video_fps,
            "num_frames": video_frames.shape[1],
            "chunk_index": sampled_chunk_index,
            "frame_start": frame_indices[0],
            "frame_end": frame_indices[-1],
            "n_orig_video_frames": n_video_frames,
        }
        return output

    return video_decoder


def construct_video_decoder(
    video_decoder_name: str = "chunked_video_decoder",
    chunk_size: int = 0,
    sequence_length: int = 34,
    min_fps_thres: int = 1,
    max_fps_thres: int = 9999,
    num_threads=4,
):
    return VIDEO_DECODER_OPTIONS[video_decoder_name](
        chunk_size=chunk_size,
        sequence_length=sequence_length,
        min_fps_thres=min_fps_thres,
        max_fps_thres=max_fps_thres,
        num_threads=num_threads,
    )
