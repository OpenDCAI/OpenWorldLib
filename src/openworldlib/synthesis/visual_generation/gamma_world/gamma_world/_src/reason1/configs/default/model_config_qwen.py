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

from typing import Dict, List, Optional, Union

import attrs
from transformers.utils import is_flash_attn_2_available

from gamma_world._src.imaginaire.lazy_config import LazyCall as L
from gamma_world._src.reason1.configs.default.model_config import FSDP2ModelConfig

_DEFAULT_ATTN_IMPLEMENTATION = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"

@attrs.define
class QwenVisionConfig:
    _attn_implementation_autoset: bool = True
    _name_or_path: str = ""
    _attn_implementation: str = _DEFAULT_ATTN_IMPLEMENTATION

    add_cross_attention: bool = False
    architectures: Optional[List[str]] = None
    bad_words_ids: Optional[List[List[int]]] = None
    begin_suppress_tokens: Optional[List[int]] = None
    bos_token_id: Optional[int] = None
    chunk_size_feed_forward: int = 0
    cross_attention_hidden_size: Optional[int] = None
    decoder_start_token_id: Optional[int] = None
    depth: int = 32
    diversity_penalty: float = 0.0
    do_sample: bool = False
    early_stopping: bool = False
    encoder_no_repeat_ngram_size: int = 0
    eos_token_id: Optional[int] = None
    exponential_decay_length_penalty: Optional[float] = None
    finetuning_task: Optional[str] = None
    forced_bos_token_id: Optional[int] = None
    forced_eos_token_id: Optional[int] = None
    fullatt_block_indexes: Optional[List[int] | None] = [7, 15, 23, 31]
    hidden_act: str = "silu"
    hidden_size: int = 1280
    id2label: Dict[int, str] = {0: "LABEL_0", 1: "LABEL_1"}
    in_channels: int = 3
    in_chans: int = 3
    intermediate_size: int = 3420
    is_decoder: bool = False
    is_encoder_decoder: bool = False
    label2id: Dict[str, int] = {"LABEL_0": 0, "LABEL_1": 1}
    length_penalty: float = 1.0
    max_length: int = 20
    min_length: int = 0
    model_type: str = "qwen2_5_vl"
    no_repeat_ngram_size: int = 0
    num_beam_groups: int = 1
    num_beams: int = 1
    num_heads: int = 16
    num_return_sequences: int = 1
    out_hidden_size: Optional[int | None] = 2048
    output_attentions: bool = False
    output_hidden_states: bool = False
    output_scores: bool = False
    pad_token_id: Optional[int] = None
    patch_size: int = 14
    prefix: Optional[str] = None
    problem_type: Optional[str] = None
    pruned_heads: Dict = attrs.field(factory=dict)
    remove_invalid_values: bool = False
    repetition_penalty: float = 1.0
    return_dict: bool = True
    return_dict_in_generate: bool = False
    sep_token_id: Optional[int] = None
    spatial_merge_size: int = 2
    spatial_patch_size: int = 14
    suppress_tokens: Optional[List[int]] = None
    task_specific_params: Optional[Dict] = None
    temperature: float = 1.0
    temporal_patch_size: int = 2
    tf_legacy_loss: bool = False
    tie_encoder_decoder: bool = False
    tie_word_embeddings: bool = True
    tokenizer_class: Optional[str] = None
    tokens_per_second: Optional[int | None] = 2
    top_k: int = 50
    top_p: float = 1.0
    torch_dtype: str = "bfloat16"
    torchscript: bool = False
    typical_p: float = 1.0
    use_bfloat16: bool = False
    window_size: Optional[int | None] = 112

    embed_dim: Optional[int | None] = None
    mlp_ratio: Optional[int | None] = None

@attrs.define
class QwenModelConfig(FSDP2ModelConfig):
    _attn_implementation: str = _DEFAULT_ATTN_IMPLEMENTATION
    _attn_implementation_autoset: bool = True
    name_or_path: str = "Qwen/Qwen2.5-VL-3B-Instruct"

    add_cross_attention: bool = False
    architectures: List[str] = ["Qwen2_5_VLForConditionalGeneration"]
    attention_dropout: float = 0.0
    bad_words_ids: Optional[List[List[int]]] = None
    begin_suppress_tokens: Optional[List[int]] = None
    bos_token_id: int = 151643
    chunk_size_feed_forward: int = 0
    cross_attention_hidden_size: Optional[int] = None
    decoder_start_token_id: Optional[int] = None
    diversity_penalty: float = 0.0
    do_sample: bool = False
    early_stopping: bool = False
    encoder_no_repeat_ngram_size: int = 0
    eos_token_id: int = 151645
    exponential_decay_length_penalty: Optional[float] = None
    finetuning_task: Optional[str] = None
    forced_bos_token_id: Optional[int] = None
    forced_eos_token_id: Optional[int] = None
    hidden_act: str = "silu"
    hidden_size: int = 2048
    id2label: Dict[int, str] = {0: "LABEL_0", 1: "LABEL_1"}
    image_token_id: int = 151655
    initializer_range: float = 0.02
    intermediate_size: Optional[int | None] = 11008
    is_decoder: bool = False
    is_encoder_decoder: bool = False
    label2id: Dict[str, int] = {"LABEL_0": 0, "LABEL_1": 1}
    length_penalty: float = 1.0
    max_length: int = 20
    max_position_embeddings: int = 128000
    max_window_layers: int = 70
    min_length: int = 0
    model_type: str = "qwen2_5_vl"
    no_repeat_ngram_size: int = 0
    num_attention_heads: int = 16
    num_beam_groups: int = 1
    num_beams: int = 1
    num_hidden_layers: int = 36
    num_key_value_heads: int = 2
    num_return_sequences: int = 1
    output_attentions: bool = False
    output_hidden_states: bool = False
    output_scores: bool = False
    pad_token_id: Optional[int] = None
    prefix: Optional[str] = None
    problem_type: Optional[str] = None
    pruned_heads: Dict = attrs.field(factory=dict)
    remove_invalid_values: bool = False
    repetition_penalty: float = 1.0
    return_dict: bool = True
    return_dict_in_generate: bool = False
    rms_norm_eps: float = 1e-6
    rope_scaling: Dict[str, Union[str, List[int]]] = {
        "mrope_section": [16, 24, 24],
        "rope_type": "default",
        "type": "default",
    }
    rope_theta: float = 1_000_000.0
    sep_token_id: Optional[int] = None
    sliding_window: int = 32768
    suppress_tokens: Optional[List[int]] = None
    task_specific_params: Optional[Dict] = None
    temperature: float = 1.0
    tf_legacy_loss: bool = False
    tie_encoder_decoder: bool = False
    tie_word_embeddings: bool = True
    tokenizer_class: Optional[str] = None
    top_k: int = 50
    top_p: float = 1.0
    torch_dtype: str = "bfloat16"
    torchscript: bool = False
    transformers_version: str = "4.51.0.dev0"
    typical_p: float = 1.0
    use_bfloat16: bool = False
    use_cache: bool = False
    use_return_dict: bool = True
    use_sliding_window: bool = False
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    vision_token_id: int = 151654
    vocab_size: int = 151936
    vision_config: QwenVisionConfig = L(QwenVisionConfig)()

    def __getitem__(self, item):
        return getattr(self, item)

    def getattr(self, item):
        if item == "_name_or_path":
            return self.name_or_path
        else:
            return super().getattr(item)
