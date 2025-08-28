# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch

from sharktank.layers.configs import LlamaHParams, LlamaModelConfig


def test_llama_hp_params_to_from_gguf_props_roundtrip():
    params = LlamaHParams(
        model_arch="llama",
        context_length=1,
        embedding_length=2,
        block_count=3,
        feed_forward_length=3,
        rope_dimension_count=4,
        rope_freq_base=5.0,
        attention_head_count=6,
        attn_head_dim=4,
        attention_layer_norm_rms_epsilon=8.0,
        attention_head_count_kv=9,
        expert_count=10,
        expert_used_count=11,
        n_dense_layers=None,
    )
    roundtripped_params = LlamaHParams.from_gguf_props(params.to_gguf_props())
    assert params == roundtripped_params


def test_llama_model_config_to_from_properties_roundtrip():
    hp = LlamaHParams(
        model_arch="llama",
        context_length=1,
        embedding_length=2,
        block_count=3,
        feed_forward_length=3,
        rope_dimension_count=4,
        rope_freq_base=5.0,
        attention_head_count=6,
        attn_head_dim=4,
        attention_layer_norm_rms_epsilon=8.0,
        attention_head_count_kv=9,
        expert_count=10,
        expert_used_count=11,
        n_dense_layers=None,
    )
    config = LlamaModelConfig(
        hp=hp,
        block_seq_stride=12,
        kv_cache_type="custom_kv_cache",
        kv_cache_dtype=torch.float8_e4m3fnuz,
        activation_dtype=torch.float16,
        attention_dtype=torch.float32,
        fake_quant=False,
        tensor_parallelism_size=13,
        block_to_pipeline_map=(
            15,
            16,
        ),
        pipeline_to_device_map=(
            (
                17,
                18,
            ),
            (19,),
        ),
        attention_kernel="custom_attention_kernel",
        use_hf=True,
        attention_chunk_size=20,
        chunked_attention_layers=set([21, 22]),
    )
    roundtripped_config = LlamaModelConfig.from_properties(config.to_properties())
    assert config == roundtripped_config
