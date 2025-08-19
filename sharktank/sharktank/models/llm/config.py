# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KVCacheConfig:
    attention_head_count_kv: int
    block_seq_stride: int
    device_block_count: int
    kv_cache_dtype: int


@dataclass
class ServiceConfig:
    module_name: str
    module_abi_version: int
    max_seq_len: int
    attn_head_dim: int
    prefill_batch_sizes: list[int]
    decode_batch_sizes: list[int]
    transformer_block_count: int
    logits_normalization: Optional[str]
    top_k: Optional[int]
    paged_kv_cache: KVCacheConfig


@dataclass
class ExportConfig:
    device_block_count: int = 512
    top_k: Optional[int] = None
    logits_normalization: Optional[str] = None
    use_attention_mask: bool = True
    prefill_final_logits: bool = False
    use_linalgext_topk: bool = True

    bs_prefill: list[int] = field(default_factory=lambda: [4])
    bs_decode: list[int] = field(default_factory=lambda: [32])
    skip_prefill: bool = False
    skip_decode: bool = False
