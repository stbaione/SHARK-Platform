# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch

from sharktank.ops import replicate, reshard_split, unshard
from sharktank.layers import *
from sharktank.types import *


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float8_e4m3fnuz,
        torch.bfloat16,
        torch.float16,
        torch.float32,
    ],
)
def test_paged(dtype: torch.dtype):
    bs = 4
    seq_length = 24
    attn_head_count = 4
    attn_head_dim = 16
    transformer_block_count = 4
    block_seq_stride = 4
    cache = PagedAttention(
        block_seq_stride=block_seq_stride,
        transformer_block_count=transformer_block_count,
        attn_head_count=attn_head_count,
        attn_head_dim=attn_head_dim,
        cache_dtype=dtype,
        attn_dtype=dtype,
        device=None,
    )

    write_seq_length = seq_length - block_seq_stride
    page_count = bs * seq_length // block_seq_stride
    page_ids = torch.arange(page_count, dtype=torch.int64)
    page_ids = page_ids.view(bs, seq_length // block_seq_stride)
    write_page_ids = page_ids[:, : write_seq_length // block_seq_stride]

    allocation = cache.allocate(page_count=page_count)
    for t in allocation:
        t[...] = torch.full(t.shape, 0.0).to(dtype=dtype)

    # Write a prefill in:
    shape = bs, write_seq_length, attn_head_count, attn_head_dim
    write_ones = torch.rand(*shape).to(dtype=dtype)
    write_twos = torch.rand(*shape).to(dtype=dtype)

    cache.write(
        allocation,
        cache_partitions=[write_ones, write_twos],
        transformer_block_index=1,
        page_ids=write_page_ids,
    )

    read_back = cache.read(
        allocation,
        transformer_block_index=1,
        page_ids=write_page_ids,
    )
    torch.testing.assert_close(write_ones, read_back[0])
    torch.testing.assert_close(write_twos, read_back[1])

    # Check the others are still zero:
    for i in range(transformer_block_count):
        if i == 1:
            continue
        read_ones = cache.read(
            allocation,
            transformer_block_index=i,
            page_ids=write_page_ids,
        )
        torch.testing.assert_close(
            read_ones[0], torch.full(read_ones[0].shape, 0.0).to(dtype=dtype)
        )
        torch.testing.assert_close(
            read_ones[1], torch.full(read_ones[0].shape, 0.0).to(dtype=dtype)
        )

    # Write timestep
    ts_shape = (bs, 1, attn_head_count, attn_head_dim)
    write_threes = torch.rand(*ts_shape).to(dtype=dtype)
    write_fours = torch.rand(*ts_shape).to(dtype=dtype)

    for i in range(block_seq_stride):
        write_pos = torch.full((bs,), write_seq_length + i, dtype=torch.int64)
        cache.write_timestep(
            allocation,
            cache_partitions=[write_threes, write_fours],
            transformer_block_index=1,
            seq_positions=write_pos,
            page_ids=page_ids,
        )

    read_back = cache.read(
        allocation,
        transformer_block_index=1,
        page_ids=page_ids,
    )

    if dtype == torch.float8_e4m3fnuz:
        check_concat_0 = torch.concat(
            [write_ones.view(torch.int8)]
            + [write_threes.view(torch.int8)] * block_seq_stride,
            dim=1,
        ).view(torch.float8_e4m3fnuz)
        check_concat_1 = torch.concat(
            [write_twos.view(torch.int8)]
            + [write_fours.view(torch.int8)] * block_seq_stride,
            dim=1,
        ).view(torch.float8_e4m3fnuz)
    else:
        check_concat_0 = torch.concat(
            [write_ones] + [write_threes] * block_seq_stride, dim=1
        )
        check_concat_1 = torch.concat(
            [write_twos] + [write_fours] * block_seq_stride, dim=1
        )

    torch.testing.assert_close(check_concat_0, read_back[0])
    torch.testing.assert_close(check_concat_1, read_back[1])


@pytest.mark.parametrize(
    "dtype,write_seq_len",
    [
        # Test all relevant dtypes
        (torch.float32, 8),
        (torch.float8_e4m3fnuz, 8),
        (torch.bfloat16, 8),
        (torch.float16, 8),
        # # Test edge cases
        (torch.float32, 4),
        (torch.float32, 12),
        (torch.float32, 0),
        (torch.float32, 24),
    ],
)
def test_write_range(dtype: torch.dtype, write_seq_len: int):
    bs = 4
    seq_length = 24
    attn_head_count = 8
    attn_head_dim = 16
    transformer_block_count = 4
    block_seq_stride = 4

    cache = PagedAttention(
        block_seq_stride=block_seq_stride,
        transformer_block_count=transformer_block_count,
        attn_head_count=attn_head_count,
        attn_head_dim=attn_head_dim,
        cache_dtype=dtype,
        attn_dtype=dtype,
        device=None,
    )

    # Allocate cache
    page_count = bs * seq_length // block_seq_stride
    page_ids = torch.arange(page_count, dtype=torch.int64).view(
        bs, seq_length // block_seq_stride
    )
    allocation = cache.allocate(page_count=page_count)
    for t in allocation:
        t[...] = torch.full(t.shape, 0.0).to(dtype=dtype)

    # Build full-random K/V
    full_ones = torch.rand(bs, seq_length, attn_head_count, attn_head_dim).to(dtype)
    full_twos = torch.rand(bs, seq_length, attn_head_count, attn_head_dim).to(dtype)

    # Compute block-aligned start‐positions
    start_positions = torch.full(
        (bs,), (seq_length - write_seq_len) // block_seq_stride, dtype=torch.int64
    )

    # Compute the token‐offsets and slice into a `tail`
    token_start = int(start_positions[0].item()) * block_seq_stride
    write_ones = full_ones[:, token_start:]
    write_twos = full_twos[:, token_start:]
    expected = [write_ones, write_twos]
    cache_partitions = [full_ones, full_twos]

    # Write tails to the cache
    cache.write_range(
        state=allocation,
        cache_partitions=cache_partitions,
        transformer_block_index=1,
        seq_positions=start_positions,  # still in blocks
        page_ids=page_ids,
    )

    # Read back the full range
    write_page_ids = page_ids  # same shape
    read_back = cache.read(
        allocation,
        transformer_block_index=1,
        page_ids=write_page_ids,
    )

    # Verify chunks were written correctly
    for part_id in (0, 1):
        suffix = read_back[part_id][:, token_start : token_start + write_seq_len]
        torch.testing.assert_close(suffix, expected[part_id])

    # verify nothing before token_start was clobbered
    token_start = int(start_positions[0].item()) * block_seq_stride
    zeros = torch.zeros(bs, token_start, attn_head_count, attn_head_dim, dtype=dtype)
    for part_id in (0, 1):
        prefix = read_back[part_id][:, :token_start]
        torch.testing.assert_close(prefix, zeros)


# TODO(stbaione): Move this to a spot that makes sense.
def test_forward_prefill_equivalence():
    bs = 4
    seq_length = 24
    query_length = 48
    attn_head_count = 8
    attn_head_dim = 16
    transformer_block_count = 3
    block_seq_stride = 4
    dtype = torch.float32
    device = torch.device("cpu")

    cache = PagedAttention(
        block_seq_stride=block_seq_stride,
        transformer_block_count=transformer_block_count,
        attn_head_count=attn_head_count,
        attn_head_dim=attn_head_dim,
        cache_dtype=dtype,
        attn_dtype=dtype,
        device=device,
    )

    # Allocate pages
    page_count = bs * (seq_length // block_seq_stride)
    seq_block_ids = torch.arange(page_count, dtype=torch.int64, device=device).view(
        bs, -1
    )
    cache_state = cache.allocate(page_count=page_count)

    # Zero the cache state
    for t in cache_state:
        t[...] = torch.full(t.shape, 0.0).to(dtype=dtype)

    # Random inputs
    q = torch.rand(
        bs, query_length, attn_head_count, attn_head_dim, dtype=dtype, device=device
    )
    k = torch.rand(
        bs, seq_length, attn_head_count, attn_head_dim, dtype=dtype, device=device
    )
    v = torch.rand(
        bs, seq_length, attn_head_count, attn_head_dim, dtype=dtype, device=device
    )

    # Prefill without offsets
    out1 = cache.forward_prefill(
        q=q,
        k=k,
        v=v,
        cache_state=cache_state,
        seq_block_ids=seq_block_ids,
        block_index=1,
        start_positions=None,
        attention_kernel="torch",
        head_count_attn=attn_head_count,
        cache_quantizer=None,
        fake_quant=None,
        softcap=None,
        scale=None,
        mask=None,
        probs_quantizer=None,
    )

    # Block aligned start positions
    start_token_positions = torch.tensor(
        [0, 3, 7, 12], dtype=torch.int64, device=device
    )
    start_positions = start_token_positions // block_seq_stride

    # Prefill with offsets
    out2 = cache.forward_prefill(
        q=q,
        k=k,
        v=v,
        cache_state=cache_state,
        seq_block_ids=seq_block_ids,
        block_index=1,
        start_positions=start_positions,
        attention_kernel="torch",
        head_count_attn=attn_head_count,
        cache_quantizer=None,
        fake_quant=None,
        softcap=None,
        scale=None,
        mask=None,
        probs_quantizer=None,
    )

    # Verify outputs are close
    torch.testing.assert_close(out1, out2, atol=1e-6, rtol=1e-5)


def test_write_range_varied_start_positions():
    bs = 4
    seq_length = 24
    attn_head_count = 8
    attn_head_dim = 16
    transformer_block_count = 3
    block_seq_stride = 4
    dtype = torch.float32
    device = torch.device("cpu")

    cache = PagedAttention(
        block_seq_stride=block_seq_stride,
        transformer_block_count=transformer_block_count,
        attn_head_count=attn_head_count,
        attn_head_dim=attn_head_dim,
        cache_dtype=dtype,
        attn_dtype=dtype,
        device=device,
    )

    # Allocate pages
    page_count = bs * (seq_length // block_seq_stride)
    page_ids = torch.arange(page_count, dtype=torch.int64, device=device).view(bs, -1)
    allocation = cache.allocate(page_count=page_count)

    # Zero the cache state
    for t in allocation:
        t[...] = torch.full(t.shape, 0.0).to(dtype=dtype)

    # Random inputs
    full_k = torch.rand(
        bs, seq_length, attn_head_count, attn_head_dim, dtype=dtype, device=device
    )
    full_v = torch.rand(
        bs, seq_length, attn_head_count, attn_head_dim, dtype=dtype, device=device
    )

    # Block-aligned start positions
    start_token_positions = torch.tensor(
        [0, 3, 7, 12], dtype=torch.int64, device=device
    )
    start_positions = start_token_positions // block_seq_stride  # shape: [bs]

    # Write suffix to the cache
    cache.write_range(
        state=allocation,
        cache_partitions=[full_k, full_v],
        transformer_block_index=1,
        seq_positions=start_positions,  # block indices
        page_ids=page_ids,  # [bs, num_blocks]
    )

    # Read back from cache
    read_k, read_v = cache.read(
        allocation,
        transformer_block_index=1,
        page_ids=page_ids,
    )

    # Verify values
    for i in range(bs):
        # compute token offset for this batch element
        offs = int(start_positions[i].item()) * block_seq_stride
        rem_len = seq_length - offs

        # Suffixes must match
        torch.testing.assert_close(
            read_k[i, offs : offs + rem_len],
            full_k[i, offs : offs + rem_len],
            atol=1e-6,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            read_v[i, offs : offs + rem_len],
            full_v[i, offs : offs + rem_len],
            atol=1e-6,
            rtol=1e-5,
        )

        # Ensure the prefix is zero
        if offs > 0:
            zeros = torch.zeros(
                offs, attn_head_count, attn_head_dim, dtype=dtype, device=device
            )
            torch.testing.assert_close(read_k[i, :offs], zeros)
            torch.testing.assert_close(read_v[i, :offs], zeros)


def test_sharded_paged():
    bs = 4
    seq_length = 24
    attn_head_count = 8
    attn_head_dim = 16
    transformer_block_count = 4
    block_seq_stride = 4
    shard_count = 4
    cache = PagedAttention(
        block_seq_stride=block_seq_stride,
        transformer_block_count=transformer_block_count,
        attn_head_count=attn_head_count,
        attn_head_dim=attn_head_dim,
        shard_count=shard_count,
        cache_dtype=torch.float32,
        attn_dtype=torch.float32,
        device=None,
    )

    write_seq_length = seq_length - block_seq_stride
    page_count = bs * seq_length // block_seq_stride
    page_ids = torch.arange(page_count, dtype=torch.int64)
    page_ids = page_ids.view(bs, seq_length // block_seq_stride)
    page_ids = replicate(page_ids, shard_count)
    write_page_ids = page_ids[:, : write_seq_length // block_seq_stride]

    allocation = cache.allocate(page_count=page_count)

    # Write a prefill in:
    shape = (bs, write_seq_length, attn_head_count, attn_head_dim)
    write_ones = reshard_split(torch.rand(shape), dim=2, count=shard_count)
    write_twos = reshard_split(torch.rand(shape), dim=2, count=shard_count)

    cache.write(
        allocation,
        cache_partitions=[write_ones, write_twos],
        transformer_block_index=1,
        page_ids=write_page_ids,
    )

    read_back = cache.read(
        allocation,
        transformer_block_index=1,
        page_ids=write_page_ids,
    )
    torch.testing.assert_close(unshard(write_ones), unshard(read_back[0]))
    torch.testing.assert_close(unshard(write_twos), unshard(read_back[1]))

    # Write timestep
    shape = (bs, 1, attn_head_count, attn_head_dim)
    write_threes = reshard_split(torch.rand(shape), dim=2, count=shard_count)
    write_fours = reshard_split(torch.rand(shape), dim=2, count=shard_count)

    for i in range(block_seq_stride):
        write_pos = replicate(
            torch.full((bs,), write_seq_length + i, dtype=torch.int64), shard_count
        )

        cache.write_timestep(
            allocation,
            cache_partitions=[write_threes, write_fours],
            transformer_block_index=1,
            seq_positions=write_pos,
            page_ids=page_ids,
        )

    read_back = cache.read(
        allocation,
        transformer_block_index=1,
        page_ids=page_ids,
    )

    check_concat_0 = torch.concat(
        [unshard(write_ones)] + block_seq_stride * [unshard(write_threes)], dim=1
    )
    check_concat_1 = torch.concat(
        [unshard(write_twos)] + block_seq_stride * [unshard(write_fours)], dim=1
    )

    torch.testing.assert_close(check_concat_0, unshard(read_back[0]))
    torch.testing.assert_close(check_concat_1, unshard(read_back[1]))
