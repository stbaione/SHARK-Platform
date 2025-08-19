# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for attention op implementations."""

import unittest
import torch
from parameterized import parameterized

from sharktank import ops
from sharktank.ops import attention_impls
from sharktank.utils.testing import OpComparisonTestBase, OpTestConfig


class TestScaledDotProductAttention(OpComparisonTestBase):
    """Test scaled dot product attention implementations."""

    @parameterized.expand(
        [
            # No causal, no mask
            (2, 8, 128, 64, torch.float16, False, False, None, None),
            (2, 8, 128, 64, torch.float32, False, False, None, None),
            # Test causal attention
            (2, 8, 128, 64, torch.float16, True, False, None, None),
            (2, 8, 128, 64, torch.float16, True, False, 0.125, None),
            # Test explicit masking
            (2, 8, 128, 64, torch.float16, False, True, None, None),
            (2, 8, 256, 64, torch.float32, False, True, None, None),
            # Test softcap
            (1, 4, 64, 32, torch.float32, False, False, None, 50.0),
        ]
    )
    def test_attention_variants(
        self,
        batch,
        heads,
        seq_len,
        head_dim,
        dtype,
        is_causal,
        has_mask,
        scale,
        softcap,
    ):
        """Test attention with various configurations."""

        torch.manual_seed(42)
        q = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype)
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype)
        v = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype)

        if has_mask:
            # Create a simple attention mask with shape [1, 1, seq_len, seq_len]
            # This broadcasts across all batches and heads
            mask = torch.triu(torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1)
            mask = mask.unsqueeze(0).unsqueeze(0)
            a = mask.to(dtype)
        else:
            a = None

        fail_on_not_implemented = True
        if softcap:
            # Other implementations don't implement softcap, so we just test that it runs for decomposed
            fail_on_not_implemented = False

        # Use decomposed as reference since it supports all features
        config = OpTestConfig(
            op=ops.scaled_dot_product_attention,
            reference_impl=attention_impls.scaled_dot_product_attention_decomposed,
            test_impls="all",
            args=[q, k, v, a],
            kwargs={
                "is_causal": is_causal,
                "scale": scale,
                "softcap": softcap,
                "impl": None,
            },
            fail_on_not_implemented=fail_on_not_implemented,
        )
        self.compare_implementations(config)


if __name__ == "__main__":
    unittest.main()
