# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import numpy as np

from dataclasses import dataclass
from typing import Tuple

import shortfin.array as sfnp


logger = logging.getLogger(__name__)


@dataclass
class Sampler:
    def sample_top_k(self, tokens: np.array, probs: np.array, k: int):
        """
        Sample k tokens from `tokens` with weights `probs`.

        Args:
            tokens: 1-D array of token IDs (shape (n_tokens,))
            probs:  1-D array of non-negative weights (shape (n_tokens,))
            k:      number of samples to draw (with replacement)

        Returns:
            choices:       1-D array of sampled tokens (shape (k,))
            chosen_probs:  1-D array of the corresponding probabilities (shape (k,))
        """
        p = probs / probs.sum()

        choices = np.random.choice(tokens, size=k, replace=True, p=p)

        token_to_p = {int(t): float(p_) for t, p_ in zip(tokens, p)}
        chosen_probs = np.array([token_to_p[int(t)] for t in choices])

        return choices, chosen_probs

    def sample_top_p(
        self,
        tokens: np.ndarray,
        probs: np.ndarray,
        p: float,
        k: int,
        return_probs=False,
    ):
        cum = np.cumsum(probs)
        idx = np.searchsorted(cum, p, side="right") + 1

        tokens, probs = tokens[:idx], probs[:idx]

        weights = probs / probs.sum()

        choices = np.random.choice(tokens, size=k, p=weights)
        chosen_probs = None
        if return_probs:
            prob_map = {tok: pr for tok, pr in zip(tokens, probs)}
            chosen_probs = np.array([prob_map[t] for t in choices])

        return choices, chosen_probs

    def select_top_k(self, logits: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function is used to get the top k tokens and their cumulative probabilities.
        """
        partitioned_tokens = np.argpartition(logits, k)
        # Slice off all axes except the last one
        zero_indices = (0,) * (partitioned_tokens.ndim - 1)

        # Obtain tokens & values from partition
        top_tokens = partitioned_tokens[zero_indices + (slice(k, None),)]
        top_values = np.take(logits, top_tokens, axis=-1)[*zero_indices]

        return top_tokens, top_values

    def select_greedy(self, logits: sfnp.device_array) -> int:
        """Greedily select a single token using `argmax`.

        Args:
            logits (sfnp.device_array): Logits from decode.

        Returns:
            int: Max token.
        """
        token = np.argmax(logits).item()
        return token
