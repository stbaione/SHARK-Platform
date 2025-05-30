# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import numpy as np
from typing import Tuple, Union

import shortfin.array as sfnp

from abc import ABC, abstractmethod
from asyncio import gather
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Set
from uuid import uuid4

from .config import DecodeConfig, LogitsNormalization
from .sampler import Sampler
from ..messages import LlmInferenceExecRequest

logger = logging.getLogger(__name__)

TOP_P_DEFAULT_SELECTION = 32


@dataclass
class BaseBeam(ABC):
    exec_req: LlmInferenceExecRequest

    decode_config: DecodeConfig

    sampler: Sampler = field(default_factory=Sampler)

    score: float = 0.0
    accumulated_normalization: float = 0.0
    last_token: int | None = None

    @abstractmethod
    def sample_logits(self):
        """Define how to sample and select tokens for a give `Beam`"""
        pass

    @abstractmethod
    def update_score(self, value: float):
        """Update the score of a `beam`.

        Args:
            value (float): Value to update the score with.
        """
        pass

    @abstractmethod
    def update_exec_req(self):
        """Update an `LlmInferenceExecRequest`, after a decode loop"""
        pass

    @abstractmethod
    def normalize_score(self, value: float):
        """Normalize the score of a `beam`.

        Args:
            value (float): Value to normalize the score with.
        """
        pass

    @abstractmethod
    def update_final_score(self):
        """Define a `final_score` for a given beam, if applicable."""
        pass

    @classmethod
    def clone(cls, beam: "BaseBeam") -> "BaseBeam":
        return cls(
            exec_req=beam.exec_req,
            score=beam.score,
            accumulated_normalization=beam.accumulated_normalization,
            last_token=beam.last_token,
            decode_config=beam.decode_config,
        )

    def apply_temperature(self, logits: np.array) -> np.array:
        """Apply temperature to the logits of a decode invocation.

        Args:
            temperature (float): Value to use for `temperature`.
        """
        if self.decode_config.temperature == 1.0:
            return logits
        return np.divide(logits, self.decode_config.temperature)

    def _softmax(self, logits: Union[np.array, sfnp.device_array]) -> np.array:
        if isinstance(logits, sfnp.device_array):
            logits = np.array(logits)

        x_max = np.max(logits)
        e_x = np.exp(logits - x_max)
        return e_x / np.sum(e_x)

    def _log_softmax(self, logits: Union[np.array, sfnp.device_array]) -> np.array:
        if isinstance(logits, sfnp.device_array):
            logits = np.array(logits)

        c = logits.max()
        shifted_logits = logits - c
        sumexp = np.log(np.exp(shifted_logits).sum())
        return shifted_logits - sumexp

    def convert_logits_normalization(
        self,
        current: LogitsNormalization,
        target: LogitsNormalization,
        logits: np.array,
        **kwargs,
    ) -> np.array:
        logits_conversion_map = {
            LogitsNormalization.NONE: {
                LogitsNormalization.LOG_SOFTMAX: self._log_softmax,
                LogitsNormalization.SOFTMAX: self._softmax,
                LogitsNormalization.NONE: lambda logits: logits,
            },
            LogitsNormalization.SOFTMAX: {
                LogitsNormalization.LOG_SOFTMAX: np.log,
                LogitsNormalization.SOFTMAX: lambda logits: logits,
            },
            LogitsNormalization.LOG_SOFTMAX: {
                LogitsNormalization.SOFTMAX: np.exp,
                LogitsNormalization.LOG_SOFTMAX: lambda logits: logits,
            },
        }

        target_conversions = logits_conversion_map.get(current)
        if target_conversions is None:
            raise KeyError(f"Cannot convert current normalization: {current}")

        conversion_function = target_conversions.get(target)
        if conversion_function is None:
            raise KeyError(f"Cannot convert {current} to {target}")

        if kwargs:
            converted_logits = conversion_function(logits, **kwargs)
        else:
            converted_logits = conversion_function(logits)

        return converted_logits

    def _to_softmax(
        self,
        values: np.array,
        logits_normalization: LogitsNormalization,
    ):
        probs = self.convert_logits_normalization(
            logits_normalization,
            LogitsNormalization.SOFTMAX,
            values,
        )

        return probs

    def _sample_logits_top_k(
        self,
        logits: np.array,
        indices: Union[np.array, None],
        top_k: int,
        num_selections: int,
    ):
        tokens, values = self.sampler.select_top_k(logits, indices, -top_k)

        probs = self._to_softmax(
            values,
            self.decode_config.logits_normalization,
        )

        if indices is None:
            sorted_order = np.argsort(probs)[::-1]
            tokens = tokens[sorted_order]
            probs = probs[sorted_order]

        return self.sampler.sample_top_k(
            tokens=tokens,
            probs=probs,
            k=num_selections,
        )

    def _sample_logits_top_p(
        self, tokens, probs, top_p, num_selections, return_probs: bool = False
    ):
        return self.sampler.sample_top_p(
            tokens=tokens,
            probs=probs,
            p=top_p,
            k=num_selections,
            return_probs=return_probs,
        )


class Beam(BaseBeam):
    def _sample_greedy(self, logits: np.array, indices: Union[np.array, None]) -> int:
        """Select the token with the highest logit value.

        Args:
            logits (np.array): The logits from which to select.
            indices (np.array | None): Optional indices to filter logits.

        Returns:
            int: The token ID of the selected token.
        """
        if indices is not None:
            return indices.items[0]

        return self.sampler.select_greedy(logits)

    def _sample_beam_search(
        self, logits: np.array, indices: Union[np.array, None], k: int
    ) -> Tuple[np.array, np.array]:
        """Select the top-k tokens based on beam search.

        Args:
            logits (np.array): The logits from which to select.
            indices (np.array | None): Optional indices to filter logits.

        Returns:
            Tuple[np.array, np.array]: The selected tokens and their probabilities.
        """
        if indices is not None:
            indices = np.array(indices)

        tokens, probs = self.sampler.select_top_k(logits, indices, -k)

        if self.decode_config.logits_normalization == LogitsNormalization.NONE:
            probs = self.apply_temperature(probs)

        log_probs = self.convert_logits_normalization(
            self.decode_config.logits_normalization,
            LogitsNormalization.LOG_SOFTMAX,
            probs,
        ).tolist()

        return tokens, log_probs

    def _convert_results_to_log_probs(
        self,
        probs: np.array,
    ):
        log_probs = self.convert_logits_normalization(
            LogitsNormalization.SOFTMAX,
            LogitsNormalization.LOG_SOFTMAX,
            probs,
        )

        return log_probs.tolist()

    def _pre_select_top_p(
        self, logits: np.array, indices: Union[np.array, None]
    ) -> Tuple[np.array, np.array]:
        top_p_selection = min(logits.shape[-1], TOP_P_DEFAULT_SELECTION)
        tokens, values = self.sampler.select_top_k(logits, indices, -top_p_selection)
        probs = self._to_softmax(
            values,
            self.decode_config.logits_normalization,
        )

        if indices is None:
            sorted_order = np.argsort(probs)[::-1]
            tokens = tokens[sorted_order]
            probs = probs[sorted_order]

        return tokens, probs

    def _sample_default(
        self, logits: np.array, indices: Union[np.array, None], k: int | None = None
    ) -> int:
        decode_config = self.decode_config
        if decode_config.use_beam_search:
            return self._sample_beam_search(logits, indices, k)
        return self._sample_greedy(logits, indices)

    def _sample_top_k(
        self,
        logits: np.array,
        indices: Union[np.array, None],
        top_k: int,
    ) -> Tuple[np.array, np.array]:
        """Sample the top-k tokens from the logits.

        Args:
            logits (np.array): The logits from which to select.
            indices (np.array | None): Optional indices to filter logits.
            top_k (int): The number of top tokens to select.
            num_selections (int): The number of selections to make.

        Returns:
            Tuple[np.array, np.array]: The selected tokens and their probabilities.
        """
        decode_config = self.decode_config
        use_beam_search = decode_config.use_beam_search
        top_p = decode_config.top_p

        num_selections = top_k if use_beam_search or (top_p is not None) else 1
        return self._sample_logits_top_k(
            logits,
            indices,
            top_k,
            num_selections,
        )

    def sample_logits(self, num_completed_beams: int) -> int:
        """Return the single highest scoring token of the logits.

        Args:
            num_active_beams (int): Number of active beams, used for beam search.

        Returns:
            int: The `argmax` of the logits.
        """
        exec_req = self.exec_req
        decode_config = self.decode_config

        use_beam_search = decode_config.use_beam_search
        num_beams = decode_config.num_beams
        k = num_beams - num_completed_beams
        top_k = decode_config.top_k
        top_p = decode_config.top_p

        logits = np.array(exec_req.result_logits)
        indices = exec_req.result_indices

        if (top_k, top_p) == (None, None):
            return self._sample_default(logits, indices, k)

        indices = np.array(indices) if indices is not None else None
        if top_k is not None:
            tokens, probs = self._sample_top_k(
                logits,
                indices,
                top_k,
            )

        if top_p is not None:
            if top_k is None:
                tokens, probs = self._pre_select_top_p(logits, indices)

            tokens, probs = self._sample_logits_top_p(
                tokens,
                probs,
                top_p,
                k if use_beam_search else 1,
                return_probs=use_beam_search,
            )

        if use_beam_search:
            log_probs = self._convert_results_to_log_probs(
                probs,
            )
            return tokens, log_probs

        return int(tokens[0])

    def update_exec_req(self):
        """Update the `LlmInferenceExecRequest` with the selected token."""
        self.exec_req.input_token_ids.append(self.last_token)
        self.exec_req.start_position += 1

    def update_score(self, log_prob: float):
        """Increment the cumulative_log_prob of the beam.

        Args:
            log_prob (float): Log probability of the token.
        """
        self.score += log_prob

    def normalize_score(self, min_log_prob: float):
        """Track the accumulated_normalization for a given beam.

        Args:
            min_log_prob (float): Minimum log probability of the selected tokens.
        """
        self.accumulated_normalization += abs(min_log_prob)

    def update_final_score(self):
        """Calculate the final score of a beam, with a brevity penalty."""
        exec_req = self.exec_req
        self.score = (self.score - self.accumulated_normalization) / (
            len(exec_req.input_token_ids) - exec_req.prompt_length
        )


class BeamGroup:
    def __init__(
        self,
        eos_token_id: int,
        num_beams: int,
        beams: List[BaseBeam],
        selection_callback: Callable[
            [List[BaseBeam], List[BaseBeam]],
            List[BaseBeam],
        ],
    ):
        self.beam_group_id = str(uuid4())
        self.eos_token_id = eos_token_id
        self.num_beams = num_beams
        self.active_beams = beams
        self.selection_callback = selection_callback
        self.completed_beams: List[BaseBeam] = []

    @property
    def active_beam_count(self):
        return len(self.active_beams)

    async def wait(self):
        done_signals = [beam.exec_req.done for beam in self.active_beams]
        return await gather(*done_signals)

    def process_beams(self):
        beam_selections = self.selection_callback(
            self.active_beams, self.completed_beams
        )
        visited_reqs: Dict[str, LlmInferenceExecRequest] = {}
        active_beams: List[BaseBeam] = []
        active_reqs: Set[LlmInferenceExecRequest] = set()
        completed_beams: List[BaseBeam] = []
        completed_reqs: Set[LlmInferenceExecRequest] = set()

        for i in range(len(beam_selections)):
            beam = beam_selections[i]
            new_req, token = beam.exec_req, beam.last_token

            if new_req.instance_id in visited_reqs:
                visited_req = visited_reqs[new_req.instance_id]
                new_req = LlmInferenceExecRequest.copy_exec_request(visited_req)
                beam.exec_req = new_req

            visited_reqs[new_req.instance_id] = new_req
            if token == self.eos_token_id:
                completed_beams.append(beam)
                completed_reqs.add(new_req)
            else:
                active_beams.append(beam)
                active_reqs.add(new_req)

        for beam in completed_beams + active_beams:
            beam.update_exec_req()
            if beam.exec_req in completed_reqs:
                beam.exec_req.free_cache_pages()

        # Free cache pages of reqs we don't need anymore
        for beam in self.active_beams:
            if beam.exec_req not in active_reqs and beam.exec_req not in completed_reqs:
                beam.exec_req.free_cache_pages()

        self.active_beams = active_beams
        self.completed_beams.extend(completed_beams)

    def clean_up(self):
        logger.debug(f"Cleaning up {self.beam_group_id}...")

        # Ensure all requests have freed their cache pages
        for beam in self.active_beams + self.completed_beams:
            beam.exec_req.free_cache_pages()
