# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
import logging

import numpy as np
from typing import List, Tuple, Union


from .beam_group import BaseBeam, BeamGroup, BaseBeamScorer
from .base_token_selection_strategy import (
    BaseTokenSelectionStrategy,
)
from .config import LogitsNormalization

from ..messages import LlmInferenceExecRequest, InferencePhase

logger = logging.getLogger(__name__)


TOP_P_DEFAULT_SELECTION = 32


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


class BeamSearchScorer(BaseBeamScorer):
    def __init__(self, config):
        self.min_log_prob: float = 0.0
        self.top_score: float | None = None
        self.top_beam: BaseBeam | None = None

        super().__init__(config)

    def update_score(
        self,
        beam: BaseBeam,
        log_prob: float,
    ) -> None:
        """Update the score of a beam with the log probability of the selected token.

        Args:
            beam (BaseBeam): The beam to update.
            log_prob (float): Log probability of the token.
        """
        if log_prob < self.min_log_prob:
            self.min_log_prob = log_prob

        beam.score += log_prob

        if self.top_score is None or beam.score > self.top_score:
            self.top_score = beam.score
            self.top_beam = beam

    def finalize_score(
        self,
        beam: BaseBeam,
    ) -> None:
        """Finalize the score of a beam after all tokens have been selected.

        Args:
            beam (BaseBeam): The beam to finalize.
        """
        beam.score = beam.score - beam.accumulated_normalization
        return self.penalize_brevity(beam)

    def normalize_score(
        self,
        beam: BaseBeam,
        min_log_prob: float,
    ) -> None:
        """Normalize the score of a beam based on the minimum log probability.

        Args:
            beam (BaseBeam): The beam to normalize.
            min_log_prob (float): Minimum log probability of the selected tokens.
        """
        beam.accumulated_normalization += abs(min_log_prob)

    def score_beams(self, beams, k: int, normalize: bool = True):
        sorted_selections = sorted(beams, key=lambda beam: beam.score, reverse=True)[:k]
        if normalize:
            for beam in sorted_selections:
                self.normalize_score(beam, self.min_log_prob)

        return sorted_selections

    def select_beams(
        self,
        active_beams: List[Beam],
        completed_beams: List[Beam],
    ) -> List[Beam]:
        """Handle the selection of the `top_k` beams within a decode step.

        Args:
            active_beams (List[IndependentBeam]): Beams that are still active.
            completed_beams (Set[IndependentBeam]): Beams that have been completed.

        Returns:
            List[IndependentBeam]: The `top_k` selections, containing necessary info for `beam_group` to handle choosing and processing beams.
        """
        config = self.config
        num_beams = config.decode_config.num_beams
        k = num_beams - len(completed_beams)
        selections: List[Beam] = []

        # Parse each beam to select the next candidates
        for beam in active_beams:
            top_tokens, top_values = beam.sample_logits(len(completed_beams))
            for token, value in zip(top_tokens, top_values):

                new_beam = Beam.clone(beam)
                new_beam.last_token = token
                self.update_score(new_beam, value)
                selections.append(new_beam)

        # Ensure we have enough beams to fill the `num_beams` requirement
        if len(selections) < k:
            beams_to_add = num_beams - len(selections)
            for _ in range(beams_to_add):
                new_beam = Beam.clone(self.scorer.top_beam)
                selections.append(new_beam)

        selections = self.score_beams(selections, k)
        self.reset()
        return selections

    def reset(self):
        """Reset the scorer state."""
        self.min_log_prob = 0.0
        self.top_score = None


@dataclass
class TokenSelector(BaseTokenSelectionStrategy):
    scorer: BeamSearchScorer
    min_log_prob: float = 0.0

    def select_independent(
        self,
        active_beams: List[Beam],
        completed_beams: List[Beam],
    ) -> List[Beam]:
        """Greedily select a token for each active beam.

        Args:
            active_beams (List[IndependentBeam]): Beams that are still active.
            _ (List[IndependentBeam]): Beams that are completed.

        Returns:
            List[IndependentBeam]: Beams with new token selected.
        """
        selections = []

        # Sample logits for each active beam for it to select its next token.
        for beam in active_beams:
            token = beam.sample_logits(len(completed_beams))
            beam.last_token = token
            selections.append(
                beam,
            )

        return selections

    def _stream_single_beam(self, beam_group: BeamGroup) -> List[Beam]:
        """Stream a single beam for the `multi_greedy` strategy.

        Args:
            beam_group (BeamGroup): The group of beams to process.

        Returns:
            List[IndependentBeam]: Beams with new token selected.
        """
        results_callback = self.token_selection_strategy_config.results_callback

        assert (
            beam_group.num_beams == 1
        ), "Streaming is not supported for multi-hypothesis yet."

        beam = beam_group.active_beams[0]
        results_callback(beam.last_token)

    async def decode(
        self,
        exec_req: LlmInferenceExecRequest,
    ):
        """Orchestrate decode loop for `multi_greedy` selection strategy.

        Args:
            exec_req (LlmInferenceExecRequest): Initial inference request, post prefill.
        """
        self._log_sampling_method()
        config = self.token_selection_strategy_config

        exec_req.reset(InferencePhase.DECODE)

        num_beams = config.decode_config.num_beams
        use_beam_search = config.decode_config.use_beam_search

        # Copy `exec_req` to `num_beams` total requests
        if num_beams > 1 and not use_beam_search:
            exec_reqs = self.replicate_inference_exec_requests(exec_req, num_beams - 1)
        else:
            exec_reqs = [exec_req]

        beams = [
            Beam(exec_req, decode_config=config.decode_config) for exec_req in exec_reqs
        ]

        selection_callback = (
            self.select_independent if not use_beam_search else self.scorer.select_beams
        )
        beam_group = BeamGroup(
            config.eos_token_id,
            config.decode_config.num_beams,
            beams,
            selection_callback,
        )

        reservations = beam_group.active_beam_count
        config.decode_begin_callback(rid=exec_req.orig_instance_id, count=reservations)
        for _ in range(config.decode_config.max_completion_tokens):
            if exec_req.status_tracker.is_disconnected():
                break

            active_beam_count = len(beam_group.active_beams)
            if reservations > active_beam_count:
                release_amount = reservations - active_beam_count
                config.decode_end_callback(
                    rid=exec_req.orig_instance_id, count=release_amount
                )
                reservations = active_beam_count

            if reservations < active_beam_count:
                acquire_amount = active_beam_count - reservations
                config.decode_begin_callback(
                    rid=exec_req.orig_instance_id, count=acquire_amount
                )
                reservations = active_beam_count

            for beam in beam_group.active_beams:
                req = beam.exec_req
                req.reset(InferencePhase.DECODE)
                config.decode_callback(req)

            await beam_group.wait()
            beam_group.process_beams()

            if not beam_group.active_beams:
                break

            if config.decode_config.num_beams == 1 and not use_beam_search:
                self._stream_single_beam(beam_group)

        config.decode_end_callback(rid=exec_req.orig_instance_id, count=reservations)
        beam_group.clean_up()

        self.get_results(beam_group)

    def _get_results_beam_search(self, beam_group: BeamGroup, results: List[List[int]]):
        for beam in beam_group.active_beams:
            beam.update_final_score()

        active_beams = sorted(
            [beam for beam in beam_group.active_beams],
            key=lambda beam: beam.score,
            reverse=True,
        )
        active_beams = beam_group.active_beams
        active_beams = self.scorer.score_beams(
            active_beams, len(active_beams), normalize=False
        )
        for i in range(beam_group.num_beams - len(results)):
            beam = active_beams[i]
            results.append(beam.exec_req.input_token_ids[beam.exec_req.prompt_length :])

        return results

    def get_results(self, beam_group: BeamGroup):
        config = self.token_selection_strategy_config
        use_beam_search = config.decode_config.use_beam_search
        if config.decode_config.num_beams == 1 and not use_beam_search:
            self._stream_single_beam(beam_group)
            return

        results = [
            beam.exec_req.input_token_ids[beam.exec_req.prompt_length :]
            for beam in beam_group.completed_beams
        ]
        if len(results) < beam_group.num_beams:
            if use_beam_search:
                results = self._get_results_beam_search(beam_group, results)
            else:
                results.extend(
                    [
                        beam.exec_req.input_token_ids[beam.exec_req.prompt_length :]
                        for beam in beam_group.active_beams
                    ]
                )

        config.results_callback(results)
