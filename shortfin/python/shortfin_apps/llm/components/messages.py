# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from enum import Enum
from uuid import uuid4

import shortfin as sf
import shortfin.array as sfnp
from shortfin.interop.fastapi import RequestStatusTracker

from .kvcache.base_attention_cache import BasePagedAttentionCache, PageAllocation
from .kvcache.trie_attention_cache import TriePagedAttentionCache
from ...utils import InferenceExecRequest


class InferencePhase(Enum):
    PREFILL = 1
    DECODE = 2


class LlmInferenceExecRequest(InferenceExecRequest):
    """Performs a prefill operation."""

    def __init__(
        self,
        phase: InferencePhase,
        input_token_ids: list[int],
        rid=None,
        orig_instance_id=None,
        page_ids: list[int] | None = None,
        status_tracker: RequestStatusTracker | None = None,
    ):
        super().__init__()
        self.phase = phase
        self.start_position: int = 0
        self.input_token_ids = input_token_ids
        self.prompt_length = len(input_token_ids)
        self.done = sf.VoidFuture()
        self.rid = rid
        # Unique `instance_id` for token selection strategies that may need
        # to differentiate between an original req and a copy of a req.
        self.instance_id = str(uuid4())

        # Unique ID for an InferenceExecRequest shared between copies and executions.
        self.orig_instance_id = (
            self.instance_id if orig_instance_id is None else orig_instance_id
        )

        # Move the result array to the host and sync to ensure data is
        # available.
        self.return_host_array: bool = True

        # Result logits as [1, sl, d] where 1 is the preserved batch dim,
        # sl is either 1 (not return_all_logits) or >=1 (return_all_logits).
        self.result_logits: sfnp.device_array | None = None
        self.result_indices: sfnp.device_array | None = None

        # Current running score of the decode req
        self.score: float = 0.0

        # Cache pages that have been locked for this request.
        self._cache: BasePagedAttentionCache | None = None
        self.allocation: PageAllocation | None = None
        self.page_ids: list[int] = page_ids
        self.status_tracker: RequestStatusTracker | None = status_tracker

    @property
    def block_count(self):
        if self.page_ids:
            return len(self.page_ids)

        if self.allocation:
            return len(self.allocation.pages)

        return 0

    @classmethod
    def copy_exec_request(
        cls, exec_req: "LlmInferenceExecRequest"
    ) -> "LlmInferenceExecRequest":
        new_exec_req = cls(
            exec_req.phase,
            exec_req.input_token_ids.copy(),
            rid=exec_req.rid,
            orig_instance_id=exec_req.orig_instance_id,
        )

        new_exec_req.start_position = exec_req.start_position
        new_exec_req.prompt_length = exec_req.prompt_length
        new_exec_req._cache = exec_req._cache

        # check if the cache is instance of TriePagedAttentionCache then
        # pass token_ids to fork_pages
        if isinstance(new_exec_req._cache, TriePagedAttentionCache):
            new_exec_req.allocation = new_exec_req._cache.fork_pages(
                exec_req.allocation.pages, exec_req.input_token_ids
            )
        else:
            new_exec_req.allocation = new_exec_req._cache.fork_pages(
                exec_req.allocation.pages
            )
        return new_exec_req

    def reset(self, phase: InferencePhase):
        """Resets all per request state in preparation for an subsequent execution."""
        self.phase = phase
        self.done = sf.VoidFuture()
        self.return_host_array = True
        self.result_logits = None

    def cache_page_indices(self, max_len: int) -> list[int]:
        if self.page_ids:
            return self.page_ids

        if not self.allocation:
            return []
        indices = [p.index for p in self.allocation.pages[:max_len]]
        return indices

    def publish_allocated_pages(self, up_to_page_index: int):
        if self.allocation is not None:
            self.allocation.publish_pages_for_tokens(
                self.input_token_ids, publish_incomplete_page=False
            )

    def free_cache_pages(self):
        if self.allocation:
            self.allocation.release_pages()
            self.allocation = None

    def __repr__(self) -> str:
        """
        String representation for logging purposes. It looks like this:

        LlmInferenceExecRequest[phase=P,pos=0,rid=test123,flags=host,input_token_ids=[1, 2, 3, 4]]

        Use
        `logging.debug("Request: %r", request)`
        and not
        `logging.debug(f"Request: {request}")
        to avoid running through this method all the time.
        """
        phase_char = "D" if self.phase == InferencePhase.DECODE else "P"
        flags = []
        if self.return_host_array:
            flags.append("host")
        flags_str = ",".join(flags)
        return f"LlmInferenceExecRequest[phase={phase_char},pos={self.start_position},rid={self.rid},instance_id={self.instance_id},flags={flags_str},input_token_ids={self.input_token_ids}]"
