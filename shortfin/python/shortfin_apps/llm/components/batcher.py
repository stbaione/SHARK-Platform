# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import math
from typing import List, Optional, Tuple, Union
from uuid import uuid4


import shortfin as sf
import shortfin.array as sfnp

from shortfin import Fiber

from .config_struct import ModelParams
from .device_array_cache import DeviceArrayCache
from .invocation import (
    LlmInvoker,
    LlmTaskInput,
    PrefillTask,
    DecodeTask,
)
from .kvcache.base_attention_cache import (
    BasePagedAttentionCache,
    CacheAllocationFailure,
)
from .messages import LlmInferenceExecRequest
from .scheduler import Scheduler

from ...utils import BatcherProcess

logger = logging.getLogger(__name__)


########################################################################################
# Batcher
########################################################################################


class LlmBatcherProcess(BatcherProcess):
    """This batcher provides a high-level mechanism for dispatching LLM tasks."""

    STROBE_SHORT_DELAY = 0.065
    STROBE_LONG_DELAY = 0.065

    def __init__(
        self,
        name: str,
        fiber: Fiber,
        page_cache: BasePagedAttentionCache,
        model_params: ModelParams,
        functions: dict[int, sf.ProgramFunction],
        ideal_batch_size: int,
        program_isolation: str,
    ):
        super().__init__(fiber=fiber)
        self.name = name
        self.page_cache: BasePagedAttentionCache = page_cache
        self.model_params = model_params
        self.functions = functions
        self.pending: set[LlmInferenceExecRequest] = set()
        # TODO: There is no "ideal" batch size. Use prefill/decode dynamic
        # batching in the scheduling algo.
        self.ideal_batch_size: int = ideal_batch_size
        self.page_seq_stride = self.model_params.paged_kv_cache.block_seq_stride
        self.scheduler = Scheduler(ideal_batch_size=self.ideal_batch_size)
        self.array_cache: DeviceArrayCache = DeviceArrayCache(fiber.device(0))

        self.program_isolation = program_isolation

        self.active_requests: dict[str, List[LlmInferenceExecRequest]] = {}

    def handle_inference_request(self, request):
        """Handle an inference request."""
        self.pending.add(request)

    def shutdown(self):
        """Shutdown the batcher process."""
        super().shutdown()
        self.array_cache.free()

    async def process_batches(self):
        """Process batches of requests."""
        await self.board_flights()

    def reserve_workload(self, *, rid, count):
        return self.scheduler.reserve_workload(batcher=self, count=count, rid=rid)

    def custom_message(self, msg):
        if self.scheduler.handle_scheduler(msg):
            return

        super().custom_message(msg)

    async def board_flights(self):
        """Make, schedule, and launch a batch of pending requests."""
        # TODO: Add lock on self.pending
        pending = self.pending
        self.pending = set()

        if len(pending) == 0:
            return

        # Determine the requested requests these jobs are for
        rids = set([j.orig_instance_id for j in pending])

        # Group jobs together under their rid
        rid_map = {rid: [] for rid in rids}
        for j in pending:
            rid_map[j.orig_instance_id].append(j)

        to_schedule = self.scheduler.should_execute(rid_map, self.strobes)

        page_cache = self.page_cache
        scheduled = []
        for job in to_schedule:
            scheduled = scheduled + job
            self.board(page_cache, self.fiber, job)
            logger.debug("Post boarding cache state: %r", page_cache)

        pending = set(pending) - set(scheduled)
        self.pending = self.pending | pending

    def make_invoker(
        self,
        page_cache: BasePagedAttentionCache,
        fiber: Fiber,
        exec_requests: list[LlmInferenceExecRequest],
        group_id: str,
        completion_callback,
    ) -> "LlmInvoker":
        """Create instance of `LlmInvoker`.

        Args:
            page_cache (BasePagedAttentionCache): KVCache instance.
            fiber (Fiber): Fiber to execute invocation on.
            exec_requests (list[LlmInferenceExecRequest]): Request batch for invocation.

        Returns:
            LlmInvoker: Process to handle execution of VMFB.
        """
        ...

    def board(
        self, page_cache: BasePagedAttentionCache, fiber: Fiber, to_schedule: set
    ):
        """Create and launch an LlmExecutorProcess for the given request batch.

        Args:
            page_cache (BasePagedAttentionCache): KVCache to use for this flight.
            fiber (Fiber): Fiber to use for invocation.
            to_schedule (set): Scheduled requests to be invoked in this flight.
        """
        # Fill prefill flights.
        assert len(to_schedule) > 0
        assert len(to_schedule) <= self.ideal_batch_size

        exec_requests = []
        for request in to_schedule:
            # Can flight this request.
            if request is not None:
                exec_requests.append(request)

        group_id = str(uuid4())
        self.active_requests[group_id] = exec_requests
        exec_process = self.make_invoker(
            page_cache, fiber, exec_requests, group_id, self.complete
        )

        # We've filled our flight. Remove from the boarding area.
        if exec_requests:
            # And takeoff.
            exec_process.launch()

    def complete(
        self,
        group_id: str,
        results_logits: List[sfnp.device_array],
        result_indices: List[sfnp.device_array],
    ):
        requests = self.active_requests[group_id]
        del self.active_requests[group_id]

        for i, req in enumerate(requests):
            req.result_logits = results_logits[i]
            req.result_indices = result_indices[i]

            total_tokens = req.start_position + len(req.input_token_ids)
            number_of_complete_pages = total_tokens // self.page_seq_stride
            req.publish_allocated_pages(number_of_complete_pages)

            req.done.set_success()


class PrefillBatcherProcess(LlmBatcherProcess):
    """The batcher is a persistent process responsible for flighting incoming work
    into batches and handling the requisite cache allocations (since every batch needs
    committed cache state).
    """

    STROBE_SHORT_DELAY = 0.065
    STROBE_LONG_DELAY = 0.065

    def __init__(
        self,
        fiber: Fiber,
        page_cache: BasePagedAttentionCache,
        model_params: ModelParams,
        prefill_functions: dict[int, sf.ProgramFunction],
        program_isolation: str,
    ):
        super().__init__(
            name="prefill",
            fiber=fiber,
            page_cache=page_cache,
            model_params=model_params,
            functions=prefill_functions,
            ideal_batch_size=max(model_params.prefill_batch_sizes),
            program_isolation=program_isolation,
        )

    def make_invoker(
        self,
        page_cache: BasePagedAttentionCache,
        fiber: Fiber,
        exec_requests: list[LlmInferenceExecRequest],
        group_id: str,
        completion_callback,
    ) -> "LlmInvoker":
        """Create instance of `LlmInvoker`.

        Args:
            page_cache (BasePagedAttentionCache): KVCache instance.
            fiber (Fiber): Fiber to execute invocation on.
            exec_requests (list[LlmInferenceExecRequest]): Request batch for invocation.

        Returns:
            LlmInvoker: Process to handle execution of VMFB.
        """
        llm_task_input = LlmTaskInput(
            array_cache=self.array_cache,
            input_tokens=[req.input_token_ids for req in exec_requests],
            page_ids=[req.page_ids for req in exec_requests],
            seq_stride=self.page_seq_stride,
            page_tables=page_cache.page_pool.page_tables,
        )
        llm_task = PrefillTask(
            task_inputs=llm_task_input,
        )
        return LlmInvoker(
            name="prefill_invocation",
            fiber=fiber,
            llm_task=llm_task,
            functions=self.functions,
            program_isolation=self.program_isolation,
            group_id=group_id,
            completion_callback=completion_callback,
        )


class DecodeBatcherProcess(LlmBatcherProcess):
    """The batcher is a persistent process responsible for flighting incoming work
    into batches and handling the requisite cache allocations (since every batch needs
    committed cache state).
    """

    STROBE_SHORT_DELAY = 0.0006
    STROBE_LONG_DELAY = 0.0006

    def __init__(
        self,
        fiber: Fiber,
        page_cache: BasePagedAttentionCache,
        model_params: ModelParams,
        decode_functions: dict[int, sf.ProgramFunction],
        program_isolation: str,
    ):
        super().__init__(
            name="decode",
            fiber=fiber,
            page_cache=page_cache,
            model_params=model_params,
            functions=decode_functions,
            ideal_batch_size=max(model_params.decode_batch_sizes),
            program_isolation=program_isolation,
        )

    def make_invoker(
        self,
        page_cache: BasePagedAttentionCache,
        fiber: Fiber,
        exec_requests: list[LlmInferenceExecRequest],
        group_id: str,
        completion_callback,
    ) -> "LlmInvoker":
        """Create instance of `LlmInvoker`.

        This method creates an instance of `LlmInvoker` to handle the
        execution of the decode function for a batch of requests.

        Args:
            page_cache (BasePagedAttentionCache): The KVCache instance to use for this flight.
            fiber (Fiber): Fiber to execute invocation on.
            exec_requests (list[LlmInferenceExecRequest]): Request batch for invocation.

        Returns:
            LlmInvoker: Process to handle execution of VMFB for decode requests.
        """
        task_inputs = LlmTaskInput(
            array_cache=self.array_cache,
            input_tokens=[req.input_token_ids for req in exec_requests],
            start_positions=[req.start_position for req in exec_requests],
            page_ids=[req.page_ids for req in exec_requests],
            seq_stride=self.page_seq_stride,
            page_tables=page_cache.page_pool.page_tables,
        )
        llm_task = DecodeTask(
            task_inputs=task_inputs,
        )
        return LlmInvoker(
            name="decode_invocation",
            fiber=fiber,
            llm_task=llm_task,
            functions=self.functions,
            program_isolation=self.program_isolation,
            group_id=group_id,
            completion_callback=completion_callback,
        )
