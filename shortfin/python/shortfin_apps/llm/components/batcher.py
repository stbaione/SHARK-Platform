# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import math
from typing import List, Optional, Tuple, Union


import shortfin as sf
import shortfin.array as sfnp

from shortfin import Fiber

from .device_array_cache import DeviceArrayCache, WrappedAllocation, Allocation
from .scheduler import Scheduler
from ...utils import BatcherProcess

from .config_struct import ModelParams
from .kvcache.base_attention_cache import (
    BasePagedAttentionCache,
    CacheAllocationFailure,
)

from .messages import LlmInferenceExecRequest

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

    def make_process(
        self,
        page_cache: BasePagedAttentionCache,
        fiber: Fiber,
        exec_requests: list[LlmInferenceExecRequest],
    ) -> "LlmInvocationProcess":
        """Create instance of `LlmInvocationProcess`.

        Args:
            page_cache (BasePagedAttentionCache): KVCache instance.
            fiber (Fiber): Fiber to execute invocation on.
            exec_requests (list[LlmInferenceExecRequest]): Request batch for invocation.

        Returns:
            LlmInvocationProcess: Process to handle execution of VMFB.
        """
        ...

    def board_request(self, page_cache, request: LlmInferenceExecRequest):
        """Prepare `LlmInferenceExecRequest` for LLM invocation.

        Args:
            page_cache (BasePagedAttentionCache): KVCache instance.
            request (LlmInferenceExecRequest): Request to prepare for invocation.
        """
        ...

    def board(
        self, page_cache: BasePagedAttentionCache, fiber: Fiber, to_schedule: set
    ):
        """Create and launch an LlmExecutorProcess for the given requests.

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
            request = self.board_request(page_cache, request)

            # Can flight this request.
            if request is not None:
                exec_requests.append(request)

        exec_process = self.make_process(page_cache, fiber, exec_requests)

        # We've filled our flight. Remove from the boarding area.
        if exec_requests:
            # And takeoff.
            exec_process.launch()


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

    def make_process(
        self,
        page_cache: BasePagedAttentionCache,
        fiber: Fiber,
        exec_requests: list[LlmInferenceExecRequest],
    ) -> "PrefillInvocationProcess":
        """Create instance of `PrefillInvocationProcess`.

        Args:
            page_cache (BasePagedAttentionCache): KVCache instance.
            fiber (Fiber): Fiber to execute invocation on.
            exec_requests (list[LlmInferenceExecRequest]): Request batch for invocation.

        Returns:
            PrefillInvocationProcess: Process to handle execution of VMFB.
        """
        return PrefillInvocationProcess(
            exec_requests,
            fiber,
            self.array_cache,
            self.functions,
            self.page_seq_stride,
            page_cache.page_pool.page_tables,
            self.program_isolation,
        )

    def board_request(
        self, page_cache: BasePagedAttentionCache, request: LlmInferenceExecRequest
    ) -> Optional[LlmInferenceExecRequest]:
        """Board a request for prefill invocation.

        Prepares the request for prefill invocation by acquiring
        the necessary pages from the page cache, calculating the number of
        pages needed based on the input token IDs and allocates them accordingly.

        Args:
            page_cache (BasePagedAttentionCache): KVCache instance.
            request (LlmInferenceExecRequest): Request to prepare for invocation.

        Returns:
            Optional[LlmInferenceExecRequest]: The request with allocated pages, or None if allocation fails.
        """
        needed_pages = math.ceil(len(request.input_token_ids) / self.page_seq_stride)
        # allocate kv cache pages
        try:
            allocation = page_cache.acquire_pages_for_tokens(
                request.input_token_ids,
                extra_token_slots=0,  # prefill needs no extra kvcache slots to write to
            )
        except CacheAllocationFailure:
            logger.debug("Cannot fulfill request for %d pages", needed_pages)
            return None

        logger.debug(f"Successfully acquired allocation: {allocation}")
        request.free_cache_pages()
        request.allocation = allocation

        return request


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

    def make_process(
        self,
        page_cache: BasePagedAttentionCache,
        fiber: Fiber,
        exec_requests: list[LlmInferenceExecRequest],
    ) -> "DecodeInvocationProcess":
        """Create instance of `DecodeInvocationProcess`.

        This method creates an instance of `DecodeInvocationProcess` to handle the
        execution of the decode function for a batch of requests.

        Args:
            page_cache (BasePagedAttentionCache): The KVCache instance to use for this flight.
            fiber (Fiber): Fiber to execute invocation on.
            exec_requests (list[LlmInferenceExecRequest]): Request batch for invocation.

        Returns:
            DecodeInvocationProcess: Process to handle execution of VMFB for decode requests.
        """
        return DecodeInvocationProcess(
            exec_requests,
            fiber,
            self.array_cache,
            self.functions,
            self.page_seq_stride,
            page_cache.page_pool.page_tables,
            self.program_isolation,
        )

    def board_request(
        self, page_cache: BasePagedAttentionCache, request: LlmInferenceExecRequest
    ) -> LlmInferenceExecRequest:
        """Prepare `LlmInferenceExecRequest` for decode invocation.

        Args:
            page_cache (BasePagedAttentionCache): KVCache instance.
            request (LlmInferenceExecRequest): Request to prepare for invocation.

        Returns:
            LlmInferenceExecRequest: The request with allocated pages.
        """
        if request.allocation is not None:
            request.allocation.extend_allocation(
                request.input_token_ids, extra_token_slots=1
            )
        return request


########################################################################################
# Inference Executor
########################################################################################


class LlmDataHandler:
    """Handles the transfer and preparation of data for VMFB invocation."""

    def __init__(
        self,
        exec_requests: List[LlmInferenceExecRequest],
        array_cache: DeviceArrayCache,
        seq_stride: int,
        is_prefill: bool,
    ):
        self.exec_requests: List[LlmInferenceExecRequest] = exec_requests
        self._array_cache: DeviceArrayCache = array_cache
        self._seq_stride: int = seq_stride
        self._is_prefill: bool = is_prefill

    async def get_args(
        self,
        page_tables: sfnp.device_array,
        batch_size: int,
    ) -> Tuple[list[sfnp.device_array], int]:
        """Get the arguments for the invocation.

        Args:
            page_tables (sfnp.device_array): Page tables in KVCache.
            batch_size (int): Size of the `exec_requests` batch.

        Returns:
            Tuple[list[sfnp.device_array], int]: A tuple containing:
                - A list of arguments for the invocation.
                - The number of requests in the batch.
        """
        ...

    async def post_process_logits(
        self,
        args: List[Union[Allocation, WrappedAllocation]],
        req_count: int,
        result: Tuple[sfnp.device_array, Optional[sfnp.device_array]],
        device0: sf.ScopedDevice,
    ):
        """Handle the post-processing of logits after a prefill invocation.

        Args:
            args (List[Union[Allocation, WrappedAllocation]]): The arguments used for the invocation.
            req_count (int): The number of requests in the batch.
            result (Tuple[sfnp.device_array, Optional[sfnp.device_array]]): The result from the invocation.
                - The 0th element should be the `logits`
                - The 1st element should be the `indices`, if any.
            device0 (sf.ScopedDevice): The device used for invocation.
        """
        seq_stride = self._seq_stride

        indices = None
        logits = result[0]
        if len(result) > 1:
            indices = result[1]

        # publish cache pages
        for r in self.exec_requests:
            total_tokens = r.start_position + len(r.input_token_ids)
            number_of_complete_pages = total_tokens // seq_stride
            r.publish_allocated_pages(number_of_complete_pages)

        logits, indices = await self.transfer_buffer(
            req_count=req_count,
            device0=device0,
            buffers=(logits, indices),
        )

        [arg.release() for arg in args]

        await self.get_result(logits, indices, req_count)

    async def get_result(
        self,
        logits: sfnp.device_array,
        indices: Optional[sfnp.device_array],
        req_count: int,
    ):
        """Get the results after a prefill invocation.

        Args:
            logits (sfnp.device_array): The logits output from prefill.
            indices (Optional[sfnp.device_array]): The indices output from prefill, if any.
            req_count (int): The number of requests in the batch.
        """
        is_prefill = self._is_prefill
        for i in range(req_count):
            req = self.exec_requests[i]
            sl = len(req.input_token_ids) - 1 if is_prefill else 0

            if logits.shape[1] == 1 and is_prefill:
                logits_item = logits.view(i)
            else:
                logits_item = logits.view(i, sl)

            index_item = None
            if indices is not None:
                if indices.shape[1] == 1 and is_prefill:
                    index_item = indices.view(i)
                else:
                    index_item = indices.view(i, sl)

            req.result_logits = logits_item
            req.result_indices = index_item

        for req in self.exec_requests:
            req.done.set_success()

    async def transfer_buffer(
        self,
        req_count: int,
        device0: sf.ScopedDevice,
        buffers: Tuple[sfnp.device_array, Optional[sfnp.device_array]],
    ) -> Tuple[sfnp.device_array, Optional[sfnp.device_array]]:
        """Transfer buffer data from device to host after invocation.

        Args:
            req_count (int): The number of requests in this batch.
            device0 (sf.ScopedDevice): The device used for invocation.
            buffers (Tuple[sfnp.device_array, Optional[sfnp.device_array]]): The buffers to be transferred.
                - The 0th buffer should be the `logits`
                - The 1st buffer should be the `indices`

        Returns:
            Tuple[sfnp.device_array, Optional[sfnp.device_array]]: A host-side copy of the given buffers.
        """
        transfer = any(
            [self.exec_requests[i].return_host_array for i in range(req_count)]
        )

        if not transfer:
            return buffers

        new_buffers = []
        for buffer in buffers:
            if buffer is None:
                new_buffers.append(None)
                continue

            host_buffer = buffer.for_transfer()
            host_buffer.copy_from(buffer)
            new_buffers.append(host_buffer)

        await device0
        return tuple(new_buffers)


class PrefillDataHandler(LlmDataHandler):
    """Handles the transfer and preparation of data for VMFB invocation."""

    def __init__(
        self,
        exec_requests: list[LlmInferenceExecRequest],
        array_cache: DeviceArrayCache,
        seq_stride: int,
    ):
        super().__init__(
            exec_requests=exec_requests,
            array_cache=array_cache,
            seq_stride=seq_stride,
            is_prefill=True,
        )

    async def get_args(
        self,
        page_tables: sfnp.device_array,
        batch_size: int,
    ) -> Tuple[list[sfnp.device_array], int]:
        """Get the arguments for the prefill invocation.

        Args:
            page_tables (sfnp.device_array): Page tables in KVCache.
            batch_size (int): Size of the `exec_requests` batch.

        Returns:
            Tuple[list[sfnp.device_array], int]: A tuple containing:
                - A list of arguments for the invocation.
                - The number of requests in the batch.
        """
        exec_requests = self.exec_requests
        seq_stride = self._seq_stride

        for r in exec_requests:
            assert r.start_position == 0

        # Compute block sequence length as maximum sequence length, rounded
        # up to the seq_stride.
        bsl = max((len(r.input_token_ids)) for r in exec_requests)
        bsl = int(math.ceil(bsl / seq_stride) * seq_stride)
        block_count = bsl // seq_stride
        req_count = len(exec_requests)
        logger.debug("Prefill bs=%d, bsl=%d", batch_size, bsl)

        # Prepare inputs.
        # TODO: Better support in shortfin for h2d. The best way to do it is
        # device dependent.
        array_cache = self._array_cache
        int_dtype = sfnp.int64

        tokens = array_cache.allocate([batch_size, bsl], int_dtype)
        seq_lens = array_cache.allocate([batch_size], int_dtype)
        seq_block_ids = array_cache.allocate([batch_size, block_count], int_dtype)

        # Populate tokens.
        for i in range(batch_size):
            with tokens.host.view(i).map(discard=True) as m:
                m.fill(0)
                if i < req_count:
                    m.items = exec_requests[i].input_token_ids

        # Populate seq_lens
        with seq_lens.host.map(discard=True) as m:
            m.fill(1)
            m.items = [len(req.input_token_ids) for req in exec_requests]

        # Populate cache pages.
        for i in range(batch_size):
            with seq_block_ids.host.view(i).map(discard=True) as m:
                m.fill(0)
                if i < req_count:
                    m.items = exec_requests[i].cache_page_indices(block_count)

        tokens.transfer_to_device()
        seq_lens.transfer_to_device()
        seq_block_ids.transfer_to_device()

        # V1 args:
        #  prefill:
        #    tokens: [bs, bsl]
        #    seq_lens: [bs]
        #    seq_block_ids: [bs, blocks]
        #    cache_slabs: ...
        args = [tokens, seq_lens, seq_block_ids]
        for page_table in page_tables:
            args.append(WrappedAllocation(sfnp.disable_barrier(page_table)))

        return args, req_count


class DecodeDataHandler(LlmDataHandler):
    """Handles the transfer and preparation of data for VMFB invocation."""

    def __init__(
        self,
        exec_requests: list[LlmInferenceExecRequest],
        array_cache: DeviceArrayCache,
        seq_stride: int,
    ):
        super().__init__(
            exec_requests=exec_requests,
            array_cache=array_cache,
            seq_stride=seq_stride,
            is_prefill=False,
        )

    async def get_args(
        self,
        page_tables: sfnp.device_array,
        batch_size: int,
    ) -> Tuple[List[Union[Allocation, WrappedAllocation]], int]:
        """Get the arguments for the decode invocation.

        Args:
            page_tables (sfnp.device_array): Page tables in KVCache.
            batch_size (int): Size of the `exec_requests` batch.

        Returns:
            Tuple[List[Union[Allocation, WrappedAllocation]], int]: A tuple containing:
                - A list of arguments for the invocation.
                - The number of requests in the batch.
        """
        # Compute block sequence length as maximum sequence length, rounded
        # up to the seq_stride.
        exec_requests = self.exec_requests
        seq_stride = self._seq_stride
        bsl = max((1 + len(r.input_token_ids)) for r in exec_requests)
        bsl = int(math.ceil(bsl / seq_stride) * seq_stride)
        block_count = bsl // seq_stride
        req_count = len(exec_requests)
        logger.debug("Decode bs=%d, bsl=%d", batch_size, bsl)

        # Prepare inputs.
        # TODO: Better support in shortfin for h2d. The best way to do it is
        # device dependent.

        array_cache = self._array_cache
        int_dtype = sfnp.int64

        tokens = array_cache.allocate([batch_size, 1], int_dtype)
        start_positions = array_cache.allocate([batch_size], int_dtype)
        seq_lens = array_cache.allocate([batch_size], int_dtype)
        seq_block_ids = array_cache.allocate([batch_size, block_count], int_dtype)

        # Populate tokens.
        with tokens.host.map(discard=True) as m:
            m.fill(0)
            vals = []
            for i in range(batch_size):
                if i < req_count:
                    vals = vals + exec_requests[i].input_token_ids[-1:]
            m.items = vals

        # For decode, populate start_positions and seq_lens.
        with start_positions.host.map(discard=True) as m:
            m.fill(0)
            m.items = [req.start_position for req in exec_requests]

        with seq_lens.host.map(discard=True) as m:
            # Pad unused requests.
            m.fill(
                1  # Must pad with a nonzero value because a division by 0 during softmax floods clobber page (page 0) in cache with NaN values.
            )
            m.items = [req.start_position + 1 for req in exec_requests]

        # Populate cache pages.
        with seq_block_ids.host.map(discard=True) as m:
            m.fill(0)
            block_ids = []
            for i in range(batch_size):
                if i < req_count:
                    batch_ids = exec_requests[i].cache_page_indices(block_count)
                    block_ids += batch_ids
                    block_ids += [0] * (block_count - len(batch_ids))
            m.items = block_ids

        # Transfer to device memory:
        tokens.transfer_to_device()
        start_positions.transfer_to_device()
        seq_lens.transfer_to_device()
        seq_block_ids.transfer_to_device()

        # V1 args:
        #  decode:
        #    tokens: [bs, 1]
        #    seq_lens: [bs]
        #    start_positions: [bs]
        #    seq_block_ids: [bs, blocks]
        #    cache_slabs: ...
        args = [tokens, seq_lens, start_positions, seq_block_ids]
        for page_table in page_tables:
            args.append(WrappedAllocation(sfnp.disable_barrier(page_table)))

        return args, req_count


class LlmInvocationProcess(sf.Process):
    """Executes the invocation of LLM for a batch of requests."""

    def __init__(
        self,
        name: str,
        fiber: Fiber,
        array_cache: DeviceArrayCache,
        data_handler: LlmDataHandler,
        functions: dict[int, sf.ProgramFunction],
        seq_stride: int,
        page_tables,
        program_isolation: sf.ProgramIsolation,
    ):
        super().__init__(fiber=fiber)
        self.name = name
        self.seq_stride = seq_stride
        self.page_tables = page_tables
        self.functions = functions
        self.program_isolation = program_isolation

        self.device0 = fiber.device(0)
        self.array_cache = array_cache
        self.data_handler = data_handler

    async def run(self):
        """Invoke `prefill` or `decode` function, with IREE, on a batch of requests.

        Raises:
            RuntimeError: No available entry point for given batch size.
        """
        try:
            exec_requests = self.data_handler.exec_requests
            req_bs = len(exec_requests)

            # Select an entrypoint for the batch.
            entrypoints = self.functions
            for bs, fn in entrypoints.items():
                if bs >= req_bs:
                    break
            else:
                raise RuntimeError(f"No available entry point for bs {req_bs}")

            args, req_count = await self.data_handler.get_args(self.page_tables, bs)

            # Invoke VMFB. Logits are of shape [bs, bsl, d].
            args_device = [arg.device for arg in args]
            result = await fn(*args_device, fiber=self.fiber)
            await self.data_handler.post_process_logits(
                args,
                req_count,
                result,
                self.device0,
            )

        except Exception:
            logger.exception("Fatal error in prefetch invocation")
            # TODO: Cancel and set error correctly
            for req in self.data_handler.exec_requests:
                req.result_logits = None
                req.free_cache_pages()
                req.done.set_success()


class PrefillInvocationProcess(LlmInvocationProcess):
    """Executes the invocation of prefill for a batch of requests."""

    def __init__(
        self,
        exec_requests: list[LlmInferenceExecRequest],
        fiber: Fiber,
        array_cache: DeviceArrayCache,
        functions: dict[int, sf.ProgramFunction],
        seq_stride: int,
        page_tables,
        program_isolation: sf.ProgramIsolation,
    ):
        data_handler = PrefillDataHandler(
            exec_requests=exec_requests,
            array_cache=array_cache,
            seq_stride=seq_stride,
            chunk_size=None,
        )
        super().__init__(
            name="prefill_process",
            fiber=fiber,
            array_cache=array_cache,
            data_handler=data_handler,
            functions=functions,
            seq_stride=seq_stride,
            page_tables=page_tables,
            program_isolation=program_isolation,
        )


class DecodeInvocationProcess(LlmInvocationProcess):
    """Executes the invocation of decode for a batch of requests."""

    def __init__(
        self,
        exec_requests: list[LlmInferenceExecRequest],
        fiber: Fiber,
        array_cache: DeviceArrayCache,
        functions: dict[int, sf.ProgramFunction],
        seq_stride: int,
        page_tables,
        program_isolation: sf.ProgramIsolation,
    ):
        data_handler = DecodeDataHandler(
            exec_requests=exec_requests,
            array_cache=array_cache,
            seq_stride=seq_stride,
        )
        super().__init__(
            name="decode_process",
            fiber=fiber,
            array_cache=array_cache,
            data_handler=data_handler,
            functions=functions,
            seq_stride=seq_stride,
            page_tables=page_tables,
            program_isolation=program_isolation,
        )
