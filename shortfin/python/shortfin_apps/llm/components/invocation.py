import logging
import math

import shortfin as sf
import shortfin.array as sfnp

from typing import List, Optional, Tuple, Union

from .buffers import copy_buffers_to_host, create_argument_buffers
from .device_array_cache import Allocation, DeviceArrayCache, WrappedAllocation
from .messages import LlmInferenceExecRequest


logger = logging.getLogger(__name__)


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

        Raises:
            NotImplementedError: This method must be implemented in subclasses.
        """
        raise NotImplementedError(
            "get_args must be implemented in subclasses of LlmDataHandler"
        )

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
        indices = None
        logits = result[0]
        if len(result) > 1:
            indices = result[1]

        logits, indices = await self.transfer_buffer(
            exec_requests=self.exec_requests,
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
        exec_requests: List[LlmInferenceExecRequest],
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
        transfer = any([req.return_host_array for req in exec_requests])

        if not transfer:
            return buffers

        new_buffers = copy_buffers_to_host(buffers)
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
        req_count = len(exec_requests)
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

        (token_vals, seq_lens_vals, seq_block_ids_vals) = [], [], []
        for req in exec_requests:
            # Populate tokens.
            seq = req.input_token_ids
            padded = seq + [0] * max(0, bsl - len(seq))
            token_vals.extend(padded)

            # Populate sequence lengths.
            seq_lens_vals.append(len(req.input_token_ids))

            # Populate cache pages.
            block_ids = req.cache_page_indices(block_count)
            padded = block_ids + [0] * max(0, block_count - len(block_ids))
            seq_block_ids_vals.extend(padded)

        args = create_argument_buffers(
            buffers=[tokens, seq_lens, seq_block_ids],
            data=[token_vals, seq_lens_vals, seq_block_ids_vals],
            defaults=[0, 1, 0],
        )

        # V1 args:
        #  prefill:
        #    tokens: [bs, bsl]
        #    seq_lens: [bs]
        #    seq_block_ids: [bs, blocks]
        #    cache_slabs: ...
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

        # Populate data for args.
        (token_data, seq_lens_data, start_positions_data, seq_block_ids_data,) = (
            [],
            [],
            [],
            [],
        )
        for req in exec_requests:
            # Populate tokens.
            token_data.extend(req.input_token_ids[-1:])

            # Populate sequence lengths.
            seq_lens_data.append(req.start_position + 1)

            # Populate start positions.
            start_positions_data.append(req.start_position)

            # Populate cache pages.
            batch_ids = req.cache_page_indices(block_count)
            seq_block_ids_data.extend(batch_ids)
            seq_block_ids_data.extend([0] * (block_count - len(batch_ids)))

        args = create_argument_buffers(
            buffers=[tokens, seq_lens, start_positions, seq_block_ids],
            data=[token_data, seq_lens_data, start_positions_data, seq_block_ids_data],
            defaults=[0, 1, 0, 0],
        )

        # V1 args:
        #  decode:
        #    tokens: [bs, 1]
        #    seq_lens: [bs]
        #    start_positions: [bs]
        #    seq_block_ids: [bs, blocks]
        #    cache_slabs: ...
        for page_table in page_tables:
            args.append(WrappedAllocation(sfnp.disable_barrier(page_table)))

        return args, req_count


class LlmInvocationProcess(sf.Process):
    """Executes the invocation of LLM for a batch of requests."""

    def __init__(
        self,
        name: str,
        fiber: sf.Fiber,
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
