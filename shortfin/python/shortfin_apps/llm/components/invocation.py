import logging
import math

import shortfin as sf
import shortfin.array as sfnp

from dataclasses import dataclass
from enum import Enum, auto
from typing import Awaitable, Callable, List, Optional, Tuple, Union

from .buffers import copy_buffers_to_host, create_argument_buffers
from .device_array_cache import Allocation, DeviceArrayCache, WrappedAllocation
from .kvcache.page_pool import PageInfo


logger = logging.getLogger(__name__)


class InvocationResult(Enum):
    SUCCESS = auto()
    FAILURE = auto()


@dataclass
class LlmTaskInput:
    array_cache: DeviceArrayCache
    block_count: int
    seq_stride: int
    input_tokens: List[List[int]]
    page_ids: List[List[PageInfo | int]]

    start_positions: Optional[List[int]] = None

    @property
    def batch_seq_len(self):
        seq_stride = self.seq_stride
        bsl = max(len(tokens) for tokens in self.input_tokens)
        return int(math.ceil(bsl / seq_stride) * seq_stride)


class LlmTask:
    """Handles the transfer and preparation of data for VMFB invocation."""

    def __init__(
        self,
        task_input: LlmTaskInput,
    ):
        self.task_input = task_input

    def get_args_data(
        self,
    ) -> Tuple[List[int | float] | List[List[int | float]]]:
        """Get the invocation data for the given requests.

        Prepare the data that will be used to create the argument_buffers
        for the invocation.

        Returns:
            Tuple[List[int | float] | List[List[int | float]]]: A tuple containing argument data.
        """

    async def get_args(
        self,
        page_tables: sfnp.device_array,
        batch_size: int,
    ) -> list[sfnp.device_array]:
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
            "get_args must be implemented in subclasses of LlmTask"
        )

    async def post_process_logits(
        self,
        args: List[Union[Allocation, WrappedAllocation]],
        result: Tuple[sfnp.device_array, Optional[sfnp.device_array]],
        device0: sf.ScopedDevice,
    ) -> Tuple[List[sfnp.device_array], Optional[List[sfnp.device_array]]]:
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

        buffers = (logits, indices)

        logits, indices = await copy_buffers_to_host(buffers, device0)

        [arg.release() for arg in args]

        return await self.get_result(logits, indices)

    async def get_result(
        self,
        logits: sfnp.device_array,
        indices: Optional[sfnp.device_array],
        req_count: int,
    ) -> Tuple[List[sfnp.device_array], Optional[List[sfnp.device_array]]]:
        """Get the results after a prefill invocation.

        Args:
            logits (sfnp.device_array): The logits output from prefill.
            indices (Optional[sfnp.device_array]): The indices output from prefill, if any.
            req_count (int): The number of requests in the batch.
        """

    async def invoke(
        self,
        page_tables: sfnp.device_array,
        batch_size: int,
        invocation_function: Callable[..., Awaitable[sfnp.device_array]],
        fiber: sf.Fiber,
        device: sf.ScopedDevice,
    ) -> Tuple[List[sfnp.device_array], Optional[List[sfnp.device_array]]]:
        args = await self.get_args(page_tables, batch_size)

        args_device = [arg.device for arg in args]
        result = await invocation_function(*args_device, fiber=fiber)

        return await self.post_process_logits(
            args,
            result,
            device,
        )


def _pad_list(
    data: List[int | float],
    target_length: int,
) -> List[int | float]:
    """Pad a list to a target length with a specified value."""
    return data + [0] * max(0, target_length - len(data))


class PrefillTask(LlmTask):
    """Handles the transfer and preparation of data for VMFB invocation."""

    def __init__(
        self,
        task_input: LlmTaskInput,
    ):
        super().__init__(
            task_input=task_input,
        )

    def get_args_data(
        self,
    ) -> Tuple[List[int]]:
        """Get the invocation data for the given requests.

        Prepare the data that will be used to create the argument_buffers
        for the invocation.

        Args:
            exec_requests (List[LlmInferenceExecRequest]): List of execution requests.
            batch_seq_len (int): The maximum sequence length for the batch.
            block_count (int): The number of blocks in the sequence.

        Returns:
            Tuple[List[int]: A tuple containing:
                - A list of token IDs for the invocation.
                - A list of sequence lengths.
                - A list of sequence block IDs.
        """
        task_input = self.task_input
        token_vals = [
            padded_tokens
            for tokens in task_input.input_tokens
            for padded_tokens in (_pad_list(tokens, task_input.batch_seq_len))
        ]

        seq_lens_vals = [len(tokens) for tokens in task_input.input_tokens]

        seq_block_ids_vals = []
        for pages in task_input.page_ids:
            block_ids = _pad_list(
                [page.index for page in pages],
                target_length=task_input.block_count,
            )

            seq_block_ids_vals.extend(block_ids)

        return token_vals, seq_lens_vals, seq_block_ids_vals

    async def get_args(
        self,
        page_tables: sfnp.device_array,
        batch_size: int,
    ) -> list[sfnp.device_array]:
        """Get the arguments for the prefill invocation.

        The prefill args that are created are:
            - tokens: [bs, bsl]
            - seq_lens: [bs]
            - seq_block_ids: [bs, blocks]
            - cache_slabs: ...

        Args:
            page_tables (sfnp.device_array): Page tables in KVCache.
            batch_size (int): Batch size of the invocation function.

        Returns:
            list[sfnp.device_array]: List of arguments for the invocation.
        """
        task_input = self.task_input

        # Compute block sequence length as maximum sequence length, rounded
        # up to the seq_stride.
        logger.debug("Prefill bs=%d, bsl=%d", batch_size, task_input.batch_seq_len)

        array_cache = task_input.array_cache
        int_dtype = sfnp.int64

        # Acquire buffers for the arguments.
        tokens = array_cache.allocate([batch_size, task_input.batch_seq_len], int_dtype)
        seq_lens = array_cache.allocate([batch_size], int_dtype)
        seq_block_ids = array_cache.allocate(
            [batch_size, task_input.block_count], int_dtype
        )

        # Populate data for args.
        arg_data = self.get_args_data()

        args = create_argument_buffers(
            buffers=[tokens, seq_lens, seq_block_ids],
            data=arg_data,
            defaults=[0, 1, 0],
        )

        for page_table in page_tables:
            args.append(WrappedAllocation(sfnp.disable_barrier(page_table)))

        return args

    async def get_result(
        self,
        logits: sfnp.device_array,
        indices: Optional[sfnp.device_array],
    ) -> Tuple[List[sfnp.device_array], Optional[List[sfnp.device_array]]]:
        """Get the results after a prefill invocation.

        Args:
            logits (sfnp.device_array): The logits output from prefill.
            indices (Optional[sfnp.device_array]): The indices output from prefill, if any.
            req_count (int): The number of requests in the batch.
        """
        result_logits = []
        result_indices = []
        task_input = self.task_input
        for i in range(len(task_input.input_tokens)):
            sl = len(task_input.input_tokens[i]) - 1

            if logits.shape[1] == 1:
                logits_item = logits.view(i)
            else:
                logits_item = logits.view(i, sl)

            index_item = None
            if indices is not None:
                if indices.shape[1] == 1:
                    index_item = indices.view(i)
                else:
                    index_item = indices.view(i, sl)

            result_logits.append(
                logits_item,
            )
            result_indices.append(
                index_item,
            )

        return result_logits, result_indices


class DecodeTask(LlmTask):
    """Handles the transfer and preparation of data for VMFB invocation."""

    def __init__(
        self,
        task_input: LlmTaskInput,
    ):
        assert (
            task_input.start_positions is not None
        ), "Must include `start_positions` for decode."
        super().__init__(
            task_input=task_input,
        )

    def get_args_data(
        self,
    ) -> Tuple[List[int | float] | List[List[int | float]]]:
        """Get the invocation data for the given requests.

        Prepare the data that will be used to create the argument_buffers
        for the invocation.

        Args:
            exec_requests (List[LlmInferenceExecRequest]): List of execution requests.
            block_count (int): The number of blocks in the sequence.

        Returns:
            Tuple[List[int | float] | List[List[int | float]]]: A tuple containing:
                - A list of token IDs for the invocation.
                - A list of sequence lengths.
                - A list of start positions.
                - A list of sequence block IDs.
        """
        task_input = self.task_input
        token_data = [
            padded_tokens
            for tokens in task_input.input_tokens
            for padded_tokens in (tokens[-1:])
        ]

        seq_block_ids_data = []
        for pages in task_input.page_ids:
            block_ids = _pad_list(
                pages,
                target_length=task_input.block_count,
            )
            seq_block_ids_data.extend(block_ids)

        return (
            token_data,
            [position + 1 for position in task_input.start_positions],
            task_input.start_positions,
            seq_block_ids_data,
        )

    async def get_args(
        self,
        page_tables: sfnp.device_array,
        batch_size: int,
    ) -> List[Union[Allocation, WrappedAllocation]]:
        """Get the arguments for the decode invocation.

        The decode args that are created are:
            - tokens: [bs, 1]
            - seq_lens: [bs]
            - start_positions: [bs]
            - seq_block_ids: [bs, blocks]
            - cache_slabs: ...

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
        task_input = self.task_input
        req_count = len(task_input.input_tokens)
        logger.debug("Decode bs=%d", req_count)

        array_cache = task_input.array_cache
        int_dtype = sfnp.int64

        # Acquire buffers for the arguments.
        tokens = array_cache.allocate([batch_size, 1], int_dtype)
        start_positions = array_cache.allocate([batch_size], int_dtype)
        seq_lens = array_cache.allocate([batch_size], int_dtype)
        seq_block_ids = array_cache.allocate(
            [batch_size, task_input.block_count], int_dtype
        )

        # Populate data for args.
        args_data = self.get_args_data()

        args = create_argument_buffers(
            buffers=[tokens, seq_lens, start_positions, seq_block_ids],
            data=args_data,
            defaults=[0, 1, 0, 0],
        )

        for page_table in page_tables:
            args.append(WrappedAllocation(sfnp.disable_barrier(page_table)))

        return args

    async def get_result(
        self,
        logits: sfnp.device_array,
        indices: Optional[sfnp.device_array],
    ) -> Tuple[List[sfnp.device_array], Optional[List[sfnp.device_array]]]:
        """Get the results after a prefill invocation.

        Args:
            logits (sfnp.device_array): The logits output from prefill.
            indices (Optional[sfnp.device_array]): The indices output from prefill, if any.
            req_count (int): The number of requests in the batch.
        """
        result_logits = []
        result_indices = []
        for i in range(len(self.task_input.input_tokens)):
            logits_item = logits.view(i, 0)

            index_item = None
            if indices is not None:
                index_item = indices.view(i, 0)

            result_logits.append(
                logits_item,
            )
            result_indices.append(
                index_item,
            )

        return result_logits, result_indices


class LlmInvoker(sf.Process):
    """Executes the invocation of LLM for a batch of requests."""

    def __init__(
        self,
        name: str,
        fiber: sf.Fiber,
        llm_task: LlmTask,
        functions: dict[int, sf.ProgramFunction],
        page_tables,
        program_isolation: sf.ProgramIsolation,
        completion_callback: Callable,
        group_id: str,
    ):
        super().__init__(fiber=fiber)
        self.name = name
        self.page_tables = page_tables
        self.functions = functions
        self.program_isolation = program_isolation

        self.device0 = fiber.device(0)
        self.llm_task = llm_task
        self.completion_callback = completion_callback
        self.group_id = group_id

    async def run(self):
        """Invoke `prefill` or `decode` function, with IREE, on a batch of requests.

        Raises:
            RuntimeError: No available entry point for given batch size.
        """
        try:
            req_bs = len(self.llm_task.task_input.input_tokens)

            # Select an entrypoint for the batch.
            entrypoints = self.functions
            for bs, fn in entrypoints.items():
                if bs >= req_bs:
                    break
            else:
                raise RuntimeError(f"No available entry point for bs {req_bs}")

            result_logits, result_indices = await self.llm_task.invoke(
                page_tables=self.page_tables,
                batch_size=bs,
                invocation_function=fn,
                fiber=self.fiber,
                device=self.device0,
            )

            self.completion_callback(
                result_logits,
                result_indices,
                InvocationResult.SUCCESS,
                self.group_id,
            )

        except Exception:
            logger.exception(
                f"Fatal error in prefetch invocation in process {self.name}"
            )
            self.completion_callback(
                None,
                None,
                InvocationResult.FAILURE,
                self.group_id,
            )
