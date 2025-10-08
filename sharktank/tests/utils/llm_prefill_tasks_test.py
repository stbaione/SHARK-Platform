from typing import List
import numpy

from unittest import TestCase
from unittest.mock import MagicMock, patch

from sharktank.utils.llm_scheduler import BasicScheduler, ChunkScheduler
from sharktank.utils.llm_tasks import (
    LlmTaskInput,
    PrefillTask,
)

from sharktank.utils.llm_utils import (
    IreeInstance,
    LlmRunner,
)


class TestPrefillTask(TestCase):
    def setUp(self) -> None:
        self._batch_size = 4
        self._mock_iree_instance = MagicMock(spec=IreeInstance)
        self._mock_iree_instance.prefill_bs = self._batch_size
        self._mock_iree_instance.decode_bs = self._batch_size
        self._page_count = 16
        self._page_sizes = [256]
        self._block_stride = 2
        self._kv_cache_dtype = "float16"

        self._llm_runner = LlmRunner(
            instance=self._mock_iree_instance,
            page_count=self._page_count,
            page_sizes=self._page_sizes,
            block_stride=self._block_stride,
            kv_cache_dtype=self._kv_cache_dtype,
        )

        self._requests = [
            [0, 1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6, 7],
            [2, 3, 4, 5],
            [3, 4, 5, 6, 7, 8, 9, 10],
        ]
        self._page_ids = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9],
            [10, 11, 12, 13],
        ]

    def _get_tasks(self, requests, page_ids) -> List[LlmTaskInput]:
        llm_runner = self._llm_runner
        llm_requests = llm_runner.make_requests(requests=requests, page_ids=page_ids)
        llm_tasks = []
        with patch.object(
            BasicScheduler,
            "schedule_task",
            side_effect=lambda task: llm_tasks.append(task),
        ):
            llm_runner.submit_prefill(llm_requests)

        return llm_tasks

    def _get_expected_tokens(self, requests: List[List[int]]) -> numpy.ndarray:
        pad = 0
        max_len = max(len(req) for req in requests)
        max_len = int(
            int(numpy.ceil(max_len / self._block_stride)) * self._block_stride
        )
        expected_tokens = numpy.full(
            (self._batch_size, max_len), fill_value=pad, dtype=numpy.int64
        )
        for i, req in enumerate(requests):
            expected_tokens[i, : len(req)] = req

        return expected_tokens

    def test_run(self):
        requests = self._requests
        page_ids = self._page_ids
        llm_tasks = self._get_tasks(requests, page_ids)

        invoked_args = []

        def _invocation_fn(*args):
            invoked_args.extend(args)
            return [i for i in range(self._batch_size)]

        prefill_task = PrefillTask(
            invocation_fn=_invocation_fn,
            llm_task_inputs=llm_tasks,
            batch_size=self._batch_size,
            block_stride=self._block_stride,
        )

        cache_state = [numpy.zeros((16, 256), dtype=numpy.float16)]
        logits, indices = prefill_task.run(*cache_state)

        expected_tokens = self._get_expected_tokens(requests)
        expected_seq_lens = numpy.array(
            [len(req) for req in requests], dtype=numpy.int64
        )
        expected_seq_block_ids = numpy.array(
            [
                page_ids[0],
                page_ids[1],
                page_ids[2] + [0, 0],
                page_ids[3],
            ]
        )
        assert numpy.array_equal(invoked_args[0], expected_tokens)
        assert numpy.array_equal(invoked_args[1], expected_seq_lens)
        assert numpy.array_equal(invoked_args[2], expected_seq_block_ids)
        assert numpy.array_equal(invoked_args[3], cache_state[0])

        assert numpy.array_equal(
            logits,
            numpy.array([i for i in range(self._batch_size)], dtype=numpy.int64),
        )
        assert indices is None

    def test_run_w_indices(self):
        requests = self._requests
        page_ids = self._page_ids
        llm_tasks = self._get_tasks(requests, page_ids)

        invoked_args = []

        def _invocation_fn(*args):
            invoked_args.extend(args)
            return (
                [i for i in range(self._batch_size)],
                [i + 1 for i in range(self._batch_size)],
            )

        prefill_task = PrefillTask(
            invocation_fn=_invocation_fn,
            llm_task_inputs=llm_tasks,
            batch_size=self._batch_size,
            block_stride=self._block_stride,
        )

        cache_state = [numpy.zeros((16, 256), dtype=numpy.float16)]
        logits, indices = prefill_task.run(*cache_state)

        assert numpy.array_equal(
            logits,
            numpy.array([i for i in range(self._batch_size)], dtype=numpy.int64),
        )
        assert numpy.array_equal(
            indices,
            numpy.array([i + 1 for i in range(self._batch_size)], dtype=numpy.int64),
        )

    def test_run_partial_batch(self):
        requests = self._requests[:2]
        page_ids = self._page_ids[:2]

        llm_tasks = self._get_tasks(requests, page_ids)

        invoked_args = []

        def _invocation_fn(*args):
            invoked_args.extend(args)
            return [i for i in range(self._batch_size)]

        prefill_task = PrefillTask(
            invocation_fn=_invocation_fn,
            llm_task_inputs=llm_tasks,
            batch_size=self._batch_size,
            block_stride=self._block_stride,
        )

        cache_state = [numpy.zeros((16, 256), dtype=numpy.float16)]
        logits, indices = prefill_task.run(*cache_state)

        expected_tokens = self._get_expected_tokens(requests)
        expected_seq_lens = numpy.array(
            [len(req) for req in requests] + [1, 1], dtype=numpy.int64
        )
        expected_seq_block_ids = numpy.array(
            [
                page_ids[0],
                page_ids[1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )
        assert numpy.array_equal(invoked_args[0], expected_tokens)
        assert numpy.array_equal(invoked_args[1], expected_seq_lens)
        assert numpy.array_equal(invoked_args[2], expected_seq_block_ids)
        assert numpy.array_equal(invoked_args[3], cache_state[0])

        assert numpy.array_equal(
            logits,
            numpy.array([i for i in range(self._batch_size)], dtype=numpy.int64),
        )
        assert indices is None


class TestChunkedPrefillTask(TestCase):
    def setUp(self) -> None:
        self._batch_size = 4
        self._mock_iree_instance = MagicMock(spec=IreeInstance)
        self._mock_iree_instance.prefill_bs = self._batch_size
        self._mock_iree_instance.decode_bs = self._batch_size
        self._page_count = 16
        self._page_sizes = [256]
        self._block_stride = 2
        self._chunk_block_size = 3
        self._kv_cache_dtype = "float16"

        self._llm_runner = LlmRunner(
            instance=self._mock_iree_instance,
            page_count=self._page_count,
            page_sizes=self._page_sizes,
            block_stride=self._block_stride,
            kv_cache_dtype=self._kv_cache_dtype,
            chunk_block_size=self._chunk_block_size,
        )

        self._requests = [
            [0, 1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6, 7],
            [2, 3, 4, 5],
            [3, 4, 5, 6, 7, 8, 9, 10],
        ]
        self._page_ids = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9],
            [10, 11, 12, 13],
        ]

    def _get_tasks(self, requests, page_ids) -> List[LlmTaskInput]:
        llm_runner = self._llm_runner
        llm_requests = llm_runner.make_requests(requests=requests, page_ids=page_ids)
        llm_tasks = []
        with patch.object(
            ChunkScheduler,
            "schedule_task",
            side_effect=lambda task: llm_tasks.append(task),
        ):
            llm_runner.submit_prefill(llm_requests)

        return llm_tasks

    def _get_expected_tokens(self, requests: List[List[int]]) -> numpy.ndarray:
        pad = 0
        max_len = max(len(req) for req in requests)
        max_len = int(
            int(numpy.ceil(max_len / self._block_stride)) * self._block_stride
        )
        expected_tokens = numpy.full(
            (self._batch_size, max_len), fill_value=pad, dtype=numpy.int64
        )
        for i, req in enumerate(requests):
            expected_tokens[i, : len(req)] = req

        return expected_tokens

    def test_run(self):
        requests = self._requests
        page_ids = self._page_ids
        llm_tasks = self._get_tasks(requests, page_ids)

        llm_tasks = [
            # First chunk of request 0
            llm_tasks[0],
            # Second chunk of request 1
            llm_tasks[3],
            # Request 2 (single chunk)
            llm_tasks[4],
            # Second chunk of request 3
            llm_tasks[6],
        ]

        invoked_args = []

        def _invocation_fn(*args):
            invoked_args.extend(args)
            return [i for i in range(self._batch_size)]

        prefill_task = PrefillTask(
            invocation_fn=_invocation_fn,
            llm_task_inputs=llm_tasks,
            batch_size=self._batch_size,
            block_stride=self._block_stride,
            has_prefill_position=True,
            chunk_block_size=self._chunk_block_size,
        )

        cache_state = [numpy.zeros((16, 256), dtype=numpy.float16)]
        logits, indices = prefill_task.run(*cache_state)

        expected_tokens = self._get_expected_tokens(
            [task_input.tokens for task_input in llm_tasks]
        )

        expected_start_positions = numpy.array(
            [task_input.start_position for task_input in llm_tasks], dtype=numpy.int64
        )
        expected_seq_lens = numpy.array(
            [task_input.seq_len for task_input in llm_tasks], dtype=numpy.int64
        )
        expected_seq_block_ids = numpy.array(
            [
                page_ids[0][: self._chunk_block_size] + [0] * 3,
                page_ids[1] + [0] * 2,
                page_ids[2] + [0] * 4,
                page_ids[3] + [0] * 2,
            ]
        )
        assert numpy.array_equal(invoked_args[0], expected_tokens)
        assert numpy.array_equal(invoked_args[1], expected_start_positions)
        assert numpy.array_equal(invoked_args[2], expected_seq_lens)
        assert numpy.array_equal(invoked_args[3], expected_seq_block_ids)
        assert numpy.array_equal(invoked_args[4], cache_state[0])

        assert numpy.array_equal(
            logits,
            numpy.array([i for i in range(self._batch_size)], dtype=numpy.int64),
        )
        assert indices is None
