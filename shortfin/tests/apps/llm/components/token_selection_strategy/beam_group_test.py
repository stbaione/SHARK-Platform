import asyncio
import random
import pytest
from typing import List
from unittest.mock import patch

from shortfin_apps.llm.components.kvcache.base_attention_cache import (
    BasePagedAttentionCacheAllocation,
)
from shortfin_apps.llm.components.messages import LlmInferenceExecRequest
from shortfin_apps.llm.components.token_selection_strategy.beam_group import (
    BeamGroup,
    ExecRequestSelection,
)


@pytest.fixture()
def exec_req_list(exec_req, cache, dummy_pages):
    exec_req._cache = cache
    allocation = BasePagedAttentionCacheAllocation(dummy_pages, cache=cache)
    exec_req.allocation = allocation
    exec_reqs = [exec_req]
    num_beams = len(dummy_pages)
    with patch.object(exec_req._cache, "fork_pages", return_value=allocation):
        for _ in range(num_beams - 1):
            exec_reqs.append(LlmInferenceExecRequest.copy_exec_request(exec_req))

    yield exec_reqs


@pytest.mark.asyncio
async def test_wait(exec_req_list):
    async def set_done(exec_reqs: List[LlmInferenceExecRequest]):
        for req in exec_reqs:
            req.done.set_success()

    beam_groups = BeamGroup(
        eos_token_id=-1,
        num_beams=len(exec_req_list),
        exec_reqs=exec_req_list,
        selection_callback=lambda x: None,
    )
    await asyncio.gather(*[beam_groups.wait(), set_done(exec_req_list)])
    for req in exec_req_list:
        assert req.done._event.is_set()


@pytest.mark.asyncio
async def test_process_beams_one_req(exec_req):
    def selection_callback(
        active_reqs: LlmInferenceExecRequest, _: LlmInferenceExecRequest
    ):
        selections = []
        for req in active_reqs:
            token = 0
            selections.append(ExecRequestSelection(req, token))

        return selections

    beam_groups = BeamGroup(
        eos_token_id=-1,
        num_beams=1,
        exec_reqs=[exec_req],
        selection_callback=selection_callback,
    )

    # Active
    await beam_groups.process_beams()
    assert beam_groups.active_exec_reqs == [exec_req]
    assert len(beam_groups.completed_reqs) == 0

    # Completed
    beam_groups.eos_token_id = 0
    with patch.object(LlmInferenceExecRequest, "free_cache_pages") as free_cache_mock:
        await beam_groups.process_beams()
        assert len(beam_groups.active_exec_reqs) == 0
        assert beam_groups.completed_reqs == {exec_req}
        free_cache_mock.assert_called_once()


@pytest.mark.asyncio
async def test_process_beams_multiple_reqs(exec_req_list):
    def selection_callback_no_completed(exec_reqs, _):
        selections = []
        for req in exec_reqs:
            token = 0
            selections.append(
                ExecRequestSelection(
                    req,
                    token,
                )
            )
        return selections

    def selection_callback_one_completed(exec_reqs, _):
        selections = [
            ExecRequestSelection(
                exec_reqs[0],
                1,
            )
        ]
        for req in exec_reqs[1:]:
            selections.append(
                ExecRequestSelection(
                    req,
                    token=0,
                )
            )
        return selections

    def selection_callback_all_completed(exec_reqs, _):
        selections = [
            ExecRequestSelection(
                exec_reqs[0],
                1,
            )
        ]
        for req in exec_reqs[1:]:
            selections.append(
                ExecRequestSelection(
                    req,
                    token=1,
                )
            )
        return selections

    req_list = exec_req_list.copy()
    beam_group = BeamGroup(
        eos_token_id=1,
        num_beams=len(req_list),
        exec_reqs=req_list,
        selection_callback=selection_callback_no_completed,
    )
    await beam_group.process_beams()
    assert set(beam_group.active_exec_reqs) == set(req_list)
    assert len(beam_group.completed_reqs) == 0

    req_list = exec_req_list.copy()
    beam_group = BeamGroup(
        eos_token_id=1,
        num_beams=len(req_list),
        exec_reqs=req_list,
        selection_callback=selection_callback_one_completed,
    )
    with patch.object(LlmInferenceExecRequest, "free_cache_pages") as free_cache_mock:
        beam_group.selection_callback = selection_callback_one_completed
        await beam_group.process_beams()
        assert beam_group.active_exec_reqs == req_list[1:]
        assert beam_group.completed_reqs == {req_list[0]}
        free_cache_mock.assert_called_once()

        # Complete another req
        await beam_group.process_beams()
        assert beam_group.active_exec_reqs == exec_req_list[2:]
        assert beam_group.completed_reqs == set(exec_req_list[:2])
        assert free_cache_mock.call_count == 2

    req_list = exec_req_list.copy()
    beam_group = BeamGroup(
        eos_token_id=1,
        num_beams=len(req_list),
        exec_reqs=req_list,
        selection_callback=selection_callback_all_completed,
    )
    # All completed
    with patch.object(LlmInferenceExecRequest, "free_cache_pages") as free_cache_mock:
        await beam_group.process_beams()
        assert len(beam_group.active_exec_reqs) == 0
        assert beam_group.completed_reqs == set(req_list)
        assert free_cache_mock.call_count == len(exec_req_list)


@pytest.mark.asyncio
async def test_clean_up(exec_req_list):
    beam_group = BeamGroup(
        eos_token_id=-1,
        num_beams=len(exec_req_list),
        exec_reqs=exec_req_list,
        selection_callback=lambda x: None,
    )
    with patch.object(LlmInferenceExecRequest, "free_cache_pages") as free_cache_mock:
        # All active
        beam_group.clean_up()
        assert free_cache_mock.call_count == len(beam_group.active_exec_reqs)

        free_cache_mock.reset_mock()

        # All completed
        beam_group.completed_reqs = set(exec_req_list)
        beam_group.active_exec_reqs = []
        beam_group.clean_up()
        assert free_cache_mock.call_count == len(beam_group.completed_reqs)

        free_cache_mock.reset_mock()

        # Mixture of both
        beam_group.completed_reqs = set(exec_req_list[: len(exec_req_list) // 2])
        beam_group.active_exec_reqs = exec_req_list[len(exec_req_list) // 2 :]
        beam_group.clean_up()
        assert free_cache_mock.call_count == len(beam_group.completed_reqs) + len(
            beam_group.active_exec_reqs
        )
