# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
import pytest
from unittest.mock import patch
from uuid import uuid4

from shortfin_apps.llm.components.messages import (
    LlmInferenceExecRequest,
    InferencePhase,
)
from shortfin_apps.llm.components.decode_strategy import (
    DecodeStrategy,
    DecodeStrategyConfig,
)


class MockVoidFuture:
    def __init__(self):
        self._event = asyncio.Event()

    def set_success(self):
        self._event.set()

    def __await__(self):
        return self._event.wait().__await__()


@pytest.fixture(scope="function")
def exec_req():
    with patch(
        "shortfin_apps.llm.components.messages.sf.VoidFuture", new=MockVoidFuture
    ):
        yield LlmInferenceExecRequest(
            phase=InferencePhase.PREFILL,
            input_token_ids=[0, 1, 2, 3, 4, 5],
            rid=str(uuid4),
        )


class DummyDecodeStrategy(DecodeStrategy):
    def __init__(self, decode_strategy_config: DecodeStrategyConfig):
        # Initialize with a dummy config instance.
        self._decode_strategy_config = decode_strategy_config

    @property
    def decode_strategy_config(self):
        return self._decode_strategy_config

    async def decode(self, exec_req):
        pass


@pytest.fixture(scope="module")
def dummy_decode_strategy():
    yield DummyDecodeStrategy(
        None,
    )
