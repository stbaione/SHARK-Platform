from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Callable, Union

from ..messages import LlmInferenceExecRequest

import shortfin.array as sfnp


class SupportedDecodeStrategies(Enum):
    """Supported decode strategies."""

    GREEDY = auto()


@dataclass
class DecodeStrategyConfig:
    """Configuration for decode strategies."""

    batcher_callback: Callable[[LlmInferenceExecRequest], None]
    results_callback: Callable[[Union[int, List[int]]], None]
    eos_token_id: int
    max_completion_tokens: int


class DecodeStrategy(ABC):
    """Abstract class for implementing decode strategies."""

    @property
    @abstractmethod
    def decode_strategy_config(self) -> DecodeStrategyConfig:
        pass

    async def prefill(self, exec_req: LlmInferenceExecRequest) -> int:
        decode_strategy_config = self.decode_strategy_config

        decode_strategy_config.batcher_callback(exec_req)
        await exec_req.done

        token = sfnp.argmax(exec_req.result_logits)
        token_int = token.items[0]
        decode_strategy_config.results_callback(token_int)

        exec_req.input_token_ids.append(token_int)
        exec_req.start_position = len(exec_req.input_token_ids) - 1

    @abstractmethod
    async def decode(self, exec_req: LlmInferenceExecRequest) -> List[int]:
        pass
