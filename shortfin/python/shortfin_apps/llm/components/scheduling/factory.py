import shortfin as sf

from typing import List

from .config import SchedulerConfig, SchedulingModes
from ..messages import LlmInferenceExecRequest
from ..scheduler import Scheduler

from shortfin_apps.utils import BatcherProcess


class _ScheduleEngineImpl:
    def __init__(self, scheduler: Scheduler) -> None:
        self.scheduler = scheduler

    def should_execute(
        self, pending: List[LlmInferenceExecRequest], strobe
    ) -> List[LlmInferenceExecRequest]:
        return self.scheduler.should_execute(pending=pending, strobe=strobe)

    def handle_message(self, message: sf.Message) -> bool:
        return self.scheduler.handle_message(message)

    def reserve_workload(self, *, batcher: BatcherProcess, count: int, rid: str):
        self.scheduler.reserve_workload(batcher=batcher, count=count, rid=rid)


def _create_scheduler(config: SchedulerConfig) -> _ScheduleEngineImpl:
    if config.mode == SchedulingModes.STROBE:
        return _ScheduleEngineImpl(
            scheduler=Scheduler(ideal_batch_size=config.ideal_batch_size),
        )

    raise ValueError(f"Unsupported scheduling mode: {config.mode}")
