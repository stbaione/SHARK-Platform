from dataclasses import dataclass
from enum import auto, Enum

from shortfin_apps.utils import BatcherProcess


class SchedulerModes(Enum):
    STROBE = auto()


@dataclass
class SchedulerConfig:
    mode: SchedulerModes
    ideal_batch_size: int
    batcher: BatcherProcess
