from dataclasses import dataclass
from enum import auto, Enum

from shortfin_apps.utils import BatcherProcess


class SchedulingModes(Enum):
    STROBE = auto()


@dataclass
class SchedulerConfig:
    mode: SchedulingModes
    ideal_batch_size: int
    batcher: BatcherProcess
