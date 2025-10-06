from sharktank.utils.llm_tasks import LlmTaskInput


class Scheduler:
    def __init__(self, batch_size: int) -> None:
        self._batch_size = batch_size
        self._pending_tasks = []

    def schedule_task(self, task: LlmTaskInput) -> None:
        self._pending_tasks.append(task)

    def get_next_batch(self) -> list[LlmTaskInput]:
        batch = self._pending_tasks[: self._batch_size]
        self._pending_tasks = self._pending_tasks[self._batch_size :]
        return batch

    def has_pending_tasks(self) -> bool:
        return len(self._pending_tasks) > 0
