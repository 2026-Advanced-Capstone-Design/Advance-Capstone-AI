import uuid
from datetime import datetime
from enum import Enum


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    ANALYZING = "ANALYZING"
    DONE = "DONE"
    FAILED = "FAILED"


class Task:
    def __init__(self, article_id: int):
        self.task_id = str(uuid.uuid4())
        self.article_id = article_id
        self.status = TaskStatus.PENDING
        self.result = None
        self.error = None
        self.created_at = datetime.utcnow().isoformat()
        self.updated_at = self.created_at

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "article_id": self.article_id,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# 인메모리 태스크 저장소 (추후 Redis로 교체)
_task_store: dict[str, Task] = {}


def create_task(article_id: int) -> Task:
    task = Task(article_id)
    _task_store[task.task_id] = task
    return task


def get_task(task_id: str) -> Task | None:
    return _task_store.get(task_id)


def update_task(task_id: str, status: TaskStatus, result=None, error=None):
    task = _task_store.get(task_id)
    if task:
        task.status = status
        task.result = result
        task.error = error
        task.updated_at = datetime.utcnow().isoformat()
