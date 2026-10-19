from __future__ import annotations

from autocut_ai.schemas import JobEvent, JobRecord
from autocut_ai.storage import JobRepository


class EventPublisher:
    def __init__(self, repository: JobRepository) -> None:
        self.repository = repository

    def publish_job(self, job: JobRecord) -> JobEvent:
        event = JobEvent(
            job_id=job.job_id,
            stage=job.stage,
            message=job.message,
            elapsed_seconds=job.elapsed_seconds,
            current_stage_elapsed_seconds=job.current_stage_elapsed_seconds,
        )
        return self.repository.append_event(event)
