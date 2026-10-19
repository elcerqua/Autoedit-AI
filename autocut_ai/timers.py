from __future__ import annotations

from datetime import datetime, timezone

from autocut_ai.schemas import JobRecord, PipelineStage, utc_now_iso


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def refresh_timers(job: JobRecord) -> JobRecord:
    now = datetime.now(timezone.utc)
    start_value = job.started_at or job.created_at
    start = parse_timestamp(start_value)
    end = parse_timestamp(job.completed_at) if job.completed_at else now
    job.elapsed_seconds = max(0.0, (end - start).total_seconds())
    if job.stage_started_at:
        stage_start = parse_timestamp(job.stage_started_at)
        stage_end = parse_timestamp(job.completed_at) if job.stage in {PipelineStage.DONE, PipelineStage.FAILED} and job.completed_at else now
        job.current_stage_elapsed_seconds = max(0.0, (stage_end - stage_start).total_seconds())
    return job


def set_stage(job: JobRecord, stage: PipelineStage, message: str | None = None) -> JobRecord:
    """Aggiorna stato e cronometri usando un unico istante coerente."""

    now = utc_now_iso()
    if job.started_at is None and stage != PipelineStage.QUEUED:
        job.started_at = now
    job.stage = stage
    job.message = message
    job.stage_started_at = now
    job.current_stage_elapsed_seconds = 0.0
    if stage in {PipelineStage.DONE, PipelineStage.FAILED}:
        job.completed_at = now
    return refresh_timers(job)
