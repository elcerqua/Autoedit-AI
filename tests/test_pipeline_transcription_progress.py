from __future__ import annotations

from pathlib import Path

from autocut_ai.events import EventPublisher
from autocut_ai.pipeline import PipelineRunner
from autocut_ai.schemas import (
    ClipTranscription,
    EditorialBrief,
    NarrativeClip,
    NarrativeMap,
    NarrativeSegment,
    ProposalReview,
    Segment,
    SegmentImportance,
    WordToken,
    JobRecord,
    PipelineStage,
)
from autocut_ai.storage import JobRepository


class ProgressTranscriber:
    def transcribe_many(self, paths: list[Path], progress_callback: object | None = None) -> list[ClipTranscription]:
        transcriptions: list[ClipTranscription] = []
        for index, path in enumerate(paths, start=1):
            if callable(progress_callback):
                progress_callback(index, len(paths), path, "whisper")
            transcriptions.append(
                ClipTranscription(
                    clip_id=index,
                    source_path=str(path),
                    duration=1.0,
                    segments=[
                        Segment(
                            id=1,
                            clip_id=index,
                            start=0.0,
                            end=1.0,
                            text=f"Clip {index}",
                            words=[
                                WordToken(word="Clip", start=0.0, end=0.3),
                                WordToken(word=str(index), start=0.4, end=0.8),
                            ],
                        )
                    ],
                )
            )
            if callable(progress_callback):
                progress_callback(index, len(paths), path, "completed")
        return transcriptions


class ProgressPlanner:
    def generate(
        self,
        transcriptions: list[ClipTranscription],
        *_args: object,
        progress_callback: object | None = None,
    ) -> NarrativeMap:
        if callable(progress_callback):
            progress_callback("Analisi narrativa chunk 1/2")
            progress_callback("Analisi narrativa chunk 2/2")
        return NarrativeMap(
            ordered_clips=[
                NarrativeClip(
                    clip_id=transcription.clip_id,
                    playback_position=index,
                    topic_summary="Clip",
                    role="development",
                    segments=[
                        NarrativeSegment(
                            segment_id=1,
                            importance=SegmentImportance.ESSENTIAL,
                            keep_verbatim=True,
                            reason="Mantieni",
                        )
                    ],
                )
                for index, transcription in enumerate(transcriptions)
            ],
            filler_words=[],
            editorial_strategy="Mantieni tutte le clip.",
            estimated_duration_s=2.0,
        )


class ProgressReviewer:
    def review(self, *_args: object) -> ProposalReview:
        return ProposalReview(approved=True, editorial_notes="ok")


class ProgressCallbackReviewer:
    def review(self, *_args: object, progress_callback: object | None = None) -> ProposalReview:
        if callable(progress_callback):
            progress_callback("Review LLM globale blocchi")
            progress_callback("Review LLM locale cut e frasi")
        return ProposalReview(approved=True, editorial_notes="ok")


class UnusedRenderer:
    pass


def test_pipeline_publishes_transcription_progress_per_clip(tmp_path: Path) -> None:
    repository = JobRepository(tmp_path / "jobs.sqlite3")
    job = JobRecord(
        job_id="progress-job",
        brief=EditorialBrief(synthesis_level=1),
        input_clip_paths=["first.mp4", "second.mp4"],
    )
    repository.save(job)
    runner = PipelineRunner(
        repository,
        EventPublisher(repository),
        ProgressTranscriber(),
        ProgressPlanner(),
        ProgressReviewer(),
        UnusedRenderer(),
    )

    saved = runner.run_phase_one(job.job_id)
    messages = [event.message or "" for event in repository.list_events_after(job.job_id, 0)]

    assert saved.stage == PipelineStage.WAITING_FOR_APPROVAL
    assert any("Trascrizione clip 1/2" in message for message in messages)
    assert any("Trascrizione clip 2/2" in message for message in messages)
    assert any("Trascrizione clip 2/2 completata" in message for message in messages)
    assert any(
        "Analisi narrativa LLM" in message and "2 clip" in message and "2 segmenti" in message
        for message in messages
    )
    assert "Analisi narrativa chunk 1/2" in messages
    assert "Analisi narrativa chunk 2/2" in messages


def test_pipeline_publishes_refining_progress_from_proposal_reviewer(tmp_path: Path) -> None:
    repository = JobRepository(tmp_path / "jobs.sqlite3")
    job = JobRecord(
        job_id="review-progress-job",
        brief=EditorialBrief(synthesis_level=1),
        input_clip_paths=["first.mp4"],
    )
    repository.save(job)
    runner = PipelineRunner(
        repository,
        EventPublisher(repository),
        ProgressTranscriber(),
        ProgressPlanner(),
        ProgressCallbackReviewer(),
        UnusedRenderer(),
    )

    saved = runner.run_phase_one(job.job_id)
    messages = [event.message or "" for event in repository.list_events_after(job.job_id, 0)]

    assert saved.stage == PipelineStage.WAITING_FOR_APPROVAL
    assert "Review LLM globale blocchi" in messages
    assert "Review LLM locale cut e frasi" in messages
    assert any("Review LLM proposta" in message for message in messages)
