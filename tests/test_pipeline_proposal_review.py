from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from autocut_ai.config import AppSettings
from autocut_ai.events import EventPublisher
from autocut_ai.pipeline import InvalidApprovalStateError, PipelineRunner
from autocut_ai.proposal_review import ProposalReviewer
from autocut_ai.schemas import (
    ClipTranscription,
    EditDecisionList,
    EditorialBrief,
    JobRecord,
    NarrativeClip,
    NarrativeMap,
    NarrativeSegment,
    PipelineStage,
    ProposalRecalculateRequest,
    ProposalIssue,
    ProposalReview,
    Segment,
    SegmentImportance,
    WordToken,
)
from autocut_ai.storage import JobRepository
from autocut_ai.narrative import MockNarrativePlanner


def _transcription() -> ClipTranscription:
    return ClipTranscription(
        clip_id=1,
        source_path="clip.mp4",
        duration=4.0,
        segments=[
            Segment(
                id=1,
                clip_id=1,
                start=0.0,
                end=1.0,
                text="Tommaso Foti introduce il tema.",
                words=[WordToken(word="Tommaso", start=0.0, end=0.3), WordToken(word="tema.", start=0.7, end=1.0)],
            ),
            Segment(
                id=2,
                clip_id=1,
                start=2.0,
                end=3.0,
                text="Il secondo passaggio completa la proposta.",
                words=[WordToken(word="Il", start=2.0, end=2.1), WordToken(word="proposta.", start=2.6, end=3.0)],
            ),
        ],
    )


def _map(segment_ids: list[int]) -> NarrativeMap:
    return NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=1,
                playback_position=0,
                topic_summary="Tema",
                role="introduction",
                segments=[
                    NarrativeSegment(
                        segment_id=segment_id,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason=f"Mantieni segmento {segment_id}",
                    )
                    for segment_id in segment_ids
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Costruisci una proposta completa.",
        estimated_duration_s=float(len(segment_ids)),
    )


class StaticTranscriber:
    def transcribe_many(self, paths: list[Path]) -> list[ClipTranscription]:
        return [_transcription()]


class FlowTranscriber:
    def transcribe_many(self, paths: list[Path]) -> list[ClipTranscription]:
        return _flow_transcriptions()


class StaticPlanner:
    def __init__(self, narrative_map: NarrativeMap) -> None:
        self.narrative_map = narrative_map

    def generate(
        self,
        transcriptions: list[ClipTranscription],
        brief: EditorialBrief,
        model: str | None,
        editing_options: object | None = None,
    ) -> NarrativeMap:
        return self.narrative_map


class CorrectingReviewer:
    def __init__(self, corrected_map: NarrativeMap) -> None:
        self.corrected_map = corrected_map
        self.seen_edl: EditDecisionList | None = None
        self.seen_edls: list[EditDecisionList] = []

    def review(
        self,
        transcriptions: list[ClipTranscription],
        narrative_map: NarrativeMap,
        edit_decision_list: EditDecisionList,
        brief: EditorialBrief,
        model: str | None,
        editing_options: object | None = None,
    ) -> ProposalReview:
        self.seen_edl = edit_decision_list
        self.seen_edls.append(edit_decision_list)
        if len(self.seen_edls) > 1:
            return ProposalReview(
                approved=True,
                issues=[],
                corrected_narrative_map=None,
                editorial_notes="Proposta corretta approvata nella review finale.",
            )
        return ProposalReview(
            approved=False,
            issues=[
                ProposalIssue(
                    severity="error",
                    code="missing_context",
                    message="La proposta manca di un passaggio necessario.",
                )
            ],
            corrected_narrative_map=self.corrected_map,
            editorial_notes="Aggiunto il segmento mancante.",
        )


class NonConvergingCorrectingReviewer:
    def __init__(self, corrected_map: NarrativeMap) -> None:
        self.corrected_map = corrected_map
        self.calls = 0

    def review(
        self,
        transcriptions: list[ClipTranscription],
        narrative_map: NarrativeMap,
        edit_decision_list: EditDecisionList,
        brief: EditorialBrief,
        model: str | None,
        editing_options: object | None = None,
    ) -> ProposalReview:
        self.calls += 1
        return ProposalReview(
            approved=False,
            issues=[
                ProposalIssue(
                    severity="warning",
                    code="still_refining",
                    message="Il reviewer continua a proporre la stessa correzione.",
                )
            ],
            corrected_narrative_map=self.corrected_map,
            editorial_notes="Correzione ripetuta.",
        )


class InvalidReferenceCorrectingReviewer:
    def __init__(self, corrected_map: NarrativeMap) -> None:
        self.corrected_map = corrected_map
        self.calls = 0

    def review(
        self,
        transcriptions: list[ClipTranscription],
        narrative_map: NarrativeMap,
        edit_decision_list: EditDecisionList,
        brief: EditorialBrief,
        model: str | None,
        editing_options: object | None = None,
        progress_callback: object | None = None,
    ) -> ProposalReview:
        self.calls += 1
        return ProposalReview(
            approved=False,
            issues=[
                ProposalIssue(
                    severity="warning",
                    code="invalid_patch_reference_ignored",
                    message="La review locale ha proposto riferimenti non validi.",
                )
            ],
            corrected_narrative_map=self.corrected_map,
            editorial_notes="Uso la correzione globale e ignoro la patch locale instabile.",
        )


class SequencedCorrectingReviewer:
    def __init__(self, corrected_maps: list[NarrativeMap]) -> None:
        self.corrected_maps = corrected_maps
        self.calls = 0

    def review(
        self,
        transcriptions: list[ClipTranscription],
        narrative_map: NarrativeMap,
        edit_decision_list: EditDecisionList,
        brief: EditorialBrief,
        model: str | None,
        editing_options: object | None = None,
    ) -> ProposalReview:
        corrected_map = self.corrected_maps[min(self.calls, len(self.corrected_maps) - 1)]
        self.calls += 1
        return ProposalReview(
            approved=False,
            issues=[
                ProposalIssue(
                    severity="warning",
                    code="still_refining",
                    message="Il reviewer propone ancora una correzione valida.",
                )
            ],
            corrected_narrative_map=corrected_map,
            editorial_notes="Correzione valida.",
        )


class RejectingReviewer:
    def review(
        self,
        transcriptions: list[ClipTranscription],
        narrative_map: NarrativeMap,
        edit_decision_list: EditDecisionList,
        brief: EditorialBrief,
        model: str | None,
        editing_options: object | None = None,
    ) -> ProposalReview:
        return ProposalReview(
            approved=False,
            issues=[
                ProposalIssue(
                    severity="error",
                    code="unsafe_opening",
                    message="La proposta non e pronta per il montaggio finale.",
                )
            ],
            corrected_narrative_map=None,
            editorial_notes="Serve una correzione LLM.",
        )


class ExcessiveSynthesisReviewVariantReviewer:
    def __init__(self, rejected_map: NarrativeMap) -> None:
        self.rejected_map = rejected_map

    def review(
        self,
        transcriptions: list[ClipTranscription],
        narrative_map: NarrativeMap,
        edit_decision_list: EditDecisionList,
        brief: EditorialBrief,
        model: str | None,
        editing_options: object | None = None,
    ) -> ProposalReview:
        return ProposalReview(
            approved=False,
            issues=[
                ProposalIssue(
                    severity="error",
                    code="review_patch_rejected_by_validation",
                    message="Patch review troppo aggressiva.",
                )
            ],
            corrected_narrative_map=None,
            rejected_narrative_map=self.rejected_map,
            editorial_notes=(
                "ProposalReviewPatch non valida: excessive_synthesis: retained_duration_s=98.6s "
                "is too short for synthesis_level=2."
            ),
        )


class ApprovingReviewer:
    def review(
        self,
        transcriptions: list[ClipTranscription],
        narrative_map: NarrativeMap,
        edit_decision_list: EditDecisionList,
        brief: EditorialBrief,
        model: str | None,
        editing_options: object | None = None,
    ) -> ProposalReview:
        return ProposalReview(
            approved=True,
            issues=[],
            corrected_narrative_map=None,
            editorial_notes="Proposta approvata.",
        )


class UnusedRenderer:
    pass


def _flow_transcriptions() -> list[ClipTranscription]:
    return [
        ClipTranscription(
            clip_id=2,
            source_path="clip2.mp4",
            duration=3.0,
            segments=[
                Segment(
                    id=1,
                    clip_id=2,
                    start=0.0,
                    end=1.5,
                    text="Effetto operativo: possono chiedere i contributi pubblici.",
                    words=[WordToken(word="La", start=0.0, end=0.1), WordToken(word="pubblici.", start=1.2, end=1.5)],
                )
            ],
        ),
        ClipTranscription(
            clip_id=3,
            source_path="clip3.mp4",
            duration=3.0,
            segments=[
                Segment(
                    id=1,
                    clip_id=3,
                    start=0.0,
                    end=1.5,
                    text="Contesto normativo: la legge del 2016, la 198, introduce il criterio.",
                    words=[WordToken(word="Solo", start=0.0, end=0.2), WordToken(word="criterio.", start=1.2, end=1.5)],
                )
            ],
        ),
    ]


def _flow_map_wrong_order() -> NarrativeMap:
    return NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=2,
                playback_position=0,
                topic_summary="Effetto della norma",
                role="development",
                segments=[
                    NarrativeSegment(
                        segment_id=1,
                        importance=SegmentImportance.VALUABLE,
                        keep_verbatim=True,
                        reason="Spiega effetto.",
                    )
                ],
            ),
            NarrativeClip(
                clip_id=3,
                playback_position=1,
                topic_summary="Definizione della norma",
                role="introduction",
                segments=[
                    NarrativeSegment(
                        segment_id=1,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason="Introduce legge 2016 198.",
                    )
                ],
            ),
        ],
        filler_words=[],
        editorial_strategy="Ordine iniziale sbagliato.",
        estimated_duration_s=3.0,
    )


def _runner(tmp_path: Path, reviewer: object, initial_map: NarrativeMap) -> tuple[PipelineRunner, JobRepository, str]:
    settings = AppSettings(data_dir=tmp_path / "data", database_path=tmp_path / "data" / "jobs.sqlite3")
    repository = JobRepository(settings.database_path)
    job = JobRecord(
        job_id="review-job",
        brief=EditorialBrief(synthesis_level=3),
        input_clip_paths=["clip.mp4"],
    )
    repository.save(job)
    runner = PipelineRunner(
        repository,
        EventPublisher(repository),
        StaticTranscriber(),
        StaticPlanner(initial_map),
        reviewer,
        UnusedRenderer(),
        settings=settings,
    )
    return runner, repository, job.job_id


def test_pipeline_rebuilds_edl_when_proposal_reviewer_corrects_narrative_map(tmp_path: Path) -> None:
    reviewer = CorrectingReviewer(corrected_map=_map([1, 2]))
    runner, repository, job_id = _runner(tmp_path, reviewer, _map([1]))

    job = runner.run_phase_one(job_id)

    saved = repository.get(job.job_id)
    assert saved.stage == PipelineStage.WAITING_FOR_APPROVAL
    assert reviewer.seen_edl is not None
    assert len(reviewer.seen_edls) == 2
    assert len(reviewer.seen_edls[0].decisions) == 1
    assert saved.proposal_review is not None
    assert saved.proposal_review.approved is True
    assert saved.proposal_review.corrected_narrative_map is None
    assert saved.edit_decision_list is not None
    assert [decision.included_text for decision in saved.edit_decision_list.decisions] == [
        "Tommaso Foti introduce il tema.",
        "Il secondo passaggio completa la proposta.",
    ]
    assert [decision.included_text for decision in reviewer.seen_edls[1].decisions] == [
        "Tommaso Foti introduce il tema.",
        "Il secondo passaggio completa la proposta.",
    ]
    proposal_txt = tmp_path / "data" / "outputs" / job_id / "proposal.txt"
    proposal_json = tmp_path / "data" / "outputs" / job_id / "proposal.json"
    assert proposal_txt.exists()
    assert proposal_json.exists()
    assert "Tommaso Foti introduce il tema." in proposal_txt.read_text(encoding="utf-8")


def test_pipeline_accepts_equivalent_repeated_reviewer_correction(tmp_path: Path) -> None:
    reviewer = NonConvergingCorrectingReviewer(corrected_map=_map([1, 2]))
    runner, repository, job_id = _runner(tmp_path, reviewer, _map([1]))

    job = runner.run_phase_one(job_id)

    saved = repository.get(job.job_id)
    assert saved.stage == PipelineStage.WAITING_FOR_APPROVAL
    assert reviewer.calls == 2
    assert saved.proposal_review is not None
    assert saved.proposal_review.approved is True
    assert saved.edit_decision_list is not None
    assert [decision.included_text for decision in saved.edit_decision_list.decisions] == [
        "Tommaso Foti introduce il tema.",
        "Il secondo passaggio completa la proposta.",
    ]


def test_pipeline_stops_review_retry_when_local_patch_has_invalid_references(tmp_path: Path) -> None:
    reviewer = InvalidReferenceCorrectingReviewer(corrected_map=_map([1, 2]))
    runner, repository, job_id = _runner(tmp_path, reviewer, _map([1]))

    job = runner.run_phase_one(job_id)

    saved = repository.get(job.job_id)
    assert saved.stage == PipelineStage.WAITING_FOR_APPROVAL
    assert reviewer.calls == 1
    assert saved.proposal_review is not None
    assert saved.proposal_review.approved is False
    assert [variant.id for variant in saved.proposal_variants] == ["initial", "review"]
    review_variant = saved.proposal_variants[1]
    assert review_variant.selectable is True
    assert any(issue.code == "invalid_patch_reference_ignored" for issue in review_variant.issues)
    assert saved.selected_proposal_variant_id == "review"
    assert saved.edit_decision_list is not None
    assert [decision.included_text for decision in saved.edit_decision_list.decisions] == [
        "Tommaso Foti introduce il tema.",
        "Il secondo passaggio completa la proposta.",
    ]


def test_pipeline_uses_last_valid_correction_when_review_hits_convergence_limit(tmp_path: Path) -> None:
    reviewer = SequencedCorrectingReviewer(
        corrected_maps=[
            _map([1, 2]),
            _map([2]),
            _map([1, 2]),
        ]
    )
    runner, repository, job_id = _runner(tmp_path, reviewer, _map([1]))

    job = runner.run_phase_one(job_id)

    saved = repository.get(job.job_id)
    assert saved.stage == PipelineStage.WAITING_FOR_APPROVAL
    assert reviewer.calls == 3
    assert saved.proposal_review is not None
    assert saved.proposal_review.approved is True
    assert [issue.code for issue in saved.proposal_review.issues] == ["proposal_review_convergence_limit"]
    assert saved.edit_decision_list is not None
    assert [decision.included_text for decision in saved.edit_decision_list.decisions] == [
        "Tommaso Foti introduce il tema.",
        "Il secondo passaggio completa la proposta.",
    ]


def test_pipeline_creates_conservative_review_variant_when_reviewer_rejects_without_correction(tmp_path: Path) -> None:
    runner, repository, job_id = _runner(tmp_path, RejectingReviewer(), _map([1]))

    job = runner.run_phase_one(job_id)

    saved = repository.get(job.job_id)
    assert saved.stage == PipelineStage.WAITING_FOR_APPROVAL
    assert saved.proposal_review is not None
    assert saved.proposal_review.approved is False
    assert [issue.code for issue in saved.proposal_review.issues] == ["unsafe_opening", "review_fallback_initial"]
    assert saved.error is None
    assert [variant.id for variant in saved.proposal_variants] == ["initial", "review"]
    initial_variant, review_variant = saved.proposal_variants
    assert initial_variant.selectable is True
    assert review_variant.selectable is True
    assert review_variant.duration_s == pytest.approx(initial_variant.duration_s)
    assert "fallback conservativo" in (review_variant.rejection_reason or "")

    approved = runner.mark_approved(job_id, proposal_variant_id="review")

    assert approved.stage == PipelineStage.COMPOSING
    assert approved.selected_proposal_variant_id == "review"


def test_pipeline_creates_review_variant_when_reviewer_approves_without_changes(tmp_path: Path) -> None:
    runner, repository, job_id = _runner(tmp_path, ApprovingReviewer(), _map([1]))

    saved = runner.run_phase_one(job_id)

    assert saved.stage == PipelineStage.WAITING_FOR_APPROVAL
    assert [variant.id for variant in saved.proposal_variants] == ["initial", "review"]
    review_variant = saved.proposal_variants[1]
    assert review_variant.selectable is True
    assert review_variant.duration_s == pytest.approx(saved.proposal_variants[0].duration_s)

    approved = runner.mark_approved(job_id, proposal_variant_id="review")

    assert approved.stage == PipelineStage.COMPOSING
    assert approved.selected_proposal_variant_id == "review"


def test_pipeline_keeps_initial_proposal_selectable_when_duration_budget_is_too_aggressive(tmp_path: Path) -> None:
    segment_duration = 25.98
    segments = [
        Segment(
            id=index,
            clip_id=1,
            start=(index - 1) * segment_duration,
            end=index * segment_duration,
            text=f"Tommaso Foti passaggio informativo completo {index}.",
            words=[
                WordToken(word="Tommaso", start=(index - 1) * segment_duration, end=(index - 1) * segment_duration + 0.4),
                WordToken(word=f"{index}.", start=index * segment_duration - 0.4, end=index * segment_duration),
            ],
        )
        for index in range(1, 11)
    ]
    transcription = ClipTranscription(clip_id=1, source_path="clip.mp4", duration=259.8, segments=segments)

    class LongTranscriber:
        def transcribe_many(self, paths: list[Path]) -> list[ClipTranscription]:
            return [transcription]

    too_short_map = NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=1,
                playback_position=0,
                topic_summary="Sequenza sintetica",
                role="development",
                segments=[
                    NarrativeSegment(
                        segment_id=segment.id,
                        importance=SegmentImportance.VALUABLE,
                        keep_verbatim=True,
                        reason="Mantieni passaggio.",
                    )
                    for segment in segments[:7]
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Proposta piu corta del budget indicativo.",
        estimated_duration_s=181.86,
    )
    settings = AppSettings(data_dir=tmp_path / "data", database_path=tmp_path / "data" / "jobs.sqlite3")
    repository = JobRepository(settings.database_path)
    job = JobRecord(
        job_id="duration-warning-job",
        brief=EditorialBrief(synthesis_level=1),
        input_clip_paths=["clip.mp4"],
    )
    repository.save(job)
    runner = PipelineRunner(
        repository,
        EventPublisher(repository),
        LongTranscriber(),
        StaticPlanner(too_short_map),
        ApprovingReviewer(),
        UnusedRenderer(),
        settings=settings,
    )

    saved = runner.run_phase_one(job.job_id)

    assert saved.stage == PipelineStage.WAITING_FOR_APPROVAL
    assert [variant.id for variant in saved.proposal_variants] == ["initial", "review"]
    assert any(issue.code == "excessive_synthesis" for issue in saved.proposal_variants[0].issues)
    assert any(issue.code == "excessive_synthesis" for issue in saved.proposal_variants[1].issues)
    approved = runner.mark_approved(job.job_id, proposal_variant_id="initial")
    assert approved.stage == PipelineStage.COMPOSING


def test_pipeline_creates_selectable_review_variant_for_excessive_synthesis_patch(tmp_path: Path) -> None:
    segment_duration = 14.09
    segments = [
        Segment(
            id=index,
            clip_id=1,
            start=(index - 1) * segment_duration,
            end=index * segment_duration,
            text=f"Segmento informativo completo {index}.",
            words=[
                WordToken(word="Segmento", start=(index - 1) * segment_duration, end=(index - 1) * segment_duration + 0.3),
                WordToken(word=f"{index}.", start=index * segment_duration - 0.3, end=index * segment_duration),
            ],
        )
        for index in range(1, 11)
    ]
    transcription = ClipTranscription(clip_id=1, source_path="clip.mp4", duration=140.9, segments=segments)

    class LongTranscriber:
        def transcribe_many(self, paths: list[Path]) -> list[ClipTranscription]:
            return [transcription]

    initial_map = NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=1,
                playback_position=0,
                topic_summary="Sequenza completa",
                role="development",
                segments=[
                    NarrativeSegment(
                        segment_id=segment.id,
                        importance=SegmentImportance.VALUABLE,
                        keep_verbatim=True,
                        reason="Mantieni contesto completo.",
                    )
                    for segment in segments
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Proposta iniziale completa.",
        estimated_duration_s=140.9,
    )
    rejected_map = NarrativeMap(
        ordered_clips=[
            initial_map.ordered_clips[0].model_copy(
                update={"segments": initial_map.ordered_clips[0].segments[:7]}
            )
        ],
        filler_words=[],
        editorial_strategy="Review LLM piu aggressiva.",
        estimated_duration_s=98.6,
    )
    settings = AppSettings(data_dir=tmp_path / "data", database_path=tmp_path / "data" / "jobs.sqlite3")
    repository = JobRepository(settings.database_path)
    job = JobRecord(
        job_id="variant-job",
        brief=EditorialBrief(synthesis_level=2),
        input_clip_paths=["clip.mp4"],
    )
    repository.save(job)
    runner = PipelineRunner(
        repository,
        EventPublisher(repository),
        LongTranscriber(),
        StaticPlanner(initial_map),
        ExcessiveSynthesisReviewVariantReviewer(rejected_map),
        UnusedRenderer(),
        settings=settings,
    )

    saved = runner.run_phase_one(job.job_id)

    assert saved.stage == PipelineStage.WAITING_FOR_APPROVAL
    assert [variant.id for variant in saved.proposal_variants] == ["initial", "review"]
    initial_variant, review_variant = saved.proposal_variants
    assert initial_variant.selectable is True
    assert review_variant.selectable is True
    assert review_variant.rejection_reason is not None
    assert "excessive_synthesis" in review_variant.rejection_reason
    assert review_variant.duration_s == pytest.approx(
        sum(decision.source_end - decision.source_start for decision in review_variant.edit_decision_list.decisions),
        abs=0.001,
    )
    assert saved.selected_proposal_variant_id == "initial"
    assert saved.message is not None
    assert "piu aggressiva del budget" in saved.message
    assert "non approvata" not in saved.message.lower()

    approved_review = runner.mark_approved(job.job_id, proposal_variant_id="review")

    assert approved_review.stage == PipelineStage.COMPOSING
    assert approved_review.selected_proposal_variant_id == "review"
    assert approved_review.edit_decision_list is not None
    assert approved_review.edit_decision_list.estimated_duration_s == pytest.approx(
        review_variant.edit_decision_list.estimated_duration_s,
        abs=0.001,
    )
    assert approved_review.edit_decision_list.estimated_duration_s < initial_variant.edit_decision_list.estimated_duration_s
    approved_review_text = " ".join(decision.included_text for decision in approved_review.edit_decision_list.decisions)
    assert "Segmento informativo completo 7." in approved_review_text
    assert "Segmento informativo completo 8." not in approved_review_text


def test_pipeline_rejects_unknown_or_unselectable_proposal_variant_on_approval(tmp_path: Path) -> None:
    runner, repository, job_id = _runner(tmp_path, ApprovingReviewer(), _map([1]))
    saved = runner.run_phase_one(job_id)
    assert saved.stage == PipelineStage.WAITING_FOR_APPROVAL

    with pytest.raises(InvalidApprovalStateError, match="Variante proposta non trovata"):
        runner.mark_approved(job_id, proposal_variant_id="missing")


def test_pipeline_falls_back_to_initial_when_review_variant_is_not_renderable(tmp_path: Path) -> None:
    invalid_review_map = _map([99])
    runner, repository, job_id = _runner(
        tmp_path,
        ExcessiveSynthesisReviewVariantReviewer(invalid_review_map),
        _map([1]),
    )

    saved = runner.run_phase_one(job_id)

    assert saved.stage == PipelineStage.WAITING_FOR_APPROVAL
    assert saved.error is None
    assert [variant.id for variant in saved.proposal_variants] == ["initial", "review"]
    assert saved.proposal_review is not None
    assert saved.proposal_review.approved is False
    assert any(issue.code == "review_variant_not_renderable" for issue in saved.proposal_review.issues)
    assert any(issue.code == "review_fallback_initial" for issue in saved.proposal_review.issues)
    review_variant = saved.proposal_variants[1]
    assert review_variant.selectable is True
    assert "fallback conservativo" in (review_variant.rejection_reason or "")
    approved = runner.mark_approved(job_id, proposal_variant_id="review")
    assert approved.stage == PipelineStage.COMPOSING


def test_pipeline_review_variant_uses_global_block_review_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    global_calls = 0

    class GlobalReviewClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def chat(self, **kwargs: object) -> dict[str, dict[str, str]]:
            nonlocal global_calls
            title = kwargs["format"]["title"]
            if title == "GlobalProposalReviewPlan":
                global_calls += 1
                if global_calls == 1:
                    return {
                        "message": {
                            "content": (
                                '{"ordered_block_ids":[1,0],"dropped_block_ids":[],"restored_segments":[],'
                                '"issues":[{"severity":"warning","code":"antecedent_order","message":"La legge deve precedere la stessa norma."}],'
                                '"editorial_notes":"Ordine globale corretto: prima legge 2016 198, poi effetto."}'
                            )
                        }
                    }
                return {
                    "message": {
                        "content": (
                            '{"ordered_block_ids":[],"dropped_block_ids":[],"restored_segments":[],'
                            '"issues":[],"editorial_notes":"Ordine globale ora coerente."}'
                        )
                    }
                }
            return {
                "message": {
                    "content": (
                        '{"approved":true,"issues":[],"segment_adjustments":[],"word_removals":[],'
                        '"clip_order":[],"editorial_notes":"Review locale senza ulteriori tagli."}'
                    )
                }
            }

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=GlobalReviewClient))

    settings = AppSettings(
        data_dir=tmp_path / "data",
        database_path=tmp_path / "data" / "jobs.sqlite3",
        ollama_run_in_subprocess=False,
    )
    repository = JobRepository(settings.database_path)
    job = JobRecord(
        job_id="global-review-pipeline",
        brief=EditorialBrief(user_prompt="Metti prima la norma e poi gli effetti.", synthesis_level=2),
        input_clip_paths=["clip2.mp4", "clip3.mp4"],
    )
    repository.save(job)
    runner = PipelineRunner(
        repository,
        EventPublisher(repository),
        FlowTranscriber(),
        StaticPlanner(_flow_map_wrong_order()),
        ProposalReviewer(settings, max_retries=0),
        UnusedRenderer(),
        settings=settings,
    )

    saved = runner.run_phase_one(job.job_id)

    assert saved.stage == PipelineStage.WAITING_FOR_APPROVAL
    assert [variant.id for variant in saved.proposal_variants] == ["initial", "review"]
    assert saved.selected_proposal_variant_id == "review"
    review_variant = saved.proposal_variants[1]
    assert [clip.clip_id for clip in review_variant.narrative_map.ordered_clips] == [3, 2]
    assert review_variant.edit_decision_list.decisions[0].clip_id == 3
    approved = runner.mark_approved(job.job_id, proposal_variant_id="review")
    assert approved.edit_decision_list is not None
    assert approved.edit_decision_list.decisions[0].included_text.startswith("Contesto normativo")


def test_pipeline_fails_before_approval_when_edl_exceeds_explicit_target_duration(tmp_path: Path) -> None:
    segments = [
        Segment(
            id=index,
            clip_id=1,
            start=float((index - 1) * 3),
            end=float((index - 1) * 3 + 2),
            text=f"Passaggio informativo {index}.",
            words=[
                WordToken(word="Passaggio", start=float((index - 1) * 3), end=float((index - 1) * 3 + 0.4)),
                WordToken(word=f"{index}.", start=float((index - 1) * 3 + 1.6), end=float((index - 1) * 3 + 2)),
            ],
        )
        for index in range(1, 21)
    ]
    transcription = ClipTranscription(clip_id=1, source_path="clip.mp4", duration=62.0, segments=segments)
    narrative_map = NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=1,
                playback_position=0,
                topic_summary="Molti passaggi",
                role="development",
                segments=[
                    NarrativeSegment(
                        segment_id=segment.id,
                        importance=SegmentImportance.VALUABLE,
                        keep_verbatim=True,
                        reason="Mantieni per test target EDL.",
                    )
                    for segment in segments
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Mappa entro target parlato ma troppo lunga dopo pre/post roll.",
        estimated_duration_s=40.0,
    )

    class LongTranscriber:
        def transcribe_many(self, paths: list[Path]) -> list[ClipTranscription]:
            return [transcription]

    settings = AppSettings(data_dir=tmp_path / "data", database_path=tmp_path / "data" / "jobs.sqlite3")
    repository = JobRepository(settings.database_path)
    job = JobRecord(
        job_id="edl-target-job",
        brief=EditorialBrief(synthesis_level=3, target_duration_s=37.0),
        input_clip_paths=["clip.mp4"],
    )
    repository.save(job)
    runner = PipelineRunner(
        repository,
        EventPublisher(repository),
        LongTranscriber(),
        StaticPlanner(narrative_map),
        ApprovingReviewer(),
        UnusedRenderer(),
        settings=settings,
    )

    saved = runner.run_phase_one(job.job_id)

    assert saved.stage == PipelineStage.FAILED
    assert "target_duration_exceeded_after_cut_safety" in (saved.error or "")
    assert "EditDecisionList" in (saved.error or "")


def test_pipeline_recalculates_proposal_without_transcription_or_global_analysis(tmp_path: Path) -> None:
    transcription = _transcription()
    initial_map = _map([1, 2])

    class ExplodingTranscriber:
        def transcribe_many(self, paths: list[Path]) -> list[ClipTranscription]:
            raise AssertionError("recalculate_proposal must not transcribe again")

    class RecalculateOnlyPlanner:
        def __init__(self) -> None:
            self.generate_calls = 0
            self.recalculate_calls = 0
            self.seen_brief: EditorialBrief | None = None

        def generate(
            self,
            transcriptions: list[ClipTranscription],
            brief: EditorialBrief,
            model: str | None,
            editing_options: object | None = None,
        ) -> NarrativeMap:
            self.generate_calls += 1
            raise AssertionError("recalculate_proposal must not run global narrative analysis")

        def recalculate_proposal(
            self,
            narrative_map: NarrativeMap,
            transcriptions: list[ClipTranscription],
            brief: EditorialBrief,
            model: str | None,
            editing_options: object | None = None,
            progress_callback: object | None = None,
        ) -> NarrativeMap:
            self.recalculate_calls += 1
            self.seen_brief = brief
            return _map([1])

    settings = AppSettings(data_dir=tmp_path / "data", database_path=tmp_path / "data" / "jobs.sqlite3")
    repository = JobRepository(settings.database_path)
    job = JobRecord(
        job_id="recalculate-job",
        stage=PipelineStage.WAITING_FOR_APPROVAL,
        brief=EditorialBrief(synthesis_level=2),
        input_clip_paths=["clip.mp4"],
        transcriptions=[transcription],
        narrative_map=initial_map,
    )
    repository.save(job)
    planner = RecalculateOnlyPlanner()
    runner = PipelineRunner(
        repository,
        EventPublisher(repository),
        ExplodingTranscriber(),
        planner,
        ApprovingReviewer(),
        UnusedRenderer(),
        settings=settings,
    )

    saved = runner.recalculate_proposal(
        job.job_id,
        ProposalRecalculateRequest(synthesis_level=5, target_duration_s=1.0),
    )

    assert saved.stage == PipelineStage.WAITING_FOR_APPROVAL
    assert saved.brief.synthesis_level == 5
    assert saved.brief.target_duration_s == 1.0
    assert planner.generate_calls == 0
    assert planner.recalculate_calls == 1
    assert planner.seen_brief is not None
    assert planner.seen_brief.synthesis_level == 5
    assert saved.edit_decision_list is not None
    assert [decision.included_text for decision in saved.edit_decision_list.decisions] == [
        "Tommaso Foti introduce il tema."
    ]
    proposal_txt = tmp_path / "data" / "outputs" / job.job_id / "proposal.txt"
    assert proposal_txt.exists()


def test_pipeline_repairs_opening_context_fragment_before_validation(tmp_path: Path) -> None:
    transcription = ClipTranscription(
        clip_id=1,
        source_path="clip.mp4",
        duration=12.0,
        segments=[
            Segment(
                id=1,
                clip_id=1,
                start=0.0,
                end=5.0,
                text="Tommaso Foti viene eletto consigliere comunale e poi ricopre",
                words=[WordToken(word="Tommaso", start=0.0, end=0.3), WordToken(word="ricopre", start=4.6, end=5.0)],
            ),
            Segment(
                id=2,
                clip_id=1,
                start=5.0,
                end=8.0,
                text="incarichi politici nella stessa area",
                words=[WordToken(word="incarichi", start=5.0, end=5.4), WordToken(word="area", start=7.6, end=8.0)],
            ),
        ],
    )

    class ContextFragmentTranscriber:
        def transcribe_many(self, paths: list[Path]) -> list[ClipTranscription]:
            return [transcription]

    initial_map = NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=1,
                playback_position=0,
                topic_summary="Carriera",
                role="introduction",
                segments=[
                    NarrativeSegment(
                        segment_id=2,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason="Apre sulla carriera",
                    )
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Apre sulla carriera.",
        estimated_duration_s=3.0,
    )
    settings = AppSettings(data_dir=tmp_path / "data", database_path=tmp_path / "data" / "jobs.sqlite3")
    repository = JobRepository(settings.database_path)
    job = JobRecord(job_id="repair-job", brief=EditorialBrief(synthesis_level=2), input_clip_paths=["clip.mp4"])
    repository.save(job)
    runner = PipelineRunner(
        repository,
        EventPublisher(repository),
        ContextFragmentTranscriber(),
        StaticPlanner(initial_map),
        ApprovingReviewer(),
        UnusedRenderer(),
    )

    saved = runner.run_phase_one(job.job_id)

    assert saved.stage == PipelineStage.WAITING_FOR_APPROVAL
    assert saved.narrative_map is not None
    assert [segment.segment_id for segment in saved.narrative_map.ordered_clips[0].segments] == [1, 2]
    assert "cut safety" in saved.narrative_map.ordered_clips[0].segments[0].reason


def test_pipeline_repairs_subjectless_opening_before_validation(tmp_path: Path) -> None:
    transcriptions = [
        ClipTranscription(
            clip_id=1,
            source_path="clip-1.mp4",
            duration=20.0,
            segments=[
                Segment(
                    id=1,
                    clip_id=1,
                    start=0.0,
                    end=7.0,
                    text="Viene eletto consigliere comunale a Piacenza nel 1980",
                    words=[WordToken(word="Viene", start=0.0, end=0.3), WordToken(word="1980", start=6.6, end=7.0)],
                ),
                Segment(
                    id=3,
                    clip_id=1,
                    start=13.0,
                    end=19.0,
                    text="area politica. Il direttore esecutivo e Italo Bocchino, quattro legislature in Parlamento",
                    words=[WordToken(word="area", start=13.0, end=13.3), WordToken(word="Parlamento", start=18.4, end=19.0)],
                ),
            ],
        ),
        ClipTranscription(
            clip_id=4,
            source_path="clip-4.mp4",
            duration=20.0,
            segments=[
                Segment(
                    id=3,
                    clip_id=4,
                    start=8.0,
                    end=14.0,
                    text="Tommaso Foti, l'attuale ministro per gli affari europei",
                    words=[WordToken(word="Tommaso", start=8.0, end=8.3), WordToken(word="europei", start=13.5, end=14.0)],
                )
            ],
        ),
    ]

    class SubjectlessOpeningTranscriber:
        def transcribe_many(self, paths: list[Path]) -> list[ClipTranscription]:
            return transcriptions

    initial_map = NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=1,
                playback_position=0,
                topic_summary="Carriera",
                role="introduction",
                segments=[
                    NarrativeSegment(
                        segment_id=1,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason="Apre con la carriera",
                    )
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Apre con la carriera.",
        estimated_duration_s=7.0,
    )
    settings = AppSettings(data_dir=tmp_path / "data", database_path=tmp_path / "data" / "jobs.sqlite3")
    repository = JobRepository(settings.database_path)
    job = JobRecord(job_id="subjectless-repair-job", brief=EditorialBrief(synthesis_level=2), input_clip_paths=["clip-1.mp4", "clip-4.mp4"])
    repository.save(job)
    runner = PipelineRunner(
        repository,
        EventPublisher(repository),
        SubjectlessOpeningTranscriber(),
        StaticPlanner(initial_map),
        ApprovingReviewer(),
        UnusedRenderer(),
    )

    saved = runner.run_phase_one(job.job_id)

    assert saved.stage == PipelineStage.WAITING_FOR_APPROVAL
    assert saved.narrative_map is not None
    assert saved.narrative_map.ordered_clips[0].clip_id == 4
    assert saved.narrative_map.ordered_clips[0].segments[0].segment_id == 3
    assert "cut safety" in saved.narrative_map.ordered_clips[0].segments[0].reason
    assert saved.edit_decision_list is not None
    assert saved.edit_decision_list.decisions[0].included_text == "Tommaso Foti, l'attuale ministro per gli affari europei"


def test_mock_narrative_planner_changes_output_for_drier_prompt() -> None:
    transcriptions = [
        ClipTranscription(
            clip_id=1,
            source_path="clip.mp4",
            duration=5.0,
            segments=[
                Segment(
                    id=1,
                    clip_id=1,
                    start=0.0,
                    end=1.0,
                    text="Tommaso Foti introduce il tema.",
                    words=[WordToken(word="Tommaso", start=0.0, end=0.3)],
                ),
                Segment(
                    id=2,
                    clip_id=1,
                    start=1.2,
                    end=2.0,
                    text="Ripete lo stesso concetto con un giro di parole.",
                    words=[WordToken(word="Ripete", start=1.2, end=1.5)],
                ),
            ],
        )
    ]
    planner = MockNarrativePlanner()

    neutral = planner.generate(transcriptions, EditorialBrief(user_prompt="", synthesis_level=2), None)
    dry = planner.generate(
        transcriptions,
        EditorialBrief(
            user_prompt="Fai un montaggio piu asciutto, evitando ripetizioni, esitazioni e giri di parole.",
            synthesis_level=2,
        ),
        None,
    )

    neutral_importances = [segment.importance for segment in neutral.ordered_clips[0].segments]
    dry_importances = [segment.importance for segment in dry.ordered_clips[0].segments]
    assert neutral_importances != dry_importances
    assert dry.ordered_clips[0].segments[1].importance == SegmentImportance.REDUNDANT
    assert "brief utente" in dry.editorial_strategy
