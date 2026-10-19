from __future__ import annotations

import inspect
from pathlib import Path

from loguru import logger

from autocut_ai.config import AppSettings
from autocut_ai.cuts import build_edit_decision_list
from autocut_ai.events import EventPublisher
from autocut_ai.ffmpeg import RenderError
from autocut_ai.housekeeping import cleanup_generated_data
from autocut_ai.narrative import (
    NarrativePlanningError,
    NarrativeValidationError,
    maximum_target_duration_s,
    repair_narrative_map_for_cut_safety,
    validate_narrative_map,
)
from autocut_ai.proposal_export import export_proposal_files
from autocut_ai.proposal_review import ProposalReviewError
from autocut_ai.schemas import (
    ClipTranscription,
    JobRecord,
    NarrativeMap,
    PipelineStage,
    ProposalIssue,
    ProposalRecalculateRequest,
    ProposalReview,
    ProposalTextPatchRequest,
    ProposalVariant,
)
from autocut_ai.storage import JobNotFoundError, JobRepository
from autocut_ai.timers import set_stage
from autocut_ai.transcription import TranscriptionError


class InvalidApprovalStateError(RuntimeError):
    pass


class InvalidProposalRecalculateStateError(RuntimeError):
    pass


class ProposalRecalculateError(RuntimeError):
    pass


class InvalidProposalTextPatchStateError(RuntimeError):
    pass


class ProposalTextPatchError(RuntimeError):
    pass


_MAX_PROPOSAL_REVIEW_PASSES = 3


def _narrative_map_review_signature(narrative_map: NarrativeMap) -> tuple[object, ...]:
    clips = sorted(narrative_map.ordered_clips, key=lambda clip: clip.playback_position)
    return tuple(
        (
            clip.clip_id,
            clip.playback_position,
            clip.role,
            tuple(
                (
                    segment.segment_id,
                    segment.importance.value,
                    segment.keep_verbatim,
                    segment.creative_zoom,
                    round(segment.creative_zoom_peak_scale, 4),
                )
                for segment in clip.segments
            ),
        )
        for clip in clips
    )


def _narrative_maps_equivalent_for_review(first: NarrativeMap, second: NarrativeMap) -> bool:
    return _narrative_map_review_signature(first) == _narrative_map_review_signature(second)


def _approved_review_after_repeated_correction(note: str, code: str) -> ProposalReview:
    return ProposalReview(
        approved=True,
        issues=[
            ProposalIssue(
                severity="warning",
                code=code,
                message=note,
            )
        ],
        corrected_narrative_map=None,
        editorial_notes=note,
    )


def _proposal_review_issue_codes(review: ProposalReview | None) -> str:
    if review is None:
        return "proposal_review_missing"
    return ", ".join(issue.code for issue in review.issues) or "proposal_review_rejected"


def _proposal_review_has_issue(review: ProposalReview | None, code: str) -> bool:
    if review is None:
        return False
    return any(issue.code == code for issue in review.issues)


def _is_excessive_synthesis_error(error: NarrativeValidationError) -> bool:
    return str(error).startswith("excessive_synthesis:")


def _excessive_synthesis_issue(error: NarrativeValidationError) -> ProposalIssue:
    return ProposalIssue(
        severity="warning",
        code="excessive_synthesis",
        message=str(error),
    )


def _merge_proposal_issues(*issue_groups: list[ProposalIssue] | None) -> list[ProposalIssue]:
    merged: list[ProposalIssue] = []
    seen: set[tuple[str, str]] = set()
    for issue_group in issue_groups:
        for issue in issue_group or []:
            key = (issue.code, issue.message)
            if key in seen:
                continue
            seen.add(key)
            merged.append(issue)
    return merged


def _validate_narrative_map_for_pipeline(
    narrative_map: NarrativeMap,
    transcriptions: list[ClipTranscription],
    editing_options: object,
    brief: object,
) -> list[ProposalIssue]:
    try:
        validate_narrative_map(narrative_map, transcriptions, editing_options, brief)
    except NarrativeValidationError as exc:
        if not _is_excessive_synthesis_error(exc):
            raise
        validate_narrative_map(
            narrative_map,
            transcriptions,
            editing_options,
            brief,
            enforce_duration_budget=False,
        )
        return [_excessive_synthesis_issue(exc)]
    return []


def _selected_video_duration_s(edit_decision_list: object) -> float:
    if not hasattr(edit_decision_list, "decisions"):
        return 0.0
    decisions = edit_decision_list.decisions
    duration_s = sum(max(0.0, decision.source_end - decision.source_start) for decision in decisions)
    if duration_s > 0:
        return duration_s
    return float(getattr(edit_decision_list, "estimated_duration_s", 0.0))


def _proposal_variant(
    *,
    variant_id: str,
    label: str,
    narrative_map: NarrativeMap,
    edit_decision_list: object,
    selectable: bool,
    issues: list[ProposalIssue] | None = None,
    rejection_reason: str | None = None,
) -> ProposalVariant:
    return ProposalVariant(
        id=variant_id,
        label=label,
        narrative_map=narrative_map,
        edit_decision_list=edit_decision_list,
        duration_s=_selected_video_duration_s(edit_decision_list),
        selectable=selectable,
        issues=issues or [],
        rejection_reason=rejection_reason,
    )


def _set_proposal_variant(job: JobRecord, variant: ProposalVariant) -> None:
    variants_by_id = {candidate.id: candidate for candidate in job.proposal_variants}
    variants_by_id[variant.id] = variant
    ordered_ids = ["initial", "review"]
    job.proposal_variants = [variants_by_id[variant_id] for variant_id in ordered_ids if variant_id in variants_by_id]


def _find_proposal_variant(job: JobRecord, variant_id: str) -> ProposalVariant | None:
    for variant in job.proposal_variants:
        if variant.id == variant_id:
            return variant
    return None


def _review_rejection_is_excessive_synthesis(review: ProposalReview | None) -> bool:
    if review is None:
        return False
    if "excessive_synthesis" in review.editorial_notes:
        return True
    return any("excessive_synthesis" in issue.message for issue in review.issues)


def _add_conservative_review_fallback_issue(review: ProposalReview) -> None:
    if not any(issue.code == "review_fallback_initial" for issue in review.issues):
        review.issues.append(
            ProposalIssue(
                severity="warning",
                code="review_fallback_initial",
                message=(
                    "La review LLM non ha prodotto una variante modificata renderizzabile; "
                    "viene esposta una review conservativa basata sulla proposta iniziale validata."
                ),
            )
        )
    fallback_note = (
        "Review LLM non modificata: fallback conservativo sulla proposta iniziale validata e renderizzabile."
    )
    if fallback_note not in review.editorial_notes:
        review.editorial_notes = f"{review.editorial_notes} {fallback_note}".strip()


class PipelineRunner:
    def __init__(
        self,
        repository: JobRepository,
        publisher: EventPublisher,
        transcriber: object,
        narrative_planner: object,
        proposal_reviewer: object,
        renderer: object,
        settings: AppSettings | None = None,
    ) -> None:
        self.repository = repository
        self.publisher = publisher
        self.transcriber = transcriber
        self.narrative_planner = narrative_planner
        self.proposal_reviewer = proposal_reviewer
        self.renderer = renderer
        self.settings = settings

    def _save_stage(self, job: JobRecord, stage: PipelineStage, message: str) -> JobRecord:
        updated = set_stage(job, stage, message)
        saved = self.repository.save(updated)
        self.publisher.publish_job(saved)
        logger.bind(job_id=job.job_id, stage=stage.value).info(message)
        return saved

    def _fail_job(self, job: JobRecord, message: str) -> JobRecord:
        job.error = message
        failed = self._save_stage(job, PipelineStage.FAILED, message)
        self._cleanup_job_data(failed)
        return failed

    def _cleanup_job_data(self, job: JobRecord) -> None:
        if self.settings is None:
            return
        final_output_path = Path(job.final_output_path) if job.final_output_path is not None else None
        cleanup_generated_data(self.settings, job.job_id, final_output_path)

    def _export_proposal(self, job: JobRecord) -> None:
        if self.settings is None:
            return
        export_proposal_files(job, self.settings.outputs_dir / job.job_id)

    def _narrative_analysis_message(self, job: JobRecord) -> str:
        clip_count = len(job.transcriptions)
        segment_count = sum(len(clip.segments) for clip in job.transcriptions)
        message = f"Analisi narrativa LLM ({clip_count} clip, {segment_count} segmenti)"
        if self.settings is not None:
            message += f", timeout {self.settings.ollama_request_timeout_s:g}s"
        return message

    def _transcribe_paths(self, job: JobRecord, paths: list[Path]) -> tuple[JobRecord, list[ClipTranscription]]:
        def progress_callback(clip_index: int, total: int, path: Path, phase: str) -> None:
            nonlocal job
            if phase == "completed":
                message = f"Trascrizione clip {clip_index}/{total} completata: {path.name}"
            else:
                phase_label = "preparazione audio" if phase == "preparing_audio" else "Whisper"
                message = f"Trascrizione clip {clip_index}/{total}: {path.name} ({phase_label})"
            job = self._save_stage(job, PipelineStage.TRANSCRIBING, message)

        transcribe_many = self.transcriber.transcribe_many
        parameters = inspect.signature(transcribe_many).parameters
        if "progress_callback" in parameters:
            return job, transcribe_many(paths, progress_callback=progress_callback)
        return job, transcribe_many(paths)

    def _validate_edit_decision_list_duration(self, job: JobRecord) -> None:
        if job.brief.target_duration_s is None or job.edit_decision_list is None:
            return
        total_spoken_duration_s = sum(
            max(0.0, segment.end - segment.start)
            for clip in job.transcriptions
            for segment in clip.segments
        )
        maximum_duration_s = maximum_target_duration_s(total_spoken_duration_s, job.brief)
        if maximum_duration_s is None:
            return
        edl_duration_s = job.edit_decision_list.estimated_duration_s
        if edl_duration_s > maximum_duration_s + 0.001:
            raise NarrativeValidationError(
                "target_duration_exceeded_after_cut_safety: "
                f"EditDecisionList duration_s={edl_duration_s:.1f}s exceeds target_duration_s="
                f"{job.brief.target_duration_s:.1f}s with maximum tolerated around {maximum_duration_s:.1f}s. "
                "La proposta e stata bloccata prima del Via libera per evitare un video finale troppo lungo; "
                "riduci il target o aumenta la sintesi."
            )

    def _generate_narrative_map(self, job: JobRecord) -> JobRecord:
        def progress_callback(message: str) -> None:
            nonlocal job
            job = self._save_stage(job, PipelineStage.NARRATIVE_ANALYSIS, message)

        generate = self.narrative_planner.generate
        parameters = inspect.signature(generate).parameters
        if "progress_callback" in parameters:
            job.narrative_map = generate(
                job.transcriptions,
                job.brief,
                job.llm_model,
                job.editing_options,
                progress_callback=progress_callback,
            )
        else:
            job.narrative_map = generate(
                job.transcriptions,
                job.brief,
                job.llm_model,
                job.editing_options,
            )
        return job

    def _review_current_proposal(
        self,
        job: JobRecord,
        initial_validation_issues: list[ProposalIssue] | None = None,
    ) -> JobRecord:
        def progress_callback(message: str) -> None:
            nonlocal job
            job = self._save_stage(job, PipelineStage.REFINING, message)

        if job.narrative_map is None:
            raise NarrativeValidationError("NarrativeMap mancante")
        initial_issues = list(initial_validation_issues or [])
        job.proposal_variants = []
        job.selected_proposal_variant_id = "initial"
        progress_callback("Costruzione proposta iniziale EDL")
        initial_narrative_map = job.narrative_map.model_copy(deep=True)
        initial_edit_decision_list = build_edit_decision_list(
            job.transcriptions,
            initial_narrative_map,
            job.editing_options,
        )
        job.edit_decision_list = initial_edit_decision_list
        self._validate_edit_decision_list_duration(job)
        _set_proposal_variant(
            job,
            _proposal_variant(
                variant_id="initial",
                label="Proposta iniziale",
                narrative_map=initial_narrative_map,
                edit_decision_list=initial_edit_decision_list,
                selectable=True,
                issues=initial_issues,
            ),
        )
        for review_pass in range(_MAX_PROPOSAL_REVIEW_PASSES):
            progress_callback(f"Review LLM proposta passaggio {review_pass + 1}/{_MAX_PROPOSAL_REVIEW_PASSES}")
            job.edit_decision_list = build_edit_decision_list(job.transcriptions, job.narrative_map, job.editing_options)
            self._validate_edit_decision_list_duration(job)
            review = self.proposal_reviewer.review
            parameters = inspect.signature(review).parameters
            if "progress_callback" in parameters:
                job.proposal_review = review(
                    job.transcriptions,
                    job.narrative_map,
                    job.edit_decision_list,
                    job.brief,
                    job.llm_model,
                    job.editing_options,
                    progress_callback=progress_callback,
                )
            else:
                job.proposal_review = review(
                    job.transcriptions,
                    job.narrative_map,
                    job.edit_decision_list,
                    job.brief,
                    job.llm_model,
                    job.editing_options,
                )
            job.proposal_review.issues = _merge_proposal_issues(initial_issues, job.proposal_review.issues)
            if job.proposal_review.corrected_narrative_map is not None:
                if _narrative_maps_equivalent_for_review(job.narrative_map, job.proposal_review.corrected_narrative_map):
                    _set_proposal_variant(
                        job,
                        _proposal_variant(
                            variant_id="review",
                            label="Review LLM",
                            narrative_map=job.narrative_map.model_copy(deep=True),
                            edit_decision_list=job.edit_decision_list,
                            selectable=True,
                            issues=job.proposal_review.issues,
                            rejection_reason=job.proposal_review.editorial_notes or None,
                        ),
                    )
                    if not _narrative_maps_equivalent_for_review(initial_narrative_map, job.narrative_map):
                        job.selected_proposal_variant_id = "review"
                    job.proposal_review = _approved_review_after_repeated_correction(
                        "La review LLM ha riproposto una correzione gia presente nella proposta; "
                        "la proposta corrente e stata considerata stabilizzata.",
                        "proposal_review_equivalent_correction",
                    )
                    return job
                if _proposal_review_has_issue(job.proposal_review, "invalid_patch_reference_ignored"):
                    progress_callback(
                        "Review LLM locale instabile: uso la review globale renderizzabile con avvisi"
                    )
                    job.narrative_map = job.proposal_review.corrected_narrative_map
                    repair_narrative_map_for_cut_safety(job.narrative_map, job.transcriptions)
                    review_validation_issues = _validate_narrative_map_for_pipeline(
                        job.narrative_map,
                        job.transcriptions,
                        job.editing_options,
                        job.brief,
                    )
                    job.proposal_review.issues = _merge_proposal_issues(
                        job.proposal_review.issues,
                        review_validation_issues,
                    )
                    job.edit_decision_list = build_edit_decision_list(
                        job.transcriptions,
                        job.narrative_map,
                        job.editing_options,
                    )
                    self._validate_edit_decision_list_duration(job)
                    _set_proposal_variant(
                        job,
                        _proposal_variant(
                            variant_id="review",
                            label="Review LLM",
                            narrative_map=job.narrative_map.model_copy(deep=True),
                            edit_decision_list=job.edit_decision_list,
                            selectable=True,
                            issues=job.proposal_review.issues,
                            rejection_reason=job.proposal_review.editorial_notes or None,
                        ),
                    )
                    job.selected_proposal_variant_id = "review"
                    return job
                if review_pass == _MAX_PROPOSAL_REVIEW_PASSES - 1:
                    job.narrative_map = job.proposal_review.corrected_narrative_map
                    repair_narrative_map_for_cut_safety(job.narrative_map, job.transcriptions)
                    review_validation_issues = _validate_narrative_map_for_pipeline(
                        job.narrative_map,
                        job.transcriptions,
                        job.editing_options,
                        job.brief,
                    )
                    job.proposal_review.issues = _merge_proposal_issues(
                        job.proposal_review.issues,
                        review_validation_issues,
                    )
                    job.edit_decision_list = build_edit_decision_list(
                        job.transcriptions,
                        job.narrative_map,
                        job.editing_options,
                    )
                    self._validate_edit_decision_list_duration(job)
                    _set_proposal_variant(
                        job,
                        _proposal_variant(
                            variant_id="review",
                            label="Review LLM",
                            narrative_map=job.narrative_map.model_copy(deep=True),
                            edit_decision_list=job.edit_decision_list,
                            selectable=True,
                            issues=job.proposal_review.issues,
                            rejection_reason=job.proposal_review.editorial_notes or None,
                        ),
                    )
                    job.selected_proposal_variant_id = "review"
                    job.proposal_review = _approved_review_after_repeated_correction(
                        "La review LLM ha raggiunto il limite di passaggi continuando a proporre correzioni "
                        "valide. L'ultima correzione validata e stata applicata e richiede controllo manuale "
                        "prima del Via libera.",
                        "proposal_review_convergence_limit",
                    )
                    return job
                job.narrative_map = job.proposal_review.corrected_narrative_map
                repair_narrative_map_for_cut_safety(job.narrative_map, job.transcriptions)
                review_validation_issues = _validate_narrative_map_for_pipeline(
                    job.narrative_map,
                    job.transcriptions,
                    job.editing_options,
                    job.brief,
                )
                if review_validation_issues:
                    job.proposal_review.issues = _merge_proposal_issues(
                        job.proposal_review.issues,
                        review_validation_issues,
                    )
                    job.edit_decision_list = build_edit_decision_list(
                        job.transcriptions,
                        job.narrative_map,
                        job.editing_options,
                    )
                    self._validate_edit_decision_list_duration(job)
                    _set_proposal_variant(
                        job,
                        _proposal_variant(
                            variant_id="review",
                            label="Review LLM",
                            narrative_map=job.narrative_map.model_copy(deep=True),
                            edit_decision_list=job.edit_decision_list,
                            selectable=True,
                            issues=job.proposal_review.issues,
                            rejection_reason=job.proposal_review.editorial_notes or None,
                        ),
                    )
                    job.selected_proposal_variant_id = "review"
                    return job
                continue
            if not job.proposal_review.approved:
                if (
                    job.proposal_review.rejected_narrative_map is not None
                    and _review_rejection_is_excessive_synthesis(job.proposal_review)
                ):
                    try:
                        review_map = job.proposal_review.rejected_narrative_map.model_copy(deep=True)
                        repair_narrative_map_for_cut_safety(review_map, job.transcriptions)
                        validate_narrative_map(
                            review_map,
                            job.transcriptions,
                            job.editing_options,
                            job.brief,
                            enforce_duration_budget=False,
                        )
                        review_edl = build_edit_decision_list(job.transcriptions, review_map, job.editing_options)
                    except (NarrativeValidationError, KeyError, ValueError) as exc:
                        job.proposal_review.issues.append(
                            ProposalIssue(
                                severity="warning",
                                code="review_variant_not_renderable",
                                message=str(exc),
                            )
                        )
                        job.proposal_review.editorial_notes = (
                            f"{job.proposal_review.editorial_notes} "
                            f"(review_variant_not_renderable: {exc})"
                        ).strip()
                    else:
                        _set_proposal_variant(
                            job,
                            _proposal_variant(
                                variant_id="review",
                                label="Review LLM",
                                narrative_map=review_map,
                                edit_decision_list=review_edl,
                                selectable=True,
                                issues=job.proposal_review.issues,
                                rejection_reason=job.proposal_review.editorial_notes or None,
                            ),
                        )
                review_variant = _find_proposal_variant(job, "review")
                if review_variant is None or not review_variant.selectable:
                    _add_conservative_review_fallback_issue(job.proposal_review)
                    _set_proposal_variant(
                        job,
                        _proposal_variant(
                            variant_id="review",
                            label="Review LLM",
                            narrative_map=job.narrative_map.model_copy(deep=True),
                            edit_decision_list=job.edit_decision_list,
                            selectable=True,
                            issues=job.proposal_review.issues,
                            rejection_reason=job.proposal_review.editorial_notes or None,
                        ),
                    )
                return job
            if not _narrative_maps_equivalent_for_review(initial_narrative_map, job.narrative_map):
                _set_proposal_variant(
                    job,
                    _proposal_variant(
                        variant_id="review",
                        label="Review LLM",
                        narrative_map=job.narrative_map.model_copy(deep=True),
                        edit_decision_list=job.edit_decision_list,
                        selectable=True,
                        issues=job.proposal_review.issues,
                        rejection_reason=job.proposal_review.editorial_notes or None,
                    ),
                )
                job.selected_proposal_variant_id = "review"
            else:
                _set_proposal_variant(
                    job,
                    _proposal_variant(
                        variant_id="review",
                        label="Review LLM",
                        narrative_map=job.narrative_map.model_copy(deep=True),
                        edit_decision_list=job.edit_decision_list,
                        selectable=True,
                        issues=job.proposal_review.issues,
                        rejection_reason=job.proposal_review.editorial_notes or None,
                    ),
                )
            return job
        raise ProposalReviewError("Proposal review did not reach a final approved proposal")

    def _waiting_for_approval_message(self, job: JobRecord, ready_message: str) -> str:
        if job.proposal_review is not None and not job.proposal_review.approved:
            issue_codes = _proposal_review_issue_codes(job.proposal_review)
            review_variant = _find_proposal_variant(job, "review")
            if (
                review_variant is not None
                and review_variant.selectable
                and _review_rejection_is_excessive_synthesis(job.proposal_review)
            ):
                return f"Proposta iniziale e review LLM pronte; review LLM selezionabile, piu aggressiva del budget ({issue_codes})"
            if review_variant is not None and review_variant.selectable:
                return f"Proposta iniziale e review LLM pronte; review LLM selezionabile con avvisi ({issue_codes})"
            if any(variant.selectable for variant in job.proposal_variants):
                return f"Proposta iniziale pronta; review LLM non renderizzabile ({issue_codes})"
            return f"Proposta da ricalcolare: review LLM non approvata ({issue_codes})"
        return ready_message

    def run_phase_one(self, job_id: str) -> JobRecord:
        """Costruisce la proposta e si arresta sempre prima del rendering."""

        job = self.repository.get(job_id)
        try:
            job = self._save_stage(job, PipelineStage.INTERPRETING_PROMPT, "Interpretazione del brief editoriale")
            job = self._save_stage(job, PipelineStage.NORMALIZING_CLIPS, "Normalizzazione tecnica delle clip")
            job = self._save_stage(job, PipelineStage.TRANSCRIBING, "Trascrizione con timestamp parola")
            paths = [Path(path) for path in job.input_clip_paths]
            job, transcriptions = self._transcribe_paths(job, paths)
            job.transcriptions = transcriptions
            job = self.repository.save(job)
            self.publisher.publish_job(job)

            job = self._save_stage(job, PipelineStage.NARRATIVE_ANALYSIS, self._narrative_analysis_message(job))
            job = self._generate_narrative_map(job)
            repair_narrative_map_for_cut_safety(job.narrative_map, job.transcriptions)
            validation_issues = _validate_narrative_map_for_pipeline(
                job.narrative_map,
                job.transcriptions,
                job.editing_options,
                job.brief,
            )
            job = self.repository.save(job)
            self.publisher.publish_job(job)

            job = self._save_stage(job, PipelineStage.REFINING, "Validazione cut safety e proposta EDL")
            job = self._review_current_proposal(job, validation_issues)
            job = self.repository.save(job)
            self.publisher.publish_job(job)

            job = self._save_stage(
                job,
                PipelineStage.WAITING_FOR_APPROVAL,
                self._waiting_for_approval_message(job, "Proposta pronta: attendo Via libera"),
            )
            # Questo è il confine obbligatorio tra intelligenza editoriale ed
            # esecuzione FFmpeg. Nessuna chiamata al renderer avviene in fase 1.
            self._cleanup_job_data(job)
            self._export_proposal(job)
            return job
        except (TranscriptionError, NarrativePlanningError, NarrativeValidationError, ProposalReviewError, KeyError, OSError, ValueError) as exc:
            return self._fail_job(job, str(exc))

    def recalculate_proposal(self, job_id: str, request: ProposalRecalculateRequest) -> JobRecord:
        job = self.repository.get(job_id)
        if job.stage != PipelineStage.WAITING_FOR_APPROVAL:
            raise InvalidProposalRecalculateStateError(
                f"Il job deve essere in waiting_for_approval, stato attuale: {job.stage.value}"
            )
        if not job.transcriptions or job.narrative_map is None:
            raise ProposalRecalculateError("Trascrizione o NarrativeMap mancanti: impossibile ricalcolare la proposta")
        backup = job.model_copy(deep=True)
        try:
            brief_updates: dict[str, object] = {}
            if request.synthesis_level is not None:
                brief_updates["synthesis_level"] = request.synthesis_level
            if "target_duration_s" in request.model_fields_set:
                brief_updates["target_duration_s"] = request.target_duration_s
            job.brief = job.brief.model_copy(update=brief_updates)
            job.error = None
            job = self._save_stage(job, PipelineStage.REFINING, "Ricalcolo proposta editoriale")

            def progress_callback(message: str) -> None:
                nonlocal job
                job = self._save_stage(job, PipelineStage.REFINING, message)

            recalculate = getattr(self.narrative_planner, "recalculate_proposal", None)
            if callable(recalculate):
                parameters = inspect.signature(recalculate).parameters
                if "progress_callback" in parameters:
                    job.narrative_map = recalculate(
                        job.narrative_map,
                        job.transcriptions,
                        job.brief,
                        job.llm_model,
                        job.editing_options,
                        progress_callback=progress_callback,
                    )
                else:
                    job.narrative_map = recalculate(
                        job.narrative_map,
                        job.transcriptions,
                        job.brief,
                        job.llm_model,
                        job.editing_options,
                    )
            else:
                job.narrative_map = job.narrative_map.model_copy(deep=True)
            repair_narrative_map_for_cut_safety(job.narrative_map, job.transcriptions)
            validation_issues = _validate_narrative_map_for_pipeline(
                job.narrative_map,
                job.transcriptions,
                job.editing_options,
                job.brief,
            )
            job = self._review_current_proposal(job, validation_issues)
            job = self.repository.save(job)
            self.publisher.publish_job(job)
            job = self._save_stage(
                job,
                PipelineStage.WAITING_FOR_APPROVAL,
                self._waiting_for_approval_message(job, "Proposta ricalcolata: attendo Via libera"),
            )
            self._cleanup_job_data(job)
            self._export_proposal(job)
            return job
        except (NarrativePlanningError, NarrativeValidationError, ProposalReviewError, KeyError, OSError, ValueError) as exc:
            backup.error = str(exc)
            backup.message = str(exc)
            restored = self.repository.save(backup)
            self.publisher.publish_job(restored)
            raise ProposalRecalculateError(str(exc)) from exc

    def mark_approved(self, job_id: str, proposal_variant_id: str | None = None) -> JobRecord:
        job = self.repository.get(job_id)
        if job.stage != PipelineStage.WAITING_FOR_APPROVAL:
            raise InvalidApprovalStateError(f"Il job deve essere in waiting_for_approval, stato attuale: {job.stage.value}")
        variant_id = proposal_variant_id or job.selected_proposal_variant_id or "initial"
        if not job.proposal_variants and job.narrative_map is not None and job.edit_decision_list is not None:
            _set_proposal_variant(
                job,
                _proposal_variant(
                    variant_id="initial",
                    label="Proposta iniziale",
                    narrative_map=job.narrative_map,
                    edit_decision_list=job.edit_decision_list,
                    selectable=True,
                ),
            )
        variant = _find_proposal_variant(job, variant_id)
        if variant is None:
            raise InvalidApprovalStateError(f"Variante proposta non trovata: {variant_id}")
        if not variant.selectable:
            raise InvalidApprovalStateError(f"Variante proposta non selezionabile: {variant_id}")
        # Copiamo la variante scelta nel job prima di passare a COMPOSING: il
        # renderer riceverà esattamente la proposta vista e approvata dall'utente.
        job.selected_proposal_variant_id = variant.id
        job.narrative_map = variant.narrative_map
        job.edit_decision_list = variant.edit_decision_list
        return self._save_stage(job, PipelineStage.COMPOSING, "Via libera ricevuto: composizione finale")

    def update_proposal_texts(self, job_id: str, request: ProposalTextPatchRequest) -> JobRecord:
        job = self.repository.get(job_id)
        if job.stage != PipelineStage.WAITING_FOR_APPROVAL:
            raise InvalidProposalTextPatchStateError(
                f"Il job deve essere in waiting_for_approval, stato attuale: {job.stage.value}"
            )
        if not job.proposal_variants and job.narrative_map is not None and job.edit_decision_list is not None:
            _set_proposal_variant(
                job,
                _proposal_variant(
                    variant_id="initial",
                    label="Proposta iniziale",
                    narrative_map=job.narrative_map,
                    edit_decision_list=job.edit_decision_list,
                    selectable=True,
                ),
            )
        variant = _find_proposal_variant(job, request.proposal_variant_id)
        if variant is None:
            raise InvalidProposalTextPatchStateError(f"Variante proposta non trovata: {request.proposal_variant_id}")
        if not variant.selectable:
            raise InvalidProposalTextPatchStateError(
                f"Variante proposta non selezionabile: {request.proposal_variant_id}"
            )

        updated_variant = variant.model_copy(deep=True)
        decisions_by_order = {
            decision.output_order: decision
            for decision in updated_variant.edit_decision_list.decisions
        }
        if len(decisions_by_order) != len(updated_variant.edit_decision_list.decisions):
            raise ProposalTextPatchError("La proposta contiene output_order duplicati")
        deleted_output_orders = set(request.deleted_output_orders)
        for output_order in deleted_output_orders:
            if output_order not in decisions_by_order:
                raise ProposalTextPatchError(f"output_order non trovato nella proposta: {output_order}")
        if deleted_output_orders and len(deleted_output_orders) >= len(updated_variant.edit_decision_list.decisions):
            raise ProposalTextPatchError("La cancellazione rimuoverebbe tutti i segmenti della proposta")
        for edit in request.edits:
            decision = decisions_by_order.get(edit.output_order)
            if decision is None:
                raise ProposalTextPatchError(f"output_order non trovato nella proposta: {edit.output_order}")
            decision.included_text = edit.included_text
        if deleted_output_orders:
            remaining_decisions = [
                decision
                for decision in sorted(updated_variant.edit_decision_list.decisions, key=lambda item: item.output_order)
                if decision.output_order not in deleted_output_orders
            ]
            for index, decision in enumerate(remaining_decisions):
                decision.output_order = index
                if index == 0:
                    decision.transition_before = None
            if len(remaining_decisions) == 1:
                remaining_decisions[0].transition_after = None
            updated_variant.edit_decision_list.decisions = remaining_decisions
            updated_variant.edit_decision_list.estimated_duration_s = sum(
                max(0.0, decision.source_end - decision.source_start)
                for decision in remaining_decisions
            )
            updated_variant.duration_s = updated_variant.edit_decision_list.estimated_duration_s

        _set_proposal_variant(job, updated_variant)
        job.selected_proposal_variant_id = updated_variant.id
        job.narrative_map = updated_variant.narrative_map
        job.edit_decision_list = updated_variant.edit_decision_list
        job.message = "Testi proposta aggiornati: attendo Via libera"
        saved = self.repository.save(job)
        self.publisher.publish_job(saved)
        logger.bind(job_id=job.job_id, variant=updated_variant.id).info("Updated proposal text edits")
        return saved

    def run_phase_two(self, job_id: str) -> JobRecord:
        job = self.repository.get(job_id)
        if job.stage != PipelineStage.COMPOSING:
            raise InvalidApprovalStateError(
                f"La composizione richiede un Via libera esplicito, stato attuale: {job.stage.value}"
            )
        try:
            if job.edit_decision_list is None:
                raise RenderError("EditDecisionList mancante")
            input_paths = [Path(path) for path in job.input_clip_paths]
            output_path = self.renderer.render(job.job_id, input_paths, job.edit_decision_list, job.editing_options)
            job.final_output_path = str(output_path)
            job = self.repository.save(job)
            self.publisher.publish_job(job)
            if job.editing_options.add_background_music:
                job = self._save_stage(job, PipelineStage.MIXING_MUSIC, "Mix musica e ducking")
            job = self._save_stage(job, PipelineStage.DONE, "Video finale pronto")
            self._cleanup_job_data(job)
            return job
        except (RenderError, JobNotFoundError, OSError, ValueError) as exc:
            return self._fail_job(job, str(exc))
