from __future__ import annotations

import json
import math
from pathlib import Path

from autocut_ai.schemas import EditDecision, EditDecisionList, JobRecord, NarrativeClip, ProposalIssue, ProposalVariant


PROPOSAL_TEXT_FILENAME = "proposal.txt"
PROPOSAL_JSON_FILENAME = "proposal.json"
PROPOSAL_EXPORT_FILENAMES = frozenset({PROPOSAL_TEXT_FILENAME, PROPOSAL_JSON_FILENAME})


def export_proposal_files(job: JobRecord, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = output_dir / PROPOSAL_TEXT_FILENAME
    json_path = output_dir / PROPOSAL_JSON_FILENAME

    txt_path.write_text(_render_proposal_text(job), encoding="utf-8")
    json_path.write_text(
        json.dumps(_proposal_payload(job), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return txt_path, json_path


def _proposal_payload(job: JobRecord) -> dict[str, object]:
    return {
        "job_id": job.job_id,
        "stage": job.stage.value,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "elapsed_seconds": job.elapsed_seconds,
        "current_stage_elapsed_seconds": job.current_stage_elapsed_seconds,
        "llm_model": job.llm_model,
        "brief": job.brief.model_dump(mode="json"),
        "editing_options": job.editing_options.model_dump(mode="json"),
        "narrative_map": job.narrative_map.model_dump(mode="json") if job.narrative_map is not None else None,
        "edit_decision_list": job.edit_decision_list.model_dump(mode="json") if job.edit_decision_list is not None else None,
        "proposal_review": job.proposal_review.model_dump(mode="json") if job.proposal_review is not None else None,
        "proposal_variants": [variant.model_dump(mode="json") for variant in job.proposal_variants],
        "selected_proposal_variant_id": job.selected_proposal_variant_id,
    }


def _selected_video_duration_s(edit_decision_list: EditDecisionList) -> float:
    decision_duration_s = sum(
        max(0.0, decision.source_end - decision.source_start)
        for decision in edit_decision_list.decisions
    )
    if decision_duration_s > 0:
        return decision_duration_s
    return edit_decision_list.estimated_duration_s


def _format_duration_minutes(duration_s: float) -> str:
    rounded_minutes = math.floor((duration_s / 60 * 100) + 0.5) / 100
    return f"{rounded_minutes:.2f} min ({duration_s:.1f}s)"


def _render_proposal_text(job: JobRecord) -> str:
    lines: list[str] = [
        "Proposta editoriale",
        f"Job: {job.job_id}",
        f"Stato: {job.stage.value}",
        f"Modello LLM: {job.llm_model or 'non specificato'}",
        f"Livello sintesi: {job.brief.synthesis_level}",
    ]
    if job.brief.user_prompt:
        lines.extend(["", "Brief utente", job.brief.user_prompt])

    if job.narrative_map is None:
        lines.extend(["", "NarrativeMap non disponibile."])
    else:
        lines.extend(
            [
                "",
                "Strategia editoriale",
                job.narrative_map.editorial_strategy,
                f"Durata stimata LLM: {job.narrative_map.estimated_duration_s:.1f}s",
            ]
        )
        if job.narrative_map.synthesis_script:
            lines.extend(["", "Testo sintetizzato", job.narrative_map.synthesis_script])
        lines.extend(["", "Ordine narrativo"])
        for clip in sorted(job.narrative_map.ordered_clips, key=lambda item: item.playback_position):
            lines.extend(_render_clip(clip))

    if job.edit_decision_list is not None:
        selected_duration_s = _selected_video_duration_s(job.edit_decision_list)
        lines.extend(
            [
                "",
                f"Lunghezza stimata video: {_format_duration_minutes(selected_duration_s)}",
                f"Durata stimata EDL: {job.edit_decision_list.estimated_duration_s:.1f}s",
                "",
                "Edit Decision List",
            ]
        )
        for decision in sorted(job.edit_decision_list.decisions, key=lambda item: item.output_order):
            lines.extend(_render_decision(decision))

    if job.proposal_variants:
        lines.extend(["", "Varianti proposta"])
        for variant in job.proposal_variants:
            lines.extend(_render_proposal_variant(variant, job.selected_proposal_variant_id))

    if job.proposal_review is not None:
        lines.extend(["", "Revisione proposta", job.proposal_review.editorial_notes or "Nessuna nota."])
        for issue in job.proposal_review.issues:
            lines.extend(_render_issue(issue))

    return "\n".join(lines).strip() + "\n"


def _render_clip(clip: NarrativeClip) -> list[str]:
    lines = [
        f"Clip {clip.clip_id} - posizione {clip.playback_position}",
        f"Ruolo: {clip.role}",
        f"Topic: {clip.topic_summary}",
    ]
    for segment in clip.segments:
        action = "mantieni" if segment.keep_verbatim else "taglia/sintetizza"
        lines.append(
            f"  Segmento {segment.segment_id}: {segment.importance.value}, {action}. Motivo: {segment.reason}"
        )
        if segment.synthesis_text:
            lines.append(f"  Sintesi: {segment.synthesis_text}")
    return lines


def _render_decision(decision: EditDecision) -> list[str]:
    lines = [
        f"Clip {decision.clip_id}",
        f"{decision.source_start:.2f}s - {decision.source_end:.2f}s",
        decision.included_text,
        f"Motivo: {decision.reason}",
    ]
    if decision.transition_before or decision.transition_after:
        lines.append(
            f"Transizioni: before={decision.transition_before or '-'}, after={decision.transition_after or '-'}"
        )
    if decision.safety_notes:
        lines.append("Note safety: " + " | ".join(decision.safety_notes))
    return lines


def _render_proposal_variant(variant: ProposalVariant, selected_variant_id: str) -> list[str]:
    selected_label = " (selezionata)" if variant.id == selected_variant_id else ""
    status = "selezionabile" if variant.selectable else "non renderizzabile"
    lines = [
        f"{variant.label}{selected_label}",
        f"ID: {variant.id}",
        f"Stato: {status}",
        f"Durata stimata: {_format_duration_minutes(variant.duration_s)}",
    ]
    if variant.rejection_reason:
        lines.append(f"Motivo review: {variant.rejection_reason}")
    for issue in variant.issues:
        lines.extend(_render_issue(issue))
    lines.append("Preview EDL variante")
    for decision in sorted(variant.edit_decision_list.decisions, key=lambda item: item.output_order):
        lines.extend(_render_decision(decision))
    return lines


def _render_issue(issue: ProposalIssue) -> list[str]:
    return [
        f"Issue {issue.severity}: {issue.code}",
        issue.message,
    ]
