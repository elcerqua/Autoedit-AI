from __future__ import annotations

import json
from pathlib import Path

from autocut_ai.proposal_export import export_proposal_files
from autocut_ai.schemas import (
    EditDecision,
    EditDecisionList,
    EditorialBrief,
    JobRecord,
    NarrativeClip,
    NarrativeMap,
    NarrativeSegment,
    PipelineStage,
    ProposalVariant,
    ProposalReview,
    SegmentImportance,
)


def _job_with_proposal() -> JobRecord:
    return JobRecord(
        job_id="proposal-job",
        stage=PipelineStage.WAITING_FOR_APPROVAL,
        brief=EditorialBrief(user_prompt="Fai un montaggio asciutto.", synthesis_level=3),
        llm_model="qwen3:8b",
        narrative_map=NarrativeMap(
            ordered_clips=[
                NarrativeClip(
                    clip_id=1,
                    playback_position=0,
                    topic_summary="Apertura",
                    role="introduction",
                    segments=[
                        NarrativeSegment(
                            segment_id=1,
                            importance=SegmentImportance.ESSENTIAL,
                            keep_verbatim=True,
                            reason="Apre con il soggetto.",
                        )
                    ],
                )
            ],
            filler_words=[],
            editorial_strategy="Apertura chiara, ritmo asciutto.",
            estimated_duration_s=8.0,
            synthesis_script="Testo sintetizzato finale.",
        ),
        edit_decision_list=EditDecisionList(
            estimated_duration_s=7.5,
            decisions=[
                EditDecision(
                    clip_id=1,
                    source_start=1.0,
                    source_end=8.5,
                    output_order=0,
                    transition_before="cut",
                    transition_after="cross_dissolve",
                    reason="Segmento essenziale.",
                    included_text="Tommaso Foti introduce il tema.",
                    safety_notes=["boundary word-safe"],
                )
            ],
        ),
        proposal_review=ProposalReview(
            approved=True,
            issues=[],
            editorial_notes="Proposta validata.",
        ),
    )


def test_export_proposal_files_writes_human_txt_and_machine_json(tmp_path: Path) -> None:
    txt_path, json_path = export_proposal_files(_job_with_proposal(), tmp_path)

    assert txt_path == tmp_path / "proposal.txt"
    assert json_path == tmp_path / "proposal.json"
    txt = txt_path.read_text(encoding="utf-8")
    assert "Proposta editoriale" in txt
    assert "Apertura chiara, ritmo asciutto." in txt
    assert "Lunghezza stimata video: 0.13 min (7.5s)" in txt
    assert "Clip 1" in txt
    assert "Tommaso Foti introduce il tema." in txt
    assert "boundary word-safe" in txt

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["job_id"] == "proposal-job"
    assert payload["brief"]["synthesis_level"] == 3
    assert payload["narrative_map"]["editorial_strategy"] == "Apertura chiara, ritmo asciutto."
    assert payload["edit_decision_list"]["decisions"][0]["included_text"] == "Tommaso Foti introduce il tema."


def test_export_proposal_files_includes_selectable_variants(tmp_path: Path) -> None:
    job = _job_with_proposal()
    assert job.narrative_map is not None
    assert job.edit_decision_list is not None
    review_edl = EditDecisionList(
        estimated_duration_s=4.0,
        decisions=[
            EditDecision(
                clip_id=1,
                source_start=1.0,
                source_end=5.0,
                output_order=0,
                reason="Review piu asciutta.",
                included_text="Tommaso Foti introduce.",
            )
        ],
    )
    job.proposal_variants = [
        ProposalVariant(
            id="initial",
            label="Proposta iniziale",
            narrative_map=job.narrative_map,
            edit_decision_list=job.edit_decision_list,
            duration_s=7.5,
            selectable=True,
        ),
        ProposalVariant(
            id="review",
            label="Review LLM",
            narrative_map=job.narrative_map.model_copy(update={"editorial_strategy": "Review piu aggressiva."}),
            edit_decision_list=review_edl,
            duration_s=4.0,
            selectable=True,
            rejection_reason="excessive_synthesis: troppo aggressiva",
        ),
    ]

    txt_path, json_path = export_proposal_files(job, tmp_path)

    txt = txt_path.read_text(encoding="utf-8")
    assert "Varianti proposta" in txt
    assert "Proposta iniziale" in txt
    assert "Review LLM" in txt
    assert "excessive_synthesis" in txt
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert [variant["id"] for variant in payload["proposal_variants"]] == ["initial", "review"]
    assert payload["selected_proposal_variant_id"] == "initial"
