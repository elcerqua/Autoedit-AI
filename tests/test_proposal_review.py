from __future__ import annotations

import sys
from types import SimpleNamespace

from autocut_ai.config import AppSettings
from autocut_ai.proposal_review import (
    MockProposalReviewer,
    ProposalReviewer,
    build_global_proposal_review_prompt,
    build_proposal_review_prompt,
)
from autocut_ai.schemas import (
    ClipTranscription,
    EditDecision,
    EditDecisionList,
    EditingOptions,
    EditorialBrief,
    NarrativeClip,
    NarrativeMap,
    NarrativeSegment,
    Segment,
    SegmentImportance,
    WordToken,
)


def _transcription() -> ClipTranscription:
    return ClipTranscription(
        clip_id=1,
        source_path=None,
        duration=3.0,
        segments=[
            Segment(
                id=1,
                clip_id=1,
                start=0.0,
                end=1.0,
                text="Tommaso Foti introduce il tema.",
                words=[WordToken(word="Tommaso", start=0.0, end=0.3)],
            )
        ],
    )


def _narrative_map() -> NarrativeMap:
    return NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=1,
                playback_position=0,
                topic_summary="Introduzione",
                role="introduction",
                segments=[
                    NarrativeSegment(
                        segment_id=1,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason="Apre con soggetto esplicito",
                    )
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Apri con il soggetto.",
        estimated_duration_s=1.0,
    )


def _edl() -> EditDecisionList:
    return EditDecisionList(
        decisions=[
            EditDecision(
                clip_id=1,
                source_start=0.0,
                source_end=1.0,
                output_order=0,
                reason="Apre con soggetto esplicito",
                included_text="Tommaso Foti introduce il tema.",
            )
        ],
        estimated_duration_s=1.0,
    )


def _many_transcription(count: int) -> ClipTranscription:
    return ClipTranscription(
        clip_id=1,
        source_path=None,
        duration=float(count),
        segments=[
            Segment(
                id=index,
                clip_id=1,
                start=float(index - 1),
                end=float(index),
                text=f"Segmento informativo molto dettagliato numero {index} con contenuto da validare.",
                words=[WordToken(word="Segmento", start=float(index - 1), end=float(index - 1) + 0.3)],
            )
            for index in range(1, count + 1)
        ],
    )


def _many_narrative_map(count: int) -> NarrativeMap:
    return NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=1,
                playback_position=0,
                topic_summary="Sequenza lunga",
                role="development",
                segments=[
                    NarrativeSegment(
                        segment_id=index,
                        importance=SegmentImportance.VALUABLE,
                        keep_verbatim=True,
                        reason="Mantieni nel montaggio lungo.",
                    )
                    for index in range(1, count + 1)
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Review di proposta lunga.",
        estimated_duration_s=float(count),
    )


def _many_edl(count: int) -> EditDecisionList:
    return EditDecisionList(
        decisions=[
            EditDecision(
                clip_id=1,
                source_start=float(index - 1),
                source_end=float(index),
                output_order=index - 1,
                reason="Mantieni nel montaggio lungo.",
                included_text=f"Segmento informativo molto dettagliato numero {index} con contenuto da validare.",
            )
            for index in range(1, count + 1)
        ],
        estimated_duration_s=float(count),
    )


def _long_transcription_for_review_patch_validation() -> ClipTranscription:
    segment_duration = 14.09
    return ClipTranscription(
        clip_id=1,
        source_path=None,
        duration=segment_duration * 10,
        segments=[
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
        ],
    )


def _long_narrative_map_for_review_patch_validation() -> NarrativeMap:
    return NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=1,
                playback_position=0,
                topic_summary="Sequenza completa",
                role="development",
                segments=[
                    NarrativeSegment(
                        segment_id=index,
                        importance=SegmentImportance.VALUABLE,
                        keep_verbatim=True,
                        reason="Mantieni il contesto completo.",
                    )
                    for index in range(1, 11)
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Mantieni una proposta completa.",
        estimated_duration_s=140.9,
    )


def _long_edl_for_review_patch_validation() -> EditDecisionList:
    transcription = _long_transcription_for_review_patch_validation()
    return EditDecisionList(
        decisions=[
            EditDecision(
                clip_id=1,
                source_start=segment.start,
                source_end=segment.end,
                output_order=index,
                reason="Mantieni il contesto completo.",
                included_text=segment.text,
            )
            for index, segment in enumerate(transcription.segments)
        ],
        estimated_duration_s=140.9,
    )


def _two_segment_transcription() -> ClipTranscription:
    return ClipTranscription(
        clip_id=1,
        source_path=None,
        duration=4.0,
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
                text="Ripete lo stesso concetto con parole diverse.",
                words=[WordToken(word="Ripete", start=1.2, end=1.5)],
            ),
        ],
    )


def _two_segment_narrative_map() -> NarrativeMap:
    return NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=1,
                playback_position=0,
                topic_summary="Tema",
                role="introduction",
                segments=[
                    NarrativeSegment(
                        segment_id=1,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason="Apre il tema.",
                    ),
                    NarrativeSegment(
                        segment_id=2,
                        importance=SegmentImportance.VALUABLE,
                        keep_verbatim=True,
                        reason="Riprende il tema.",
                    ),
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Apertura con ripetizione da valutare.",
        estimated_duration_s=2.0,
    )


def _two_segment_edl() -> EditDecisionList:
    return EditDecisionList(
        decisions=[
            EditDecision(
                clip_id=1,
                source_start=0.0,
                source_end=1.0,
                output_order=0,
                reason="Apre il tema.",
                included_text="Tommaso Foti introduce il tema.",
            ),
            EditDecision(
                clip_id=1,
                source_start=1.2,
                source_end=2.0,
                output_order=1,
                reason="Riprende il tema.",
                included_text="Ripete lo stesso concetto con parole diverse.",
            ),
        ],
        estimated_duration_s=1.8,
    )


def _micro_repetition_transcription() -> ClipTranscription:
    return ClipTranscription(
        clip_id=3,
        source_path=None,
        duration=3.0,
        segments=[
            Segment(
                id=1,
                clip_id=3,
                start=0.0,
                end=2.0,
                text="Nel 24 ha perso, ha rimesso, ha una perdita dell'esercizio.",
                words=[
                    WordToken(word="Nel", start=0.0, end=0.10),
                    WordToken(word="24", start=0.12, end=0.25),
                    WordToken(word="ha", start=0.30, end=0.38),
                    WordToken(word="perso,", start=0.40, end=0.65),
                    WordToken(word="ha", start=0.70, end=0.78),
                    WordToken(word="rimesso,", start=0.80, end=1.10),
                    WordToken(word="ha", start=1.20, end=1.28),
                    WordToken(word="una", start=1.30, end=1.38),
                    WordToken(word="perdita", start=1.40, end=1.65),
                    WordToken(word="dell'esercizio.", start=1.68, end=2.00),
                ],
            )
        ],
    )


def _micro_repetition_map() -> NarrativeMap:
    return NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=3,
                playback_position=0,
                topic_summary="Dati bilancio",
                role="development",
                segments=[
                    NarrativeSegment(
                        segment_id=1,
                        importance=SegmentImportance.VALUABLE,
                        keep_verbatim=True,
                        reason="Mantieni dato di bilancio.",
                    )
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Mantieni il dato e lima autoriparazioni.",
        estimated_duration_s=2.0,
    )


def _micro_repetition_edl() -> EditDecisionList:
    return EditDecisionList(
        decisions=[
            EditDecision(
                clip_id=3,
                source_start=0.0,
                source_end=2.0,
                output_order=0,
                reason="Mantieni dato di bilancio.",
                included_text="Nel 24 ha perso, ha rimesso, ha una perdita dell'esercizio.",
            )
        ],
        estimated_duration_s=2.0,
    )


def _narrative_flow_transcriptions() -> list[ClipTranscription]:
    return [
        ClipTranscription(
            clip_id=2,
            source_path=None,
            duration=4.0,
            segments=[
                Segment(
                    id=1,
                    clip_id=2,
                    start=0.0,
                    end=2.0,
                    text="La stessa norma poi ci dice che possono prendere i contributi pubblici.",
                    words=[WordToken(word="La", start=0.0, end=0.1), WordToken(word="pubblici.", start=1.7, end=2.0)],
                )
            ],
        ),
        ClipTranscription(
            clip_id=3,
            source_path=None,
            duration=4.0,
            segments=[
                Segment(
                    id=1,
                    clip_id=3,
                    start=0.0,
                    end=2.0,
                    text="Solo che la legge del 2016, la 198, ridefinisce i beneficiari.",
                    words=[WordToken(word="Solo", start=0.0, end=0.2), WordToken(word="beneficiari.", start=1.6, end=2.0)],
                )
            ],
        ),
    ]


def _narrative_flow_map_wrong_order() -> NarrativeMap:
    return NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=2,
                playback_position=0,
                topic_summary="Riferimento alla stessa norma",
                role="development",
                segments=[
                    NarrativeSegment(
                        segment_id=1,
                        importance=SegmentImportance.VALUABLE,
                        keep_verbatim=True,
                        reason="Spiega cosa consente la norma.",
                    )
                ],
            ),
            NarrativeClip(
                clip_id=3,
                playback_position=1,
                topic_summary="Introduzione della legge 2016 198",
                role="introduction",
                segments=[
                    NarrativeSegment(
                        segment_id=1,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason="Introduce la norma citata dopo.",
                    )
                ],
            ),
        ],
        filler_words=[],
        editorial_strategy="Ordine iniziale da verificare.",
        estimated_duration_s=4.0,
    )


def _narrative_flow_edl_wrong_order() -> EditDecisionList:
    return EditDecisionList(
        decisions=[
            EditDecision(
                clip_id=2,
                source_start=0.0,
                source_end=2.0,
                output_order=0,
                reason="Spiega cosa consente la norma.",
                included_text="La stessa norma poi ci dice che possono prendere i contributi pubblici.",
            ),
            EditDecision(
                clip_id=3,
                source_start=0.0,
                source_end=2.0,
                output_order=1,
                reason="Introduce la norma citata dopo.",
                included_text="Solo che la legge del 2016, la 198, ridefinisce i beneficiari.",
            ),
        ],
        estimated_duration_s=4.0,
    )


def _subject_context_transcription() -> ClipTranscription:
    return ClipTranscription(
        clip_id=4,
        source_path=None,
        duration=5.0,
        segments=[
            Segment(
                id=1,
                clip_id=4,
                start=0.0,
                end=1.5,
                text="Tommaso Foti, l'attuale ministro per gli affari europei.",
                words=[WordToken(word="Tommaso", start=0.0, end=0.3), WordToken(word="europei.", start=1.2, end=1.5)],
            ),
            Segment(
                id=2,
                clip_id=4,
                start=1.6,
                end=3.0,
                text="ma risulta come fiduciario della Fondazione Alleanza Nazionale.",
                words=[WordToken(word="ma", start=1.6, end=1.7), WordToken(word="Nazionale.", start=2.7, end=3.0)],
            ),
        ],
    )


def _subject_context_map_missing_subject() -> NarrativeMap:
    return NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=4,
                playback_position=0,
                topic_summary="Fiduciario senza soggetto esplicito",
                role="development",
                segments=[
                    NarrativeSegment(
                        segment_id=2,
                        importance=SegmentImportance.VALUABLE,
                        keep_verbatim=True,
                        reason="Spiega il ruolo fiduciario.",
                    )
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Manca il soggetto esplicito.",
        estimated_duration_s=1.4,
    )


def _subject_context_edl_missing_subject() -> EditDecisionList:
    return EditDecisionList(
        decisions=[
            EditDecision(
                clip_id=4,
                source_start=1.6,
                source_end=3.0,
                output_order=0,
                reason="Spiega il ruolo fiduciario.",
                included_text="ma risulta come fiduciario della Fondazione Alleanza Nazionale.",
            )
        ],
        estimated_duration_s=1.4,
    )


def test_mock_proposal_reviewer_approves_without_correction() -> None:
    review = MockProposalReviewer().review([_transcription()], _narrative_map(), _edl(), EditorialBrief(), "qwen3:8b")

    assert review.approved is True
    assert review.corrected_narrative_map is None
    assert review.issues == []


def test_proposal_review_prompt_includes_actual_edl_text() -> None:
    prompt = build_proposal_review_prompt([_transcription()], _narrative_map(), _edl(), EditorialBrief(synthesis_level=3))

    assert "Tommaso Foti introduce il tema." in prompt
    assert "EditDecisionList" in prompt
    assert "synthesis_level=3" in prompt


def test_proposal_review_prompt_preserves_creative_zoom_requirement() -> None:
    prompt = build_proposal_review_prompt(
        [_transcription()],
        _narrative_map(),
        _edl(),
        EditorialBrief(synthesis_level=3),
        editing_options=EditingOptions(enable_creative_zoom=True),
    )

    assert "pause" in prompt
    assert "zoom fisso" in prompt
    assert "corrected_narrative_map" in prompt


def test_proposal_review_prompt_checks_user_prompt_compliance() -> None:
    prompt = build_proposal_review_prompt(
        [_transcription()],
        _narrative_map(),
        _edl(),
        EditorialBrief(
            user_prompt=(
                "Fai un montaggio piu asciutto, evitando ripetizioni, esitazioni e giri di parole."
            ),
            synthesis_level=2,
        ),
    )

    assert "brief utente" in prompt.casefold()
    assert "vincolante" in prompt
    assert "piu asciutto" in prompt
    assert "corrected_narrative_map" in prompt


def test_global_proposal_review_prompt_uses_compact_blocks_not_full_narrative_map() -> None:
    prompt = build_global_proposal_review_prompt(
        _narrative_flow_transcriptions(),
        _narrative_flow_map_wrong_order(),
        _narrative_flow_edl_wrong_order(),
        EditorialBrief(user_prompt="Correggi la logica narrativa.", synthesis_level=2),
    )

    assert "Review globale a blocchi" in prompt
    assert "block_id=0" in prompt
    assert "block_id=1" in prompt
    assert "La stessa norma" in prompt
    assert "legge del 2016" in prompt
    assert '"ordered_clips"' not in prompt
    assert "GlobalProposalReviewPlan" in prompt


def test_large_proposal_review_prompt_uses_compact_context() -> None:
    prompt = build_proposal_review_prompt(
        [_many_transcription(80)],
        _many_narrative_map(80),
        _many_edl(80),
        EditorialBrief(synthesis_level=2),
    )

    assert "Trascrizione sorgente compatta" in prompt
    assert "NarrativeMap corrente compatta" in prompt
    assert '"ordered_clips"' not in prompt
    assert len(prompt) < 18000


def test_large_proposal_review_prompt_still_requires_llm_coherence_correction() -> None:
    prompt = build_proposal_review_prompt(
        [_many_transcription(80)],
        _many_narrative_map(80),
        _many_edl(80),
        EditorialBrief(
            user_prompt="Fai un montaggio piu asciutto, evitando ripetizioni.",
            synthesis_level=4,
        ),
    )

    folded = prompt.casefold()
    assert "coerenza temporale" in folded
    assert "grammaticale" in folded
    assert "sintattica" in folded
    assert "spezzettare" in folded
    assert "limare" in folded
    assert "corrected_narrative_map=null" not in prompt


def test_very_large_proposal_review_prompt_stays_small_enough_for_fast_review() -> None:
    prompt = build_proposal_review_prompt(
        [_many_transcription(500)],
        _many_narrative_map(500),
        _many_edl(500),
        EditorialBrief(synthesis_level=2, target_duration_s=420.0),
    )

    assert "decision_count=500" in prompt
    assert "omesse" in prompt
    assert len(prompt) < 10000


def test_large_proposal_under_ten_minutes_runs_llm_review_in_chunks(
    tmp_path,
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []

    class ReviewClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def chat(self, **kwargs: object) -> dict[str, dict[str, str]]:
            calls.append(kwargs)
            if kwargs["format"]["title"] == "GlobalProposalReviewPlan":
                return {
                    "message": {
                        "content": (
                            '{"ordered_block_ids":[],"dropped_block_ids":[],"restored_segments":[],'
                            '"issues":[],"editorial_notes":"Review globale eseguita."}'
                        )
                    }
                }
            return {
                "message": {
                    "content": (
                        '{"approved":true,"issues":[],"segment_adjustments":[],"word_removals":[],'
                        '"clip_order":[],"editorial_notes":"Chunk locale validato."}'
                    )
                }
            }

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=ReviewClient))

    review = ProposalReviewer(
        AppSettings(
            data_dir=tmp_path / "data",
            database_path=tmp_path / "data" / "jobs.sqlite3",
            ollama_run_in_subprocess=False,
        ),
        max_retries=0,
    ).review(
        [_many_transcription(80)],
        _many_narrative_map(80),
        _many_edl(80),
        EditorialBrief(synthesis_level=3, target_duration_s=600.0),
        "qwen3:8b",
    )

    assert [call["format"]["title"] for call in calls] == [
        "GlobalProposalReviewPlan",
        "ProposalReviewPatch",
        "ProposalReviewPatch",
        "ProposalReviewPatch",
        "ProposalReviewPatch",
        "ProposalReviewPatch",
    ]
    assert review.approved is True
    assert review.issues == []


def test_large_proposal_over_ten_minutes_skips_final_llm_review_to_avoid_stalls(
    tmp_path,
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []

    class ReviewClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def chat(self, **kwargs: object) -> dict[str, dict[str, str]]:
            calls.append(kwargs)
            review = MockProposalReviewer().review(
                [_transcription()],
                _narrative_map(),
                _edl(),
                EditorialBrief(),
                "qwen3:8b",
            )
            return {"message": {"content": review.model_dump_json()}}

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=ReviewClient))

    review = ProposalReviewer(
        AppSettings(
            data_dir=tmp_path / "data",
            database_path=tmp_path / "data" / "jobs.sqlite3",
            ollama_run_in_subprocess=False,
        ),
        max_retries=0,
    ).review(
        [_many_transcription(601)],
        _many_narrative_map(601),
        _many_edl(601),
        EditorialBrief(synthesis_level=3, target_duration_s=900.0),
        "qwen3:8b",
    )

    assert calls == []
    assert review.approved is True
    assert review.corrected_narrative_map is None
    assert [issue.code for issue in review.issues] == ["proposal_review_skipped_large_proposal"]


def test_proposal_reviewer_uses_lightweight_patch_schema(
    tmp_path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class PatchClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def chat(self, **kwargs: object) -> dict[str, dict[str, str]]:
            captured["format"] = kwargs["format"]
            captured["prompt"] = kwargs["messages"][0]["content"]
            return {
                "message": {
                    "content": (
                        '{"approved":true,"issues":[],"segment_adjustments":[],'
                        '"clip_order":[],"editorial_notes":"Proposta coerente."}'
                    )
                }
            }

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=PatchClient))

    review = ProposalReviewer(
        AppSettings(
            data_dir=tmp_path / "data",
            database_path=tmp_path / "data" / "jobs.sqlite3",
            ollama_run_in_subprocess=False,
        ),
        max_retries=0,
    ).review([_transcription()], _narrative_map(), _edl(), EditorialBrief(), "qwen3:8b")

    assert review.approved is True
    response_format = captured["format"]
    assert isinstance(response_format, dict)
    assert response_format["title"] == "ProposalReviewPatch"
    assert "corrected_narrative_map" not in str(captured["prompt"])
    assert "clip_id=1 segment_ids=[1]" in str(captured["prompt"])


def test_proposal_reviewer_applies_global_block_order_before_local_patch(
    tmp_path,
    monkeypatch,
) -> None:
    formats: list[str] = []
    local_prompts: list[str] = []

    class GlobalThenLocalClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def chat(self, **kwargs: object) -> dict[str, dict[str, str]]:
            title = kwargs["format"]["title"]
            formats.append(title)
            if title == "GlobalProposalReviewPlan":
                return {
                    "message": {
                        "content": (
                            '{"ordered_block_ids":[1,0],"dropped_block_ids":[],"restored_segments":[],'
                            '"issues":[{"severity":"warning","code":"antecedent_order","message":"La norma va introdotta prima del riferimento alla stessa norma."}],'
                            '"editorial_notes":"Portata prima la legge 2016 198."}'
                        )
                    }
                }
            prompt = kwargs["messages"][0]["content"]
            assert isinstance(prompt, str)
            local_prompts.append(prompt)
            return {
                "message": {
                    "content": (
                        '{"approved":true,"issues":[],"segment_adjustments":[],"word_removals":[],'
                        '"clip_order":[],"editorial_notes":"Ordine globale coerente, nessuna limatura locale."}'
                    )
                }
            }

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=GlobalThenLocalClient))

    review = ProposalReviewer(
        AppSettings(
            data_dir=tmp_path / "data",
            database_path=tmp_path / "data" / "jobs.sqlite3",
            ollama_run_in_subprocess=False,
        ),
        max_retries=0,
    ).review(
        _narrative_flow_transcriptions(),
        _narrative_flow_map_wrong_order(),
        _narrative_flow_edl_wrong_order(),
        EditorialBrief(user_prompt="Metti prima la norma e poi gli effetti.", synthesis_level=2),
        "qwen3:8b",
    )

    assert formats == ["GlobalProposalReviewPlan", "ProposalReviewPatch"]
    assert local_prompts
    edl_prompt_section = local_prompts[0].split("EditDecisionList corrente:", maxsplit=1)[1]
    assert edl_prompt_section.index("legge del 2016") < edl_prompt_section.index("La stessa norma")
    assert review.approved is False
    assert review.corrected_narrative_map is not None
    assert [clip.clip_id for clip in review.corrected_narrative_map.ordered_clips] == [3, 2]
    assert "Portata prima" in review.editorial_notes


def test_proposal_reviewer_global_review_restores_missing_subject_segment(
    tmp_path,
    monkeypatch,
) -> None:
    class RestoreSubjectClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def chat(self, **kwargs: object) -> dict[str, dict[str, str]]:
            if kwargs["format"]["title"] == "GlobalProposalReviewPlan":
                return {
                    "message": {
                        "content": (
                            '{"ordered_block_ids":[0],"dropped_block_ids":[],'
                            '"restored_segments":[{"clip_id":4,"segment_id":1,"importance":"essential",'
                            '"reason":"Ripristina il soggetto Tommaso Foti prima del riferimento fiduciario."}],'
                            '"issues":[{"severity":"warning","code":"missing_subject","message":"Serve soggetto esplicito."}],'
                            '"editorial_notes":"Ripristinato soggetto esplicito."}'
                        )
                    }
                }
            return {
                "message": {
                    "content": (
                        '{"approved":true,"issues":[],"segment_adjustments":[],"word_removals":[],'
                        '"clip_order":[],"editorial_notes":"Frase completa dopo ripristino."}'
                    )
                }
            }

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=RestoreSubjectClient))

    review = ProposalReviewer(
        AppSettings(
            data_dir=tmp_path / "data",
            database_path=tmp_path / "data" / "jobs.sqlite3",
            ollama_run_in_subprocess=False,
        ),
        max_retries=0,
    ).review(
        [_subject_context_transcription()],
        _subject_context_map_missing_subject(),
        _subject_context_edl_missing_subject(),
        EditorialBrief(user_prompt="Evita frasi senza soggetto.", synthesis_level=2),
        "qwen3:8b",
    )

    assert review.approved is False
    assert review.corrected_narrative_map is not None
    segment_ids = [segment.segment_id for segment in review.corrected_narrative_map.ordered_clips[0].segments]
    assert segment_ids == [1, 2]
    assert review.corrected_narrative_map.ordered_clips[0].segments[0].importance == SegmentImportance.ESSENTIAL


def test_proposal_reviewer_global_restore_clears_stale_synthesis_text(
    tmp_path,
    monkeypatch,
) -> None:
    narrative_map = _subject_context_map_missing_subject()
    narrative_map.ordered_clips[0].segments.insert(
        0,
        NarrativeSegment(
            segment_id=1,
            importance=SegmentImportance.OPTIONAL,
            keep_verbatim=False,
            synthesis_text="Tommaso Foti",
            reason="Sintesi precedente non renderizzabile direttamente.",
        ),
    )

    class RestoreSubjectClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def chat(self, **kwargs: object) -> dict[str, dict[str, str]]:
            if kwargs["format"]["title"] == "GlobalProposalReviewPlan":
                return {
                    "message": {
                        "content": (
                            '{"ordered_block_ids":[0],"dropped_block_ids":[],' 
                            '"restored_segments":[{"clip_id":4,"segment_id":1,"importance":"essential",'
                            '"reason":"Ripristina il soggetto usando il testo audio originale."}],'
                            '"issues":[{"severity":"warning","code":"missing_subject","message":"Serve soggetto esplicito."}],'
                            '"editorial_notes":"Ripristinato soggetto esplicito."}'
                        )
                    }
                }
            return {
                "message": {
                    "content": (
                        '{"approved":true,"issues":[],"segment_adjustments":[],"word_removals":[],' 
                        '"clip_order":[],"editorial_notes":"Ripristino verbatim valido."}'
                    )
                }
            }

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=RestoreSubjectClient))

    review = ProposalReviewer(
        AppSettings(
            data_dir=tmp_path / "data",
            database_path=tmp_path / "data" / "jobs.sqlite3",
            ollama_run_in_subprocess=False,
        ),
        max_retries=0,
    ).review(
        [_subject_context_transcription()],
        narrative_map,
        _subject_context_edl_missing_subject(),
        EditorialBrief(user_prompt="Evita frasi senza soggetto.", synthesis_level=2),
        "qwen3:8b",
    )

    assert review.corrected_narrative_map is not None
    restored_segment = review.corrected_narrative_map.ordered_clips[0].segments[0]
    assert restored_segment.segment_id == 1
    assert restored_segment.keep_verbatim is True
    assert restored_segment.synthesis_text is None
    assert [issue.code for issue in review.issues] == ["missing_subject"]


def test_proposal_reviewer_global_review_blocks_invalid_block_ids_without_local_review(
    tmp_path,
    monkeypatch,
) -> None:
    formats: list[str] = []

    class InvalidGlobalClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def chat(self, **kwargs: object) -> dict[str, dict[str, str]]:
            formats.append(kwargs["format"]["title"])
            return {
                "message": {
                    "content": (
                        '{"ordered_block_ids":[99],"dropped_block_ids":[],"restored_segments":[],'
                        '"issues":[],"editorial_notes":"ID blocco inventato."}'
                    )
                }
            }

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=InvalidGlobalClient))

    review = ProposalReviewer(
        AppSettings(
            data_dir=tmp_path / "data",
            database_path=tmp_path / "data" / "jobs.sqlite3",
            ollama_run_in_subprocess=False,
        ),
        max_retries=0,
    ).review(
        _narrative_flow_transcriptions(),
        _narrative_flow_map_wrong_order(),
        _narrative_flow_edl_wrong_order(),
        EditorialBrief(synthesis_level=2),
        "qwen3:8b",
    )

    assert formats == ["GlobalProposalReviewPlan"]
    assert review.approved is False
    assert review.corrected_narrative_map is None
    assert review.rejected_narrative_map is None
    assert [issue.code for issue in review.issues] == ["global_review_rejected_by_validation"]


def test_proposal_reviewer_applies_lightweight_segment_patch(
    tmp_path,
    monkeypatch,
) -> None:
    class PatchClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def chat(self, **_kwargs: object) -> dict[str, dict[str, str]]:
            return {
                "message": {
                    "content": (
                        '{"approved":false,'
                        '"issues":[{"severity":"warning","code":"repetition","message":"Segmento ripetitivo","clip_id":1,"segment_id":2}],'
                        '"segment_adjustments":[{"clip_id":1,"segment_id":2,"importance":"redundant",'
                        '"keep_verbatim":false,"reason":"Ripetizione sicura da limare."}],'
                        '"clip_order":[],"editorial_notes":"Limata ripetizione."}'
                    )
                }
            }

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=PatchClient))

    review = ProposalReviewer(
        AppSettings(
            data_dir=tmp_path / "data",
            database_path=tmp_path / "data" / "jobs.sqlite3",
            ollama_run_in_subprocess=False,
        ),
        max_retries=0,
    ).review(
        [_two_segment_transcription()],
        _two_segment_narrative_map(),
        _two_segment_edl(),
        EditorialBrief(user_prompt="Fai un montaggio piu asciutto.", synthesis_level=3),
        "qwen3:8b",
    )

    assert review.approved is False
    assert review.corrected_narrative_map is not None
    patched_segments = review.corrected_narrative_map.ordered_clips[0].segments
    patched_by_id = {segment.segment_id: segment for segment in patched_segments}
    assert patched_by_id[2].importance == SegmentImportance.REDUNDANT
    assert patched_by_id[2].keep_verbatim is False


def test_proposal_reviewer_applies_word_removal_patch_for_micro_repetition(
    tmp_path,
    monkeypatch,
) -> None:
    prompts: list[str] = []

    class WordPatchClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def chat(self, **kwargs: object) -> dict[str, dict[str, str]]:
            prompt = kwargs["messages"][0]["content"]
            assert isinstance(prompt, str)
            prompts.append(prompt)
            return {
                "message": {
                    "content": (
                        '{"approved":false,'
                        '"issues":[{"severity":"warning","code":"micro_repetition","message":"Autoriparazione interna al segmento","clip_id":3,"segment_id":1}],'
                        '"segment_adjustments":[],'
                        '"word_removals":[{"clip_id":3,"segment_id":1,"start_word_index":2,"end_word_index":5,'
                        '"reason":"Taglia ha perso, ha rimesso e mantieni ha una perdita dell esercizio."}],'
                        '"clip_order":[],"editorial_notes":"Micro-ripetizione limata a livello parola."}'
                    )
                }
            }

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=WordPatchClient))

    review = ProposalReviewer(
        AppSettings(
            data_dir=tmp_path / "data",
            database_path=tmp_path / "data" / "jobs.sqlite3",
            ollama_run_in_subprocess=False,
        ),
        max_retries=0,
    ).review(
        [_micro_repetition_transcription()],
        _micro_repetition_map(),
        _micro_repetition_edl(),
        EditorialBrief(
            user_prompt="Fai un montaggio piu asciutto, evitando ripetizioni e giri di parole.",
            synthesis_level=3,
        ),
        "qwen3:8b",
    )

    assert review.approved is False
    assert review.corrected_narrative_map is not None
    patched_segment = review.corrected_narrative_map.ordered_clips[0].segments[0]
    assert [(item.start_word_index, item.end_word_index) for item in patched_segment.remove_word_ranges] == [(2, 5)]
    assert "Review globale a blocchi" in prompts[0]
    assert "word_removals" in prompts[1]
    assert "segment_id=1 words=[0:Nel" in prompts[1]


def test_proposal_reviewer_converts_full_segment_word_removal_to_segment_cut(
    tmp_path,
    monkeypatch,
) -> None:
    class FullSegmentWordRemovalClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def chat(self, **kwargs: object) -> dict[str, dict[str, str]]:
            if kwargs["format"]["title"] == "GlobalProposalReviewPlan":
                return {
                    "message": {
                        "content": (
                            '{"ordered_block_ids":[],"dropped_block_ids":[],"restored_segments":[],'
                            '"issues":[],"editorial_notes":"Ordine globale ok."}'
                        )
                    }
                }
            return {
                "message": {
                    "content": (
                        '{"approved":false,'
                        '"issues":[{"severity":"warning","code":"low_value_segment","message":"Segmento eliminabile","clip_id":1,"segment_id":2}],'
                        '"segment_adjustments":[],'
                        '"word_removals":[{"clip_id":1,"segment_id":2,"start_word_index":0,"end_word_index":0,'
                        '"reason":"Rimuovi tutto il segmento ripetitivo."}],'
                        '"clip_order":[],"editorial_notes":"Taglio segmento espresso come word removal."}'
                    )
                }
            }

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=FullSegmentWordRemovalClient))

    review = ProposalReviewer(
        AppSettings(
            data_dir=tmp_path / "data",
            database_path=tmp_path / "data" / "jobs.sqlite3",
            ollama_run_in_subprocess=False,
        ),
        max_retries=0,
    ).review(
        [_two_segment_transcription()],
        _two_segment_narrative_map(),
        _two_segment_edl(),
        EditorialBrief(user_prompt="Taglia i passaggi ripetitivi.", synthesis_level=3),
        "qwen3:8b",
    )

    assert review.approved is False
    assert review.corrected_narrative_map is not None
    patched_segments = review.corrected_narrative_map.ordered_clips[0].segments
    patched_by_id = {segment.segment_id: segment for segment in patched_segments}
    assert patched_by_id[2].importance == SegmentImportance.REDUNDANT
    assert patched_by_id[2].keep_verbatim is False
    assert patched_by_id[2].remove_word_ranges == []
    assert any(issue.code == "full_segment_word_removal_converted" for issue in review.issues)


def test_proposal_reviewer_converts_full_word_removal_when_patch_also_restores_segment(
    tmp_path,
    monkeypatch,
) -> None:
    narrative_map = _two_segment_narrative_map()
    narrative_map.ordered_clips[0].segments = narrative_map.ordered_clips[0].segments[:1]
    edit_decision_list = EditDecisionList(
        decisions=[_two_segment_edl().decisions[0]],
        estimated_duration_s=1.0,
    )

    class RestoreThenRemoveClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def chat(self, **kwargs: object) -> dict[str, dict[str, str]]:
            if kwargs["format"]["title"] == "GlobalProposalReviewPlan":
                return {
                    "message": {
                        "content": (
                            '{"ordered_block_ids":[],"dropped_block_ids":[],"restored_segments":[],'
                            '"issues":[],"editorial_notes":"Ordine globale ok."}'
                        )
                    }
                }
            return {
                "message": {
                    "content": (
                        '{"approved":false,'
                        '"issues":[{"severity":"warning","code":"low_value_segment","message":"Segmento da scartare","clip_id":1,"segment_id":2}],'
                        '"segment_adjustments":[{"clip_id":1,"segment_id":2,"importance":"valuable",'
                        '"keep_verbatim":true,"reason":"Segmento valutato nella review."}],'
                        '"word_removals":[{"clip_id":1,"segment_id":2,"start_word_index":0,"end_word_index":0,'
                        '"reason":"Rimuovi tutte le parole del segmento."}],'
                        '"clip_order":[],"editorial_notes":"Ripristino e rimozione totale nello stesso patch."}'
                    )
                }
            }

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=RestoreThenRemoveClient))

    review = ProposalReviewer(
        AppSettings(
            data_dir=tmp_path / "data",
            database_path=tmp_path / "data" / "jobs.sqlite3",
            ollama_run_in_subprocess=False,
        ),
        max_retries=0,
    ).review(
        [_two_segment_transcription()],
        narrative_map,
        edit_decision_list,
        EditorialBrief(user_prompt="Valuta il segmento ripetitivo.", synthesis_level=3),
        "qwen3:8b",
    )

    assert review.approved is False
    assert review.corrected_narrative_map is not None
    patched_segments = review.corrected_narrative_map.ordered_clips[0].segments
    patched_by_id = {segment.segment_id: segment for segment in patched_segments}
    assert patched_by_id[2].importance == SegmentImportance.REDUNDANT
    assert patched_by_id[2].keep_verbatim is False
    assert patched_by_id[2].remove_word_ranges == []
    assert "ProposalReviewPatch non valida" not in review.editorial_notes


def test_proposal_reviewer_blocks_too_aggressive_patch_without_raising(
    tmp_path,
    monkeypatch,
) -> None:
    class TooAggressivePatchClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def chat(self, **_kwargs: object) -> dict[str, dict[str, str]]:
            redundant_adjustments = ",".join(
                (
                    f'{{"clip_id":1,"segment_id":{segment_id},"importance":"redundant",'
                    '"keep_verbatim":false,"reason":"Taglio troppo aggressivo proposto dalla review."}'
                )
                for segment_id in range(1, 5)
            )
            return {
                "message": {
                    "content": (
                        '{"approved":false,'
                        '"issues":[{"severity":"warning","code":"too_aggressive_review","message":"Patch troppo sintetica"}],'
                        f'"segment_adjustments":[{redundant_adjustments}],'
                        '"word_removals":[],"clip_order":[],"editorial_notes":"Taglia troppi segmenti."}'
                    )
                }
            }

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=TooAggressivePatchClient))

    review = ProposalReviewer(
        AppSettings(
            data_dir=tmp_path / "data",
            database_path=tmp_path / "data" / "jobs.sqlite3",
            ollama_run_in_subprocess=False,
        ),
        max_retries=0,
    ).review(
        [_long_transcription_for_review_patch_validation()],
        _long_narrative_map_for_review_patch_validation(),
        _long_edl_for_review_patch_validation(),
        EditorialBrief(synthesis_level=2),
        "qwen3:8b",
    )

    assert review.approved is False
    assert review.corrected_narrative_map is None
    assert any(issue.code == "review_patch_rejected_by_validation" for issue in review.issues)
    assert "excessive_synthesis" in review.editorial_notes


def test_proposal_reviewer_blocks_truncated_json_without_raising(
    tmp_path,
    monkeypatch,
) -> None:
    class TruncatedJsonClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def chat(self, **_kwargs: object) -> dict[str, dict[str, str]]:
            return {
                "message": {
                    "content": '{"approved":false,"issues":[{"severity":"warning","code":"truncated"'
                }
            }

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=TruncatedJsonClient))

    review = ProposalReviewer(
        AppSettings(
            data_dir=tmp_path / "data",
            database_path=tmp_path / "data" / "jobs.sqlite3",
            ollama_run_in_subprocess=False,
        ),
        max_retries=0,
    ).review(
        [_transcription()],
        _narrative_map(),
        _edl(),
        EditorialBrief(synthesis_level=2),
        "qwen3:8b",
    )

    assert review.approved is False
    assert review.corrected_narrative_map is None
    assert [issue.code for issue in review.issues] == ["proposal_review_invalid_json"]
    assert "Output JSON troncato" in review.editorial_notes


def test_proposal_reviewer_treats_repeated_already_applied_patch_as_approved(
    tmp_path,
    monkeypatch,
) -> None:
    already_patched_map = _two_segment_narrative_map()
    already_patched_map.ordered_clips[0].segments[1] = already_patched_map.ordered_clips[0].segments[1].model_copy(
        update={
            "importance": SegmentImportance.REDUNDANT,
            "keep_verbatim": False,
            "reason": "Ripetizione gia limata.",
        }
    )

    class RepeatedPatchClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def chat(self, **_kwargs: object) -> dict[str, dict[str, str]]:
            return {
                "message": {
                    "content": (
                        '{"approved":false,'
                        '"issues":[{"severity":"warning","code":"repetition","message":"Patch gia applicata","clip_id":1,"segment_id":2}],'
                        '"segment_adjustments":[{"clip_id":1,"segment_id":2,"importance":"redundant",'
                        '"keep_verbatim":false,"reason":"Ripetizione gia limata."}],'
                        '"clip_order":[],"editorial_notes":"Conferma patch gia applicata."}'
                    )
                }
            }

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=RepeatedPatchClient))

    review = ProposalReviewer(
        AppSettings(
            data_dir=tmp_path / "data",
            database_path=tmp_path / "data" / "jobs.sqlite3",
            ollama_run_in_subprocess=False,
        ),
        max_retries=0,
    ).review(
        [_two_segment_transcription()],
        already_patched_map,
        _two_segment_edl(),
        EditorialBrief(user_prompt="Fai un montaggio piu asciutto.", synthesis_level=3),
        "qwen3:8b",
    )

    assert review.approved is True
    assert review.corrected_narrative_map is None


def test_proposal_reviewer_ignores_invalid_patch_segment_ids_without_failing(
    tmp_path,
    monkeypatch,
) -> None:
    transcriptions = [
        ClipTranscription(
            clip_id=3,
            source_path=None,
            duration=4.0,
            segments=[
                Segment(
                    id=index,
                    clip_id=3,
                    start=float(index - 1),
                    end=float(index),
                    text=f"Segmento valido {index}.",
                    words=[WordToken(word="Segmento", start=float(index - 1), end=float(index - 1) + 0.3)],
                )
                for index in range(1, 5)
            ],
        )
    ]
    narrative_map = NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=3,
                playback_position=0,
                topic_summary="Clip con quattro segmenti",
                role="development",
                segments=[
                    NarrativeSegment(
                        segment_id=index,
                        importance=SegmentImportance.VALUABLE,
                        keep_verbatim=True,
                        reason="Mantieni segmento valido.",
                    )
                    for index in range(1, 5)
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Mappa valida.",
        estimated_duration_s=4.0,
    )
    edit_decision_list = EditDecisionList(
        decisions=[
            EditDecision(
                clip_id=3,
                source_start=float(index - 1),
                source_end=float(index),
                output_order=index - 1,
                reason="Mantieni segmento valido.",
                included_text=f"Segmento valido {index}.",
            )
            for index in range(1, 5)
        ],
        estimated_duration_s=4.0,
    )

    class InvalidPatchClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def chat(self, **_kwargs: object) -> dict[str, dict[str, str]]:
            return {
                "message": {
                    "content": (
                        '{"approved":false,'
                        '"issues":[{"severity":"warning","code":"invalid_reference","message":"ID non valido","clip_id":3,"segment_id":5}],'
                        '"segment_adjustments":[{"clip_id":3,"segment_id":5,"importance":"redundant",'
                        '"keep_verbatim":false,"reason":"ID inventato dal modello."}],'
                        '"clip_order":[],"editorial_notes":"Patch con ID non valido."}'
                    )
                }
            }

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=InvalidPatchClient))

    review = ProposalReviewer(
        AppSettings(
            data_dir=tmp_path / "data",
            database_path=tmp_path / "data" / "jobs.sqlite3",
            ollama_run_in_subprocess=False,
        ),
        max_retries=0,
    ).review(
        transcriptions,
        narrative_map,
        edit_decision_list,
        EditorialBrief(synthesis_level=3),
        "qwen3:8b",
    )

    assert review.approved is True
    assert review.corrected_narrative_map is None
    assert any(issue.code == "invalid_patch_reference_ignored" for issue in review.issues)


def test_proposal_reviewer_skips_mixed_patch_when_any_reference_is_invalid(
    tmp_path,
    monkeypatch,
) -> None:
    transcriptions = [
        ClipTranscription(
            clip_id=3,
            source_path=None,
            duration=4.0,
            segments=[
                Segment(
                    id=index,
                    clip_id=3,
                    start=float(index - 1),
                    end=float(index),
                    text=f"Segmento valido {index}.",
                    words=[WordToken(word="Segmento", start=float(index - 1), end=float(index - 1) + 0.3)],
                )
                for index in range(1, 5)
            ],
        )
    ]
    narrative_map = NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=3,
                playback_position=0,
                topic_summary="Clip con quattro segmenti",
                role="development",
                segments=[
                    NarrativeSegment(
                        segment_id=index,
                        importance=SegmentImportance.VALUABLE,
                        keep_verbatim=True,
                        reason="Mantieni segmento valido.",
                    )
                    for index in range(1, 5)
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Mappa valida.",
        estimated_duration_s=4.0,
    )
    edit_decision_list = EditDecisionList(
        decisions=[
            EditDecision(
                clip_id=3,
                source_start=float(index - 1),
                source_end=float(index),
                output_order=index - 1,
                reason="Mantieni segmento valido.",
                included_text=f"Segmento valido {index}.",
            )
            for index in range(1, 5)
        ],
        estimated_duration_s=4.0,
    )

    class MixedPatchClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def chat(self, **_kwargs: object) -> dict[str, dict[str, str]]:
            return {
                "message": {
                    "content": (
                        '{"approved":false,'
                        '"issues":[{"severity":"warning","code":"mixed_patch","message":"Patch mista"}],'
                        '"segment_adjustments":['
                        '{"clip_id":3,"segment_id":2,"importance":"redundant",'
                        '"keep_verbatim":false,"reason":"Taglio valido ma patch instabile."},'
                        '{"clip_id":3,"segment_id":5,"importance":"redundant",'
                        '"keep_verbatim":false,"reason":"ID inventato dal modello."}'
                        '],"clip_order":[],"editorial_notes":"Patch mista con ID non valido."}'
                    )
                }
            }

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=MixedPatchClient))

    review = ProposalReviewer(
        AppSettings(
            data_dir=tmp_path / "data",
            database_path=tmp_path / "data" / "jobs.sqlite3",
            ollama_run_in_subprocess=False,
        ),
        max_retries=0,
    ).review(
        transcriptions,
        narrative_map,
        edit_decision_list,
        EditorialBrief(synthesis_level=3),
        "qwen3:8b",
    )

    assert review.approved is True
    assert review.corrected_narrative_map is None
    assert any(issue.code == "invalid_patch_reference_ignored" for issue in review.issues)


def test_medium_proposal_review_is_split_into_lightweight_chunks(
    tmp_path,
    monkeypatch,
) -> None:
    prompts: list[str] = []

    class ChunkClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def chat(self, **kwargs: object) -> dict[str, dict[str, str]]:
            prompt = kwargs["messages"][0]["content"]
            assert isinstance(prompt, str)
            prompts.append(prompt)
            return {
                "message": {
                    "content": (
                        '{"approved":true,"issues":[],"segment_adjustments":[],'
                        '"clip_order":[],"editorial_notes":"Chunk coerente."}'
                    )
                }
            }

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=ChunkClient))

    review = ProposalReviewer(
        AppSettings(
            data_dir=tmp_path / "data",
            database_path=tmp_path / "data" / "jobs.sqlite3",
            ollama_run_in_subprocess=False,
        ),
        max_retries=0,
    ).review(
        [_many_transcription(32)],
        _many_narrative_map(32),
        _many_edl(32),
        EditorialBrief(synthesis_level=3),
        "qwen3:8b",
    )

    assert review.approved is True
    assert len(prompts) == 3
    assert "Review globale a blocchi" in prompts[0]
    local_prompts = prompts[1:]
    assert all(prompt.count("output_order=") <= 16 for prompt in local_prompts)
    assert all("massimo 6 modifiche" in prompt for prompt in local_prompts)


def test_large_proposal_under_ten_minutes_reviews_all_local_chunks(
    tmp_path,
    monkeypatch,
) -> None:
    prompts: list[str] = []
    titles: list[str] = []
    progress_messages: list[str] = []

    class LargeProposalClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def chat(self, **kwargs: object) -> dict[str, dict[str, str]]:
            prompt = kwargs["messages"][0]["content"]
            assert isinstance(prompt, str)
            prompts.append(prompt)
            title = kwargs["format"]["title"]
            titles.append(title)
            if title == "ProposalReviewPatch":
                return {
                    "message": {
                        "content": (
                            '{"approved":true,"issues":[],"segment_adjustments":[],"word_removals":[],'
                            '"clip_order":[],"editorial_notes":"Chunk locale validato."}'
                        )
                    }
                }
            return {
                "message": {
                    "content": (
                        '{"ordered_block_ids":[],"dropped_block_ids":[],"restored_segments":[],'
                        '"issues":[],"editorial_notes":"Review globale sufficiente."}'
                    )
                }
            }

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=LargeProposalClient))

    review = ProposalReviewer(
        AppSettings(
            data_dir=tmp_path / "data",
            database_path=tmp_path / "data" / "jobs.sqlite3",
            ollama_run_in_subprocess=False,
        ),
        max_retries=0,
    ).review(
        [_many_transcription(95)],
        _many_narrative_map(95),
        _many_edl(95),
        EditorialBrief(synthesis_level=3),
        "qwen3:8b",
        progress_callback=progress_messages.append,
    )

    assert review.approved is True
    assert review.corrected_narrative_map is None
    assert titles == [
        "GlobalProposalReviewPlan",
        "ProposalReviewPatch",
        "ProposalReviewPatch",
        "ProposalReviewPatch",
        "ProposalReviewPatch",
        "ProposalReviewPatch",
        "ProposalReviewPatch",
    ]
    assert len(prompts) == 7
    assert review.issues == []
    assert not any("Review LLM saltata" in message for message in progress_messages)


def test_large_final_review_over_ten_minutes_returns_before_calling_ollama(
    tmp_path,
    monkeypatch,
) -> None:
    class BlockingClient:
        def __init__(self, **_kwargs: object) -> None:
            raise AssertionError("large proposal review must not instantiate Ollama client")

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=BlockingClient))
    progress_messages: list[str] = []

    review = ProposalReviewer(
        AppSettings(
            data_dir=tmp_path / "data",
            database_path=tmp_path / "data" / "jobs.sqlite3",
            ollama_run_in_subprocess=False,
        ),
        max_retries=0,
    ).review(
        [_many_transcription(601)],
        _many_narrative_map(601),
        _many_edl(601),
        EditorialBrief(synthesis_level=3),
        "qwen3:8b",
        progress_callback=progress_messages.append,
    )

    assert review.approved is True
    assert review.corrected_narrative_map is None
    assert [issue.code for issue in review.issues] == ["proposal_review_skipped_large_proposal"]
    assert "Review LLM saltata" in progress_messages[-1]


def test_proposal_reviewer_increases_patch_budget_after_truncated_json(
    tmp_path,
    monkeypatch,
) -> None:
    local_num_predicts: list[int] = []

    class RetryBudgetClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def chat(self, **kwargs: object) -> dict[str, dict[str, str]]:
            if kwargs["format"]["title"] == "GlobalProposalReviewPlan":
                return {
                    "message": {
                        "content": (
                            '{"ordered_block_ids":[],"dropped_block_ids":[],"restored_segments":[],'
                            '"issues":[],"editorial_notes":"Ordine globale ok."}'
                        )
                    }
                }
            local_num_predicts.append(kwargs["options"]["num_predict"])
            if len(local_num_predicts) == 1:
                return {
                    "message": {
                        "content": '{"approved":false,"issues":[{"severity":"warning","code":"truncated"'
                    }
                }
            return {
                "message": {
                    "content": (
                        '{"approved":true,"issues":[],"segment_adjustments":[],"word_removals":[],'
                        '"clip_order":[],"editorial_notes":"Patch compatta valida al retry."}'
                    )
                }
            }

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=RetryBudgetClient))

    review = ProposalReviewer(
        AppSettings(
            data_dir=tmp_path / "data",
            database_path=tmp_path / "data" / "jobs.sqlite3",
            ollama_run_in_subprocess=False,
        ),
        max_retries=1,
    ).review(
        [_transcription()],
        _narrative_map(),
        _edl(),
        EditorialBrief(synthesis_level=2),
        "qwen3:8b",
    )

    assert review.approved is True
    assert local_num_predicts == [2048, 4096]


def test_proposal_reviewer_falls_back_to_ultra_light_plan_after_patch_json_failure(
    tmp_path,
    monkeypatch,
) -> None:
    formats: list[str] = []

    class UltraLightFallbackClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def chat(self, **kwargs: object) -> dict[str, dict[str, str]]:
            title = kwargs["format"]["title"]
            formats.append(title)
            if title == "GlobalProposalReviewPlan":
                if formats.count("GlobalProposalReviewPlan") == 1:
                    return {
                        "message": {
                            "content": (
                                '{"ordered_block_ids":[],"dropped_block_ids":[],"restored_segments":[],'
                                '"issues":[],"editorial_notes":"Ordine globale ok."}'
                            )
                        }
                    }
                return {
                    "message": {
                        "content": (
                            '{"ordered_block_ids":[0],"dropped_block_ids":[],"restored_segments":[],'
                            '"issues":[{"severity":"warning","code":"ultra_light_review","message":"Review ultra-light validata."}],'
                            '"editorial_notes":"Fallback ultra-light: proposta coerente e renderizzabile."}'
                        )
                    }
                }
            return {
                "message": {
                    "content": '{"approved":false,"issues":[{"severity":"warning","code":"truncated"'
                }
            }

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=UltraLightFallbackClient))

    review = ProposalReviewer(
        AppSettings(
            data_dir=tmp_path / "data",
            database_path=tmp_path / "data" / "jobs.sqlite3",
            ollama_run_in_subprocess=False,
        ),
        max_retries=0,
    ).review(
        [_transcription()],
        _narrative_map(),
        _edl(),
        EditorialBrief(synthesis_level=2),
        "qwen3:8b",
    )

    assert formats == ["GlobalProposalReviewPlan", "ProposalReviewPatch", "GlobalProposalReviewPlan"]
    assert review.approved is False
    assert review.corrected_narrative_map is not None
    assert [issue.code for issue in review.issues] == ["ultra_light_review"]
    assert "Fallback ultra-light" in review.editorial_notes
