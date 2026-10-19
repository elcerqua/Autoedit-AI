from __future__ import annotations

import pytest

from autocut_ai.cuts import _merge_overlapping_same_clip_decisions, build_edit_decision_list, snap_interval_to_word_boundaries
from autocut_ai.schemas import (
    ClipTranscription,
    EditDecision,
    EditingOptions,
    NarrativeClip,
    NarrativeMap,
    NarrativeSegment,
    Segment,
    SegmentImportance,
    WordRemovalRange,
    WordToken,
)


def _transcription() -> ClipTranscription:
    return ClipTranscription(
        clip_id=1,
        source_path="tests/fixtures/1.mp4",
        duration=5.0,
        segments=[
            Segment(
                id=10,
                clip_id=1,
                start=1.0,
                end=3.2,
                text="Nel 2026 costera 10 euro",
                words=[
                    WordToken(word="Nel", start=1.0, end=1.15),
                    WordToken(word="2026", start=1.2, end=1.55),
                    WordToken(word="costera", start=1.6, end=2.0),
                    WordToken(word="10", start=2.1, end=2.3),
                    WordToken(word="euro", start=2.35, end=2.7),
                ],
            )
        ],
    )


def _narrative_map(importance: SegmentImportance = SegmentImportance.ESSENTIAL) -> NarrativeMap:
    return NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=1,
                playback_position=0,
                topic_summary="Prezzo",
                role="development",
                segments=[
                    NarrativeSegment(
                        segment_id=10,
                        importance=importance,
                        keep_verbatim=True,
                        reason="Contiene cifra e unita",
                    )
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Preserva numeri e unita.",
        estimated_duration_s=2.2,
    )


def test_snap_interval_does_not_cut_inside_word_or_number() -> None:
    segment = _transcription().segments[0]
    start, end, notes = snap_interval_to_word_boundaries(
        segment=segment,
        proposed_start=1.35,
        proposed_end=2.2,
        preserve_numbers=True,
    )

    assert start == pytest.approx(1.2)
    assert end == pytest.approx(2.3)
    assert any("start" in note for note in notes)
    assert any("end" in note for note in notes)


def test_edit_decision_list_applies_pre_roll_post_roll_and_min_duration() -> None:
    options = EditingOptions(pre_roll_ms=100, post_roll_ms=150, min_segment_duration_ms=500)
    edl = build_edit_decision_list([_transcription()], _narrative_map(), options)

    assert len(edl.decisions) == 1
    decision = edl.decisions[0]
    assert decision.source_start == pytest.approx(0.9)
    assert decision.source_end == pytest.approx(3.55)
    assert decision.source_end - decision.source_start >= 0.5
    assert decision.included_text == "Nel 2026 costera 10 euro"
    assert any("tail guard" in note for note in decision.safety_notes)


def test_edit_decision_list_adds_transition_tail_guard_for_soft_transitions() -> None:
    options = EditingOptions(
        pre_roll_ms=0,
        post_roll_ms=0,
        default_transition="cross_dissolve",
        transition_duration_ms=400,
        min_segment_duration_ms=0,
    )
    edl = build_edit_decision_list([_transcription()], _narrative_map(), options)

    decision = edl.decisions[0]
    assert decision.source_end == pytest.approx(3.95)
    assert any("tail guard" in note for note in decision.safety_notes)


def test_edit_decision_list_does_not_place_padding_inside_adjacent_words() -> None:
    transcription = ClipTranscription(
        clip_id=1,
        source_path="tests/fixtures/1.mp4",
        duration=3.0,
        segments=[
            Segment(
                id=1,
                clip_id=1,
                start=0.70,
                end=0.96,
                text="prima",
                words=[WordToken(word="prima", start=0.70, end=0.96)],
            ),
            Segment(
                id=2,
                clip_id=1,
                start=1.00,
                end=1.50,
                text="parola scelta",
                words=[
                    WordToken(word="parola", start=1.00, end=1.20),
                    WordToken(word="scelta", start=1.30, end=1.50),
                ],
            ),
            Segment(
                id=3,
                clip_id=1,
                start=1.58,
                end=1.95,
                text="dopo",
                words=[WordToken(word="dopo", start=1.58, end=1.95)],
            ),
        ],
    )
    narrative_map = NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=1,
                playback_position=0,
                topic_summary="Segmento centrale",
                role="development",
                segments=[
                    NarrativeSegment(
                        segment_id=2,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason="Mantieni solo il segmento centrale.",
                    )
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Evita parole troncate ai margini.",
        estimated_duration_s=0.5,
    )

    edl = build_edit_decision_list(
        [transcription],
        narrative_map,
        EditingOptions(
            pre_roll_ms=100,
            post_roll_ms=120,
            min_segment_duration_ms=0,
            avoid_hard_cut_inside_sentence=False,
        ),
    )

    decision = edl.decisions[0]
    assert decision.source_start == pytest.approx(0.96)
    assert decision.source_end == pytest.approx(1.58)
    assert any("pre-roll" in note for note in decision.safety_notes)
    assert any("Clamped tail guard before adjacent word" in note for note in decision.safety_notes)


def test_edit_decision_list_min_duration_extension_does_not_end_inside_next_word() -> None:
    transcription = ClipTranscription(
        clip_id=1,
        source_path="tests/fixtures/1.mp4",
        duration=3.0,
        segments=[
            Segment(
                id=1,
                clip_id=1,
                start=1.00,
                end=1.10,
                text="Si",
                words=[WordToken(word="Si", start=1.00, end=1.10)],
            ),
            Segment(
                id=2,
                clip_id=1,
                start=1.40,
                end=1.80,
                text="dopo",
                words=[WordToken(word="dopo", start=1.40, end=1.80)],
            ),
        ],
    )
    narrative_map = NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=1,
                playback_position=0,
                topic_summary="Risposta breve",
                role="development",
                segments=[
                    NarrativeSegment(
                        segment_id=1,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason="Mantieni risposta breve.",
                    )
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Evita estensioni a meta parola.",
        estimated_duration_s=0.1,
    )

    edl = build_edit_decision_list(
        [transcription],
        narrative_map,
        EditingOptions(
            pre_roll_ms=0,
            post_roll_ms=0,
            min_segment_duration_ms=600,
            avoid_hard_cut_inside_sentence=False,
        ),
    )

    decision = edl.decisions[0]
    assert decision.source_end == pytest.approx(1.80)
    assert any("minimum duration" in note for note in decision.safety_notes)


def test_edit_decision_list_skips_too_short_optional_segments() -> None:
    transcription = _transcription()
    transcription.segments[0].words = [WordToken(word="Si", start=1.0, end=1.1)]
    transcription.segments[0].text = "Si"
    options = EditingOptions(pre_roll_ms=0, post_roll_ms=0, min_segment_duration_ms=500)

    edl = build_edit_decision_list([transcription], _narrative_map(SegmentImportance.OPTIONAL), options)

    assert edl.decisions == []


def test_edit_decision_list_merges_overlapping_same_clip_decisions() -> None:
    transcription = ClipTranscription(
        clip_id=1,
        source_path="tests/fixtures/1.mp4",
        duration=5.0,
        segments=[
            Segment(
                id=10,
                clip_id=1,
                start=0.0,
                end=1.0,
                text="Primo concetto",
                words=[
                    WordToken(word="Primo", start=0.1, end=0.4),
                    WordToken(word="concetto", start=0.5, end=0.9),
                ],
            ),
            Segment(
                id=11,
                clip_id=1,
                start=1.05,
                end=2.0,
                text="Secondo concetto",
                words=[
                    WordToken(word="Secondo", start=1.1, end=1.45),
                    WordToken(word="concetto", start=1.55, end=1.9),
                ],
            ),
        ],
    )
    narrative_map = NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=1,
                playback_position=0,
                topic_summary="Due concetti vicini",
                role="development",
                segments=[
                    NarrativeSegment(
                        segment_id=10,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason="Primo punto",
                    ),
                    NarrativeSegment(
                        segment_id=11,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason="Secondo punto",
                    ),
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Mantieni entrambi senza ripetizione.",
        estimated_duration_s=2.0,
    )
    options = EditingOptions(pre_roll_ms=200, post_roll_ms=250, min_segment_duration_ms=100)

    edl = build_edit_decision_list([transcription], narrative_map, options)

    assert len(edl.decisions) == 1
    decision = edl.decisions[0]
    assert decision.source_start == pytest.approx(0.0)
    assert decision.source_end == pytest.approx(2.75)
    assert decision.included_text == "Primo concetto Secondo concetto"
    assert any("Merged overlapping same-clip decision" in note for note in decision.safety_notes)


def test_overlapping_merge_does_not_duplicate_text_overlap() -> None:
    decisions = [
        EditDecision(
            clip_id=1,
            source_start=0.0,
            source_end=2.0,
            output_order=0,
            reason="Prima decisione estesa per frase completa.",
            included_text="Prima frase. Seconda frase.",
        ),
        EditDecision(
            clip_id=1,
            source_start=1.8,
            source_end=3.0,
            output_order=1,
            reason="Decisione successiva sovrapposta.",
            included_text="Seconda frase. Terza frase.",
        ),
    ]

    merged = _merge_overlapping_same_clip_decisions(decisions)

    assert len(merged) == 1
    assert merged[0].included_text == "Prima frase. Seconda frase. Terza frase."
    assert any("Skipped duplicate text" in note for note in merged[0].safety_notes)


def test_edit_decision_list_uses_source_text_for_verbatim_segments() -> None:
    narrative_map = _narrative_map()
    narrative_map.ordered_clips[0].segments[0].synthesis_text = "Testo riscritto con errore sintattico"

    edl = build_edit_decision_list([_transcription()], narrative_map, EditingOptions())

    assert edl.decisions[0].included_text == "Nel 2026 costera 10 euro"


def test_edit_decision_list_applies_llm_word_removal_ranges_without_merging_repetition() -> None:
    transcription = ClipTranscription(
        clip_id=3,
        source_path="tests/fixtures/3.mp4",
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
    narrative_map = NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=3,
                playback_position=0,
                topic_summary="Dati di bilancio",
                role="development",
                segments=[
                    NarrativeSegment(
                        segment_id=1,
                        importance=SegmentImportance.VALUABLE,
                        keep_verbatim=True,
                        remove_word_ranges=[
                            WordRemovalRange(
                                start_word_index=2,
                                end_word_index=5,
                                reason="Autoriparazione ripetitiva: mantieni solo la formulazione precisa della perdita.",
                            )
                        ],
                        reason="Mantieni dato di bilancio limato.",
                    )
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Montaggio asciutto con micro-taglio LLM.",
        estimated_duration_s=1.0,
    )

    edl = build_edit_decision_list(
        [transcription],
        narrative_map,
        EditingOptions(pre_roll_ms=0, post_roll_ms=0, min_segment_duration_ms=0, avoid_hard_cut_inside_sentence=False),
    )

    assert [decision.included_text for decision in edl.decisions] == [
        "Nel 24",
        "ha una perdita dell'esercizio.",
    ]
    assert [(decision.source_start, decision.source_end) for decision in edl.decisions] == [
        (0.0, 0.30),
        (1.20, 2.85),
    ]
    assert any("Clamped tail guard before adjacent word" in note for note in edl.decisions[0].safety_notes)
    assert any("tail guard" in note for note in edl.decisions[1].safety_notes)


def test_creative_zoom_uses_long_average_pauses_instead_of_llm_flags() -> None:
    transcription = ClipTranscription(
        clip_id=1,
        source_path="tests/fixtures/1.mp4",
        duration=8.0,
        segments=[
            Segment(
                id=10,
                clip_id=1,
                start=0.0,
                end=1.0,
                text="Primo contesto",
                words=[
                    WordToken(word="Primo", start=0.1, end=0.4),
                    WordToken(word="contesto", start=0.5, end=0.9),
                ],
            ),
            Segment(
                id=11,
                clip_id=1,
                start=1.05,
                end=2.0,
                text="Secondo passaggio",
                words=[
                    WordToken(word="Secondo", start=1.1, end=1.45),
                    WordToken(word="passaggio", start=1.55, end=1.9),
                ],
            ),
            Segment(
                id=12,
                clip_id=1,
                start=3.2,
                end=4.0,
                text="Frase dopo pausa lunga",
                words=[
                    WordToken(word="Frase", start=3.25, end=3.45),
                    WordToken(word="dopo", start=3.50, end=3.60),
                    WordToken(word="pausa", start=3.65, end=3.72),
                    WordToken(word="lunga", start=3.75, end=3.95),
                ],
            ),
        ],
    )
    narrative_map = NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=1,
                playback_position=0,
                topic_summary="Zoom interno",
                role="development",
                segments=[
                    NarrativeSegment(
                        segment_id=10,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason="Contesto necessario",
                    ),
                    NarrativeSegment(
                        segment_id=11,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        creative_zoom=True,
                        reason="Flag LLM ignorato per pause brevi",
                    ),
                    NarrativeSegment(
                        segment_id=12,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason="Passaggio dopo pausa lunga",
                    ),
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Zoom sulle pause lunghe.",
        estimated_duration_s=4.0,
    )

    edl = build_edit_decision_list(
        [transcription],
        narrative_map,
        EditingOptions(enable_creative_zoom=True, pre_roll_ms=200, post_roll_ms=250, min_segment_duration_ms=100),
    )

    assert len(edl.decisions) == 2
    decision = edl.decisions[1]
    assert decision.source_start == pytest.approx(3.05)
    assert decision.source_end == pytest.approx(4.8)
    assert decision.creative_zoom is True
    assert len(decision.creative_zoom_spans) == 1
    span = decision.creative_zoom_spans[0]
    assert span.source_start == pytest.approx(3.25)
    assert span.source_end == pytest.approx(3.95)
    assert span.start_scale == pytest.approx(span.end_scale)
    assert span.text == "Frase dopo pausa lunga"
    assert "pausa" in (span.reason or "")
    assert span.source_start > decision.source_start
    assert span.source_end < decision.source_end


def test_creative_zoom_uses_long_word_level_pause_inside_single_retained_segment() -> None:
    transcription = ClipTranscription(
        clip_id=1,
        source_path="tests/fixtures/1.mp4",
        duration=3.0,
        segments=[
            Segment(
                id=10,
                clip_id=1,
                start=0.0,
                end=2.0,
                text="Prima parte Seconda parte",
                words=[
                    WordToken(word="Prima", start=0.10, end=0.30),
                    WordToken(word="parte", start=0.35, end=0.60),
                    WordToken(word="Seconda", start=0.80, end=1.10),
                    WordToken(word="parte", start=1.15, end=1.50),
                ],
            )
        ],
    )
    narrative_map = NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=1,
                playback_position=0,
                topic_summary="Segmento unico",
                role="development",
                segments=[
                    NarrativeSegment(
                        segment_id=10,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason="Un solo segmento mantenuto",
                    )
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Zoom sulla pausa word-level piu lunga.",
        estimated_duration_s=2.0,
    )

    edl = build_edit_decision_list(
        [transcription],
        narrative_map,
        EditingOptions(enable_creative_zoom=True, pre_roll_ms=0, post_roll_ms=0, min_pause_to_cut_ms=300),
    )

    assert len(edl.decisions) == 1
    decision = edl.decisions[0]
    assert decision.creative_zoom is True
    assert len(decision.creative_zoom_spans) == 1
    span = decision.creative_zoom_spans[0]
    assert span.source_start == pytest.approx(0.80)
    assert span.source_end == pytest.approx(1.50)
    assert span.text == "Seconda parte"
    assert span.start_scale == pytest.approx(span.end_scale)


def test_creative_zoom_skips_zero_length_word_spans_without_failing() -> None:
    transcription = ClipTranscription(
        clip_id=1,
        source_path="tests/fixtures/1.mp4",
        duration=90.0,
        segments=[
            Segment(
                id=10,
                clip_id=1,
                start=83.0,
                end=85.0,
                text="Contesto valido elemento",
                words=[
                    WordToken(word="Contesto", start=83.10, end=83.40),
                    WordToken(word="valido", start=83.45, end=83.70),
                    WordToken(word="elemento", start=84.25, end=84.25),
                ],
            )
        ],
    )
    narrative_map = NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=1,
                playback_position=0,
                topic_summary="Zoom con timestamp degenerato",
                role="development",
                segments=[
                    NarrativeSegment(
                        segment_id=10,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason="Mantieni il segmento anche se una parola ha durata zero.",
                    )
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Non fallire per uno span zoom nullo.",
        estimated_duration_s=2.0,
    )

    edl = build_edit_decision_list(
        [transcription],
        narrative_map,
        EditingOptions(enable_creative_zoom=True, min_creative_zoom_count=2, pre_roll_ms=0, post_roll_ms=0),
    )

    assert len(edl.decisions) == 1
    assert edl.decisions[0].source_start == pytest.approx(83.10)
    assert edl.decisions[0].source_end >= 84.25
    assert all(
        span.source_end > span.source_start
        for span in edl.decisions[0].creative_zoom_spans
    )


def test_creative_zoom_llm_flag_is_ignored_when_option_is_disabled() -> None:
    narrative_map = _narrative_map()
    narrative_map.ordered_clips[0].segments[0].creative_zoom = True

    edl = build_edit_decision_list([_transcription()], narrative_map, EditingOptions(enable_creative_zoom=False))

    assert edl.decisions[0].creative_zoom is False
    assert edl.decisions[0].creative_zoom_spans == []


def test_creative_zoom_respects_requested_minimum_with_phrase_fallback() -> None:
    transcription = ClipTranscription(
        clip_id=1,
        source_path="tests/fixtures/1.mp4",
        duration=7.0,
        segments=[
            Segment(
                id=10,
                clip_id=1,
                start=0.0,
                end=0.9,
                text="Primo punto",
                words=[
                    WordToken(word="Primo", start=0.1, end=0.3),
                    WordToken(word="punto", start=0.4, end=0.8),
                ],
            ),
            Segment(
                id=11,
                clip_id=1,
                start=1.2,
                end=2.0,
                text="Secondo punto",
                words=[
                    WordToken(word="Secondo", start=1.25, end=1.55),
                    WordToken(word="punto", start=1.65, end=1.9),
                ],
            ),
            Segment(
                id=12,
                clip_id=1,
                start=2.4,
                end=3.2,
                text="Terzo punto",
                words=[
                    WordToken(word="Terzo", start=2.45, end=2.7),
                    WordToken(word="punto", start=2.8, end=3.1),
                ],
            ),
            Segment(
                id=13,
                clip_id=1,
                start=3.6,
                end=4.4,
                text="Quarto punto",
                words=[
                    WordToken(word="Quarto", start=3.65, end=3.95),
                    WordToken(word="punto", start=4.05, end=4.3),
                ],
            ),
            Segment(
                id=14,
                clip_id=1,
                start=4.8,
                end=5.6,
                text="Quinto punto",
                words=[
                    WordToken(word="Quinto", start=4.85, end=5.15),
                    WordToken(word="punto", start=5.25, end=5.5),
                ],
            ),
        ],
    )
    narrative_map = NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=1,
                playback_position=0,
                topic_summary="Cinque punti",
                role="development",
                segments=[
                    NarrativeSegment(
                        segment_id=segment_id,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason=f"Punto {segment_id}",
                    )
                    for segment_id in [10, 11, 12, 13, 14]
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Usa almeno cinque zoom creativi.",
        estimated_duration_s=5.0,
    )

    edl = build_edit_decision_list(
        [transcription],
        narrative_map,
        EditingOptions(
            enable_creative_zoom=True,
            min_creative_zoom_count=5,
            pre_roll_ms=0,
            post_roll_ms=0,
            min_segment_duration_ms=100,
        ),
    )

    spans = [span for decision in edl.decisions for span in decision.creative_zoom_spans]

    assert len(spans) == 5
    assert sum(1 for decision in edl.decisions if decision.creative_zoom) == 5
    assert any("minimo richiesto" in (span.reason or "") for span in spans)


def test_edit_decision_list_extends_dangling_preposition_to_complete_phrase() -> None:
    transcription = ClipTranscription(
        clip_id=1,
        source_path="tests/fixtures/1.mp4",
        duration=30.0,
        segments=[
            Segment(
                id=3,
                clip_id=1,
                start=15.62,
                end=22.40,
                text="europei e il proprietario unico del capitale sociale ma risulta come fiduciario della",
                words=[
                    WordToken(word="europei", start=15.62, end=16.00),
                    WordToken(word="della", start=22.10, end=22.40),
                ],
            ),
            Segment(
                id=4,
                clip_id=1,
                start=22.40,
                end=28.08,
                text="fondazione alleanza nazionale la vera proprietaria del secolo d'italia",
                words=[
                    WordToken(word="fondazione", start=22.40, end=22.90),
                    WordToken(word="italia", start=27.70, end=28.08),
                ],
            ),
        ],
    )
    narrative_map = NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=1,
                playback_position=0,
                topic_summary="Assetto societario",
                role="development",
                segments=[
                    NarrativeSegment(
                        segment_id=3,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason="Spiega il ruolo fiduciario",
                    )
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Mantieni il passaggio senza troncare la frase.",
        estimated_duration_s=12.0,
    )

    edl = build_edit_decision_list([transcription], narrative_map, EditingOptions(post_roll_ms=0))

    assert edl.decisions[0].source_end == pytest.approx(28.93)
    assert edl.decisions[0].included_text.endswith("proprietaria del secolo d'italia")
    assert any("incomplete sentence" in note for note in edl.decisions[0].safety_notes)
    assert any("tail guard" in note for note in edl.decisions[0].safety_notes)


def test_edit_decision_list_extends_when_next_segment_starts_as_continuation() -> None:
    transcription = ClipTranscription(
        clip_id=2,
        source_path="tests/fixtures/2.mp4",
        duration=26.0,
        segments=[
            Segment(
                id=3,
                clip_id=2,
                start=13.62,
                end=19.92,
                text="area politica. Il direttore esecutivo e Italo Bocchino, quattro legislature in Parlamento",
                words=[
                    WordToken(word="area", start=13.62, end=14.00),
                    WordToken(word="Parlamento", start=19.30, end=19.92),
                ],
            ),
            Segment(
                id=4,
                clip_id=2,
                start=19.92,
                end=25.80,
                text="come fedelissimo di fine, insomma tutti politici e tutti da sempre schierati.",
                words=[
                    WordToken(word="come", start=19.92, end=20.20),
                    WordToken(word="schierati.", start=25.20, end=25.80),
                ],
            ),
        ],
    )
    narrative_map = NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=2,
                playback_position=0,
                topic_summary="Profilo politico",
                role="development",
                segments=[
                    NarrativeSegment(
                        segment_id=3,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason="Spiega il ruolo parlamentare",
                    )
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Mantieni il passaggio senza troncare la frase.",
        estimated_duration_s=12.0,
    )

    edl = build_edit_decision_list([transcription], narrative_map, EditingOptions(post_roll_ms=0))

    assert edl.decisions[0].source_end == pytest.approx(26.0)
    assert edl.decisions[0].included_text.endswith("da sempre schierati.")
    assert "Parlamento come fedelissimo" in edl.decisions[0].included_text
