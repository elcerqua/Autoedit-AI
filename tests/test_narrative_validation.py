from __future__ import annotations

import pytest

from autocut_ai.narrative import (
    NarrativeValidationError,
    build_editor_prompt,
    repair_narrative_map_for_cut_safety,
    validate_narrative_map,
)
from autocut_ai.schemas import (
    ClipTranscription,
    EditorialBrief,
    EditingOptions,
    NarrativeClip,
    NarrativeMap,
    NarrativeSegment,
    Segment,
    SegmentImportance,
    WordToken,
)


def _clip(clip_id: int, segment_id: int) -> ClipTranscription:
    return ClipTranscription(
        clip_id=clip_id,
        source_path=None,
        duration=3.0,
        segments=[
            Segment(
                id=segment_id,
                clip_id=clip_id,
                start=0.2,
                end=1.4,
                text="Questo e un test",
                words=[
                    WordToken(word="Questo", start=0.2, end=0.5),
                    WordToken(word="e", start=0.55, end=0.65),
                    WordToken(word="un", start=0.7, end=0.8),
                    WordToken(word="test", start=0.9, end=1.4),
                ],
            )
        ],
    )


def _map(clip_id: int = 1, segment_id: int = 10, playback_position: int = 0) -> NarrativeMap:
    return NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=clip_id,
                playback_position=playback_position,
                topic_summary="Tema",
                role="development",
                segments=[
                    NarrativeSegment(
                        segment_id=segment_id,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason="Nodo essenziale",
                    )
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Mantieni il punto centrale.",
        estimated_duration_s=1.2,
    )


def _source_segment(clip_id: int, segment_id: int, start: float, end: float, text: str) -> Segment:
    tokens = text.split()
    first_word = tokens[0] if tokens else "segmento"
    last_word = tokens[-1] if tokens else "segmento"
    return Segment(
        id=segment_id,
        clip_id=clip_id,
        start=start,
        end=end,
        text=text,
        words=[
            WordToken(word=first_word, start=start, end=min(end, start + 0.2)),
            WordToken(word=last_word, start=max(start, end - 0.2), end=end),
        ],
    )


def test_narrative_map_accepts_known_clip_and_segment_ids() -> None:
    validate_narrative_map(_map(), [_clip(1, 10)])


def test_editor_prompt_requests_creative_zoom_flags_when_enabled() -> None:
    prompt = build_editor_prompt(
        [_clip(1, 10)],
        EditorialBrief(user_prompt="Dai enfasi ai passaggi piu importanti"),
        editing_options=EditingOptions(enable_creative_zoom=True),
    )

    assert "pause" in prompt
    assert "zoom fisso" in prompt
    assert "20%" in prompt
    assert "50%" in prompt


def test_editor_prompt_treats_user_prompt_as_binding_editorial_direction() -> None:
    prompt = build_editor_prompt(
        [_clip(1, 10)],
        EditorialBrief(
            user_prompt=(
                "Fai un montaggio piu asciutto, evitando ripetizioni, esitazioni e giri di parole. "
                "Mantieni solo i passaggi necessari alla comprensione."
            ),
            synthesis_level=2,
        ),
    )

    assert "Brief utente vincolante" in prompt
    assert "montaggio piu asciutto" in prompt
    assert "Non restituire la stessa classificazione" in prompt
    assert "editorial_strategy" in prompt


def test_editor_prompt_includes_minimum_retained_duration_budget() -> None:
    prompt = build_editor_prompt(
        [
            ClipTranscription(
                clip_id=1,
                source_path=None,
                duration=90.0,
                segments=[
                    Segment(
                        id=index,
                        clip_id=1,
                        start=float(index * 10),
                        end=float(index * 10 + 10),
                        text=f"Segmento {index}",
                        words=[WordToken(word="Segmento", start=float(index * 10), end=float(index * 10 + 1))],
                    )
                    for index in range(1, 7)
                ],
            )
        ],
        EditorialBrief(synthesis_level=3),
    )

    assert "durata parlata totale" in prompt
    assert "non creare una mini-compilation" in prompt
    assert "durata minima attesa" in prompt


def test_editor_prompt_includes_explicit_target_duration_budget() -> None:
    prompt = build_editor_prompt(
        [
            ClipTranscription(
                clip_id=1,
                source_path=None,
                duration=900.0,
                segments=[
                    Segment(
                        id=index,
                        clip_id=1,
                        start=float(index * 60),
                        end=float(index * 60 + 60),
                        text=f"Blocco informativo {index}",
                        words=[WordToken(word="Blocco", start=float(index * 60), end=float(index * 60 + 1))],
                    )
                    for index in range(1, 11)
                ],
            )
        ],
        EditorialBrief(synthesis_level=3, target_duration_s=300.0),
    )

    assert "durata target finale richiesta=300.0s" in prompt
    assert "5.00 min" in prompt
    assert "non superare il target" in prompt


def test_editor_prompt_requests_compact_json_for_large_transcripts() -> None:
    prompt = build_editor_prompt(
        [
            ClipTranscription(
                clip_id=1,
                source_path=None,
                duration=500.0,
                segments=[
                    Segment(
                        id=index,
                        clip_id=1,
                        start=float(index),
                        end=float(index + 1),
                        text=f"Segmento informativo {index}",
                        words=[WordToken(word="Segmento", start=float(index), end=float(index) + 0.3)],
                    )
                    for index in range(1, 81)
                ],
            )
        ],
        EditorialBrief(synthesis_level=2),
    )

    assert "JSON compatto" in prompt
    assert "synthesis_script=null" in prompt
    assert "reason <= 90 caratteri" in prompt


def test_narrative_map_rejects_excessive_synthesis_for_long_transcript() -> None:
    transcription = ClipTranscription(
        clip_id=1,
        source_path=None,
        duration=100.0,
        segments=[
            Segment(
                id=index,
                clip_id=1,
                start=float((index - 1) * 10),
                end=float(index * 10),
                text=f"Passaggio informativo {index}",
                words=[WordToken(word="Passaggio", start=float((index - 1) * 10), end=float((index - 1) * 10 + 1))],
            )
            for index in range(1, 7)
        ],
    )
    narrative_map = NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=1,
                playback_position=0,
                topic_summary="Sintesi eccessiva",
                role="development",
                segments=[
                    NarrativeSegment(
                        segment_id=1,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason="Tiene solo un passaggio",
                    )
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Mini compilation troppo corta.",
        estimated_duration_s=10.0,
    )

    with pytest.raises(NarrativeValidationError, match="excessive_synthesis"):
        validate_narrative_map(
            narrative_map,
            [transcription],
            brief=EditorialBrief(synthesis_level=3),
        )


def test_narrative_map_allows_shorter_edit_when_target_duration_is_explicit() -> None:
    transcription = ClipTranscription(
        clip_id=1,
        source_path=None,
        duration=100.0,
        segments=[
            Segment(
                id=index,
                clip_id=1,
                start=float((index - 1) * 10),
                end=float(index * 10),
                text=f"Passaggio informativo {index}",
                words=[WordToken(word="Passaggio", start=float((index - 1) * 10), end=float((index - 1) * 10 + 1))],
            )
            for index in range(1, 7)
        ],
    )
    narrative_map = NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=1,
                playback_position=0,
                topic_summary="Target breve",
                role="development",
                segments=[
                    NarrativeSegment(
                        segment_id=1,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason="Target duration esplicito",
                    )
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Clip breve richiesta.",
        estimated_duration_s=10.0,
    )

    validate_narrative_map(
        narrative_map,
        [transcription],
        brief=EditorialBrief(synthesis_level=3, target_duration_s=12.0),
    )


def test_narrative_map_rejects_edit_far_above_explicit_target_duration() -> None:
    transcription = ClipTranscription(
        clip_id=1,
        source_path=None,
        duration=600.0,
        segments=[
            Segment(
                id=index,
                clip_id=1,
                start=float((index - 1) * 60),
                end=float(index * 60),
                text=f"Passaggio informativo {index}",
                words=[WordToken(word="Passaggio", start=float((index - 1) * 60), end=float((index - 1) * 60 + 1))],
            )
            for index in range(1, 7)
        ],
    )
    narrative_map = NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=1,
                playback_position=0,
                topic_summary="Target ignorato",
                role="development",
                segments=[
                    NarrativeSegment(
                        segment_id=index,
                        importance=SegmentImportance.VALUABLE,
                        keep_verbatim=True,
                        reason="Tiene troppo materiale",
                    )
                    for index in range(1, 7)
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Ignora target durata.",
        estimated_duration_s=360.0,
    )

    with pytest.raises(NarrativeValidationError, match="target_duration_exceeded"):
        validate_narrative_map(
            narrative_map,
            [transcription],
            brief=EditorialBrief(synthesis_level=3, target_duration_s=180.0),
        )


def test_narrative_map_rejects_edit_more_than_ten_percent_above_explicit_target_duration() -> None:
    transcription = ClipTranscription(
        clip_id=1,
        source_path=None,
        duration=392.0,
        segments=[
            Segment(
                id=index,
                clip_id=1,
                start=float((index - 1) * 56),
                end=float(index * 56),
                text=f"Passaggio informativo {index}",
                words=[WordToken(word="Passaggio", start=float((index - 1) * 56), end=float((index - 1) * 56 + 1))],
            )
            for index in range(1, 8)
        ],
    )
    narrative_map = NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=1,
                playback_position=0,
                topic_summary="Target quasi ignorato",
                role="development",
                segments=[
                    NarrativeSegment(
                        segment_id=index,
                        importance=SegmentImportance.VALUABLE if index <= 6 else SegmentImportance.REDUNDANT,
                        keep_verbatim=index <= 6,
                        reason="Tiene ancora troppo materiale",
                    )
                    for index in range(1, 8)
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Resta oltre la tolleranza del target.",
        estimated_duration_s=336.0,
    )

    with pytest.raises(NarrativeValidationError, match="target_duration_exceeded"):
        validate_narrative_map(
            narrative_map,
            [transcription],
            brief=EditorialBrief(synthesis_level=3, target_duration_s=300.0),
        )


def test_narrative_map_does_not_require_llm_zoom_flag_when_pause_detection_is_enabled() -> None:
    narrative_map = _map()

    validate_narrative_map(narrative_map, [_clip(1, 10)], EditingOptions(enable_creative_zoom=True))


def test_narrative_map_accepts_creative_zoom_flag_on_retained_important_segment() -> None:
    narrative_map = _map()
    narrative_map.ordered_clips[0].segments[0].creative_zoom = True

    validate_narrative_map(narrative_map, [_clip(1, 10)], EditingOptions(enable_creative_zoom=True))


def test_narrative_map_rejects_unknown_clip_id() -> None:
    with pytest.raises(NarrativeValidationError, match="clip_id"):
        validate_narrative_map(_map(clip_id=99), [_clip(1, 10)])


def test_narrative_map_rejects_unknown_segment_id() -> None:
    with pytest.raises(NarrativeValidationError, match="segment_id"):
        validate_narrative_map(_map(segment_id=99), [_clip(1, 10)])


def test_narrative_map_normalizes_duplicate_playback_positions() -> None:
    narrative_map = NarrativeMap(
        ordered_clips=[
            _map(clip_id=1, segment_id=10, playback_position=28820).ordered_clips[0],
            _map(clip_id=2, segment_id=20, playback_position=28820).ordered_clips[0],
        ],
        filler_words=[],
        editorial_strategy="Ordine narrativo",
        estimated_duration_s=2.4,
    )

    validate_narrative_map(narrative_map, [_clip(1, 10), _clip(2, 20)])

    assert [clip.playback_position for clip in narrative_map.ordered_clips] == [0, 1]


def test_narrative_map_rejects_rewritten_essential_audio_segments() -> None:
    narrative_map = _map()
    narrative_map.ordered_clips[0].segments[0].keep_verbatim = False
    narrative_map.ordered_clips[0].segments[0].synthesis_text = "Una frase nuova non presente nell'audio."

    with pytest.raises(NarrativeValidationError, match="source audio"):
        validate_narrative_map(narrative_map, [_clip(1, 10)])


def test_narrative_map_rejects_synthesis_text_on_verbatim_segments() -> None:
    narrative_map = _map()
    narrative_map.ordered_clips[0].segments[0].synthesis_text = "Questo testo non coincide con l'audio."

    with pytest.raises(NarrativeValidationError, match="synthesis_text"):
        validate_narrative_map(narrative_map, [_clip(1, 10)])


def test_narrative_map_rejects_opening_without_required_subject_context() -> None:
    transcriptions = [
        ClipTranscription(
            clip_id=1,
            source_path=None,
            duration=4.0,
            segments=[
                Segment(
                    id=10,
                    clip_id=1,
                    start=0.0,
                    end=2.0,
                    text="Tommaso Foti e l'attuale ministro per gli affari europei",
                    words=[WordToken(word="Tommaso", start=0.0, end=0.4), WordToken(word="Foti", start=0.5, end=0.8)],
                )
            ],
        ),
        ClipTranscription(
            clip_id=2,
            source_path=None,
            duration=4.0,
            segments=[
                Segment(
                    id=20,
                    clip_id=2,
                    start=0.0,
                    end=3.0,
                    text="Viene eletto consigliere comunale a Piacenza nel 1980",
                    words=[WordToken(word="Viene", start=0.0, end=0.3), WordToken(word="1980", start=2.6, end=3.0)],
                )
            ],
        ),
    ]
    narrative_map = NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=2,
                playback_position=0,
                topic_summary="Carriera politica",
                role="introduction",
                segments=[
                    NarrativeSegment(
                        segment_id=20,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason="Inizio carriera",
                    )
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Apri con la carriera politica.",
        estimated_duration_s=3.0,
    )

    with pytest.raises(NarrativeValidationError) as exc_info:
        validate_narrative_map(narrative_map, transcriptions)

    message = str(exc_info.value)
    assert "subjectless_opening" in message
    assert "clip_id=2" in message
    assert "segment_id=20" in message
    assert "Candidate subject openings" in message
    assert "clip_id=1 segment_id=10" in message
    assert "Tommaso Foti" in message


def test_narrative_map_rejects_non_monotonic_segments_within_same_clip() -> None:
    transcription = ClipTranscription(
        clip_id=1,
        source_path=None,
        duration=30.0,
        segments=[
            Segment(
                id=1,
                clip_id=1,
                start=0.0,
                end=7.68,
                text="Viene eletto consigliere comunale a Piacenza nel 1980 e da allora ad oggi ha solo ricoperto",
                words=[WordToken(word="Viene", start=0.0, end=0.3)],
            ),
            Segment(
                id=2,
                clip_id=1,
                start=7.68,
                end=13.64,
                text="incarichi politici, con sei legislature consecutive alla Camera dei Deputati, sempre nella stessa",
                words=[WordToken(word="incarichi", start=7.68, end=8.1)],
            ),
            Segment(
                id=3,
                clip_id=1,
                start=13.64,
                end=19.94,
                text="area politica. Il direttore esecutivo e Italo Bocchino",
                words=[WordToken(word="area", start=13.64, end=14.0)],
            ),
        ],
    )
    narrative_map = NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=1,
                playback_position=0,
                topic_summary="Carriera politica",
                role="introduction",
                segments=[
                    NarrativeSegment(
                        segment_id=3,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason="Apre con il direttore",
                    ),
                    NarrativeSegment(
                        segment_id=1,
                        importance=SegmentImportance.VALUABLE,
                        keep_verbatim=True,
                        reason="Recupera inizio carriera",
                    ),
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Ordine non cronologico dentro la clip.",
        estimated_duration_s=20.0,
    )

    with pytest.raises(NarrativeValidationError) as exc_info:
        validate_narrative_map(narrative_map, [transcription])

    message = str(exc_info.value)
    assert "non_monotonic_clip_segments" in message
    assert "clip_id=1" in message
    assert "Current retained order: [3, 1]" in message
    assert "Required source order: [1, 3]" in message
    assert "Do not place segment_id=1 after segment_id=3" in message


def test_narrative_map_rejects_opening_from_sentence_fragment() -> None:
    transcription = ClipTranscription(
        clip_id=1,
        source_path=None,
        duration=30.0,
        segments=[
            Segment(
                id=2,
                clip_id=1,
                start=7.68,
                end=13.64,
                text="incarichi politici, con sei legislature consecutive alla Camera dei Deputati, sempre nella stessa",
                words=[WordToken(word="incarichi", start=7.68, end=8.1)],
            ),
            Segment(
                id=3,
                clip_id=1,
                start=13.64,
                end=19.94,
                text="area politica. Il direttore esecutivo e Italo Bocchino",
                words=[WordToken(word="area", start=13.64, end=14.0)],
            ),
        ],
    )
    narrative_map = NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=1,
                playback_position=0,
                topic_summary="Carriera politica",
                role="introduction",
                segments=[
                    NarrativeSegment(
                        segment_id=3,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason="Apre con un frammento",
                    )
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Apre da un frammento della frase precedente.",
        estimated_duration_s=6.0,
    )

    with pytest.raises(NarrativeValidationError) as exc_info:
        validate_narrative_map(narrative_map, [transcription])

    message = str(exc_info.value)
    assert "opening_context_fragment" in message
    assert "clip_id=1" in message
    assert "segment_id=3" in message
    assert "preceding segment_id=2" in message
    assert "Do not start" in message


def test_cut_safety_repair_includes_preceding_opening_context_segment() -> None:
    transcription = ClipTranscription(
        clip_id=1,
        source_path=None,
        duration=12.0,
        segments=[
            Segment(
                id=1,
                clip_id=1,
                start=0.0,
                end=5.0,
                text="Tommaso Foti viene eletto consigliere comunale a Piacenza e da allora ricopre",
                words=[WordToken(word="Tommaso", start=0.0, end=0.4), WordToken(word="ricopre", start=4.6, end=5.0)],
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
    narrative_map = NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=1,
                playback_position=0,
                topic_summary="Carriera politica",
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
        editorial_strategy="Apre sulla carriera politica.",
        estimated_duration_s=3.0,
    )

    repaired = repair_narrative_map_for_cut_safety(narrative_map, [transcription])

    validate_narrative_map(repaired, [transcription])
    repaired_segments = repaired.ordered_clips[0].segments
    assert [segment.segment_id for segment in repaired_segments] == [1, 2]
    assert repaired_segments[0].importance == SegmentImportance.VALUABLE
    assert repaired_segments[0].keep_verbatim is True
    assert "cut safety" in repaired_segments[0].reason


def test_cut_safety_removes_duplicate_retained_source_segments() -> None:
    transcription = ClipTranscription(
        clip_id=1,
        source_path=None,
        duration=4.0,
        segments=[
            Segment(
                id=1,
                clip_id=1,
                start=0.0,
                end=2.0,
                text="Frase importante da tenere una sola volta.",
                words=[WordToken(word="Frase", start=0.0, end=0.3), WordToken(word="volta.", start=1.6, end=2.0)],
            )
        ],
    )
    narrative_map = NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=1,
                playback_position=0,
                topic_summary="Prima occorrenza",
                role="development",
                segments=[
                    NarrativeSegment(
                        segment_id=1,
                        importance=SegmentImportance.VALUABLE,
                        keep_verbatim=True,
                        reason="Mantieni il passaggio.",
                    )
                ],
            ),
            NarrativeClip(
                clip_id=1,
                playback_position=1,
                topic_summary="Duplicato",
                role="supporting",
                segments=[
                    NarrativeSegment(
                        segment_id=1,
                        importance=SegmentImportance.VALUABLE,
                        keep_verbatim=True,
                        reason="Duplicato proposto dalla review.",
                    )
                ],
            ),
        ],
        filler_words=[],
        editorial_strategy="Evita duplicati.",
        estimated_duration_s=2.0,
    )

    repaired = repair_narrative_map_for_cut_safety(narrative_map, [transcription])

    retained_ids = [
        segment.segment_id
        for clip in repaired.ordered_clips
        for segment in clip.segments
        if segment.keep_verbatim and segment.importance != SegmentImportance.REDUNDANT
    ]
    assert retained_ids == [1]
    assert len(repaired.ordered_clips) == 1


def test_narrative_map_rejects_detached_internal_sentence_fragment() -> None:
    transcriptions = [
        ClipTranscription(
            clip_id=4,
            source_path=None,
            duration=30.0,
            segments=[
                Segment(
                    id=2,
                    clip_id=4,
                    start=12.0,
                    end=18.8,
                    text="Tommaso Foti, l'attuale ministro per gli affari europei, e il proprietario unico del capitale sociale,",
                    words=[WordToken(word="Tommaso", start=12.0, end=12.4), WordToken(word="sociale,", start=18.2, end=18.8)],
                ),
                Segment(
                    id=3,
                    clip_id=4,
                    start=18.94,
                    end=28.0,
                    text="ma risulta come fiduciario della Fondazione Alleanza Nazionale, la vera proprietaria del Secolo d'Italia.",
                    words=[WordToken(word="ma", start=18.94, end=19.1), WordToken(word="Italia.", start=27.5, end=28.0)],
                ),
            ],
        ),
        ClipTranscription(
            clip_id=5,
            source_path=None,
            duration=5.0,
            segments=[
                Segment(
                    id=1,
                    clip_id=5,
                    start=0.0,
                    end=2.0,
                    text="Un altro passaggio interrompe il contesto.",
                    words=[WordToken(word="Un", start=0.0, end=0.2), WordToken(word="contesto.", start=1.6, end=2.0)],
                )
            ],
        ),
    ]
    narrative_map = NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=4,
                playback_position=0,
                topic_summary="Soggetto",
                role="introduction",
                segments=[
                    NarrativeSegment(
                        segment_id=2,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason="Presenta Tommaso Foti.",
                    )
                ],
            ),
            NarrativeClip(
                clip_id=5,
                playback_position=1,
                topic_summary="Interruzione",
                role="supporting",
                segments=[
                    NarrativeSegment(
                        segment_id=1,
                        importance=SegmentImportance.VALUABLE,
                        keep_verbatim=True,
                        reason="Passaggio intermedio.",
                    )
                ],
            ),
            NarrativeClip(
                clip_id=4,
                playback_position=2,
                topic_summary="Fiduciario",
                role="development",
                segments=[
                    NarrativeSegment(
                        segment_id=3,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason="Spiega la fiduciaria.",
                    )
                ],
            ),
        ],
        filler_words=[],
        editorial_strategy="Segmento dipendente staccato dal soggetto.",
        estimated_duration_s=18.0,
    )

    with pytest.raises(NarrativeValidationError, match="detached_context_fragment"):
        validate_narrative_map(narrative_map, transcriptions)


def test_cut_safety_repair_moves_internal_context_fragment_next_to_subject_segment() -> None:
    transcriptions = [
        ClipTranscription(
            clip_id=4,
            source_path=None,
            duration=30.0,
            segments=[
                Segment(
                    id=2,
                    clip_id=4,
                    start=12.0,
                    end=18.8,
                    text="Tommaso Foti, l'attuale ministro per gli affari europei, e il proprietario unico del capitale sociale,",
                    words=[WordToken(word="Tommaso", start=12.0, end=12.4), WordToken(word="sociale,", start=18.2, end=18.8)],
                ),
                Segment(
                    id=3,
                    clip_id=4,
                    start=18.94,
                    end=28.0,
                    text="ma risulta come fiduciario della Fondazione Alleanza Nazionale, la vera proprietaria del Secolo d'Italia.",
                    words=[WordToken(word="ma", start=18.94, end=19.1), WordToken(word="Italia.", start=27.5, end=28.0)],
                ),
            ],
        ),
        ClipTranscription(
            clip_id=5,
            source_path=None,
            duration=5.0,
            segments=[
                Segment(
                    id=1,
                    clip_id=5,
                    start=0.0,
                    end=2.0,
                    text="Un altro passaggio interrompe il contesto.",
                    words=[WordToken(word="Un", start=0.0, end=0.2), WordToken(word="contesto.", start=1.6, end=2.0)],
                )
            ],
        ),
    ]
    narrative_map = NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=4,
                playback_position=0,
                topic_summary="Soggetto",
                role="introduction",
                segments=[
                    NarrativeSegment(
                        segment_id=2,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason="Presenta Tommaso Foti.",
                    )
                ],
            ),
            NarrativeClip(
                clip_id=5,
                playback_position=1,
                topic_summary="Interruzione",
                role="supporting",
                segments=[
                    NarrativeSegment(
                        segment_id=1,
                        importance=SegmentImportance.VALUABLE,
                        keep_verbatim=True,
                        reason="Passaggio intermedio.",
                    )
                ],
            ),
            NarrativeClip(
                clip_id=4,
                playback_position=2,
                topic_summary="Fiduciario",
                role="development",
                segments=[
                    NarrativeSegment(
                        segment_id=3,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason="Spiega la fiduciaria.",
                    )
                ],
            ),
        ],
        filler_words=[],
        editorial_strategy="Segmento dipendente staccato dal soggetto.",
        estimated_duration_s=18.0,
    )

    repaired = repair_narrative_map_for_cut_safety(narrative_map, transcriptions)

    validate_narrative_map(repaired, transcriptions)
    assert [segment.segment_id for segment in repaired.ordered_clips[0].segments] == [2, 3]
    assert repaired.ordered_clips[1].clip_id == 5
    assert all(segment.segment_id != 3 for segment in repaired.ordered_clips[-1].segments)
    assert "cut safety" in repaired.ordered_clips[0].segments[1].reason


def test_narrative_map_rejects_cross_clip_norm_reference_before_antecedent() -> None:
    transcriptions = [
        ClipTranscription(
            clip_id=2,
            source_path=None,
            duration=24.0,
            segments=[
                _source_segment(
                    2,
                    1,
                    0.58,
                    23.30,
                    (
                        "La stessa norma poi ci dice che possono prendere i contributi pubblici "
                        "le imprese editrici il cui capitale sociale sia detenuto da una fondazione."
                    ),
                )
            ],
        ),
        ClipTranscription(
            clip_id=3,
            source_path=None,
            duration=18.0,
            segments=[
                _source_segment(
                    3,
                    1,
                    0.74,
                    17.22,
                    (
                        "Solo che la legge del 2016, la 198, ridefinendo la platea dei beneficiari "
                        "dei contributi pubblici ai giornali, ha escluso gli organi collegati ai partiti politici."
                    ),
                )
            ],
        ),
    ]
    narrative_map = NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=2,
                playback_position=0,
                topic_summary="Norma richiamata",
                role="development",
                segments=[
                    NarrativeSegment(
                        segment_id=1,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason="Spiega chi puo prendere contributi pubblici.",
                    )
                ],
            ),
            NarrativeClip(
                clip_id=3,
                playback_position=1,
                topic_summary="Legge 2016 198",
                role="supporting",
                segments=[
                    NarrativeSegment(
                        segment_id=1,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason="Nomina la legge di riferimento.",
                    )
                ],
            ),
        ],
        filler_words=[],
        editorial_strategy="Ordine errato: riferimento prima dell'antecedente.",
        estimated_duration_s=39.0,
    )

    with pytest.raises(NarrativeValidationError, match="missing_context_antecedent"):
        validate_narrative_map(narrative_map, transcriptions)


def test_cut_safety_repair_moves_cross_clip_norm_antecedent_before_reference() -> None:
    transcriptions = [
        ClipTranscription(
            clip_id=2,
            source_path=None,
            duration=24.0,
            segments=[
                _source_segment(
                    2,
                    1,
                    0.58,
                    23.30,
                    (
                        "La stessa norma poi ci dice che possono prendere i contributi pubblici "
                        "le imprese editrici il cui capitale sociale sia detenuto da una fondazione."
                    ),
                )
            ],
        ),
        ClipTranscription(
            clip_id=3,
            source_path=None,
            duration=18.0,
            segments=[
                _source_segment(
                    3,
                    1,
                    0.74,
                    17.22,
                    (
                        "Solo che la legge del 2016, la 198, ridefinendo la platea dei beneficiari "
                        "dei contributi pubblici ai giornali, ha escluso gli organi collegati ai partiti politici."
                    ),
                )
            ],
        ),
    ]
    narrative_map = NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=2,
                playback_position=0,
                topic_summary="Norma richiamata",
                role="development",
                segments=[
                    NarrativeSegment(
                        segment_id=1,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason="Spiega chi puo prendere contributi pubblici.",
                    )
                ],
            ),
            NarrativeClip(
                clip_id=3,
                playback_position=1,
                topic_summary="Legge 2016 198",
                role="supporting",
                segments=[
                    NarrativeSegment(
                        segment_id=1,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason="Nomina la legge di riferimento.",
                    )
                ],
            ),
        ],
        filler_words=[],
        editorial_strategy="Ripara l'ordine per preservare il riferimento.",
        estimated_duration_s=39.0,
    )

    repaired = repair_narrative_map_for_cut_safety(narrative_map, transcriptions)

    validate_narrative_map(repaired, transcriptions)
    assert [(clip.clip_id, clip.segments[0].segment_id) for clip in repaired.ordered_clips] == [(3, 1), (2, 1)]
    assert "cut safety" in repaired.ordered_clips[0].segments[0].reason


def test_cut_safety_repair_inserts_missing_cross_clip_norm_antecedent() -> None:
    transcriptions = [
        ClipTranscription(
            clip_id=2,
            source_path=None,
            duration=24.0,
            segments=[
                _source_segment(
                    2,
                    1,
                    0.58,
                    23.30,
                    (
                        "La stessa norma poi ci dice che possono prendere i contributi pubblici "
                        "le imprese editrici il cui capitale sociale sia detenuto da una fondazione."
                    ),
                )
            ],
        ),
        ClipTranscription(
            clip_id=3,
            source_path=None,
            duration=18.0,
            segments=[
                _source_segment(
                    3,
                    1,
                    0.74,
                    17.22,
                    (
                        "Solo che la legge del 2016, la 198, ridefinendo la platea dei beneficiari "
                        "dei contributi pubblici ai giornali, ha escluso gli organi collegati ai partiti politici."
                    ),
                )
            ],
        ),
    ]
    narrative_map = NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=2,
                playback_position=0,
                topic_summary="Norma richiamata",
                role="development",
                segments=[
                    NarrativeSegment(
                        segment_id=1,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason="Spiega chi puo prendere contributi pubblici.",
                    )
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Antecedente normativo omesso.",
        estimated_duration_s=23.0,
    )

    repaired = repair_narrative_map_for_cut_safety(narrative_map, transcriptions)

    validate_narrative_map(repaired, transcriptions)
    assert [(clip.clip_id, clip.segments[0].segment_id) for clip in repaired.ordered_clips] == [(3, 1), (2, 1)]
    inserted_segment = repaired.ordered_clips[0].segments[0]
    assert inserted_segment.importance == SegmentImportance.VALUABLE
    assert inserted_segment.keep_verbatim is True
    assert "cut safety" in inserted_segment.reason


def test_cut_safety_repair_prepends_best_subject_candidate_for_subjectless_opening() -> None:
    transcriptions = [
        ClipTranscription(
            clip_id=1,
            source_path=None,
            duration=30.0,
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
            source_path=None,
            duration=30.0,
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
    narrative_map = NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=1,
                playback_position=0,
                topic_summary="Carriera politica",
                role="introduction",
                segments=[
                    NarrativeSegment(
                        segment_id=1,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason="Apre con la carriera politica",
                    )
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Apre con la carriera.",
        estimated_duration_s=7.0,
    )

    repaired = repair_narrative_map_for_cut_safety(narrative_map, transcriptions)

    validate_narrative_map(repaired, transcriptions)
    assert repaired.ordered_clips[0].clip_id == 4
    assert repaired.ordered_clips[0].segments[0].segment_id == 3
    assert "cut safety" in repaired.ordered_clips[0].segments[0].reason
    retained_texts = [
        segment.text
        for segment in [
            next(
                source_segment
                for clip in transcriptions
                for source_segment in clip.segments
                if source_segment.clip_id == narrative_clip.clip_id and source_segment.id == narrative_segment.segment_id
            )
            for narrative_clip in repaired.ordered_clips
            for narrative_segment in narrative_clip.segments
            if narrative_segment.keep_verbatim and narrative_segment.importance != SegmentImportance.REDUNDANT
        ]
    ]
    assert retained_texts[:2] == [
        "Tommaso Foti, l'attuale ministro per gli affari europei",
        "Viene eletto consigliere comunale a Piacenza nel 1980",
    ]


def test_narrative_map_rejects_contextless_pronoun_opening() -> None:
    transcription = ClipTranscription(
        clip_id=1,
        source_path=None,
        duration=12.0,
        segments=[
            _source_segment(1, 1, 0.0, 3.0, "Siamo pronti? Buonasera, con chi siamo questa sera?"),
            _source_segment(1, 2, 3.2, 5.0, "Ve lo dividerete alla fine del gioco."),
        ],
    )
    narrative_map = NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=1,
                playback_position=0,
                topic_summary="Premio senza contesto",
                role="development",
                segments=[
                    NarrativeSegment(
                        segment_id=2,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason="Apre con il premio.",
                    )
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Apertura pronominale.",
        estimated_duration_s=2.0,
    )

    with pytest.raises(NarrativeValidationError) as exc_info:
        validate_narrative_map(narrative_map, [transcription])

    message = str(exc_info.value)
    assert "contextless_opening" in message
    assert "Ve lo dividerete" in message
    assert "clip_id=1 segment_id=1" in message


def test_cut_safety_repair_prepends_intro_candidate_for_contextless_opening() -> None:
    transcription = ClipTranscription(
        clip_id=1,
        source_path=None,
        duration=12.0,
        segments=[
            _source_segment(1, 1, 0.0, 3.0, "Siamo pronti? Buonasera, con chi siamo questa sera?"),
            _source_segment(1, 2, 3.2, 5.0, "Ve lo dividerete alla fine del gioco."),
        ],
    )
    narrative_map = NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=1,
                playback_position=0,
                topic_summary="Premio senza contesto",
                role="development",
                segments=[
                    NarrativeSegment(
                        segment_id=2,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason="Apre con il premio.",
                    )
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Apertura pronominale.",
        estimated_duration_s=2.0,
    )

    repaired = repair_narrative_map_for_cut_safety(narrative_map, [transcription])

    validate_narrative_map(repaired, [transcription])
    assert [segment.segment_id for segment in repaired.ordered_clips[0].segments] == [1, 2]
    assert "cut safety" in repaired.ordered_clips[0].segments[0].reason


def test_cut_safety_repair_replaces_same_clip_contextless_opening_with_later_intro() -> None:
    transcription = ClipTranscription(
        clip_id=2,
        source_path=None,
        duration=18.0,
        segments=[
            _source_segment(2, 1, 0.0, 1.2, "Ok, siamo pronti."),
            _source_segment(2, 3, 2.0, 3.5, "Piacere, Giulia."),
            _source_segment(2, 4, 4.0, 6.0, "Oggi parliamo del gioco."),
        ],
    )
    narrative_map = NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=2,
                playback_position=0,
                topic_summary="Apertura senza contesto",
                role="introduction",
                segments=[
                    NarrativeSegment(
                        segment_id=1,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason="Apertura scelta dal modello.",
                    ),
                    NarrativeSegment(
                        segment_id=4,
                        importance=SegmentImportance.VALUABLE,
                        keep_verbatim=True,
                        reason="Segmento informativo.",
                    ),
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Apertura da riparare.",
        estimated_duration_s=3.2,
    )

    repaired = repair_narrative_map_for_cut_safety(narrative_map, [transcription])

    validate_narrative_map(repaired, [transcription])
    retained_ids = [
        segment.segment_id
        for segment in repaired.ordered_clips[0].segments
        if segment.keep_verbatim and segment.importance != SegmentImportance.REDUNDANT
    ]
    assert retained_ids == [3, 4]
    assert repaired.ordered_clips[0].segments[0].segment_id == 1
    assert repaired.ordered_clips[0].segments[0].importance == SegmentImportance.REDUNDANT
    assert "cut safety" in repaired.ordered_clips[0].segments[0].reason


def test_cut_safety_repair_replaces_ok_opening_with_best_later_intro_candidate() -> None:
    transcription = ClipTranscription(
        clip_id=2,
        source_path=None,
        duration=70.0,
        segments=[
            _source_segment(2, 1, 0.0, 1.2, "Ok, siamo pronti."),
            _source_segment(2, 3, 2.0, 3.5, "Piacere, Giulia."),
            _source_segment(2, 22, 22.0, 23.0, "Ma Londra non e in America!"),
            _source_segment(2, 35, 35.0, 36.0, "Cristoforo Colombo o..."),
            _source_segment(2, 60, 60.0, 61.0, "Van Gogh!"),
        ],
    )
    narrative_map = NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=2,
                playback_position=0,
                topic_summary="Apertura senza contesto",
                role="introduction",
                segments=[
                    NarrativeSegment(
                        segment_id=1,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason="Apertura scelta dal modello.",
                    )
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Apertura da riparare.",
        estimated_duration_s=1.2,
    )

    repaired = repair_narrative_map_for_cut_safety(narrative_map, [transcription])

    validate_narrative_map(repaired, [transcription])
    retained_ids = [
        segment.segment_id
        for segment in repaired.ordered_clips[0].segments
        if segment.keep_verbatim and segment.importance != SegmentImportance.REDUNDANT
    ]
    assert retained_ids == [3]
    assert repaired.ordered_clips[0].segments[0].segment_id == 1
    assert repaired.ordered_clips[0].segments[0].importance == SegmentImportance.REDUNDANT


def test_cut_safety_repair_prepends_intro_to_exact_first_same_clip_occurrence() -> None:
    transcription = ClipTranscription(
        clip_id=1,
        source_path=None,
        duration=18.0,
        segments=[
            _source_segment(1, 1, 0.0, 2.0, "Siamo pronti? Allora, buonasera a tutti, con chi siamo questa sera?"),
            _source_segment(1, 2, 2.2, 3.6, "Con Elisa Zacchini, buonasera."),
            _source_segment(1, 4, 4.0, 6.0, "Si, siamo di Tarquigna. Elisa Zacchini, buonasera a tutti."),
            _source_segment(1, 29, 12.0, 14.0, "Laura Piroli, si fiera di me per favore questa sera."),
        ],
    )
    narrative_map = NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=1,
                playback_position=0,
                topic_summary="Apertura scelta",
                role="introduction",
                segments=[
                    NarrativeSegment(
                        segment_id=4,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason="Apertura scelta dal modello.",
                    )
                ],
            ),
            NarrativeClip(
                clip_id=1,
                playback_position=1,
                topic_summary="Secondo blocco stesso clip",
                role="development",
                segments=[
                    NarrativeSegment(
                        segment_id=29,
                        importance=SegmentImportance.VALUABLE,
                        keep_verbatim=True,
                        reason="Segmento successivo.",
                    )
                ],
            ),
        ],
        filler_words=[],
        editorial_strategy="Apertura contextless con clip duplicato.",
        estimated_duration_s=4.0,
    )

    repaired = repair_narrative_map_for_cut_safety(narrative_map, [transcription])

    validate_narrative_map(repaired, [transcription])
    retained_first_clip_ids = [
        segment.segment_id
        for segment in repaired.ordered_clips[0].segments
        if segment.keep_verbatim and segment.importance != SegmentImportance.REDUNDANT
    ]
    assert retained_first_clip_ids[:2] == [2, 4]
    assert "cut safety" in repaired.ordered_clips[0].segments[0].reason


def test_cut_safety_repair_leaves_unknown_segments_to_validation() -> None:
    transcription = _clip(1, 10)
    narrative_map = _map(segment_id=99)

    repaired = repair_narrative_map_for_cut_safety(narrative_map, [transcription])

    with pytest.raises(NarrativeValidationError, match="segment_id"):
        validate_narrative_map(repaired, [transcription])
