from __future__ import annotations

import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from autocut_ai.config import AppSettings
from autocut_ai.narrative import NarrativePlanner, NarrativePlanningError
from autocut_ai.ollama_runtime import OllamaTimeoutError, structured_chat
from autocut_ai.proposal_review import ProposalReviewer
from autocut_ai.schemas import (
    ClipTranscription,
    EditDecision,
    EditDecisionList,
    EditorialBrief,
    NarrativeClip,
    NarrativeMap,
    NarrativeSegment,
    ProposalReview,
    Segment,
    SegmentImportance,
    WordToken,
)


def _transcription() -> ClipTranscription:
    return ClipTranscription(
        clip_id=1,
        source_path=None,
        duration=1.0,
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


def _many_segment_transcription(count: int) -> ClipTranscription:
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
                text=f"Segmento informativo {index}",
                words=[WordToken(word="Segmento", start=float(index - 1), end=float(index - 1) + 0.4)],
            )
            for index in range(1, count + 1)
        ],
    )


def _long_segment_transcription(count: int, duration_s: float) -> ClipTranscription:
    return ClipTranscription(
        clip_id=1,
        source_path=None,
        duration=float(count) * duration_s,
        segments=[
            Segment(
                id=index,
                clip_id=1,
                start=float(index - 1) * duration_s,
                end=float(index) * duration_s,
                text=f"Tommaso Foti passaggio informativo {index}.",
                words=[
                    WordToken(word="Tommaso", start=float(index - 1) * duration_s, end=float(index - 1) * duration_s + 0.4)
                ],
            )
            for index in range(1, count + 1)
        ],
    )


def _narrative_map() -> NarrativeMap:
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
                        reason="Apre con il soggetto.",
                    )
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Apertura chiara.",
        estimated_duration_s=1.0,
    )


def _edl() -> EditDecisionList:
    return EditDecisionList(
        estimated_duration_s=1.0,
        decisions=[
            EditDecision(
                clip_id=1,
                source_start=0.0,
                source_end=1.0,
                output_order=0,
                reason="Apre con il soggetto.",
                included_text="Tommaso Foti introduce il tema.",
            )
        ],
    )


def _settings(tmp_path: Path) -> AppSettings:
    return AppSettings(
        data_dir=tmp_path / "data",
        database_path=tmp_path / "data" / "jobs.sqlite3",
        ollama_request_timeout_s=12.0,
        ollama_num_ctx=4096,
        ollama_num_predict=2048,
        ollama_temperature=0.2,
        ollama_think=False,
        ollama_run_in_subprocess=False,
        narrative_hierarchical_threshold_segments=999,
        narrative_chunk_max_segments=80,
    )


def test_narrative_planner_uses_bounded_ollama_runtime_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            captured["client_kwargs"] = kwargs

        def chat(self, **kwargs: object) -> dict[str, dict[str, str]]:
            captured["chat_kwargs"] = kwargs
            return {"message": {"content": _narrative_map().model_dump_json()}}

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=FakeClient))

    planner = NarrativePlanner(_settings(tmp_path))
    planner.generate([_transcription()], EditorialBrief(), "qwen3:8b")

    assert captured["client_kwargs"] == {"timeout": 12.0}
    chat_kwargs = captured["chat_kwargs"]
    assert isinstance(chat_kwargs, dict)
    assert chat_kwargs["think"] is False
    assert chat_kwargs["keep_alive"] == "5m"
    assert chat_kwargs["options"] == {"num_ctx": 4096, "num_predict": 2048, "temperature": 0.2}


def test_narrative_planner_turns_ollama_timeout_into_clear_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TimeoutClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def chat(self, **_kwargs: object) -> object:
            raise httpx.TimeoutException("read timeout")

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=TimeoutClient))

    planner = NarrativePlanner(_settings(tmp_path), max_retries=0)

    with pytest.raises(NarrativePlanningError, match="Timeout Ollama dopo 12s"):
        planner.generate([_transcription()], EditorialBrief(), "qwen3:8b")


def test_narrative_planner_retries_truncated_json_with_larger_generation_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options_seen: list[dict[str, object]] = []

    class RetryClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def chat(self, **kwargs: object) -> dict[str, dict[str, str]]:
            options = kwargs["options"]
            assert isinstance(options, dict)
            options_seen.append(options)
            if len(options_seen) == 1:
                return {"message": {"content": '{"ordered_clips": [{"clip_id":'}}
            return {"message": {"content": _narrative_map().model_dump_json()}}

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=RetryClient))

    planner = NarrativePlanner(_settings(tmp_path), max_retries=1)
    planner.generate([_transcription()], EditorialBrief(), "qwen3:8b")

    assert options_seen[0]["num_predict"] == 2048
    assert options_seen[1]["num_predict"] == 4096


def test_large_transcript_uses_chunked_planning_instead_of_one_huge_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class ChunkingClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def chat(self, **kwargs: object) -> dict[str, dict[str, str]]:
            calls.append(kwargs)
            schema = kwargs["format"]
            assert isinstance(schema, dict)
            prompt = kwargs["messages"][0]["content"]
            assert isinstance(prompt, str)
            if schema.get("title") == "NarrativeOrderPlan":
                block_ids = [
                    int(line.split()[1].rstrip(":"))
                    for line in prompt.splitlines()
                    if line.startswith("BLOCK ")
                ]
                return {
                    "message": {
                        "content": {
                            "ordered_blocks": [
                                {
                                    "block_id": block_id,
                                    "playback_position": position,
                                    "role": "development",
                                    "topic_summary": f"Blocco {block_id}",
                                }
                                for position, block_id in enumerate(block_ids)
                            ],
                            "editorial_strategy": "Ordine globale compatto.",
                        }.__repr__().replace("'", '"')
                    }
                }

            if schema.get("title") == "CompactNarrativePlan":
                segment_ids = [
                    segment_id
                    for segment_id in range(1, 6)
                    if f"segment_id={segment_id} " in prompt or f"segment_id={segment_id}\n" in prompt
                ]
                return {
                    "message": {
                        "content": {
                            "clip_id": 1,
                            "topic_summary": "Chunk",
                            "role": "development",
                            "essential_segment_ids": segment_ids,
                            "valuable_segment_ids": [],
                            "optional_segment_ids": [],
                            "redundant_segment_ids": [],
                            "filler_words": [],
                            "editorial_strategy": "Chunk compatto.",
                            "estimated_duration_s": float(len(segment_ids)),
                        }.__repr__().replace("'", '"')
                    }
                }

            segment_ids = [
                segment_id
                for segment_id in range(1, 6)
                if f"segment_id={segment_id} " in prompt or f"segment_id={segment_id}\n" in prompt
            ]
            narrative_map = NarrativeMap(
                ordered_clips=[
                    NarrativeClip(
                        clip_id=1,
                        playback_position=0,
                        topic_summary="Chunk",
                        role="development",
                        segments=[
                            NarrativeSegment(
                                segment_id=segment_id,
                                importance=SegmentImportance.ESSENTIAL,
                                keep_verbatim=True,
                                reason="Mantieni",
                            )
                            for segment_id in segment_ids
                        ],
                    )
                ],
                filler_words=[],
                editorial_strategy="Chunk locale.",
                estimated_duration_s=float(len(segment_ids)),
            )
            return {"message": {"content": narrative_map.model_dump_json()}}

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=ChunkingClient))
    settings = _settings(tmp_path).model_copy(
        update={
            "narrative_hierarchical_threshold_segments": 3,
            "narrative_chunk_max_segments": 2,
        }
    )

    result = NarrativePlanner(settings).generate([_many_segment_transcription(5)], EditorialBrief(), "qwen3:8b")

    map_calls = [call for call in calls if call["format"]["title"] == "CompactNarrativePlan"]  # type: ignore[index]
    order_calls = [call for call in calls if call["format"]["title"] == "NarrativeOrderPlan"]  # type: ignore[index]
    assert len(map_calls) == 3
    assert len(order_calls) == 1
    assert [
        segment.segment_id
        for clip in sorted(result.ordered_clips, key=lambda item: item.playback_position)
        for segment in clip.segments
    ] == [1, 2, 3, 4, 5]
    assert all(str(call["messages"][0]["content"]).count("segment_id=") <= 2 for call in map_calls)  # type: ignore[index]


def test_chunked_planning_enforces_duration_budget_only_on_final_map(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    map_call_count = 0

    class ChunkBudgetClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def chat(self, **kwargs: object) -> dict[str, dict[str, str]]:
            nonlocal map_call_count
            schema = kwargs["format"]
            assert isinstance(schema, dict)
            if schema.get("title") == "NarrativeOrderPlan":
                return {
                    "message": {
                        "content": (
                            '{"ordered_blocks":['
                            '{"block_id":0,"playback_position":0,"role":"introduction","topic_summary":"Primo blocco"},'
                            '{"block_id":1,"playback_position":1,"role":"development","topic_summary":"Secondo blocco"}'
                            '],"editorial_strategy":"Ordine globale."}'
                        )
                    }
                }

            if schema.get("title") == "CompactNarrativePlan":
                map_call_count += 1
                essential_ids = [1, 2] if map_call_count == 1 else [5, 6, 7, 8]
                redundant_ids = [3, 4] if map_call_count == 1 else []
                return {
                    "message": {
                        "content": {
                            "clip_id": 1,
                            "topic_summary": "Chunk",
                            "role": "development",
                            "essential_segment_ids": essential_ids,
                            "valuable_segment_ids": [],
                            "optional_segment_ids": [],
                            "redundant_segment_ids": redundant_ids,
                            "filler_words": [],
                            "editorial_strategy": "Chunk locale.",
                            "estimated_duration_s": float(len(essential_ids) * 10),
                        }.__repr__().replace("'", '"')
                    }
                }

            map_call_count += 1
            segment_ids = [1, 2] if map_call_count == 1 else [5, 6, 7, 8]
            narrative_map = NarrativeMap(
                ordered_clips=[
                    NarrativeClip(
                        clip_id=1,
                        playback_position=0,
                        topic_summary="Chunk",
                        role="development",
                        segments=[
                            NarrativeSegment(
                                segment_id=segment_id,
                                importance=SegmentImportance.ESSENTIAL,
                                keep_verbatim=True,
                                reason="Mantieni",
                            )
                            for segment_id in segment_ids
                        ],
                    )
                ],
                filler_words=[],
                editorial_strategy="Chunk locale.",
                estimated_duration_s=float(len(segment_ids) * 10),
            )
            return {"message": {"content": narrative_map.model_dump_json()}}

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=ChunkBudgetClient))
    settings = _settings(tmp_path).model_copy(
        update={
            "narrative_hierarchical_threshold_segments": 5,
            "narrative_chunk_max_segments": 4,
        }
    )

    result = NarrativePlanner(settings).generate(
        [_long_segment_transcription(count=8, duration_s=10.0)],
        EditorialBrief(synthesis_level=2),
        "qwen3:8b",
    )

    retained_ids = [
        segment.segment_id
        for clip in sorted(result.ordered_clips, key=lambda item: item.playback_position)
        for segment in clip.segments
        if segment.keep_verbatim
    ]
    assert retained_ids == [1, 2, 5, 6, 7, 8]


def test_compact_plan_duplicate_ids_are_normalized_conservatively(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DuplicateCompactClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def chat(self, **kwargs: object) -> dict[str, dict[str, str]]:
            schema = kwargs["format"]
            assert isinstance(schema, dict)
            if schema.get("title") == "NarrativeOrderPlan":
                return {
                    "message": {
                        "content": (
                            '{"ordered_blocks":['
                            '{"block_id":0,"playback_position":0,"role":"development","topic_summary":"Chunk"}'
                            '],"editorial_strategy":"Ordine globale."}'
                        )
                    }
                }
            return {
                "message": {
                    "content": (
                        '{"clip_id":1,"topic_summary":"Chunk","role":"development",'
                        '"essential_segment_ids":[1,2,2],'
                        '"valuable_segment_ids":[3],'
                        '"optional_segment_ids":[3],'
                        '"redundant_segment_ids":[2,5,999],'
                        '"filler_words":[],'
                        '"editorial_strategy":"Classificazione con duplicati.",'
                        '"estimated_duration_s":4.0}'
                    )
                }
            }

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=DuplicateCompactClient))
    settings = _settings(tmp_path).model_copy(
        update={
            "narrative_hierarchical_threshold_segments": 3,
            "narrative_chunk_max_segments": 5,
        }
    )

    result = NarrativePlanner(settings).generate([_many_segment_transcription(5)], EditorialBrief(), "qwen3:8b")
    segments = result.ordered_clips[0].segments
    importance_by_id = {segment.segment_id: segment.importance for segment in segments}
    keep_by_id = {segment.segment_id: segment.keep_verbatim for segment in segments}

    assert importance_by_id[1] == SegmentImportance.ESSENTIAL
    assert importance_by_id[2] == SegmentImportance.ESSENTIAL
    assert importance_by_id[3] == SegmentImportance.VALUABLE
    assert importance_by_id[4] == SegmentImportance.VALUABLE
    assert importance_by_id[5] == SegmentImportance.REDUNDANT
    assert keep_by_id[2] is True
    assert keep_by_id[4] is True
    assert keep_by_id[5] is False
    assert 999 not in importance_by_id


def test_compact_plan_filler_refs_are_normalized_to_current_clip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FillerRefClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def chat(self, **kwargs: object) -> dict[str, dict[str, str]]:
            schema = kwargs["format"]
            assert isinstance(schema, dict)
            if schema.get("title") == "NarrativeOrderPlan":
                return {
                    "message": {
                        "content": (
                            '{"ordered_blocks":['
                            '{"block_id":0,"playback_position":0,"role":"development","topic_summary":"Chunk"}'
                            '],"editorial_strategy":"Ordine globale."}'
                        )
                    }
                }
            return {
                "message": {
                    "content": (
                        '{"clip_id":2,"topic_summary":"Chunk","role":"development",'
                        '"essential_segment_ids":[61,62],'
                        '"valuable_segment_ids":[],'
                        '"optional_segment_ids":[],'
                        '"redundant_segment_ids":[],'
                        '"filler_words":['
                        '{"clip_id":61,"segment_id":61,"word_index":0,"word_text":"eh"},'
                        '{"clip_id":2,"segment_id":62,"word_index":1,"word_text":"cioe"},'
                        '{"clip_id":999,"segment_id":999,"word_index":0,"word_text":"skip"}'
                        '],'
                        '"editorial_strategy":"Normalizza filler.",'
                        '"estimated_duration_s":2.0}'
                    )
                }
            }

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=FillerRefClient))
    transcription = ClipTranscription(
        clip_id=2,
        source_path=None,
        duration=2.0,
        segments=[
            Segment(
                id=61,
                clip_id=2,
                start=0.0,
                end=1.0,
                text="Segmento sessantuno",
                words=[WordToken(word="eh", start=0.0, end=0.1)],
            ),
            Segment(
                id=62,
                clip_id=2,
                start=1.0,
                end=2.0,
                text="Segmento sessantadue",
                words=[WordToken(word="cioe", start=1.0, end=1.1)],
            ),
        ],
    )
    settings = _settings(tmp_path).model_copy(
        update={
            "narrative_hierarchical_threshold_segments": 1,
            "narrative_chunk_max_segments": 2,
        }
    )

    result = NarrativePlanner(settings).generate([transcription], EditorialBrief(), "qwen3:8b")

    assert [(filler.clip_id, filler.segment_id, filler.word_text) for filler in result.filler_words] == [
        (2, 61, "eh"),
        (2, 62, "cioe"),
    ]


def test_timeouted_compact_chunk_is_split_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compact_prompt_sizes: list[int] = []

    class AdaptiveChunkClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def chat(self, **kwargs: object) -> dict[str, dict[str, str]]:
            schema = kwargs["format"]
            prompt = kwargs["messages"][0]["content"]
            assert isinstance(schema, dict)
            assert isinstance(prompt, str)
            if schema.get("title") == "NarrativeOrderPlan":
                block_ids = [
                    int(line.split()[1].rstrip(":"))
                    for line in prompt.splitlines()
                    if line.startswith("BLOCK ")
                ]
                return {
                    "message": {
                        "content": {
                            "ordered_blocks": [
                                {
                                    "block_id": block_id,
                                    "playback_position": position,
                                    "role": "development",
                                    "topic_summary": f"Blocco {block_id}",
                                }
                                for position, block_id in enumerate(block_ids)
                            ],
                            "editorial_strategy": "Ordine globale.",
                        }.__repr__().replace("'", '"')
                    }
                }

            segment_ids = [
                segment_id
                for segment_id in range(1, 5)
                if f"segment_id={segment_id} " in prompt or f"segment_id={segment_id}\n" in prompt
            ]
            compact_prompt_sizes.append(len(segment_ids))
            if len(segment_ids) > 2:
                raise httpx.TimeoutException("chunk troppo grande")
            return {
                "message": {
                    "content": {
                        "clip_id": 1,
                        "topic_summary": "Chunk adattivo",
                        "role": "development",
                        "essential_segment_ids": segment_ids,
                        "valuable_segment_ids": [],
                        "optional_segment_ids": [],
                        "redundant_segment_ids": [],
                        "filler_words": [],
                        "editorial_strategy": "Chunk piccolo completato.",
                        "estimated_duration_s": float(len(segment_ids)),
                    }.__repr__().replace("'", '"')
                }
            }

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=AdaptiveChunkClient))
    settings = _settings(tmp_path).model_copy(
        update={
            "narrative_hierarchical_threshold_segments": 1,
            "narrative_chunk_max_segments": 4,
        }
    )

    result = NarrativePlanner(settings, max_retries=0).generate(
        [_many_segment_transcription(4)],
        EditorialBrief(),
        "qwen3:8b",
    )

    retained_ids = [
        segment.segment_id
        for clip in sorted(result.ordered_clips, key=lambda item: item.playback_position)
        for segment in clip.segments
        if segment.keep_verbatim
    ]
    assert compact_prompt_sizes == [4, 2, 2]
    assert retained_ids == [1, 2, 3, 4]


def test_order_plan_timeout_falls_back_to_upload_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OrderTimeoutClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def chat(self, **kwargs: object) -> dict[str, dict[str, str]]:
            schema = kwargs["format"]
            prompt = kwargs["messages"][0]["content"]
            assert isinstance(schema, dict)
            assert isinstance(prompt, str)
            if schema.get("title") == "NarrativeOrderPlan":
                raise httpx.TimeoutException("order timeout")
            segment_ids = [
                segment_id
                for segment_id in range(1, 4)
                if f"segment_id={segment_id} " in prompt or f"segment_id={segment_id}\n" in prompt
            ]
            return {
                "message": {
                    "content": {
                        "clip_id": 1,
                        "topic_summary": "Chunk",
                        "role": "development",
                        "essential_segment_ids": segment_ids,
                        "valuable_segment_ids": [],
                        "optional_segment_ids": [],
                        "redundant_segment_ids": [],
                        "filler_words": [],
                        "editorial_strategy": "Chunk ok.",
                        "estimated_duration_s": float(len(segment_ids)),
                    }.__repr__().replace("'", '"')
                }
            }

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=OrderTimeoutClient))
    settings = _settings(tmp_path).model_copy(
        update={
            "narrative_hierarchical_threshold_segments": 1,
            "narrative_chunk_max_segments": 1,
        }
    )

    result = NarrativePlanner(settings, max_retries=0).generate(
        [_many_segment_transcription(3)],
        EditorialBrief(clip_order="ai_narrative"),
        "qwen3:8b",
    )

    ordered_ids = [
        segment.segment_id
        for clip in sorted(result.ordered_clips, key=lambda item: item.playback_position)
        for segment in clip.segments
    ]
    assert ordered_ids == [1, 2, 3]
    assert "ordine di upload" in result.editorial_strategy.lower()


def test_hierarchical_planning_refines_map_when_target_duration_is_exceeded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refinement_calls = 0

    class TargetRefinementClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def chat(self, **kwargs: object) -> dict[str, dict[str, str]]:
            nonlocal refinement_calls
            schema = kwargs["format"]
            prompt = kwargs["messages"][0]["content"]
            assert isinstance(schema, dict)
            assert isinstance(prompt, str)
            if schema.get("title") == "NarrativeOrderPlan":
                return {
                    "message": {
                        "content": (
                            '{"ordered_blocks":['
                            '{"block_id":0,"playback_position":0,"role":"introduction","topic_summary":"Primo blocco"},'
                            '{"block_id":1,"playback_position":1,"role":"development","topic_summary":"Secondo blocco"}'
                            '],"editorial_strategy":"Ordine globale."}'
                        )
                    }
                }
            if schema.get("title") == "DurationRefinementPlan":
                refinement_calls += 1
                assert "Durata trattenuta corrente=800.0s" in prompt
                assert "target richiesto=300.0s" in prompt
                return {
                    "message": {
                        "content": (
                            '{"redundant_segments":['
                            '{"clip_id":1,"segment_id":4,"reason":"Riduce blocco accessorio."},'
                            '{"clip_id":1,"segment_id":5,"reason":"Riduce blocco accessorio."},'
                            '{"clip_id":1,"segment_id":6,"reason":"Riduce blocco accessorio."},'
                            '{"clip_id":1,"segment_id":7,"reason":"Riduce blocco accessorio."},'
                            '{"clip_id":1,"segment_id":8,"reason":"Riduce blocco accessorio."}'
                            '],"editorial_strategy":"Rientro nel target richiesto.",'
                            '"estimated_duration_s":300.0}'
                        )
                    }
                }

            segment_ids = [
                segment_id
                for segment_id in range(1, 9)
                if f"segment_id={segment_id} " in prompt or f"segment_id={segment_id}\n" in prompt
            ]
            return {
                "message": {
                    "content": {
                        "clip_id": 1,
                        "topic_summary": "Chunk",
                        "role": "development",
                        "essential_segment_ids": [],
                        "valuable_segment_ids": segment_ids,
                        "optional_segment_ids": [],
                        "redundant_segment_ids": [],
                        "filler_words": [],
                        "editorial_strategy": "Chunk inizialmente troppo conservativo.",
                        "estimated_duration_s": float(len(segment_ids) * 100),
                    }.__repr__().replace("'", '"')
                }
            }

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=TargetRefinementClient))
    settings = _settings(tmp_path).model_copy(
        update={
            "narrative_hierarchical_threshold_segments": 1,
            "narrative_chunk_max_segments": 4,
        }
    )

    result = NarrativePlanner(settings, max_retries=0).generate(
        [_long_segment_transcription(count=8, duration_s=100.0)],
        EditorialBrief(synthesis_level=3, target_duration_s=300.0),
        "qwen3:8b",
    )

    retained_ids = [
        segment.segment_id
        for clip in sorted(result.ordered_clips, key=lambda item: item.playback_position)
        for segment in clip.segments
        if segment.keep_verbatim
    ]
    assert refinement_calls == 1
    assert retained_ids == [1, 2, 3]
    assert result.estimated_duration_s == 300.0
    assert "Rientro nel target richiesto" in result.editorial_strategy


def test_duration_refinement_accumulates_partial_llm_cuts_across_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refinement_prompts: list[str] = []

    class PartialRefinementClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def chat(self, **kwargs: object) -> dict[str, dict[str, str]]:
            schema = kwargs["format"]
            prompt = kwargs["messages"][0]["content"]
            assert isinstance(schema, dict)
            assert isinstance(prompt, str)
            if schema.get("title") == "NarrativeOrderPlan":
                return {
                    "message": {
                        "content": (
                            '{"ordered_blocks":['
                            '{"block_id":0,"playback_position":0,"role":"introduction","topic_summary":"Primo blocco"},'
                            '{"block_id":1,"playback_position":1,"role":"development","topic_summary":"Secondo blocco"}'
                            '],"editorial_strategy":"Ordine globale."}'
                        )
                    }
                }
            if schema.get("title") == "DurationRefinementPlan":
                refinement_prompts.append(prompt)
                if len(refinement_prompts) == 1:
                    return {
                        "message": {
                            "content": (
                                '{"redundant_segments":['
                                '{"clip_id":1,"segment_id":7,"reason":"Primo taglio accessorio."},'
                                '{"clip_id":1,"segment_id":8,"reason":"Secondo taglio accessorio."}'
                                '],"editorial_strategy":"Prima riduzione parziale.",'
                                '"estimated_duration_s":600.0}'
                            )
                        }
                    }
                assert "Durata trattenuta corrente=600.0s" in prompt
                assert "segment_id=7" not in prompt
                assert "segment_id=8" not in prompt
                return {
                    "message": {
                        "content": (
                            '{"redundant_segments":['
                            '{"clip_id":1,"segment_id":4,"reason":"Ulteriore blocco accessorio."},'
                            '{"clip_id":1,"segment_id":5,"reason":"Ulteriore blocco accessorio."},'
                            '{"clip_id":1,"segment_id":6,"reason":"Ulteriore blocco accessorio."}'
                            '],"editorial_strategy":"Seconda riduzione fino al target.",'
                            '"estimated_duration_s":300.0}'
                        )
                    }
                }

            segment_ids = [
                segment_id
                for segment_id in range(1, 9)
                if f"segment_id={segment_id} " in prompt or f"segment_id={segment_id}\n" in prompt
            ]
            return {
                "message": {
                    "content": {
                        "clip_id": 1,
                        "topic_summary": "Chunk",
                        "role": "development",
                        "essential_segment_ids": [],
                        "valuable_segment_ids": segment_ids,
                        "optional_segment_ids": [],
                        "redundant_segment_ids": [],
                        "filler_words": [],
                        "editorial_strategy": "Chunk conservativo.",
                        "estimated_duration_s": float(len(segment_ids) * 100),
                    }.__repr__().replace("'", '"')
                }
            }

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=PartialRefinementClient))
    settings = _settings(tmp_path).model_copy(
        update={
            "narrative_hierarchical_threshold_segments": 1,
            "narrative_chunk_max_segments": 4,
        }
    )

    result = NarrativePlanner(settings, max_retries=1).generate(
        [_long_segment_transcription(count=8, duration_s=100.0)],
        EditorialBrief(synthesis_level=3, target_duration_s=300.0),
        "qwen3:8b",
    )

    retained_ids = [
        segment.segment_id
        for clip in sorted(result.ordered_clips, key=lambda item: item.playback_position)
        for segment in clip.segments
        if segment.keep_verbatim
    ]
    assert len(refinement_prompts) == 2
    assert retained_ids == [1, 2, 3]


def test_adaptive_upload_order_keeps_source_chunk_sequence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UploadOrderClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def chat(self, **kwargs: object) -> dict[str, dict[str, str]]:
            schema = kwargs["format"]
            prompt = kwargs["messages"][0]["content"]
            assert isinstance(schema, dict)
            assert isinstance(prompt, str)
            assert schema.get("title") == "CompactNarrativePlan"
            segment_ids = [
                segment_id
                for segment_id in range(1, 9)
                if f"segment_id={segment_id} " in prompt or f"segment_id={segment_id}\n" in prompt
            ]
            if len(segment_ids) > 2:
                raise httpx.TimeoutException("split required")
            return {
                "message": {
                    "content": {
                        "clip_id": 1,
                        "topic_summary": "Chunk",
                        "role": "development",
                        "essential_segment_ids": segment_ids,
                        "valuable_segment_ids": [],
                        "optional_segment_ids": [],
                        "redundant_segment_ids": [],
                        "filler_words": [],
                        "editorial_strategy": "Chunk ok.",
                        "estimated_duration_s": float(len(segment_ids)),
                    }.__repr__().replace("'", '"')
                }
            }

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=UploadOrderClient))
    settings = _settings(tmp_path).model_copy(
        update={
            "narrative_hierarchical_threshold_segments": 1,
            "narrative_chunk_max_segments": 4,
        }
    )

    result = NarrativePlanner(settings, max_retries=0).generate(
        [_many_segment_transcription(8)],
        EditorialBrief(clip_order="upload"),
        "qwen3:8b",
    )

    ordered_ids = [
        segment.segment_id
        for clip in sorted(result.ordered_clips, key=lambda item: item.playback_position)
        for segment in clip.segments
    ]
    assert ordered_ids == [1, 2, 3, 4, 5, 6, 7, 8]


def test_compact_planning_uses_small_generation_budget_even_when_global_budget_is_large(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compact_num_predict: list[int] = []

    class BudgetClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def chat(self, **kwargs: object) -> dict[str, dict[str, str]]:
            schema = kwargs["format"]
            options = kwargs["options"]
            prompt = kwargs["messages"][0]["content"]
            assert isinstance(schema, dict)
            assert isinstance(options, dict)
            assert isinstance(prompt, str)
            if schema.get("title") == "NarrativeOrderPlan":
                return {
                    "message": {
                        "content": (
                            '{"ordered_blocks":['
                            '{"block_id":0,"playback_position":0,"role":"development","topic_summary":"Chunk"}'
                            '],"editorial_strategy":"Ordine globale."}'
                        )
                    }
                }
            compact_num_predict.append(int(options["num_predict"]))
            segment_ids = [
                segment_id
                for segment_id in range(1, 3)
                if f"segment_id={segment_id} " in prompt or f"segment_id={segment_id}\n" in prompt
            ]
            return {
                "message": {
                    "content": {
                        "clip_id": 1,
                        "topic_summary": "Chunk",
                        "role": "development",
                        "essential_segment_ids": segment_ids,
                        "valuable_segment_ids": [],
                        "optional_segment_ids": [],
                        "redundant_segment_ids": [],
                        "filler_words": [],
                        "editorial_strategy": "Budget compatto.",
                        "estimated_duration_s": float(len(segment_ids)),
                    }.__repr__().replace("'", '"')
                }
            }

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=BudgetClient))
    settings = _settings(tmp_path).model_copy(
        update={
            "ollama_num_predict": 32768,
            "narrative_hierarchical_threshold_segments": 1,
            "narrative_chunk_max_segments": 2,
        }
    )

    NarrativePlanner(settings).generate([_many_segment_transcription(2)], EditorialBrief(), "qwen3:8b")

    assert compact_num_predict == [4096]


def test_proposal_reviewer_uses_bounded_ollama_runtime_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            captured["client_kwargs"] = kwargs

        def chat(self, **kwargs: object) -> dict[str, dict[str, str]]:
            captured["chat_kwargs"] = kwargs
            review = ProposalReview(approved=True, editorial_notes="ok")
            return {"message": {"content": review.model_dump_json()}}

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=FakeClient))

    reviewer = ProposalReviewer(_settings(tmp_path))
    reviewer.review([_transcription()], _narrative_map(), _edl(), EditorialBrief(), "qwen3:8b")

    assert captured["client_kwargs"] == {"timeout": 12.0}
    chat_kwargs = captured["chat_kwargs"]
    assert isinstance(chat_kwargs, dict)
    assert chat_kwargs["think"] is False
    assert chat_kwargs["options"] == {"num_ctx": 4096, "num_predict": 2048, "temperature": 0.2}


def test_proposal_reviewer_timeout_returns_warning_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TimeoutClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def chat(self, **_kwargs: object) -> object:
            raise httpx.TimeoutException("review timeout")

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=TimeoutClient))

    review = ProposalReviewer(_settings(tmp_path), max_retries=0).review(
        [_transcription()],
        _narrative_map(),
        _edl(),
        EditorialBrief(),
        "qwen3:8b",
    )

    assert review.approved is False
    assert review.corrected_narrative_map is None
    assert [issue.code for issue in review.issues] == ["proposal_review_timeout"]
    assert "non approvata" in review.editorial_notes


def test_default_ollama_budget_allows_large_structured_outputs() -> None:
    settings = AppSettings()

    assert settings.ollama_request_timeout_s <= 300
    assert settings.ollama_num_ctx >= 49152
    assert settings.ollama_num_predict >= 32768
    assert settings.narrative_hierarchical_threshold_segments <= 30
    assert settings.narrative_chunk_max_segments <= 30


def test_structured_chat_subprocess_timeout_is_wall_clock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*_args: object, **_kwargs: object) -> object:
        raise subprocess.TimeoutExpired(cmd=["python", "-m", "autocut_ai.ollama_worker"], timeout=3)

    monkeypatch.setattr(subprocess, "run", fake_run)
    settings = AppSettings(
        data_dir=tmp_path / "data",
        database_path=tmp_path / "data" / "jobs.sqlite3",
        ollama_request_timeout_s=3,
        ollama_run_in_subprocess=True,
    )

    with pytest.raises(OllamaTimeoutError, match="Timeout Ollama dopo 3s"):
        structured_chat(
            SimpleNamespace(Client=object),
            settings,
            model="qwen3:8b",
            messages=[{"role": "user", "content": "ciao"}],
            response_format=NarrativeMap.model_json_schema(),
        )
