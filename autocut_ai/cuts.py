from __future__ import annotations

from loguru import logger

from autocut_ai.schemas import (
    ClipTranscription,
    CreativeZoomSpan,
    EditDecision,
    EditDecisionList,
    EditingOptions,
    NarrativeMap,
    NarrativeSegment,
    Segment,
    SegmentImportance,
    WordToken,
)


def _clip_lookup(transcriptions: list[ClipTranscription]) -> dict[int, ClipTranscription]:
    return {clip.clip_id: clip for clip in transcriptions}


def _segment_lookup(transcriptions: list[ClipTranscription]) -> dict[tuple[int, int], Segment]:
    return {
        (clip.clip_id, segment.id): segment
        for clip in transcriptions
        for segment in clip.segments
    }


def _is_numeric_like(value: str) -> bool:
    cleaned = value.strip(".,:%€$£")
    return cleaned.replace(",", "").replace(".", "").isdigit()


_TERMINAL_PUNCTUATION = (".", "!", "?", "...")
_TOKEN_STRIP_CHARS = " \t\r\n.,;:!?...'\"()[]{}"
_DANGLING_END_WORDS = {
    "a",
    "ad",
    "ai",
    "al",
    "alla",
    "alle",
    "allo",
    "agli",
    "con",
    "da",
    "dai",
    "dal",
    "dalla",
    "dalle",
    "degli",
    "dei",
    "del",
    "della",
    "delle",
    "di",
    "e",
    "ed",
    "fra",
    "gli",
    "i",
    "il",
    "in",
    "la",
    "le",
    "lo",
    "ma",
    "nel",
    "nella",
    "nelle",
    "per",
    "su",
    "sui",
    "sul",
    "sulla",
    "tra",
    "un",
    "una",
}
_CONTINUATION_START_WORDS = {
    "a",
    "ad",
    "anche",
    "che",
    "come",
    "con",
    "da",
    "dai",
    "dal",
    "della",
    "di",
    "e",
    "ed",
    "in",
    "ma",
    "mentre",
    "o",
    "per",
    "perche",
    "quindi",
    "su",
}
_MIN_RENDER_TAIL_GUARD_S = 0.85


def _tail_guard_seconds(options: EditingOptions) -> float:
    transition_guard_s = (
        0.0
        if options.default_transition == "cut"
        else max(0.0, options.transition_duration_ms / 1000)
    )
    configured_post_roll_s = max(0.0, options.post_roll_ms / 1000)
    return max(configured_post_roll_s, transition_guard_s + _MIN_RENDER_TAIL_GUARD_S)


def _tokens(text: str) -> list[str]:
    return [token.strip(_TOKEN_STRIP_CHARS) for token in text.split() if token.strip(_TOKEN_STRIP_CHARS)]


def _normalized_text_part(value: str) -> str:
    return value.strip(_TOKEN_STRIP_CHARS).casefold()


def _append_without_duplicate_overlap(first: str, second: str) -> tuple[str, bool]:
    first_parts = first.split()
    second_parts = second.split()
    if not first_parts or not second_parts:
        return f"{first} {second}".strip(), False
    first_normalized = [_normalized_text_part(part) for part in first_parts]
    second_normalized = [_normalized_text_part(part) for part in second_parts]
    max_overlap = min(len(first_normalized), len(second_normalized))
    for overlap in range(max_overlap, 0, -1):
        if first_normalized[-overlap:] != second_normalized[:overlap]:
            continue
        remaining = second_parts[overlap:]
        if not remaining:
            return first.strip(), True
        return f"{first} {' '.join(remaining)}".strip(), True
    return f"{first} {second}".strip(), False


def _ends_with_terminal_punctuation(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and stripped.endswith(_TERMINAL_PUNCTUATION)


def _needs_sentence_completion(current_text: str, next_text: str) -> bool:
    current_tokens = _tokens(current_text)
    next_tokens = _tokens(next_text)
    if not current_tokens or not next_tokens:
        return False
    if _ends_with_terminal_punctuation(current_text):
        return False
    last_token = current_tokens[-1].casefold()
    first_token = next_tokens[0].casefold()
    if last_token in _DANGLING_END_WORDS:
        return True
    if first_token in _CONTINUATION_START_WORDS:
        return True
    first_character = next_tokens[0][0]
    return first_character.casefold() == first_character


def _extend_segment_to_sentence_boundary(
    clip: ClipTranscription,
    segment: Segment,
    included_text: str,
    source_end: float,
    options: EditingOptions,
) -> tuple[float, str, list[str]]:
    if not options.avoid_hard_cut_inside_sentence:
        return source_end, included_text, []
    ordered_segments = sorted(clip.segments, key=lambda item: item.start)
    try:
        current_index = next(index for index, item in enumerate(ordered_segments) if item.id == segment.id)
    except StopIteration:
        return source_end, included_text, []

    notes: list[str] = []
    completed_text = included_text
    completed_end = source_end
    max_gap = max(0.75, options.min_pause_to_cut_ms / 1000)
    for next_segment in ordered_segments[current_index + 1 :]:
        gap = max(0.0, next_segment.start - completed_end)
        if gap > max_gap:
            break
        if not _needs_sentence_completion(completed_text, next_segment.text):
            break
        if not next_segment.words:
            break
        completed_text = f"{completed_text} {next_segment.text}".strip()
        completed_end = next_segment.words[-1].end
        notes.append(f"Extended to segment {next_segment.id} to avoid incomplete sentence at cut boundary")
        if _ends_with_terminal_punctuation(completed_text):
            break
    return completed_end, completed_text, notes


def _word_removal_indexes(narrative_segment: NarrativeSegment, segment: Segment) -> set[int]:
    if not narrative_segment.remove_word_ranges:
        return set()
    word_count = len(segment.words)
    removal_indexes: set[int] = set()
    for removal_range in narrative_segment.remove_word_ranges:
        if removal_range.start_word_index >= word_count:
            continue
        end_index = min(removal_range.end_word_index, word_count - 1)
        removal_indexes.update(range(removal_range.start_word_index, end_index + 1))
    return removal_indexes


def _retained_word_runs(segment: Segment, narrative_segment: NarrativeSegment) -> list[list[WordToken]]:
    removal_indexes = _word_removal_indexes(narrative_segment, segment)
    if not removal_indexes:
        return [segment.words]

    runs: list[list[WordToken]] = []
    current_run: list[WordToken] = []
    for index, word in enumerate(segment.words):
        if index in removal_indexes:
            if current_run:
                runs.append(current_run)
                current_run = []
            continue
        current_run.append(word)
    if current_run:
        runs.append(current_run)
    return runs


def _word_run_text(segment: Segment, run: list[WordToken], has_word_removals: bool) -> str:
    if not has_word_removals and run == segment.words:
        return segment.text
    return " ".join(word.word for word in run).strip()


def snap_interval_to_word_boundaries(
    segment: Segment,
    proposed_start: float,
    proposed_end: float,
    preserve_numbers: bool = True,
) -> tuple[float, float, list[str]]:
    """Allarga un intervallo solo quanto basta per non troncare una parola."""

    notes: list[str] = []
    start = proposed_start
    end = proposed_end
    for word in segment.words:
        if word.start < start < word.end:
            start = word.start
            notes.append(f"Adjusted start to word boundary before '{word.word}'")
            if preserve_numbers and _is_numeric_like(word.word):
                notes.append(f"Preserved numeric token '{word.word}' at cut start")
            break
    for word in segment.words:
        if word.start < end < word.end:
            end = word.end
            notes.append(f"Adjusted end to word boundary after '{word.word}'")
            if preserve_numbers and _is_numeric_like(word.word):
                notes.append(f"Preserved numeric token '{word.word}' at cut end")
            break
    return start, end, notes


def _clip_words(clip: ClipTranscription) -> list[WordToken]:
    return sorted(
        [word for segment in clip.segments for word in segment.words],
        key=lambda word: (word.start, word.end),
    )


def _snap_padding_to_clip_word_boundaries(
    clip: ClipTranscription,
    source_start: float,
    source_end: float,
    content_start: float,
    content_end: float,
) -> tuple[float, float, list[str]]:
    notes: list[str] = []
    start = source_start
    end = source_end
    for word in _clip_words(clip):
        if word.start < start < word.end:
            if word.end <= content_start:
                start = min(content_start, word.end)
                notes.append(f"Adjusted pre-roll start after adjacent word '{word.word}'")
            else:
                start = word.start
                notes.append(f"Adjusted start to word boundary before overlapping word '{word.word}'")
            break
    for word in _clip_words(clip):
        if word.start < end < word.end:
            if word.start >= content_end:
                end = max(content_end, word.start)
                notes.append(f"Adjusted post-roll end before adjacent word '{word.word}'")
            else:
                end = word.end
                notes.append(f"Adjusted end to word boundary after overlapping word '{word.word}'")
            break
    return start, end, notes


def _extend_min_duration_end_to_clip_word_boundary(
    clip: ClipTranscription,
    source_end: float,
) -> tuple[float, list[str]]:
    for word in _clip_words(clip):
        if word.start < source_end < word.end:
            return min(clip.duration, word.end), [
                f"Extended minimum duration end to avoid truncating adjacent word '{word.word}'"
            ]
    return source_end, []


def _apply_tail_guard_to_clip_word_boundary(
    clip: ClipTranscription,
    source_end: float,
    content_end: float,
    options: EditingOptions,
) -> tuple[float, list[str]]:
    requested_end = min(clip.duration, content_end + _tail_guard_seconds(options))
    if requested_end <= source_end:
        return source_end, []
    clamped_before_word: str | None = None
    for word in _clip_words(clip):
        if word.start >= content_end and word.start < requested_end:
            requested_end = max(content_end, word.start)
            clamped_before_word = word.word
            break
    _start, guarded_end, notes = _snap_padding_to_clip_word_boundaries(
        clip=clip,
        source_start=content_end,
        source_end=requested_end,
        content_start=content_end,
        content_end=content_end,
    )
    guard_notes = []
    if guarded_end > source_end:
        guard_notes.append(
            f"Extended tail guard to {guarded_end - content_end:.2f}s after retained speech to prevent truncated final words"
        )
    if clamped_before_word is not None:
        guard_notes.append(f"Clamped tail guard before adjacent word '{clamped_before_word}'")
    guard_notes.extend(notes)
    return max(source_end, guarded_end), guard_notes


def _merge_overlapping_same_clip_decisions(decisions: list[EditDecision]) -> list[EditDecision]:
    merged: list[EditDecision] = []
    for decision in decisions:
        if merged and decision.clip_id == merged[-1].clip_id and decision.source_start < merged[-1].source_end:
            previous = merged[-1]
            if decision.source_end <= previous.source_end:
                previous.creative_zoom_spans.extend(decision.creative_zoom_spans)
                previous.creative_zoom = previous.creative_zoom or bool(previous.creative_zoom_spans)
                previous.safety_notes.append(
                    f"Dropped fully overlapped same-clip decision at output order {decision.output_order} to avoid repeated audio"
                )
                continue
            previous.source_end = decision.source_end
            previous.transition_after = decision.transition_after
            previous.creative_zoom_spans.extend(decision.creative_zoom_spans)
            previous.creative_zoom = previous.creative_zoom or bool(previous.creative_zoom_spans)
            previous.creative_zoom_start_scale = min(previous.creative_zoom_start_scale, decision.creative_zoom_start_scale)
            previous.creative_zoom_end_scale = max(previous.creative_zoom_end_scale, decision.creative_zoom_end_scale)
            previous.reason = f"{previous.reason}; {decision.reason}"
            previous.included_text, removed_duplicate_text = _append_without_duplicate_overlap(
                previous.included_text,
                decision.included_text,
            )
            previous.safety_notes.extend(decision.safety_notes)
            if removed_duplicate_text:
                previous.safety_notes.append(
                    f"Skipped duplicate text while merging output order {decision.output_order}"
                )
            previous.safety_notes.append(
                f"Merged overlapping same-clip decision at output order {decision.output_order} to avoid repeated audio"
            )
            logger.bind(clip_id=decision.clip_id).info("Merged overlapping same-clip decision")
            continue
        merged.append(decision)
    return merged


def _creative_zoom_candidates_for_segment(
    segment: Segment,
    narrative_segment: NarrativeSegment,
    options: EditingOptions,
) -> list[CreativeZoomSpan]:
    if not options.enable_creative_zoom:
        return []
    if not segment.words:
        return []
    sorted_words = sorted(segment.words, key=lambda word: word.start)
    gaps = [
        max(0.0, current.start - previous.end)
        for previous, current in zip(sorted_words, sorted_words[1:])
    ]
    split_indexes: set[int] = set()
    if len(gaps) > 1:
        average_gap = sum(gaps) / len(gaps)
        meaningful_pause_s = max(0.12, average_gap * 1.5)
        split_indexes = {index + 1 for index, gap in enumerate(gaps) if gap >= meaningful_pause_s}

    chunks: list[list[WordToken]] = []
    current_chunk: list[WordToken] = []
    for index, word in enumerate(sorted_words):
        if index in split_indexes and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
        current_chunk.append(word)
    if current_chunk:
        chunks.append(current_chunk)

    spans: list[CreativeZoomSpan] = []
    for chunk in chunks:
        if not chunk:
            continue
        source_start = chunk[0].start
        source_end = chunk[-1].end
        if source_end <= source_start:
            logger.bind(
                clip_id=segment.clip_id,
                segment_id=segment.id,
                source_start=source_start,
                source_end=source_end,
            ).info("Skipped invalid creative zoom span with non-positive duration")
            continue
        spans.append(
            CreativeZoomSpan(
                source_start=source_start,
                source_end=source_end,
                start_scale=1.2,
                end_scale=1.2,
                text=" ".join(word.word for word in chunk).strip(),
                reason=narrative_segment.reason,
            )
        )
    return spans


def _fixed_zoom_scale_for_pause(pause_s: float, average_pause_s: float) -> float:
    if average_pause_s <= 0:
        return 1.2
    ratio = pause_s / average_pause_s
    if ratio >= 1.5:
        return 1.5
    if ratio >= 1.2:
        return 1.35
    return 1.2


def _apply_pause_based_creative_zoom(decisions: list[EditDecision], options: EditingOptions) -> None:
    if not options.enable_creative_zoom:
        for decision in decisions:
            decision.creative_zoom = False
            decision.creative_zoom_spans = []
        return

    candidates: list[tuple[int, CreativeZoomSpan, float, int]] = []
    previous_span_by_clip: dict[int, CreativeZoomSpan] = {}
    ordinal = 0
    for decision_index, decision in enumerate(decisions):
        spans = sorted(decision.creative_zoom_spans, key=lambda span: (span.source_start, span.source_end))
        for span in spans:
            previous_span = previous_span_by_clip.get(decision.clip_id)
            pause_s = 0.0
            if previous_span is not None:
                pause_s = max(0.0, span.source_start - previous_span.source_end)
            candidates.append((decision_index, span, pause_s, ordinal))
            ordinal += 1
            previous_span_by_clip[decision.clip_id] = span

    if not candidates:
        for decision in decisions:
            decision.creative_zoom = False
            decision.creative_zoom_spans = []
        return

    pauses = [pause_s for _decision_index, _span, pause_s, _ordinal in candidates if pause_s > 0]
    requested_minimum = min(options.min_creative_zoom_count, len(candidates))
    if not pauses and requested_minimum == 0:
        for decision in decisions:
            decision.creative_zoom = False
            decision.creative_zoom_spans = []
        return

    average_pause_s = sum(pauses) / len(pauses) if pauses else 0.0
    selected_by_decision: dict[int, list[CreativeZoomSpan]] = {index: [] for index in range(len(decisions))}
    selected_ordinals: set[int] = set()

    def add_selected(
        decision_index: int,
        span: CreativeZoomSpan,
        pause_s: float,
        candidate_ordinal: int,
        reason: str,
    ) -> None:
        if candidate_ordinal in selected_ordinals:
            return
        scale = _fixed_zoom_scale_for_pause(pause_s, average_pause_s)
        selected_by_decision[decision_index].append(
            CreativeZoomSpan(
                source_start=span.source_start,
                source_end=span.source_end,
                start_scale=scale,
                end_scale=scale,
                text=span.text,
                reason=reason,
            )
        )
        selected_ordinals.add(candidate_ordinal)

    if pauses:
        for decision_index, span, pause_s, candidate_ordinal in candidates:
            if pause_s < average_pause_s:
                continue
            add_selected(
                decision_index,
                span,
                pause_s,
                candidate_ordinal,
                f"zoom creativo: pausa lunga {pause_s:.2f}s sopra media {average_pause_s:.2f}s",
            )

    if len(selected_ordinals) < requested_minimum:
        remaining_candidates = sorted(
            candidates,
            key=lambda item: (
                item[3] in selected_ordinals,
                -item[2],
                -(item[1].source_end - item[1].source_start),
                item[3],
            ),
        )
        for decision_index, span, pause_s, candidate_ordinal in remaining_candidates:
            if len(selected_ordinals) >= requested_minimum:
                break
            if candidate_ordinal in selected_ordinals:
                continue
            if pause_s > 0:
                reason = f"zoom creativo: minimo richiesto {requested_minimum}, pausa {pause_s:.2f}s"
            else:
                reason = f"zoom creativo: minimo richiesto {requested_minimum}, struttura frase"
            add_selected(decision_index, span, pause_s, candidate_ordinal, reason)

    for index, decision in enumerate(decisions):
        decision.creative_zoom_spans = sorted(
            selected_by_decision[index],
            key=lambda span: (span.source_start, span.source_end),
        )
        decision.creative_zoom = bool(decision.creative_zoom_spans)
        if decision.creative_zoom_spans:
            decision.creative_zoom_start_scale = decision.creative_zoom_spans[0].start_scale
            decision.creative_zoom_end_scale = max(span.end_scale for span in decision.creative_zoom_spans)


def build_edit_decision_list(
    transcriptions: list[ClipTranscription],
    narrative_map: NarrativeMap,
    options: EditingOptions,
) -> EditDecisionList:
    """Traduce le scelte editoriali dell'LLM in tagli tecnicamente sicuri.

    Questa funzione può correggere esclusivamente i bordi temporali. Non
    introduce segmenti che la ``NarrativeMap`` abbia escluso.
    """

    clips = _clip_lookup(transcriptions)
    segments = _segment_lookup(transcriptions)
    decisions: list[EditDecision] = []
    output_order = 0

    for narrative_clip in sorted(narrative_map.ordered_clips, key=lambda item: item.playback_position):
        clip = clips[narrative_clip.clip_id]
        for narrative_segment in narrative_clip.segments:
            if narrative_segment.importance == SegmentImportance.REDUNDANT:
                continue
            if not narrative_segment.keep_verbatim:
                continue
            segment = segments[(narrative_clip.clip_id, narrative_segment.segment_id)]
            if not segment.words:
                continue
            word_runs = _retained_word_runs(segment, narrative_segment)
            has_word_removals = bool(narrative_segment.remove_word_ranges)
            if has_word_removals and not word_runs:
                logger.bind(clip_id=clip.clip_id, segment_id=segment.id).info("Skipping segment after word removals")
                continue
            for run_index, word_run in enumerate(word_runs):
                word_start = word_run[0].start
                word_end = word_run[-1].end
                source_start, source_end, notes = snap_interval_to_word_boundaries(
                    segment=segment,
                    proposed_start=word_start,
                    proposed_end=word_end,
                    preserve_numbers=options.preserve_numbers,
                )
                included_text = _word_run_text(segment, word_run, has_word_removals)
                if has_word_removals:
                    notes.append("Applied LLM word-level removal range")
                    content_end = source_end
                    source_end, tail_guard_notes = _apply_tail_guard_to_clip_word_boundary(
                        clip=clip,
                        source_end=source_end,
                        content_end=content_end,
                        options=options,
                    )
                    notes.extend(tail_guard_notes)
                else:
                    source_end, included_text, completion_notes = _extend_segment_to_sentence_boundary(
                        clip=clip,
                        segment=segment,
                        included_text=included_text,
                        source_end=source_end,
                        options=options,
                    )
                    notes.extend(completion_notes)
                    content_start = source_start
                    content_end = source_end
                    source_start = max(0.0, source_start - options.pre_roll_ms / 1000)
                    source_start, source_end, padding_notes = _snap_padding_to_clip_word_boundaries(
                        clip=clip,
                        source_start=source_start,
                        source_end=source_end,
                        content_start=content_start,
                        content_end=content_end,
                    )
                    notes.extend(padding_notes)
                    duration = source_end - source_start
                    min_duration = options.min_segment_duration_ms / 1000
                    if duration < min_duration:
                        notes.append(f"Skipped segment shorter than minimum duration {min_duration:.2f}s")
                        if narrative_segment.importance in {SegmentImportance.ESSENTIAL, SegmentImportance.VALUABLE}:
                            source_end = min(clip.duration, source_start + min_duration)
                            source_end, min_duration_notes = _extend_min_duration_end_to_clip_word_boundary(
                                clip=clip,
                                source_end=source_end,
                            )
                            notes.extend(min_duration_notes)
                            notes.append("Extended important segment to minimum duration")
                        else:
                            logger.bind(clip_id=clip.clip_id, segment_id=segment.id).info("Skipping short optional segment")
                            continue
                    source_end, tail_guard_notes = _apply_tail_guard_to_clip_word_boundary(
                        clip=clip,
                        source_end=source_end,
                        content_end=content_end,
                        options=options,
                    )
                    notes.extend(tail_guard_notes)
                if notes:
                    logger.bind(clip_id=clip.clip_id, segment_id=segment.id).info("Applied cut safety corrections")
                zoom_segment = segment
                if has_word_removals:
                    zoom_segment = segment.model_copy(
                        update={
                            "start": source_start,
                            "end": source_end,
                            "text": included_text,
                            "words": word_run,
                        }
                    )
                creative_zoom_spans = _creative_zoom_candidates_for_segment(zoom_segment, narrative_segment, options)
                internal_word_cut = has_word_removals and run_index > 0
                decisions.append(
                    EditDecision(
                        clip_id=clip.clip_id,
                        source_start=source_start,
                        source_end=source_end,
                        output_order=output_order,
                        transition_before="cut" if internal_word_cut else (options.default_transition if output_order > 0 else None),
                        transition_after="cut" if has_word_removals and run_index < len(word_runs) - 1 else options.default_transition,
                        creative_zoom=bool(creative_zoom_spans),
                        creative_zoom_start_scale=creative_zoom_spans[0].start_scale if creative_zoom_spans else 1.0,
                        creative_zoom_end_scale=max((span.end_scale for span in creative_zoom_spans), default=1.35),
                        creative_zoom_spans=creative_zoom_spans,
                        reason=narrative_segment.reason,
                        included_text=included_text,
                        safety_notes=notes,
                    )
                )
                output_order += 1

    # Due intervalli sovrapposti della stessa clip produrrebbero audio ripetuto:
    # li fondiamo conservando testo, note di sicurezza ed effetti già decisi.
    decisions = _merge_overlapping_same_clip_decisions(decisions)
    for index, decision in enumerate(decisions):
        decision.output_order = index
        if index == 0:
            decision.transition_before = None
        elif decision.transition_before is None:
            decision.transition_before = options.default_transition
        decision.creative_zoom_spans = sorted(decision.creative_zoom_spans, key=lambda span: (span.source_start, span.source_end))
        decision.creative_zoom = bool(decision.creative_zoom_spans)
    _apply_pause_based_creative_zoom(decisions, options)
    estimated = sum(decision.source_end - decision.source_start for decision in decisions)
    return EditDecisionList(decisions=decisions, estimated_duration_s=estimated)
