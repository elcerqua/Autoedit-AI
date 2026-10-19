from __future__ import annotations

from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from autocut_ai.config import AppSettings
from autocut_ai.ollama_runtime import (
    OllamaRuntimeError,
    OllamaTimeoutError,
    ollama_num_predict_for_attempt,
    structured_chat,
    validation_error_is_json_invalid,
)
from autocut_ai.schemas import (
    ClipTranscription,
    EditorialBrief,
    EditingOptions,
    FillerRef,
    NarrativeClip,
    NarrativeMap,
    NarrativeSegment,
    Segment,
    SegmentImportance,
)
from autocut_ai.synthesis import target_reduction_ratio


class NarrativeValidationError(ValueError):
    pass


class NarrativePlanningError(RuntimeError):
    pass


class NarrativeOrderBlock(BaseModel):
    block_id: int
    playback_position: int = Field(ge=0)
    role: Literal["introduction", "development", "climax", "conclusion", "supporting", "skip"]
    topic_summary: str


class NarrativeOrderPlan(BaseModel):
    ordered_blocks: list[NarrativeOrderBlock]
    editorial_strategy: str


class CompactNarrativePlan(BaseModel):
    clip_id: int
    topic_summary: str
    role: Literal["introduction", "development", "climax", "conclusion", "supporting", "skip"]
    essential_segment_ids: list[int] = Field(default_factory=list)
    valuable_segment_ids: list[int] = Field(default_factory=list)
    optional_segment_ids: list[int] = Field(default_factory=list)
    redundant_segment_ids: list[int] = Field(default_factory=list)
    filler_words: list[FillerRef] = Field(default_factory=list)
    editorial_strategy: str
    estimated_duration_s: float = Field(ge=0)


class DurationRefinementSegment(BaseModel):
    clip_id: int
    segment_id: int
    reason: str


class DurationRefinementPlan(BaseModel):
    redundant_segments: list[DurationRefinementSegment] = Field(default_factory=list)
    editorial_strategy: str
    estimated_duration_s: float = Field(ge=0)


_TOKEN_STRIP_CHARS = " \t\r\n.,;:!?...'\"()[]{}"
_TERMINAL_PUNCTUATION = (".", "!", "?", "...")
_DURATION_REFINEMENT_CANDIDATE_LIMIT = 180
_SUBJECTLESS_OPENING_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("viene", "eletto"),
    ("viene", "eletta"),
    ("viene", "nominato"),
    ("viene", "nominata"),
    ("viene", "scelto"),
    ("viene", "scelta"),
)
_CONTEXTLESS_OPENING_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("ve", "lo"),
    ("ve", "la"),
    ("ve", "li"),
    ("ve", "le"),
    ("ci",),
    ("ne",),
    ("si",),
    ("sì",),
    ("no",),
    ("ok",),
)
_DRY_PROMPT_MARKERS = (
    "asciutto",
    "asciutta",
    "ripetizion",
    "esitazion",
    "giri di parole",
    "solo i passaggi necessari",
    "solo contenuto essenziale",
)
_REDUNDANCY_MARKERS = (
    "ripete",
    "ripetizione",
    "ripetizioni",
    "stesso concetto",
    "giro di parole",
    "giri di parole",
    "esitazione",
    "esitazioni",
    "ridondante",
    "filler",
)
_SUBJECT_CANDIDATE_LIMIT = 5
_NAME_TOKEN_STOPWORDS = {"Ah", "Allora", "I", "Il", "La", "Le", "Lo", "Ma", "Ok", "Siamo", "Un", "Una", "Vado"}
_PERSON_CONTEXT_MARKERS = (
    "attuale",
    "ministro",
    "deputato",
    "deputati",
    "direttore",
    "presidente",
    "proprietario",
    "fiduciario",
    "consigliere",
)
_ORGANIZATION_CONTEXT_MARKERS = (
    "giornale",
    "fondazione",
    "partito",
    "srl",
    "fratelli d'italia",
    "secolo d'italia",
)
_INTRO_CONTEXT_MARKERS = (
    "buonasera",
    "buongiorno",
    "benvenuti",
    "piacere",
    "siamo pronti",
    "con chi siamo",
    "questa sera",
    "questa mattina",
    "oggi",
    "parliamo",
    "vediamo",
    "iniziamo",
    "pronti",
    "domanda",
    "gioco",
    "intervista",
    "tema",
)
_CONTEXT_CONTINUATION_START_WORDS = {
    "a",
    "ad",
    "anche",
    "che",
    "come",
    "con",
    "da",
    "dai",
    "dal",
    "dalla",
    "delle",
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
_LEGAL_NORM_REFERENCE_SEQUENCES: tuple[tuple[str, ...], ...] = (
    ("la", "stessa", "norma"),
    ("lo", "stesso", "decreto"),
    ("lo", "stesso", "articolo"),
    ("la", "stessa", "legge"),
    ("questa", "norma"),
    ("questo", "decreto"),
    ("questo", "articolo"),
    ("questa", "legge"),
    ("quella", "norma"),
    ("quel", "decreto"),
    ("quell", "articolo"),
    ("quella", "legge"),
    ("tale", "norma"),
    ("tale", "decreto"),
    ("tale", "articolo"),
    ("tale", "legge"),
    ("la", "norma", "poi"),
    ("il", "decreto", "poi"),
    ("l", "articolo", "poi"),
    ("la", "legge", "poi"),
)
_LEGAL_NORM_ANTECEDENT_TOKENS = {
    "articolo",
    "decreto",
    "legge",
    "norma",
    "provvedimento",
}
_LEGAL_NORM_TOPIC_TOKENS = {
    "beneficiari",
    "contributi",
    "editrici",
    "fondazione",
    "giornali",
    "impresa",
    "imprese",
    "movimenti",
    "partiti",
    "politici",
    "pubblici",
}
_CONTEXT_ANTECEDENT_CANDIDATE_LIMIT = 5


def _normalized_tokens(text: str) -> list[str]:
    return [token.strip(_TOKEN_STRIP_CHARS).casefold() for token in text.split() if token.strip(_TOKEN_STRIP_CHARS)]


def _raw_tokens(text: str) -> list[str]:
    return [token.strip(_TOKEN_STRIP_CHARS) for token in text.split() if token.strip(_TOKEN_STRIP_CHARS)]


def _short_text_preview(text: str, max_chars: int = 120) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_chars:
        return collapsed
    return f"{collapsed[: max_chars - 3].rstrip()}..."


def _starts_with_subjectless_opening(text: str) -> bool:
    tokens = _normalized_tokens(text)
    return any(tokens[: len(prefix)] == list(prefix) for prefix in _SUBJECTLESS_OPENING_PREFIXES)


def _starts_with_contextless_opening(text: str) -> bool:
    tokens = _normalized_tokens(text)
    return any(tokens[: len(prefix)] == list(prefix) for prefix in _CONTEXTLESS_OPENING_PREFIXES)


def _contains_token_sequence(tokens: list[str], sequence: tuple[str, ...]) -> bool:
    if not sequence or len(tokens) < len(sequence):
        return False
    sequence_length = len(sequence)
    return any(tokens[index : index + sequence_length] == list(sequence) for index in range(len(tokens) - sequence_length + 1))


def _text_has_digit(text: str) -> bool:
    return any(character.isdigit() for character in text)


def _context_reference_kind(segment: Segment) -> Literal["legal_norm"] | None:
    tokens = _normalized_tokens(segment.text)
    if bool(set(tokens) & _LEGAL_NORM_ANTECEDENT_TOKENS) and _text_has_digit(segment.text):
        return None
    if any(_contains_token_sequence(tokens, sequence) for sequence in _LEGAL_NORM_REFERENCE_SEQUENCES):
        return "legal_norm"
    return None


def _segment_has_context_antecedent(segment: Segment, kind: Literal["legal_norm"]) -> bool:
    tokens = set(_normalized_tokens(segment.text))
    if kind == "legal_norm":
        return bool(tokens & _LEGAL_NORM_ANTECEDENT_TOKENS) and _text_has_digit(segment.text)
    return False


def _context_antecedent_candidate_score(
    segment: Segment,
    dependent_segment: Segment,
    kind: Literal["legal_norm"],
) -> int:
    if segment.clip_id == dependent_segment.clip_id and segment.id == dependent_segment.id:
        return 0
    if not _segment_has_context_antecedent(segment, kind):
        return 0
    tokens = set(_normalized_tokens(segment.text))
    dependent_tokens = set(_normalized_tokens(dependent_segment.text))
    score = 100
    if "legge" in tokens:
        score += 30
    if "norma" in tokens:
        score += 20
    if "decreto" in tokens or "articolo" in tokens:
        score += 10
    score += 5 * len((tokens & dependent_tokens) & _LEGAL_NORM_TOPIC_TOKENS)
    if segment.clip_id == dependent_segment.clip_id and segment.start < dependent_segment.start:
        score += 15
    return score


def _context_antecedent_candidates(
    transcriptions: list[ClipTranscription],
    dependent_segment: Segment,
    kind: Literal["legal_norm"],
) -> list[Segment]:
    scored_candidates: list[tuple[int, int, float, Segment]] = []
    for clip in transcriptions:
        for segment in clip.segments:
            score = _context_antecedent_candidate_score(segment, dependent_segment, kind)
            if score <= 0:
                continue
            same_clip_bonus = 1 if segment.clip_id == dependent_segment.clip_id else 0
            scored_candidates.append((score, same_clip_bonus, -abs(segment.start - dependent_segment.start), segment))
    scored_candidates.sort(key=lambda item: item[:3], reverse=True)
    return [segment for _score, _same_clip_bonus, _distance, segment in scored_candidates[:_CONTEXT_ANTECEDENT_CANDIDATE_LIMIT]]


def _starts_with_lowercase_word(text: str) -> bool:
    tokens = _raw_tokens(text)
    if not tokens:
        return False
    first_character = tokens[0][0]
    return first_character.isalpha() and first_character.islower()


def _ends_with_terminal_punctuation(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and stripped.endswith(_TERMINAL_PUNCTUATION)


def _brief_requests_dry_edit(brief: EditorialBrief) -> bool:
    prompt = brief.user_prompt.casefold()
    return any(marker in prompt for marker in _DRY_PROMPT_MARKERS)


def _segment_matches_dry_cleanup(segment: Segment) -> bool:
    text = segment.text.casefold()
    return any(marker in text for marker in _REDUNDANCY_MARKERS)


def _is_name_like_token(token: str) -> bool:
    if not token:
        return False
    if token in _NAME_TOKEN_STOPWORDS:
        return False
    return token[0].isupper() and any(character.islower() for character in token[1:])


def _first_name_like_pair_index(segment: Segment) -> int | None:
    tokens = _raw_tokens(segment.text)
    for index, (first, second) in enumerate(zip(tokens, tokens[1:])):
        if _is_name_like_token(first) and _is_name_like_token(second):
            return index
    return None


def _subject_candidate_score(segment: Segment) -> int:
    pair_index = _first_name_like_pair_index(segment)
    if pair_index is None:
        return 0
    text = segment.text.casefold()
    score = max(10, 100 - min(pair_index, 8) * 10)
    if pair_index == 0:
        score += 60
    if any(marker in text for marker in _PERSON_CONTEXT_MARKERS):
        score += 20
    if any(marker in text for marker in _ORGANIZATION_CONTEXT_MARKERS):
        score -= 40
    return score


def _subject_candidate_segments(transcriptions: list[ClipTranscription], excluded: Segment) -> list[Segment]:
    scored_candidates: list[tuple[int, int, int, Segment]] = []
    for clip in transcriptions:
        for segment in clip.segments:
            if segment.clip_id == excluded.clip_id and segment.id == excluded.id:
                continue
            if _starts_with_subjectless_opening(segment.text):
                continue
            score = _subject_candidate_score(segment)
            if score > 0:
                scored_candidates.append((score, -clip.clip_id, -segment.id, segment))
    scored_candidates.sort(key=lambda item: item[:3], reverse=True)
    return [segment for _score, _clip_id, _segment_id, segment in scored_candidates[:_SUBJECT_CANDIDATE_LIMIT]]


def _intro_candidate_score(segment: Segment, dependent_segment: Segment) -> int:
    if segment.clip_id == dependent_segment.clip_id and segment.id == dependent_segment.id:
        return 0
    text = segment.text.casefold()
    score = _subject_candidate_score(segment)
    for marker in _INTRO_CONTEXT_MARKERS:
        if marker in text:
            score += 35
    if "?" in segment.text:
        score += 20
    if segment.clip_id == dependent_segment.clip_id and segment.start < dependent_segment.start:
        score += 30
    if _starts_with_contextless_opening(segment.text):
        score -= 80
    return score


def _intro_candidate_segments(transcriptions: list[ClipTranscription], excluded: Segment) -> list[Segment]:
    scored_candidates: list[tuple[int, int, float, Segment]] = []
    for clip in transcriptions:
        for segment in clip.segments:
            score = _intro_candidate_score(segment, excluded)
            if score <= 0:
                continue
            same_clip_bonus = 1 if segment.clip_id == excluded.clip_id else 0
            scored_candidates.append((score, same_clip_bonus, -abs(segment.start - excluded.start), segment))
    scored_candidates.sort(key=lambda item: item[:3], reverse=True)
    return [segment for _score, _same_clip_bonus, _distance, segment in scored_candidates[:_SUBJECT_CANDIDATE_LIMIT]]


def _contextless_opening_message(segment: Segment, transcriptions: list[ClipTranscription]) -> str | None:
    if not _starts_with_contextless_opening(segment.text):
        return None
    candidates = _intro_candidate_segments(transcriptions, segment)
    candidate_text = "No obvious introduction candidate found; choose a complete opening with explicit context."
    if candidates:
        candidate_text = "Candidate context openings: " + "; ".join(
            f"clip_id={candidate.clip_id} segment_id={candidate.id} text='{_short_text_preview(candidate.text)}'"
            for candidate in candidates
        )
    return (
        "contextless_opening: first retained segment "
        f"clip_id={segment.clip_id} segment_id={segment.id} text='{_short_text_preview(segment.text)}' "
        "starts with a pronoun, answer or internal reference that needs prior context. "
        "Do not start the edit there; prepend an introduction/question/context segment or choose another opening. "
        f"{candidate_text}"
    )


def _subjectless_opening_message(segment: Segment, transcriptions: list[ClipTranscription]) -> str | None:
    if not _starts_with_subjectless_opening(segment.text):
        return None
    candidates = _subject_candidate_segments(transcriptions, segment)
    candidate_text = "No obvious subject candidate found; choose a complete opening that explicitly names the subject."
    if candidates:
        candidate_text = "Candidate subject openings: " + "; ".join(
            f"clip_id={candidate.clip_id} segment_id={candidate.id} text='{_short_text_preview(candidate.text)}'"
            for candidate in candidates
        )
    return (
        "subjectless_opening: first retained segment "
        f"clip_id={segment.clip_id} segment_id={segment.id} starts without naming the subject. "
        f"Do not start the edit with clip_id={segment.clip_id} segment_id={segment.id} alone. "
        "Place a segment that explicitly names the subject before it, or choose a different complete opening. "
        f"{candidate_text}"
    )


def _segments_by_clip_order(transcriptions: list[ClipTranscription]) -> dict[int, list[Segment]]:
    return {
        clip.clip_id: sorted(clip.segments, key=lambda item: item.start)
        for clip in transcriptions
    }


def _previous_source_segment(segment: Segment, ordered_segments: list[Segment]) -> Segment | None:
    for index, candidate in enumerate(ordered_segments):
        if candidate.id == segment.id:
            if index == 0:
                return None
            return ordered_segments[index - 1]
    return None


def _starts_with_context_continuation(text: str) -> bool:
    tokens = _normalized_tokens(text)
    if not tokens:
        return False
    return tokens[0] in _CONTEXT_CONTINUATION_START_WORDS or _starts_with_lowercase_word(text)


def _previous_context_segment_if_required(segment: Segment, ordered_segments: list[Segment]) -> Segment | None:
    previous = _previous_source_segment(segment, ordered_segments)
    if previous is None:
        return None
    if _ends_with_terminal_punctuation(previous.text):
        return None
    if not _starts_with_context_continuation(segment.text):
        return None
    return previous


def _opening_context_fragment_message(segment: Segment, ordered_segments: list[Segment]) -> str | None:
    previous = _previous_source_segment(segment, ordered_segments)
    if previous is None:
        return None
    if _ends_with_terminal_punctuation(previous.text) or not _starts_with_lowercase_word(segment.text):
        return None
    return (
        "opening_context_fragment: first retained segment "
        f"clip_id={segment.clip_id} segment_id={segment.id} starts from an unclosed sentence fragment. "
        f"The preceding segment_id={previous.id} ends without terminal punctuation and provides required context. "
        f"Do not start the edit with clip_id={segment.clip_id} segment_id={segment.id} alone; "
        f"include preceding segment_id={previous.id} immediately before it, or choose a different complete opening."
    )


def _detached_context_fragment_message(
    retained_segments: list[Segment],
    ordered_segments_by_clip: dict[int, list[Segment]],
) -> str | None:
    for index, segment in enumerate(retained_segments):
        previous = _previous_context_segment_if_required(segment, ordered_segments_by_clip[segment.clip_id])
        if previous is None:
            continue
        if (
            index > 0
            and retained_segments[index - 1].clip_id == previous.clip_id
            and retained_segments[index - 1].id == previous.id
        ):
            continue
        return (
            "detached_context_fragment: retained segment "
            f"clip_id={segment.clip_id} segment_id={segment.id} starts from a sentence continuation "
            f"without its immediate source context. The preceding segment_id={previous.id} provides "
            "the required subject/context and must be placed immediately before it in playback order."
        )
    return None


def _missing_context_antecedent_message(
    retained_segments: list[Segment],
    transcriptions: list[ClipTranscription],
) -> str | None:
    for index, segment in enumerate(retained_segments):
        reference_kind = _context_reference_kind(segment)
        if reference_kind is None:
            continue
        if any(_segment_has_context_antecedent(prior_segment, reference_kind) for prior_segment in retained_segments[:index]):
            continue
        candidates = _context_antecedent_candidates(transcriptions, segment, reference_kind)
        if not candidates:
            continue
        candidate_text = "; ".join(
            f"clip_id={candidate.clip_id} segment_id={candidate.id} text='{_short_text_preview(candidate.text)}'"
            for candidate in candidates
        )
        return (
            "missing_context_antecedent: retained segment "
            f"clip_id={segment.clip_id} segment_id={segment.id} references a legal/normative antecedent "
            "before that antecedent has been introduced. "
            "Place an explicit law/norm/decree/article segment before references such as "
            "'la stessa norma', 'questa norma' or 'la stessa legge'. "
            f"Candidate antecedents: {candidate_text}."
        )
    return None


def _retained_segments_in_playback_order(
    narrative_map: NarrativeMap,
    segments_by_key: dict[tuple[int, int], Segment],
) -> list[Segment]:
    retained_segments: list[Segment] = []
    for clip in sorted(narrative_map.ordered_clips, key=lambda item: item.playback_position):
        for segment in clip.segments:
            if segment.importance == SegmentImportance.REDUNDANT:
                continue
            if not segment.keep_verbatim:
                continue
            source_segment = segments_by_key.get((clip.clip_id, segment.segment_id))
            if source_segment is not None:
                retained_segments.append(source_segment)
    return retained_segments


def _retained_duration_s(narrative_map: NarrativeMap, segments_by_key: dict[tuple[int, int], Segment]) -> float:
    return sum(
        max(0.0, segment.end - segment.start)
        for segment in _retained_segments_in_playback_order(narrative_map, segments_by_key)
    )


def _retained_segment_ids(narrative_clip: NarrativeClip) -> list[int]:
    return [
        segment.segment_id
        for segment in narrative_clip.segments
        if segment.keep_verbatim and segment.importance != SegmentImportance.REDUNDANT
    ]


def _spoken_duration(transcriptions: list[ClipTranscription]) -> float:
    return sum(
        max(0.0, segment.end - segment.start)
        for clip in transcriptions
        for segment in clip.segments
    )


def _segment_count(transcriptions: list[ClipTranscription]) -> int:
    return sum(len(clip.segments) for clip in transcriptions)


def _minimum_retained_duration(total_spoken_duration_s: float, brief: EditorialBrief) -> float:
    reduction = target_reduction_ratio(brief.synthesis_level)
    default_minimum = total_spoken_duration_s * max(0.0, 1.0 - reduction - 0.20)
    if brief.target_duration_s is None:
        return default_minimum
    return min(default_minimum, max(0.0, brief.target_duration_s * 0.75))


def maximum_target_duration_s(total_spoken_duration_s: float, brief: EditorialBrief) -> float | None:
    if brief.target_duration_s is None:
        return None
    return min(total_spoken_duration_s, max(brief.target_duration_s * 1.10, brief.target_duration_s + 5.0))


def _maximum_retained_duration(total_spoken_duration_s: float, brief: EditorialBrief) -> float | None:
    return maximum_target_duration_s(total_spoken_duration_s, brief)


def _duration_budget_prompt_line(transcriptions: list[ClipTranscription], brief: EditorialBrief) -> str:
    total_spoken_duration_s = _spoken_duration(transcriptions)
    minimum_duration_s = _minimum_retained_duration(total_spoken_duration_s, brief)
    reduction = target_reduction_ratio(brief.synthesis_level)
    if brief.target_duration_s is not None:
        return (
            f"durata parlata totale stimata={total_spoken_duration_s:.1f}s; "
            f"durata target finale richiesta={brief.target_duration_s:.1f}s "
            f"({brief.target_duration_s / 60:.2f} min). "
            f"Obiettivo operativo: avvicinati al target, idealmente tra "
            f"{brief.target_duration_s * 0.95:.1f}s e {brief.target_duration_s * 1.05:.1f}s; "
            "non superare il target quando esistono tagli sicuri, e "
            f"non superare {max(brief.target_duration_s * 1.10, brief.target_duration_s + 5.0):.1f}s "
            "se esistono tagli sicuri. "
            "Se il target e troppo aggressivo, preserva coerenza narrativa, nomi, numeri e passaggi logici."
        )
    if total_spoken_duration_s < 30:
        return (
            f"durata parlata totale stimata={total_spoken_duration_s:.1f}s; "
            f"synthesis_level={brief.synthesis_level} rimozione target={reduction:.0%}."
        )
    return (
        f"durata parlata totale stimata={total_spoken_duration_s:.1f}s; "
        f"per synthesis_level={brief.synthesis_level} rimozione target={reduction:.0%}, "
        f"durata minima attesa del parlato mantenuto circa {minimum_duration_s:.1f}s. "
        "non creare una mini-compilation o highlights troppo brevi se target_duration_s non lo richiede esplicitamente."
    )


def _brief_with_implicit_synthesis_target(
    transcriptions: list[ClipTranscription],
    brief: EditorialBrief,
) -> EditorialBrief:
    if brief.target_duration_s is not None:
        return brief
    reduction = target_reduction_ratio(brief.synthesis_level)
    if reduction <= 0:
        return brief
    total_spoken_duration_s = _spoken_duration(transcriptions)
    if total_spoken_duration_s <= 0:
        return brief
    return brief.model_copy(update={"target_duration_s": total_spoken_duration_s * (1.0 - reduction)})


def _is_creative_zoom_candidate(segment: NarrativeSegment) -> bool:
    return segment.keep_verbatim and segment.importance in {SegmentImportance.ESSENTIAL, SegmentImportance.VALUABLE}


def normalize_playback_positions(narrative_map: NarrativeMap) -> None:
    ordered_clips = sorted(
        enumerate(narrative_map.ordered_clips),
        key=lambda indexed_clip: (indexed_clip[1].playback_position, indexed_clip[0]),
    )
    narrative_map.ordered_clips = [clip for _index, clip in ordered_clips]
    for position, clip in enumerate(narrative_map.ordered_clips):
        clip.playback_position = position


def _find_narrative_segment(narrative_clip: NarrativeClip, segment_id: int) -> NarrativeSegment | None:
    for segment in narrative_clip.segments:
        if segment.segment_id == segment_id:
            return segment
    return None


def _retain_segment_for_cut_safety(
    narrative_clip: NarrativeClip,
    segment_id: int,
    before_segment_id: int,
) -> None:
    existing_segment = _find_narrative_segment(narrative_clip, segment_id)
    reason = f"Added by cut safety to complete the opening sentence before segment_id={before_segment_id}."
    if existing_segment is not None:
        existing_segment.keep_verbatim = True
        existing_segment.synthesis_text = None
        if existing_segment.importance == SegmentImportance.REDUNDANT:
            existing_segment.importance = SegmentImportance.VALUABLE
        if "cut safety" not in existing_segment.reason.casefold():
            existing_segment.reason = f"{existing_segment.reason} {reason}"
        return

    insert_index = 0
    for index, segment in enumerate(narrative_clip.segments):
        if segment.segment_id == before_segment_id:
            insert_index = index
            break
    narrative_clip.segments.insert(
        insert_index,
        NarrativeSegment(
            segment_id=segment_id,
            importance=SegmentImportance.VALUABLE,
            keep_verbatim=True,
            synthesis_text=None,
            reason=reason,
        ),
    )


def _remove_narrative_segment(narrative_map: NarrativeMap, clip_id: int, segment_id: int) -> None:
    for narrative_clip in narrative_map.ordered_clips:
        if narrative_clip.clip_id == clip_id:
            narrative_clip.segments = [
                segment
                for segment in narrative_clip.segments
                if segment.segment_id != segment_id
            ]
    narrative_map.ordered_clips = [
        narrative_clip
        for narrative_clip in narrative_map.ordered_clips
        if narrative_clip.segments
    ]


def _best_subject_candidate_for_opening(
    transcriptions: list[ClipTranscription],
    first_retained_segment: Segment,
) -> Segment | None:
    for candidate in _subject_candidate_segments(transcriptions, first_retained_segment):
        if candidate.clip_id != first_retained_segment.clip_id or candidate.start <= first_retained_segment.start:
            return candidate
    return None


def _prepend_subject_candidate_for_cut_safety(
    narrative_map: NarrativeMap,
    candidate: Segment,
    before_segment: Segment,
) -> None:
    _remove_narrative_segment(narrative_map, candidate.clip_id, candidate.id)
    narrative_map.ordered_clips.insert(
        0,
        NarrativeClip(
            clip_id=candidate.clip_id,
            playback_position=0,
            topic_summary="Contesto soggetto",
            role="introduction",
            segments=[
                NarrativeSegment(
                    segment_id=candidate.id,
                    importance=SegmentImportance.VALUABLE,
                    keep_verbatim=True,
                    synthesis_text=None,
                    reason=(
                        "Added by cut safety to name the subject before "
                        f"subjectless opening clip_id={before_segment.clip_id} segment_id={before_segment.id}."
                    ),
                )
            ],
        ),
    )
    normalize_playback_positions(narrative_map)


def _prepend_intro_candidate_for_cut_safety(
    narrative_map: NarrativeMap,
    candidate: Segment,
    before_segment: Segment,
    narrative_clip: NarrativeClip | None,
) -> None:
    if narrative_clip is not None and narrative_clip.clip_id == candidate.clip_id:
        if candidate.start > before_segment.start:
            _replace_contextless_opening_with_intro_candidate(narrative_clip, candidate, before_segment)
            return
        _retain_segment_for_cut_safety(
            narrative_clip,
            candidate.id,
            before_segment.id,
        )
        return
    _remove_narrative_segment(narrative_map, candidate.clip_id, candidate.id)
    narrative_map.ordered_clips.insert(
        0,
        NarrativeClip(
            clip_id=candidate.clip_id,
            playback_position=0,
            topic_summary="Contesto apertura",
            role="introduction",
            segments=[
                NarrativeSegment(
                    segment_id=candidate.id,
                    importance=SegmentImportance.VALUABLE,
                    keep_verbatim=True,
                    synthesis_text=None,
                    reason=(
                        "Added by cut safety to provide narrative context before "
                        f"contextless opening clip_id={before_segment.clip_id} segment_id={before_segment.id}."
                    ),
                )
            ],
        ),
    )
    normalize_playback_positions(narrative_map)


def _replace_contextless_opening_with_intro_candidate(
    narrative_clip: NarrativeClip,
    candidate: Segment,
    before_segment: Segment,
) -> None:
    removed_reason = (
        "Removed by cut safety because this contextless opening cannot be prepended to a later "
        f"same-clip context segment_id={candidate.id} without breaking source order."
    )
    before_narrative_segment = _find_narrative_segment(narrative_clip, before_segment.id)
    if before_narrative_segment is not None:
        before_narrative_segment.keep_verbatim = False
        before_narrative_segment.importance = SegmentImportance.REDUNDANT
        before_narrative_segment.creative_zoom = False
        before_narrative_segment.synthesis_text = None
        if "cut safety" not in before_narrative_segment.reason.casefold():
            before_narrative_segment.reason = f"{before_narrative_segment.reason} {removed_reason}"

    existing_candidate = _find_narrative_segment(narrative_clip, candidate.id)
    candidate_reason = (
        "Added by cut safety as a same-clip replacement opening after removing "
        f"contextless segment_id={before_segment.id}."
    )
    if existing_candidate is not None:
        existing_candidate.keep_verbatim = True
        existing_candidate.synthesis_text = None
        if existing_candidate.importance == SegmentImportance.REDUNDANT:
            existing_candidate.importance = SegmentImportance.VALUABLE
        if "cut safety" not in existing_candidate.reason.casefold():
            existing_candidate.reason = f"{existing_candidate.reason} {candidate_reason}"
        return

    insert_index = 0
    for index, segment in enumerate(narrative_clip.segments):
        if segment.segment_id == before_segment.id:
            insert_index = index + 1
            break
    narrative_clip.segments.insert(
        insert_index,
        NarrativeSegment(
            segment_id=candidate.id,
            importance=SegmentImportance.VALUABLE,
            keep_verbatim=True,
            synthesis_text=None,
            reason=candidate_reason,
        ),
    )


def _sort_narrative_clip_segments_by_source_order(
    narrative_clip: NarrativeClip,
    segments_by_key: dict[tuple[int, int], Segment],
) -> None:
    indexed_segments = sorted(
        enumerate(narrative_clip.segments),
        key=lambda indexed_segment: (
            segments_by_key[(narrative_clip.clip_id, indexed_segment[1].segment_id)].start,
            indexed_segment[0],
        ),
    )
    narrative_clip.segments = [segment for _index, segment in indexed_segments]


def _retained_segment_occurrences(
    narrative_map: NarrativeMap,
    segments_by_key: dict[tuple[int, int], Segment],
) -> list[tuple[NarrativeClip, NarrativeSegment, Segment]]:
    occurrences: list[tuple[NarrativeClip, NarrativeSegment, Segment]] = []
    for narrative_clip in sorted(narrative_map.ordered_clips, key=lambda item: item.playback_position):
        for narrative_segment in narrative_clip.segments:
            if narrative_segment.importance == SegmentImportance.REDUNDANT or not narrative_segment.keep_verbatim:
                continue
            source_segment = segments_by_key.get((narrative_clip.clip_id, narrative_segment.segment_id))
            if source_segment is not None:
                occurrences.append((narrative_clip, narrative_segment, source_segment))
    return occurrences


def _remove_empty_narrative_clips(narrative_map: NarrativeMap) -> None:
    narrative_map.ordered_clips = [clip for clip in narrative_map.ordered_clips if clip.segments]
    normalize_playback_positions(narrative_map)


def _move_narrative_segment_after_context(
    narrative_map: NarrativeMap,
    current_clip: NarrativeClip,
    current_segment: NarrativeSegment,
    context_clip: NarrativeClip,
    context_segment_id: int,
) -> None:
    current_clip.segments = [segment for segment in current_clip.segments if segment is not current_segment]
    reason = f"Moved by cut safety to keep sentence context after segment_id={context_segment_id}."
    if "cut safety" not in current_segment.reason.casefold():
        current_segment.reason = f"{current_segment.reason} {reason}"
    insert_index = len(context_clip.segments)
    for index, segment in enumerate(context_clip.segments):
        if segment.segment_id == context_segment_id:
            insert_index = index + 1
            break
    context_clip.segments.insert(insert_index, current_segment)
    _remove_empty_narrative_clips(narrative_map)


def _insert_clip_before_reference(
    narrative_map: NarrativeMap,
    inserted_clip: NarrativeClip,
    reference_clip: NarrativeClip,
) -> None:
    insert_index = next(
        (
            index
            for index, narrative_clip in enumerate(narrative_map.ordered_clips)
            if narrative_clip is reference_clip
        ),
        len(narrative_map.ordered_clips),
    )
    narrative_map.ordered_clips.insert(insert_index, inserted_clip)
    normalize_playback_positions(narrative_map)


def _move_narrative_segment_before_reference(
    narrative_map: NarrativeMap,
    antecedent_clip: NarrativeClip,
    antecedent_segment: NarrativeSegment,
    reference_clip: NarrativeClip,
    reference_segment: Segment,
) -> None:
    antecedent_clip.segments = [segment for segment in antecedent_clip.segments if segment is not antecedent_segment]
    reason = (
        "Moved by cut safety before "
        f"clip_id={reference_segment.clip_id} segment_id={reference_segment.id} "
        "because it defines the legal/normative reference used there."
    )
    if "cut safety" not in antecedent_segment.reason.casefold():
        antecedent_segment.reason = f"{antecedent_segment.reason} {reason}"
    inserted_clip = NarrativeClip(
        clip_id=antecedent_clip.clip_id,
        playback_position=reference_clip.playback_position,
        topic_summary=antecedent_clip.topic_summary,
        role=antecedent_clip.role,
        segments=[antecedent_segment],
    )
    _insert_clip_before_reference(narrative_map, inserted_clip, reference_clip)
    _remove_empty_narrative_clips(narrative_map)


def _insert_context_antecedent_for_cut_safety(
    narrative_map: NarrativeMap,
    candidate: Segment,
    reference_clip: NarrativeClip,
    reference_segment: Segment,
) -> None:
    _remove_narrative_segment(narrative_map, candidate.clip_id, candidate.id)
    inserted_clip = NarrativeClip(
        clip_id=candidate.clip_id,
        playback_position=reference_clip.playback_position,
        topic_summary="Contesto normativo",
        role="supporting",
        segments=[
            NarrativeSegment(
                segment_id=candidate.id,
                importance=SegmentImportance.VALUABLE,
                keep_verbatim=True,
                synthesis_text=None,
                reason=(
                    "Added by cut safety before "
                    f"clip_id={reference_segment.clip_id} segment_id={reference_segment.id} "
                    "because it defines the legal/normative reference used there."
                ),
            )
        ],
    )
    _insert_clip_before_reference(narrative_map, inserted_clip, reference_clip)


def _repair_detached_context_fragments(
    narrative_map: NarrativeMap,
    segments_by_key: dict[tuple[int, int], Segment],
    ordered_segments_by_clip: dict[int, list[Segment]],
) -> None:
    while True:
        occurrences = _retained_segment_occurrences(narrative_map, segments_by_key)
        changed = False
        for index, (narrative_clip, narrative_segment, source_segment) in enumerate(occurrences):
            previous_source_segment = _previous_context_segment_if_required(
                source_segment,
                ordered_segments_by_clip[source_segment.clip_id],
            )
            if previous_source_segment is None:
                continue
            if (
                index > 0
                and occurrences[index - 1][2].clip_id == previous_source_segment.clip_id
                and occurrences[index - 1][2].id == previous_source_segment.id
            ):
                continue
            previous_occurrence = next(
                (
                    occurrence
                    for occurrence in occurrences
                    if occurrence[2].clip_id == previous_source_segment.clip_id
                    and occurrence[2].id == previous_source_segment.id
                ),
                None,
            )
            if previous_occurrence is None:
                _retain_segment_for_cut_safety(
                    narrative_clip,
                    previous_source_segment.id,
                    source_segment.id,
                )
            else:
                _move_narrative_segment_after_context(
                    narrative_map,
                    narrative_clip,
                    narrative_segment,
                    previous_occurrence[0],
                    previous_source_segment.id,
                )
            changed = True
            break
        if not changed:
            return


def _repair_context_antecedent_references(
    narrative_map: NarrativeMap,
    transcriptions: list[ClipTranscription],
    segments_by_key: dict[tuple[int, int], Segment],
) -> None:
    while True:
        occurrences = _retained_segment_occurrences(narrative_map, segments_by_key)
        changed = False
        for index, (reference_clip, _reference_narrative_segment, reference_segment) in enumerate(occurrences):
            reference_kind = _context_reference_kind(reference_segment)
            if reference_kind is None:
                continue
            if any(
                _segment_has_context_antecedent(prior_occurrence[2], reference_kind)
                for prior_occurrence in occurrences[:index]
            ):
                continue
            retained_antecedent = next(
                (
                    occurrence
                    for occurrence in occurrences[index + 1 :]
                    if _segment_has_context_antecedent(occurrence[2], reference_kind)
                ),
                None,
            )
            if retained_antecedent is not None:
                _move_narrative_segment_before_reference(
                    narrative_map,
                    retained_antecedent[0],
                    retained_antecedent[1],
                    reference_clip,
                    reference_segment,
                )
                changed = True
                break
            candidates = _context_antecedent_candidates(transcriptions, reference_segment, reference_kind)
            if candidates:
                _insert_context_antecedent_for_cut_safety(
                    narrative_map,
                    candidates[0],
                    reference_clip,
                    reference_segment,
                )
                changed = True
                break
        if not changed:
            return


def _remove_duplicate_retained_source_segments(narrative_map: NarrativeMap) -> None:
    seen: set[tuple[int, int]] = set()
    removed: list[tuple[int, int]] = []
    for narrative_clip in sorted(narrative_map.ordered_clips, key=lambda item: item.playback_position):
        next_segments: list[NarrativeSegment] = []
        for segment in narrative_clip.segments:
            key = (narrative_clip.clip_id, segment.segment_id)
            if segment.keep_verbatim and segment.importance != SegmentImportance.REDUNDANT:
                if key in seen:
                    removed.append(key)
                    continue
                seen.add(key)
            next_segments.append(segment)
        narrative_clip.segments = next_segments
    if removed:
        logger.bind(removed_duplicate_segments=removed).info(
            "Removed duplicate retained source segments during cut-safety repair"
        )
        _remove_empty_narrative_clips(narrative_map)


def _demote_contextless_opening_for_cut_safety(
    narrative_segment: NarrativeSegment,
    replacement: Segment | None,
) -> None:
    replacement_text = (
        f" in favor of clip_id={replacement.clip_id} segment_id={replacement.id}"
        if replacement is not None
        else ""
    )
    reason = (
        "Removed by cut safety because this contextless opening still appeared first"
        f"{replacement_text}."
    )
    narrative_segment.keep_verbatim = False
    narrative_segment.importance = SegmentImportance.REDUNDANT
    narrative_segment.creative_zoom = False
    narrative_segment.synthesis_text = None
    if "cut safety" not in narrative_segment.reason.casefold():
        narrative_segment.reason = f"{narrative_segment.reason} {reason}"


def _repair_first_contextless_opening_until_stable(
    narrative_map: NarrativeMap,
    transcriptions: list[ClipTranscription],
    segments_by_key: dict[tuple[int, int], Segment],
) -> None:
    max_attempts = max(1, len(segments_by_key))
    for _attempt in range(max_attempts):
        occurrences = _retained_segment_occurrences(narrative_map, segments_by_key)
        if not occurrences:
            return
        first_clip, first_narrative_segment, first_segment = occurrences[0]
        if not _starts_with_contextless_opening(first_segment.text):
            return

        candidates = [
            candidate
            for candidate in _intro_candidate_segments(transcriptions, first_segment)
            if (candidate.clip_id, candidate.id) != (first_segment.clip_id, first_segment.id)
        ]
        previous_key = (first_segment.clip_id, first_segment.id)
        if candidates:
            _prepend_intro_candidate_for_cut_safety(
                narrative_map,
                candidates[0],
                first_segment,
                first_clip,
            )
        elif len(occurrences) > 1:
            _demote_contextless_opening_for_cut_safety(first_narrative_segment, None)
        else:
            return

        _remove_duplicate_retained_source_segments(narrative_map)
        for narrative_clip in narrative_map.ordered_clips:
            if all((narrative_clip.clip_id, segment.segment_id) in segments_by_key for segment in narrative_clip.segments):
                _sort_narrative_clip_segments_by_source_order(narrative_clip, segments_by_key)

        next_occurrences = _retained_segment_occurrences(narrative_map, segments_by_key)
        if not next_occurrences:
            return
        next_key = (next_occurrences[0][2].clip_id, next_occurrences[0][2].id)
        if next_key == previous_key:
            _demote_contextless_opening_for_cut_safety(
                next_occurrences[0][1],
                candidates[0] if candidates else None,
            )


def repair_narrative_map_for_cut_safety(
    narrative_map: NarrativeMap,
    transcriptions: list[ClipTranscription],
) -> NarrativeMap:
    segments_by_key = {
        (clip.clip_id, segment.id): segment
        for clip in transcriptions
        for segment in clip.segments
    }
    ordered_segments_by_clip = _segments_by_clip_order(transcriptions)

    retained_occurrences = _retained_segment_occurrences(narrative_map, segments_by_key)
    if retained_occurrences:
        first_narrative_clip, _first_narrative_segment, first_retained_segment = retained_occurrences[0]
        if _starts_with_contextless_opening(first_retained_segment.text):
            intro_candidate = next(iter(_intro_candidate_segments(transcriptions, first_retained_segment)), None)
            if intro_candidate is not None:
                _prepend_intro_candidate_for_cut_safety(
                    narrative_map,
                    intro_candidate,
                    first_retained_segment,
                    first_narrative_clip,
                )
                retained_occurrences = _retained_segment_occurrences(narrative_map, segments_by_key)
                if not retained_occurrences:
                    first_retained_segment = None
                else:
                    first_narrative_clip, _first_narrative_segment, first_retained_segment = retained_occurrences[0]
        if first_retained_segment is not None and _starts_with_subjectless_opening(first_retained_segment.text):
            subject_candidate = _best_subject_candidate_for_opening(transcriptions, first_retained_segment)
            if subject_candidate is not None:
                _prepend_subject_candidate_for_cut_safety(
                    narrative_map,
                    subject_candidate,
                    first_retained_segment,
                )
                retained_occurrences = _retained_segment_occurrences(narrative_map, segments_by_key)
                if retained_occurrences:
                    first_narrative_clip, _first_narrative_segment, first_retained_segment = retained_occurrences[0]
        if first_retained_segment is not None:
            previous_segment = _previous_source_segment(
                first_retained_segment,
                ordered_segments_by_clip[first_retained_segment.clip_id],
            )
            if (
                previous_segment is not None
                and not _ends_with_terminal_punctuation(previous_segment.text)
                and _starts_with_lowercase_word(first_retained_segment.text)
            ):
                _retain_segment_for_cut_safety(
                    first_narrative_clip,
                    previous_segment.id,
                    first_retained_segment.id,
                )

    _repair_detached_context_fragments(
        narrative_map,
        segments_by_key,
        ordered_segments_by_clip,
    )
    _repair_context_antecedent_references(
        narrative_map,
        transcriptions,
        segments_by_key,
    )
    _repair_detached_context_fragments(
        narrative_map,
        segments_by_key,
        ordered_segments_by_clip,
    )
    _remove_duplicate_retained_source_segments(narrative_map)

    for narrative_clip in narrative_map.ordered_clips:
        if all((narrative_clip.clip_id, segment.segment_id) in segments_by_key for segment in narrative_clip.segments):
            _sort_narrative_clip_segments_by_source_order(narrative_clip, segments_by_key)

    _repair_first_contextless_opening_until_stable(
        narrative_map,
        transcriptions,
        segments_by_key,
    )

    return narrative_map


def validate_narrative_map(
    narrative_map: NarrativeMap,
    transcriptions: list[ClipTranscription],
    editing_options: EditingOptions | None = None,
    brief: EditorialBrief | None = None,
    enforce_opening: bool = True,
    enforce_duration_budget: bool = True,
) -> None:
    normalize_playback_positions(narrative_map)
    clip_ids = {clip.clip_id for clip in transcriptions}
    segment_ids_by_clip = {
        clip.clip_id: {segment.id for segment in clip.segments}
        for clip in transcriptions
    }
    segments_by_key = {
        (clip.clip_id, segment.id): segment
        for clip in transcriptions
        for segment in clip.segments
    }
    ordered_segments_by_clip = _segments_by_clip_order(transcriptions)

    for clip in narrative_map.ordered_clips:
        if clip.clip_id not in clip_ids:
            raise NarrativeValidationError(f"Invalid clip_id {clip.clip_id}")
        valid_segment_ids = segment_ids_by_clip[clip.clip_id]
        retained_source_segments_for_clip: list[Segment] = []
        for segment in clip.segments:
            if segment.segment_id not in valid_segment_ids:
                raise NarrativeValidationError(f"Invalid segment_id {segment.segment_id} for clip_id {clip.clip_id}")
            source_segment = segments_by_key[(clip.clip_id, segment.segment_id)]
            if segment.remove_word_ranges:
                word_count = len(source_segment.words)
                removed_word_indexes: set[int] = set()
                for removal_range in segment.remove_word_ranges:
                    if removal_range.end_word_index >= word_count:
                        raise NarrativeValidationError(
                            "Invalid word removal range "
                            f"{removal_range.start_word_index}-{removal_range.end_word_index} "
                            f"for clip_id={clip.clip_id} segment_id={segment.segment_id}; "
                            f"segment has {word_count} words"
                        )
                    removed_word_indexes.update(
                        range(removal_range.start_word_index, removal_range.end_word_index + 1)
                    )
                if word_count > 0 and len(removed_word_indexes) >= word_count:
                    raise NarrativeValidationError(
                        "Word removal cannot remove the entire retained segment; "
                        f"mark clip_id={clip.clip_id} segment_id={segment.segment_id} as redundant instead"
                    )
            if segment.creative_zoom and not _is_creative_zoom_candidate(segment):
                raise NarrativeValidationError(
                    f"creative_zoom true is only valid on retained essential or valuable segments "
                    f"(clip_id={clip.clip_id}, segment_id={segment.segment_id})"
                )
            if segment.keep_verbatim and segment.importance != SegmentImportance.REDUNDANT:
                previous_segment = retained_source_segments_for_clip[-1] if retained_source_segments_for_clip else None
                if previous_segment is not None and source_segment.start < previous_segment.start:
                    current_segments = retained_source_segments_for_clip + [source_segment]
                    current_order = [candidate.id for candidate in current_segments]
                    source_order = [candidate.id for candidate in sorted(current_segments, key=lambda item: item.start)]
                    raise NarrativeValidationError(
                        "non_monotonic_clip_segments: "
                        f"clip_id={clip.clip_id} retained segments must follow source timeline order. "
                        f"Current retained order: {current_order}. Required source order: {source_order}. "
                        f"Do not place segment_id={source_segment.id} after segment_id={previous_segment.id}; "
                        "inside the same clip, keep retained segment IDs in increasing source timestamp order."
                    )
                retained_source_segments_for_clip.append(source_segment)
            if segment.importance in {SegmentImportance.ESSENTIAL, SegmentImportance.VALUABLE} and not segment.keep_verbatim:
                raise NarrativeValidationError(
                    f"Segment {segment.segment_id} is marked as rewritten, but source audio rendering requires keep_verbatim"
                )
            if segment.keep_verbatim and segment.synthesis_text is not None:
                source_text = " ".join(source_segment.text.split()).casefold()
                synthesis_text = " ".join(segment.synthesis_text.split()).casefold()
                if synthesis_text and synthesis_text != source_text:
                    raise NarrativeValidationError(
                        f"Segment {segment.segment_id} has synthesis_text that does not match source audio"
                    )

    retained_segments = _retained_segments_in_playback_order(narrative_map, segments_by_key)
    if enforce_opening and retained_segments:
        contextless_message = _contextless_opening_message(retained_segments[0], transcriptions)
        if contextless_message is not None:
            raise NarrativeValidationError(contextless_message)
        subjectless_message = _subjectless_opening_message(retained_segments[0], transcriptions)
        if subjectless_message is not None:
            raise NarrativeValidationError(subjectless_message)
        opening_context_message = _opening_context_fragment_message(
            retained_segments[0],
            ordered_segments_by_clip[retained_segments[0].clip_id],
        )
        if opening_context_message is not None:
            raise NarrativeValidationError(opening_context_message)
        detached_context_message = _detached_context_fragment_message(retained_segments, ordered_segments_by_clip)
        if detached_context_message is not None:
            raise NarrativeValidationError(detached_context_message)

    missing_antecedent_message = _missing_context_antecedent_message(retained_segments, transcriptions)
    if missing_antecedent_message is not None:
        raise NarrativeValidationError(missing_antecedent_message)

    if brief is not None and enforce_duration_budget:
        total_spoken_duration_s = _spoken_duration(transcriptions)
        if total_spoken_duration_s >= 30:
            retained_duration_s = sum(max(0.0, segment.end - segment.start) for segment in retained_segments)
            minimum_duration_s = _minimum_retained_duration(total_spoken_duration_s, brief)
            if retained_duration_s + 0.001 < minimum_duration_s:
                raise NarrativeValidationError(
                    "excessive_synthesis: "
                    f"retained_duration_s={retained_duration_s:.1f}s is too short for "
                    f"synthesis_level={brief.synthesis_level}; minimum expected around "
                    f"{minimum_duration_s:.1f}s from total spoken duration {total_spoken_duration_s:.1f}s. "
                    "Do not produce only highlights or a mini-compilation; retain a complete edit proposal "
                    "unless target_duration_s explicitly requests a shorter video."
                )
            maximum_duration_s = _maximum_retained_duration(total_spoken_duration_s, brief)
            if maximum_duration_s is not None and retained_duration_s > maximum_duration_s + 0.001:
                raise NarrativeValidationError(
                    "target_duration_exceeded: "
                    f"retained_duration_s={retained_duration_s:.1f}s exceeds target_duration_s="
                    f"{brief.target_duration_s:.1f}s with maximum tolerated around {maximum_duration_s:.1f}s. "
                    "Classify additional non-essential repetitions, pauses, digressions and optional blocks as redundant "
                    "while preserving narrative coherence."
                )

    for filler in narrative_map.filler_words:
        if filler.clip_id not in clip_ids:
            raise NarrativeValidationError(f"Invalid filler clip_id {filler.clip_id}")
        valid_segment_ids = segment_ids_by_clip[filler.clip_id]
        if filler.segment_id not in valid_segment_ids:
            raise NarrativeValidationError(f"Invalid filler segment_id {filler.segment_id}")


def _validate_narrative_map_allowing_excessive_synthesis(
    narrative_map: NarrativeMap,
    transcriptions: list[ClipTranscription],
    editing_options: EditingOptions | None,
    brief: EditorialBrief,
) -> None:
    try:
        validate_narrative_map(narrative_map, transcriptions, editing_options, brief)
    except NarrativeValidationError as exc:
        if not str(exc).startswith("excessive_synthesis:"):
            raise
        validate_narrative_map(
            narrative_map,
            transcriptions,
            editing_options,
            brief,
            enforce_duration_budget=False,
        )


def _creative_zoom_prompt_lines(editing_options: EditingOptions | None) -> list[str]:
    if editing_options is not None and editing_options.enable_creative_zoom:
        return [
            "Zoom creativo attivo: non usarlo come scelta editoriale autonoma nella NarrativeMap.",
            "Il renderer applichera zoom fisso, non progressivo, sui segmenti individuati dalle pause mediamente piu lunghe nella proposta editoriale.",
            "Le scale saranno variabili tra 20% e 50% per creare stacchi con diverso grado di zoom avanti o indietro.",
            "Puoi lasciare creative_zoom=false: il rilevamento finale usa i timestamp parola e le pause della proposta.",
        ]
    return ["Zoom creativo non attivo: imposta creative_zoom=false per tutti i segmenti."]


def _compact_output_prompt_lines(transcriptions: list[ClipTranscription]) -> list[str]:
    segment_count = sum(len(clip.segments) for clip in transcriptions)
    if segment_count < 50:
        return []
    return [
        "Output grande: usa JSON compatto e completa sempre tutti gli array e oggetti.",
        "Per ogni reason <= 90 caratteri; topic_summary <= 120 caratteri; editorial_strategy <= 900 caratteri.",
        "Se i segmenti sono molti imposta synthesis_script=null; non copiare tutta la trascrizione nel JSON.",
    ]


def build_editor_prompt(
    transcriptions: list[ClipTranscription],
    brief: EditorialBrief,
    validation_feedback: str | None = None,
    editing_options: EditingOptions | None = None,
) -> str:
    transcript_lines: list[str] = []
    for clip in transcriptions:
        transcript_lines.append(f"CLIP {clip.clip_id} durata={clip.duration:.2f}s")
        for segment in clip.segments:
            transcript_lines.append(
                f"- segment_id={segment.id} start={segment.start:.2f} end={segment.end:.2f} text={segment.text}"
            )
    reduction = target_reduction_ratio(brief.synthesis_level)
    feedback = f"\nCorrezione richiesta: {validation_feedback}\n" if validation_feedback else ""
    lines = [
        "Sei un editor video professionista per clip informative in italiano.",
        "Devi prima costruire l'ordine cronologico/narrativo dei periodi del testo completo.",
        "Solo dopo classifica i segmenti e decidi cosa mantenere, sintetizzare o tagliare.",
        "Classifica ogni segmento come essential, valuable, optional o redundant.",
        f"Rispetta synthesis_level={brief.synthesis_level}, target indicativo rimozione testo={reduction:.0%}.",
        _duration_budget_prompt_line(transcriptions, brief),
        "Preserva concetti essenziali, numeri, date, importi, nomi, sigle e passaggi logici.",
        "Rimuovi ripetizioni, filler, ridondanze e divagazioni solo quando il taglio resta naturale.",
        "Non ignorare il brief utente: deve modificare ritmo, classificazione e strategia quando contiene direttive concrete.",
        "Questa pipeline monta solo audio/video sorgente: non inventare frasi nuove e non riscrivere il parlato.",
        "Per ogni segmento essential o valuable imposta keep_verbatim=true e synthesis_text=null.",
        "Evita tagli che lasciano frasi senza soggetto, verbo, oggetto o passaggi logici necessari.",
        "La prima frase del montaggio deve nominare esplicitamente il soggetto prima di forme come 'Viene eletto'.",
        "Non mettere riferimenti come 'la stessa norma', 'questa norma' o 'la stessa legge' prima del segmento che nomina esplicitamente legge, numero, anno o decreto.",
        "Assegna playback_position come interi consecutivi 0, 1, 2... senza duplicati.",
        "Dentro una stessa clip mantieni i segmenti in ordine temporale sorgente.",
        "Non iniziare da un segmento che continua una frase precedente non inclusa.",
    ]
    if brief.user_prompt.strip():
        lines.extend(
            [
                f"Brief utente vincolante: {brief.user_prompt}",
                "Applica il brief utente come direttiva editoriale primaria insieme a coerenza narrativa e cut safety.",
                "Se il brief chiede un montaggio piu asciutto, classifica come redundant/optional ripetizioni, esitazioni, autoriparazioni e giri di parole non necessari.",
                "Non restituire la stessa classificazione di un montaggio neutro quando il brief chiede tagli piu asciutti, salvo assenza di tagli sicuri.",
                "Nel campo editorial_strategy spiega esplicitamente come il brief utente ha modificato tagli, segmenti mantenuti e ritmo.",
            ]
        )
    else:
        lines.append("Brief utente assente: applica solo synthesis_level e criteri editoriali generali.")
    lines.extend(_creative_zoom_prompt_lines(editing_options))
    lines.extend(_compact_output_prompt_lines(transcriptions))
    lines.extend(
        [
            "Restituisci solo output strutturato conforme allo schema NarrativeMap.",
            feedback,
            "Trascrizione:",
            "\n".join(transcript_lines),
        ]
    )
    return "\n".join(lines)


def _chunk_transcriptions(
    transcriptions: list[ClipTranscription],
    max_segments: int,
) -> list[list[ClipTranscription]]:
    chunks: list[list[ClipTranscription]] = []
    for clip in transcriptions:
        for start_index in range(0, len(clip.segments), max_segments):
            segment_chunk = clip.segments[start_index : start_index + max_segments]
            if not segment_chunk:
                continue
            chunks.append(
                [
                    ClipTranscription(
                        clip_id=clip.clip_id,
                        source_path=clip.source_path,
                        duration=clip.duration,
                        segments=segment_chunk,
                    )
                ]
            )
    return chunks


def _message_content(response: object) -> str:
    return response["message"]["content"] if isinstance(response, dict) else response.message.content  # type: ignore[attr-defined]


def _bounded_num_predict_for_attempt(settings: AppSettings, attempt: int, first_attempt_limit: int) -> int:
    initial_budget = min(settings.ollama_num_predict, first_attempt_limit)
    if attempt <= 0:
        return initial_budget
    return min(settings.ollama_num_ctx, settings.ollama_num_predict, initial_budget * (2**attempt))


def _compact_num_predict_for_attempt(settings: AppSettings, attempt: int) -> int:
    return _bounded_num_predict_for_attempt(settings, attempt, 4096)


def _order_num_predict_for_attempt(settings: AppSettings, attempt: int) -> int:
    return _bounded_num_predict_for_attempt(settings, attempt, 4096)


def _clone_clip_with_segments(clip: ClipTranscription, segments: list[Segment]) -> ClipTranscription:
    return ClipTranscription(
        clip_id=clip.clip_id,
        source_path=clip.source_path,
        duration=clip.duration,
        segments=segments,
    )


def _split_single_clip_chunk(transcriptions: list[ClipTranscription]) -> list[list[ClipTranscription]]:
    if len(transcriptions) != 1:
        raise NarrativeValidationError("Adaptive compact planning expects one clip per chunk")
    clip = transcriptions[0]
    if len(clip.segments) <= 1:
        return [transcriptions]
    midpoint = max(1, len(clip.segments) // 2)
    return [
        [_clone_clip_with_segments(clip, clip.segments[:midpoint])],
        [_clone_clip_with_segments(clip, clip.segments[midpoint:])],
    ]


def _merge_narrative_maps(maps: list[NarrativeMap], editorial_strategy: str) -> NarrativeMap:
    ordered_clips: list[NarrativeClip] = []
    filler_words: list[FillerRef] = []
    estimated_duration_s = 0.0
    playback_position = 0
    for narrative_map in maps:
        filler_words.extend(narrative_map.filler_words)
        estimated_duration_s += narrative_map.estimated_duration_s
        for narrative_clip in sorted(narrative_map.ordered_clips, key=lambda item: item.playback_position):
            ordered_clips.append(narrative_clip.model_copy(update={"playback_position": playback_position}, deep=True))
            playback_position += 1
    return NarrativeMap(
        ordered_clips=ordered_clips,
        filler_words=filler_words,
        editorial_strategy=editorial_strategy,
        estimated_duration_s=estimated_duration_s,
        synthesis_script=None,
    )


def _conservative_chunk_map(transcriptions: list[ClipTranscription], reason: str) -> NarrativeMap:
    if len(transcriptions) != 1:
        raise NarrativeValidationError("Conservative compact fallback expects one clip")
    clip = transcriptions[0]
    estimated_duration_s = sum(max(0.0, segment.end - segment.start) for segment in clip.segments)
    return NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=clip.clip_id,
                playback_position=0,
                topic_summary=f"Fallback conservativo clip {clip.clip_id}",
                role="supporting",
                segments=[
                    NarrativeSegment(
                        segment_id=segment.id,
                        importance=SegmentImportance.VALUABLE,
                        keep_verbatim=True,
                        synthesis_text=None,
                        reason=reason[:90] or "Fallback conservativo: contenuto mantenuto.",
                    )
                    for segment in clip.segments
                ],
            )
        ],
        filler_words=[],
        editorial_strategy=reason[:900] or "Fallback conservativo: contenuto mantenuto senza tagli editoriali.",
        estimated_duration_s=estimated_duration_s,
        synthesis_script=None,
    )


def _build_order_prompt(
    block_clips: list[NarrativeClip],
    transcriptions: list[ClipTranscription],
    brief: EditorialBrief,
) -> str:
    segments_by_key = {
        (clip.clip_id, segment.id): segment
        for clip in transcriptions
        for segment in clip.segments
    }
    lines = [
        "Sei un editor video professionista. Devi ordinare blocchi narrativi gia analizzati.",
        "Non riclassificare segmenti e non inventare tagli: scegli solo l'ordine globale dei block_id.",
        "Assegna playback_position consecutivi 0, 1, 2... senza duplicati.",
        "Restituisci solo output strutturato conforme allo schema NarrativeOrderPlan.",
        f"Brief utente: {brief.user_prompt or 'assente'}",
        f"clip_order richiesto: {brief.clip_order}",
        "Blocchi disponibili:",
    ]
    for block_id, clip in enumerate(block_clips):
        retained_ids = _retained_segment_ids(clip)
        preview_segments = [
            segments_by_key[(clip.clip_id, segment_id)].text
            for segment_id in retained_ids[:3]
            if (clip.clip_id, segment_id) in segments_by_key
        ]
        lines.append(
            f"BLOCK {block_id}: clip_id={clip.clip_id} role={clip.role} "
            f"topic={clip.topic_summary} retained_segment_ids={retained_ids} "
            f"preview={' | '.join(_short_text_preview(text, 80) for text in preview_segments)}"
        )
    return "\n".join(lines)


def _duration_refinement_candidates(
    narrative_map: NarrativeMap,
    transcriptions: list[ClipTranscription],
    required_reduction_s: float,
    *,
    limit: int = _DURATION_REFINEMENT_CANDIDATE_LIMIT,
) -> list[tuple[NarrativeSegment, Segment]]:
    segments_by_key = {
        (clip.clip_id, segment.id): segment
        for clip in transcriptions
        for segment in clip.segments
    }
    retained_pairs: list[tuple[int, NarrativeSegment, Segment]] = []
    playback_index = 0
    for narrative_clip in sorted(narrative_map.ordered_clips, key=lambda item: item.playback_position):
        for narrative_segment in narrative_clip.segments:
            if narrative_segment.importance == SegmentImportance.REDUNDANT or not narrative_segment.keep_verbatim:
                continue
            source_segment = segments_by_key.get((narrative_clip.clip_id, narrative_segment.segment_id))
            if source_segment is None:
                continue
            retained_pairs.append((playback_index, narrative_segment, source_segment))
            playback_index += 1

    if not retained_pairs:
        return []

    first_retained_index = retained_pairs[0][0]
    priority_by_importance = {
        SegmentImportance.OPTIONAL: 0,
        SegmentImportance.VALUABLE: 1,
        SegmentImportance.ESSENTIAL: 3,
        SegmentImportance.REDUNDANT: 4,
    }
    ranked_pairs = sorted(
        (
            pair
            for pair in retained_pairs
            if pair[0] != first_retained_index
        ),
        key=lambda pair: (
            priority_by_importance[pair[1].importance],
            -max(0.0, pair[2].end - pair[2].start),
            pair[2].clip_id,
            pair[2].start,
        ),
    )
    selected: list[tuple[NarrativeSegment, Segment]] = []
    selected_duration_s = 0.0
    desired_candidate_pool_s = max(required_reduction_s * 1.6, required_reduction_s + 30.0)
    for _playback_index, narrative_segment, source_segment in ranked_pairs:
        selected.append((narrative_segment, source_segment))
        selected_duration_s += max(0.0, source_segment.end - source_segment.start)
        if len(selected) >= limit:
            break
        if selected_duration_s >= desired_candidate_pool_s and len(selected) >= 12:
            break
    return selected


def _build_duration_refinement_prompt(
    narrative_map: NarrativeMap,
    transcriptions: list[ClipTranscription],
    brief: EditorialBrief,
    validation_feedback: str | None = None,
) -> tuple[str, set[tuple[int, int]]]:
    if brief.target_duration_s is None:
        raise NarrativePlanningError("Duration refinement requires target_duration_s")
    segments_by_key = {
        (clip.clip_id, segment.id): segment
        for clip in transcriptions
        for segment in clip.segments
    }
    current_duration_s = _retained_duration_s(narrative_map, segments_by_key)
    maximum_duration_s = _maximum_retained_duration(_spoken_duration(transcriptions), brief)
    if maximum_duration_s is None:
        raise NarrativePlanningError("Duration refinement requires a maximum duration")
    required_reduction_s = max(0.0, current_duration_s - maximum_duration_s)
    ideal_reduction_s = max(0.0, current_duration_s - brief.target_duration_s)
    candidates = _duration_refinement_candidates(narrative_map, transcriptions, required_reduction_s)
    candidate_keys = {(source_segment.clip_id, source_segment.id) for _narrative_segment, source_segment in candidates}
    candidate_duration_s = sum(
        max(0.0, source_segment.end - source_segment.start)
        for _narrative_segment, source_segment in candidates
    )
    feedback = f"\nCorrezione richiesta: {validation_feedback}\n" if validation_feedback else ""

    lines = [
        "Sei un editor video professionista. Devi accorciare una NarrativeMap gia validata semanticamente.",
        "Non riordinare clip o segmenti. Non inventare nuovi tagli.",
        "Scegli solo segmenti tra i candidati elencati da riclassificare come redundant.",
        "Devi selezionare abbastanza candidati per coprire almeno la riduzione minima necessaria.",
        "Mantieni soggetto, nomi, numeri, date, passaggi logici e apertura comprensibile.",
        "Preferisci ripetizioni, esitazioni, pause, risposte di servizio, digressioni e blocchi opzionali.",
        "Non eliminare un segmento solo per avvicinarti matematicamente al target se rompe la comprensione.",
        f"Durata trattenuta corrente={current_duration_s:.1f}s; target richiesto={brief.target_duration_s:.1f}s; "
        f"massimo tollerato={maximum_duration_s:.1f}s.",
        f"Riduzione minima necessaria={required_reduction_s:.1f}s; riduzione ideale={ideal_reduction_s:.1f}s.",
        f"Candidati disponibili={len(candidates)}; durata totale candidati={candidate_duration_s:.1f}s.",
        "Se la precedente selezione non e bastata, aggiungi altri tagli dai candidati rimanenti.",
        f"synthesis_level={brief.synthesis_level}; brief utente={brief.user_prompt or 'assente'}.",
        "Restituisci solo output strutturato conforme allo schema DurationRefinementPlan.",
        feedback,
        "Candidati tagliabili:",
    ]
    for narrative_segment, source_segment in candidates:
        duration_s = max(0.0, source_segment.end - source_segment.start)
        lines.append(
            f"- clip_id={source_segment.clip_id} segment_id={source_segment.id} "
            f"duration={duration_s:.1f}s importance={narrative_segment.importance.value} "
            f"start={source_segment.start:.2f} end={source_segment.end:.2f} "
            f"text={_short_text_preview(source_segment.text, 120)}"
        )
    return "\n".join(lines), candidate_keys


def _apply_duration_refinement_plan(
    narrative_map: NarrativeMap,
    plan: DurationRefinementPlan,
    transcriptions: list[ClipTranscription],
    candidate_keys: set[tuple[int, int]],
) -> NarrativeMap:
    redundant_keys = {(segment.clip_id, segment.segment_id) for segment in plan.redundant_segments}
    invalid_keys = sorted(redundant_keys - candidate_keys)
    if invalid_keys:
        raise NarrativeValidationError(f"Invalid duration refinement segment refs {invalid_keys}")

    reason_by_key = {
        (segment.clip_id, segment.segment_id): segment.reason
        for segment in plan.redundant_segments
        if (segment.clip_id, segment.segment_id) in candidate_keys
    }
    refined = narrative_map.model_copy(deep=True)
    for narrative_clip in refined.ordered_clips:
        for narrative_segment in narrative_clip.segments:
            key = (narrative_clip.clip_id, narrative_segment.segment_id)
            reason = reason_by_key.get(key)
            if reason is None:
                continue
            narrative_segment.importance = SegmentImportance.REDUNDANT
            narrative_segment.keep_verbatim = False
            narrative_segment.synthesis_text = None
            narrative_segment.creative_zoom = False
            narrative_segment.reason = f"{narrative_segment.reason} Target duration refinement: {reason}"

    segments_by_key = {
        (clip.clip_id, segment.id): segment
        for clip in transcriptions
        for segment in clip.segments
    }
    refined.estimated_duration_s = _retained_duration_s(refined, segments_by_key)
    refined.editorial_strategy = (
        f"{refined.editorial_strategy} Rifinitura durata target: {plan.editorial_strategy}"
    )[:900]
    return refined


def _build_compact_chunk_prompt(
    transcriptions: list[ClipTranscription],
    brief: EditorialBrief,
    validation_feedback: str | None = None,
    editing_options: EditingOptions | None = None,
) -> str:
    if len(transcriptions) != 1:
        raise NarrativePlanningError("Compact chunk planning expects one clip per chunk")
    clip = transcriptions[0]
    reduction = target_reduction_ratio(brief.synthesis_level)
    feedback = f"\nCorrezione richiesta: {validation_feedback}\n" if validation_feedback else ""
    lines = [
        "Sei un editor video professionista per clip informative in italiano.",
        "Analizza questo chunk locale e classifica ogni segment_id esattamente una volta.",
        "Non generare una NarrativeMap completa: restituisci solo CompactNarrativePlan.",
        "Usa liste di segment_id compatte per essential, valuable, optional, redundant.",
        "Metti in redundant_segment_ids solo segmenti da tagliare; gli altri verranno mantenuti.",
        f"Rispetta synthesis_level={brief.synthesis_level}, target indicativo rimozione testo={reduction:.0%}.",
        _duration_budget_prompt_line(transcriptions, brief),
        "Preserva numeri, nomi, date, passaggi logici e frasi necessarie alla comprensione.",
        "Taglia ripetizioni, esitazioni, filler e giri di parole solo se il taglio e naturale.",
        "reason non serve per segmento: usa editorial_strategy breve per spiegare il criterio.",
        "Restituisci JSON strutturato compatto e completo.",
    ]
    if brief.user_prompt.strip():
        lines.append(f"Brief utente vincolante: {brief.user_prompt}")
    else:
        lines.append("Brief utente assente.")
    lines.extend(_creative_zoom_prompt_lines(editing_options))
    lines.extend(
        [
            feedback,
            f"CLIP {clip.clip_id} durata={clip.duration:.2f}s",
            "Segmenti:",
        ]
    )
    for segment in clip.segments:
        lines.append(f"- segment_id={segment.id} start={segment.start:.2f} end={segment.end:.2f} text={segment.text}")
    return "\n".join(lines)


def _validate_compact_plan(plan: CompactNarrativePlan, transcriptions: list[ClipTranscription]) -> None:
    if len(transcriptions) != 1:
        raise NarrativeValidationError("Compact plan validation expects one clip")
    clip = transcriptions[0]
    if plan.clip_id != clip.clip_id:
        raise NarrativeValidationError(f"Invalid compact clip_id {plan.clip_id}")
    valid_ids = {segment.id for segment in clip.segments}
    categorized_ids = set(
        plan.essential_segment_ids
        + plan.valuable_segment_ids
        + plan.optional_segment_ids
        + plan.redundant_segment_ids
    )
    unknown_ids = sorted(categorized_ids - valid_ids)
    if unknown_ids:
        raise NarrativeValidationError(f"Invalid compact segment_ids {unknown_ids}")
    missing_ids = sorted(valid_ids - set(categorized_ids))
    if missing_ids:
        raise NarrativeValidationError(f"Missing compact segment_ids {missing_ids}")


def _normalize_compact_plan(plan: CompactNarrativePlan, transcriptions: list[ClipTranscription]) -> CompactNarrativePlan:
    if len(transcriptions) != 1:
        raise NarrativeValidationError("Compact plan normalization expects one clip")
    clip = transcriptions[0]
    if plan.clip_id != clip.clip_id:
        raise NarrativeValidationError(f"Invalid compact clip_id {plan.clip_id}")

    valid_ids = {segment.id for segment in clip.segments}
    assigned: dict[int, SegmentImportance] = {}
    duplicate_ids: set[int] = set()
    unknown_ids: set[int] = set()

    def assign(segment_ids: list[int], importance: SegmentImportance) -> None:
        for segment_id in segment_ids:
            if segment_id not in valid_ids:
                unknown_ids.add(segment_id)
                continue
            if segment_id in assigned:
                duplicate_ids.add(segment_id)
                continue
            assigned[segment_id] = importance

    assign(plan.essential_segment_ids, SegmentImportance.ESSENTIAL)
    assign(plan.valuable_segment_ids, SegmentImportance.VALUABLE)
    assign(plan.optional_segment_ids, SegmentImportance.OPTIONAL)
    assign(plan.redundant_segment_ids, SegmentImportance.REDUNDANT)

    missing_ids = sorted(valid_ids - set(assigned))
    for segment_id in missing_ids:
        assigned[segment_id] = SegmentImportance.VALUABLE

    normalized_fillers: list[FillerRef] = []
    repaired_filler_refs: list[tuple[int, int, int]] = []
    skipped_filler_refs: list[tuple[int, int]] = []
    for filler in plan.filler_words:
        if filler.clip_id == clip.clip_id and filler.segment_id in valid_ids:
            normalized_fillers.append(filler)
            continue
        if filler.clip_id in valid_ids:
            normalized_fillers.append(
                FillerRef(
                    clip_id=clip.clip_id,
                    segment_id=filler.clip_id,
                    word_index=filler.word_index,
                    word_text=filler.word_text,
                )
            )
            repaired_filler_refs.append((filler.clip_id, filler.segment_id, filler.clip_id))
            continue
        if filler.segment_id in valid_ids:
            normalized_fillers.append(
                FillerRef(
                    clip_id=clip.clip_id,
                    segment_id=filler.segment_id,
                    word_index=filler.word_index,
                    word_text=filler.word_text,
                )
            )
            repaired_filler_refs.append((filler.clip_id, filler.segment_id, filler.segment_id))
            continue
        skipped_filler_refs.append((filler.clip_id, filler.segment_id))

    if duplicate_ids or unknown_ids or missing_ids or repaired_filler_refs or skipped_filler_refs:
        logger.bind(clip_id=clip.clip_id).info(
            "Normalized compact narrative plan",
            duplicate_segment_ids=sorted(duplicate_ids),
            unknown_segment_ids=sorted(unknown_ids),
            missing_segment_ids=missing_ids,
            repaired_filler_refs=repaired_filler_refs,
            skipped_filler_refs=skipped_filler_refs,
        )

    return plan.model_copy(
        update={
            "essential_segment_ids": [
                segment.id
                for segment in clip.segments
                if assigned[segment.id] == SegmentImportance.ESSENTIAL
            ],
            "valuable_segment_ids": [
                segment.id
                for segment in clip.segments
                if assigned[segment.id] == SegmentImportance.VALUABLE
            ],
            "optional_segment_ids": [
                segment.id
                for segment in clip.segments
                if assigned[segment.id] == SegmentImportance.OPTIONAL
            ],
            "redundant_segment_ids": [
                segment.id
                for segment in clip.segments
                if assigned[segment.id] == SegmentImportance.REDUNDANT
            ],
            "filler_words": normalized_fillers,
        }
    )


def _compact_plan_to_narrative_map(
    plan: CompactNarrativePlan,
    transcriptions: list[ClipTranscription],
) -> NarrativeMap:
    plan = _normalize_compact_plan(plan, transcriptions)
    _validate_compact_plan(plan, transcriptions)
    clip = transcriptions[0]
    importance_by_segment_id: dict[int, SegmentImportance] = {}
    for segment_id in plan.essential_segment_ids:
        importance_by_segment_id[segment_id] = SegmentImportance.ESSENTIAL
    for segment_id in plan.valuable_segment_ids:
        importance_by_segment_id[segment_id] = SegmentImportance.VALUABLE
    for segment_id in plan.optional_segment_ids:
        importance_by_segment_id[segment_id] = SegmentImportance.OPTIONAL
    for segment_id in plan.redundant_segment_ids:
        importance_by_segment_id[segment_id] = SegmentImportance.REDUNDANT

    narrative_segments: list[NarrativeSegment] = []
    retained_duration_s = 0.0
    for segment in sorted(clip.segments, key=lambda item: item.start):
        importance = importance_by_segment_id[segment.id]
        keep_verbatim = importance != SegmentImportance.REDUNDANT
        if keep_verbatim:
            retained_duration_s += max(0.0, segment.end - segment.start)
        narrative_segments.append(
            NarrativeSegment(
                segment_id=segment.id,
                importance=importance,
                keep_verbatim=keep_verbatim,
                synthesis_text=None,
                reason=plan.editorial_strategy[:90] or "Classificazione compatta LLM.",
            )
        )

    return NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=plan.clip_id,
                playback_position=0,
                topic_summary=plan.topic_summary,
                role=plan.role,
                segments=narrative_segments,
            )
        ],
        filler_words=plan.filler_words,
        editorial_strategy=plan.editorial_strategy,
        estimated_duration_s=retained_duration_s if retained_duration_s > 0 else plan.estimated_duration_s,
        synthesis_script=None,
    )


def _validate_order_plan(order_plan: NarrativeOrderPlan, block_count: int) -> None:
    valid_block_ids = set(range(block_count))
    seen_block_ids: set[int] = set()
    seen_positions: set[int] = set()
    for block in order_plan.ordered_blocks:
        if block.block_id not in valid_block_ids:
            raise NarrativeValidationError(f"Invalid block_id {block.block_id}")
        if block.block_id in seen_block_ids:
            raise NarrativeValidationError(f"Duplicate block_id {block.block_id}")
        if block.playback_position in seen_positions:
            raise NarrativeValidationError(f"Duplicate block playback_position {block.playback_position}")
        seen_block_ids.add(block.block_id)
        seen_positions.add(block.playback_position)
    missing = sorted(valid_block_ids - seen_block_ids)
    if missing:
        raise NarrativeValidationError(f"Missing block_ids {missing}")


class NarrativePlanner:
    def __init__(self, settings: AppSettings, max_retries: int = 2) -> None:
        self.settings = settings
        self.max_retries = max_retries

    def generate(
        self,
        transcriptions: list[ClipTranscription],
        brief: EditorialBrief,
        model: str | None,
        editing_options: EditingOptions | None = None,
        progress_callback: object | None = None,
    ) -> NarrativeMap:
        try:
            import ollama
        except ImportError as exc:
            raise NarrativePlanningError("Il pacchetto ollama non e installato") from exc

        selected_model = model or self.settings.default_ollama_model
        if _segment_count(transcriptions) > self.settings.narrative_hierarchical_threshold_segments:
            return self._generate_hierarchical(
                ollama,
                transcriptions,
                brief,
                selected_model,
                editing_options,
                progress_callback=progress_callback,
            )
        return self._generate_single_pass(
            ollama,
            transcriptions,
            brief,
            selected_model,
            editing_options,
            enforce_opening=True,
        )

    def recalculate_proposal(
        self,
        narrative_map: NarrativeMap,
        transcriptions: list[ClipTranscription],
        brief: EditorialBrief,
        model: str | None,
        editing_options: EditingOptions | None = None,
        progress_callback: object | None = None,
    ) -> NarrativeMap:
        try:
            import ollama
        except ImportError as exc:
            raise NarrativePlanningError("Il pacchetto ollama non e installato") from exc

        selected_model = model or self.settings.default_ollama_model
        working_map = narrative_map.model_copy(deep=True)
        repair_narrative_map_for_cut_safety(working_map, transcriptions)
        effective_brief = _brief_with_implicit_synthesis_target(transcriptions, brief)
        refined_map = self._refine_duration_budget(
            ollama,
            working_map,
            transcriptions,
            effective_brief,
            selected_model,
            editing_options,
            progress_callback=progress_callback,
        )
        _validate_narrative_map_allowing_excessive_synthesis(
            refined_map,
            transcriptions,
            editing_options,
            effective_brief,
        )
        refined_map.editorial_strategy = (
            f"{refined_map.editorial_strategy} Proposta ricalcolata con synthesis_level={brief.synthesis_level}"
            + (
                f" e target_duration_s={brief.target_duration_s:.1f}s."
                if brief.target_duration_s is not None
                else "."
            )
        )
        return refined_map

    def _generate_single_pass(
        self,
        ollama_module: object,
        transcriptions: list[ClipTranscription],
        brief: EditorialBrief,
        selected_model: str,
        editing_options: EditingOptions | None,
        *,
        enforce_opening: bool,
        enforce_duration_budget: bool = True,
    ) -> NarrativeMap:
        feedback: str | None = None
        for attempt in range(self.max_retries + 1):
            prompt = build_editor_prompt(transcriptions, brief, feedback, editing_options)
            try:
                response = structured_chat(
                    ollama_module,
                    self.settings,
                    model=selected_model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=NarrativeMap.model_json_schema(),
                    num_predict=ollama_num_predict_for_attempt(self.settings, attempt),
                )
                content = _message_content(response)
                narrative_map = NarrativeMap.model_validate_json(content)
                repair_narrative_map_for_cut_safety(narrative_map, transcriptions)
                validate_narrative_map(
                    narrative_map,
                    transcriptions,
                    editing_options,
                    brief,
                    enforce_opening=enforce_opening,
                    enforce_duration_budget=enforce_duration_budget,
                )
                return narrative_map
            except (OllamaRuntimeError, OllamaTimeoutError) as exc:
                raise NarrativePlanningError(str(exc)) from exc
            except ValidationError as exc:
                if validation_error_is_json_invalid(exc):
                    feedback = (
                        "Output JSON troncato o non valido: restituisci una NarrativeMap completa, "
                        "usa JSON compatto, chiudi tutti gli array/oggetti e imposta synthesis_script=null se serve."
                    )
                else:
                    feedback = f"Schema non valido: {exc}"
            except NarrativeValidationError as exc:
                available = [
                    f"clip_id={clip.clip_id}: segment_ids={[segment.id for segment in clip.segments]}"
                    for clip in transcriptions
                ]
                feedback = f"{exc}. Usa solo questi ID: {'; '.join(available)}"
        raise NarrativePlanningError(feedback or "NarrativeMap non valida")

    def _generate_compact_chunk(
        self,
        ollama_module: object,
        transcriptions: list[ClipTranscription],
        brief: EditorialBrief,
        selected_model: str,
        editing_options: EditingOptions | None,
    ) -> NarrativeMap:
        feedback: str | None = None
        for attempt in range(self.max_retries + 1):
            prompt = _build_compact_chunk_prompt(transcriptions, brief, feedback, editing_options)
            try:
                response = structured_chat(
                    ollama_module,
                    self.settings,
                    model=selected_model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=CompactNarrativePlan.model_json_schema(),
                    num_predict=_compact_num_predict_for_attempt(self.settings, attempt),
                )
                compact_plan = CompactNarrativePlan.model_validate_json(_message_content(response))
                narrative_map = _compact_plan_to_narrative_map(compact_plan, transcriptions)
                validate_narrative_map(
                    narrative_map,
                    transcriptions,
                    editing_options,
                    brief,
                    enforce_opening=False,
                    enforce_duration_budget=False,
                )
                return narrative_map
            except OllamaTimeoutError:
                raise
            except OllamaRuntimeError as exc:
                raise NarrativePlanningError(str(exc)) from exc
            except ValidationError as exc:
                if validation_error_is_json_invalid(exc):
                    feedback = "Output JSON troncato o non valido: restituisci CompactNarrativePlan completo e compatto."
                else:
                    feedback = f"Schema CompactNarrativePlan non valido: {exc}"
            except NarrativeValidationError as exc:
                if str(exc).startswith("excessive_synthesis:"):
                    validate_narrative_map(
                        narrative_map,
                        transcriptions,
                        editing_options,
                        brief,
                        enforce_opening=enforce_opening,
                        enforce_duration_budget=False,
                    )
                    return narrative_map
                available = [
                    f"clip_id={clip.clip_id}: segment_ids={[segment.id for segment in clip.segments]}"
                    for clip in transcriptions
                ]
                feedback = f"{exc}. Classifica tutti e soli questi ID: {'; '.join(available)}"
        raise NarrativePlanningError(feedback or "CompactNarrativePlan non valido")

    def _generate_compact_chunk_adaptive(
        self,
        ollama_module: object,
        transcriptions: list[ClipTranscription],
        brief: EditorialBrief,
        selected_model: str,
        editing_options: EditingOptions | None,
        *,
        progress_callback: object | None,
    ) -> NarrativeMap:
        try:
            return self._generate_compact_chunk(
                ollama_module,
                transcriptions,
                brief,
                selected_model,
                editing_options,
            )
        except OllamaTimeoutError as exc:
            segment_count = _segment_count(transcriptions)
            if segment_count <= 1:
                if callable(progress_callback):
                    progress_callback("Timeout su micro-chunk LLM: mantengo il segmento senza tagli editoriali")
                logger.bind(clip_id=transcriptions[0].clip_id if transcriptions else None).info(
                    "Timeout on single-segment compact narrative chunk; using conservative keep fallback"
                )
                return _conservative_chunk_map(
                    transcriptions,
                    "Fallback conservativo dopo timeout LLM: contenuto mantenuto senza tagli editoriali.",
                )
            if callable(progress_callback):
                progress_callback(f"Timeout chunk LLM da {segment_count} segmenti: divisione in sotto-chunk")
            sub_maps = [
                self._generate_compact_chunk_adaptive(
                    ollama_module,
                    subchunk,
                    brief,
                    selected_model,
                    editing_options,
                    progress_callback=progress_callback,
                )
                for subchunk in _split_single_clip_chunk(transcriptions)
            ]
            return _merge_narrative_maps(
                sub_maps,
                f"Analisi adattiva: chunk diviso dopo timeout Ollama ({exc}).",
            )

    def _generate_hierarchical(
        self,
        ollama_module: object,
        transcriptions: list[ClipTranscription],
        brief: EditorialBrief,
        selected_model: str,
        editing_options: EditingOptions | None,
        *,
        progress_callback: object | None = None,
    ) -> NarrativeMap:
        block_clips: list[NarrativeClip] = []
        filler_words = []
        local_strategies: list[str] = []
        estimated_duration_s = 0.0
        chunks = _chunk_transcriptions(transcriptions, self.settings.narrative_chunk_max_segments)
        total_chunks = len(chunks)
        for chunk_index, chunk in enumerate(chunks, start=1):
            if callable(progress_callback):
                progress_callback(f"Analisi narrativa chunk {chunk_index}/{total_chunks}")
            chunk_map = self._generate_compact_chunk_adaptive(
                ollama_module,
                chunk,
                brief,
                selected_model,
                editing_options,
                progress_callback=progress_callback,
            )
            if callable(progress_callback):
                progress_callback(f"Analisi narrativa chunk {chunk_index}/{total_chunks} completata")
            local_strategies.append(chunk_map.editorial_strategy)
            filler_words.extend(chunk_map.filler_words)
            estimated_duration_s += chunk_map.estimated_duration_s
            block_clips.extend(
                clip.model_copy(deep=True)
                for clip in sorted(chunk_map.ordered_clips, key=lambda item: item.playback_position)
                if clip.segments
            )

        if not block_clips:
            raise NarrativePlanningError("Analisi gerarchica senza blocchi narrativi validi")

        if brief.clip_order == "upload":
            ordered_clips = [
                clip.model_copy(update={"playback_position": position}, deep=True)
                for position, clip in enumerate(block_clips)
            ]
            editorial_strategy = "Ordine di upload conservato; analisi locale LLM per chunk. "
            editorial_strategy += " ".join(local_strategies[:3])
        else:
            if callable(progress_callback):
                progress_callback("Analisi narrativa ordinamento globale")
            try:
                order_plan = self._generate_order_plan(ollama_module, block_clips, transcriptions, brief, selected_model)
            except (OllamaTimeoutError, NarrativePlanningError) as exc:
                if callable(progress_callback):
                    progress_callback("Ordinamento globale LLM non completato: uso ordine di upload conservativo")
                ordered_clips = [
                    clip.model_copy(update={"playback_position": position}, deep=True)
                    for position, clip in enumerate(block_clips)
                ]
                editorial_strategy = (
                    "Ordinamento globale LLM non completato; ordine di upload conservato. "
                    f"Analisi locale LLM mantenuta. Dettaglio: {exc}"
                )
            else:
                if callable(progress_callback):
                    progress_callback("Analisi narrativa ordinamento globale completato")
                clips_by_block_id = {block_id: clip for block_id, clip in enumerate(block_clips)}
                ordered_clips = []
                for order_block in sorted(order_plan.ordered_blocks, key=lambda item: item.playback_position):
                    narrative_clip = clips_by_block_id[order_block.block_id].model_copy(deep=True)
                    narrative_clip.playback_position = order_block.playback_position
                    narrative_clip.role = order_block.role
                    narrative_clip.topic_summary = order_block.topic_summary
                    ordered_clips.append(narrative_clip)
                editorial_strategy = order_plan.editorial_strategy

        narrative_map = NarrativeMap(
            ordered_clips=ordered_clips,
            filler_words=filler_words,
            editorial_strategy=editorial_strategy,
            estimated_duration_s=estimated_duration_s,
            synthesis_script=None,
        )
        repair_narrative_map_for_cut_safety(narrative_map, transcriptions)
        narrative_map = self._refine_duration_budget(
            ollama_module,
            narrative_map,
            transcriptions,
            brief,
            selected_model,
            editing_options,
            progress_callback=progress_callback,
        )
        _validate_narrative_map_allowing_excessive_synthesis(
            narrative_map,
            transcriptions,
            editing_options,
            brief,
        )
        return narrative_map

    def _refine_duration_budget(
        self,
        ollama_module: object,
        narrative_map: NarrativeMap,
        transcriptions: list[ClipTranscription],
        brief: EditorialBrief,
        selected_model: str,
        editing_options: EditingOptions | None,
        *,
        progress_callback: object | None,
    ) -> NarrativeMap:
        if brief.target_duration_s is None:
            return narrative_map

        segments_by_key = {
            (clip.clip_id, segment.id): segment
            for clip in transcriptions
            for segment in clip.segments
        }
        maximum_duration_s = _maximum_retained_duration(_spoken_duration(transcriptions), brief)
        current_duration_s = _retained_duration_s(narrative_map, segments_by_key)
        if maximum_duration_s is None or current_duration_s <= maximum_duration_s + 0.001:
            return narrative_map

        working_map = narrative_map
        feedback: str | None = None
        refinement_attempts = max(self.max_retries + 1, 5)
        for attempt in range(refinement_attempts):
            current_duration_s = _retained_duration_s(working_map, segments_by_key)
            if current_duration_s <= maximum_duration_s + 0.001:
                validate_narrative_map(working_map, transcriptions, editing_options, brief)
                return working_map
            if callable(progress_callback):
                progress_callback(
                    f"Rifinitura durata target LLM ({current_duration_s:.0f}s -> {brief.target_duration_s:.0f}s)"
                )
            prompt, candidate_keys = _build_duration_refinement_prompt(
                working_map,
                transcriptions,
                brief,
                feedback,
            )
            if not candidate_keys:
                raise NarrativePlanningError(
                    "Target duration non raggiungibile: nessun segmento candidato tagliabile senza rompere l'apertura."
                )
            refined_map: NarrativeMap | None = None
            try:
                response = structured_chat(
                    ollama_module,
                    self.settings,
                    model=selected_model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=DurationRefinementPlan.model_json_schema(),
                    num_predict=_order_num_predict_for_attempt(self.settings, attempt),
                )
                plan = DurationRefinementPlan.model_validate_json(_message_content(response))
                refined_map = _apply_duration_refinement_plan(
                    working_map,
                    plan,
                    transcriptions,
                    candidate_keys,
                )
                repair_narrative_map_for_cut_safety(refined_map, transcriptions)
                validate_narrative_map(refined_map, transcriptions, editing_options, brief)
                if callable(progress_callback):
                    progress_callback("Rifinitura durata target completata")
                return refined_map
            except (OllamaRuntimeError, OllamaTimeoutError) as exc:
                raise NarrativePlanningError(str(exc)) from exc
            except ValidationError as exc:
                if validation_error_is_json_invalid(exc):
                    feedback = (
                        "Output JSON troncato o non valido: restituisci un DurationRefinementPlan completo e compatto."
                    )
                else:
                    feedback = f"Schema DurationRefinementPlan non valido: {exc}"
            except NarrativeValidationError as exc:
                refined_duration_s = _retained_duration_s(refined_map, segments_by_key) if refined_map is not None else current_duration_s
                if str(exc).startswith("target_duration_exceeded") and refined_duration_s < current_duration_s - 0.001:
                    working_map = refined_map
                    remaining_reduction_s = max(0.0, refined_duration_s - maximum_duration_s)
                    feedback = (
                        f"Riduzione parziale accettata: durata ora {refined_duration_s:.1f}s. "
                        f"Servono ancora almeno {remaining_reduction_s:.1f}s di tagli aggiuntivi dai candidati rimanenti. "
                        "Non ripetere segmenti gia riclassificati redundant."
                    )
                    continue
                available = sorted(candidate_keys)
                feedback = (
                    f"{exc}. Usa solo queste coppie clip_id/segment_id candidate: {available}. "
                    "Riduci abbastanza per rientrare nel target ma non sotto la durata minima."
                )
        raise NarrativePlanningError(feedback or "Rifinitura durata target non valida")

    def _generate_order_plan(
        self,
        ollama_module: object,
        block_clips: list[NarrativeClip],
        transcriptions: list[ClipTranscription],
        brief: EditorialBrief,
        selected_model: str,
    ) -> NarrativeOrderPlan:
        feedback: str | None = None
        for attempt in range(self.max_retries + 1):
            prompt = _build_order_prompt(block_clips, transcriptions, brief)
            if feedback:
                prompt = f"{prompt}\nCorrezione richiesta: {feedback}"
            try:
                response = structured_chat(
                    ollama_module,
                    self.settings,
                    model=selected_model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=NarrativeOrderPlan.model_json_schema(),
                    num_predict=_order_num_predict_for_attempt(self.settings, attempt),
                )
                order_plan = NarrativeOrderPlan.model_validate_json(_message_content(response))
                _validate_order_plan(order_plan, len(block_clips))
                return order_plan
            except OllamaTimeoutError:
                raise
            except OllamaRuntimeError as exc:
                raise NarrativePlanningError(str(exc)) from exc
            except ValidationError as exc:
                if validation_error_is_json_invalid(exc):
                    feedback = "Output JSON troncato o non valido: restituisci un NarrativeOrderPlan completo e compatto."
                else:
                    feedback = f"Schema NarrativeOrderPlan non valido: {exc}"
            except NarrativeValidationError as exc:
                feedback = f"{exc}. Usa tutti e soli i block_id da 0 a {len(block_clips) - 1}."
        raise NarrativePlanningError(feedback or "NarrativeOrderPlan non valido")


class MockNarrativePlanner:
    def generate(
        self,
        transcriptions: list[ClipTranscription],
        brief: EditorialBrief,
        model: str | None,
        editing_options: EditingOptions | None = None,
    ) -> NarrativeMap:
        ordered_clips: list[NarrativeClip] = []
        estimated_duration = 0.0
        dry_edit_requested = _brief_requests_dry_edit(brief)
        for position, clip in enumerate(transcriptions):
            narrative_segments: list[NarrativeSegment] = []
            for segment_index, segment in enumerate(clip.segments):
                importance = SegmentImportance.ESSENTIAL
                if dry_edit_requested and segment_index > 0 and _segment_matches_dry_cleanup(segment):
                    importance = SegmentImportance.REDUNDANT
                elif brief.synthesis_level >= 4 and position > 0:
                    importance = SegmentImportance.VALUABLE
                narrative_segments.append(
                    NarrativeSegment(
                        segment_id=segment.id,
                        importance=importance,
                        keep_verbatim=True,
                        creative_zoom=bool(editing_options and editing_options.enable_creative_zoom and position == 0),
                        reason="Segmento mantenuto nella proposta mockabile.",
                    )
                )
                estimated_duration += max(0.0, segment.end - segment.start)
            ordered_clips.append(
                NarrativeClip(
                    clip_id=clip.clip_id,
                    playback_position=position,
                    topic_summary=f"Clip {clip.clip_id}",
                    role="introduction" if position == 0 else "development",
                    segments=narrative_segments,
                )
            )
        strategy = "Proposta deterministica di sviluppo: conserva i segmenti informativi e prepara tagli sicuri."
        if dry_edit_requested:
            strategy = (
                "Proposta deterministica guidata dal brief utente: taglia ripetizioni, "
                "esitazioni e giri di parole quando riconoscibili."
            )
        return NarrativeMap(
            ordered_clips=ordered_clips,
            filler_words=[],
            editorial_strategy=strategy,
            estimated_duration_s=estimated_duration,
            synthesis_script=" ".join(segment.text for clip in transcriptions for segment in clip.segments),
        )

    def recalculate_proposal(
        self,
        narrative_map: NarrativeMap,
        transcriptions: list[ClipTranscription],
        brief: EditorialBrief,
        model: str | None,
        editing_options: EditingOptions | None = None,
        progress_callback: object | None = None,
    ) -> NarrativeMap:
        refined = narrative_map.model_copy(deep=True)
        target_duration_s = brief.target_duration_s
        if target_duration_s is None and brief.synthesis_level > 0:
            target_duration_s = _spoken_duration(transcriptions) * (1.0 - target_reduction_ratio(brief.synthesis_level))
        retained_duration_s = _retained_duration_s(
            refined,
            {
                (clip.clip_id, segment.id): segment
                for clip in transcriptions
                for segment in clip.segments
            },
        )
        if target_duration_s is not None and retained_duration_s > target_duration_s:
            for narrative_clip in reversed(refined.ordered_clips):
                for narrative_segment in reversed(narrative_clip.segments):
                    if narrative_segment.importance == SegmentImportance.ESSENTIAL:
                        continue
                    narrative_segment.importance = SegmentImportance.REDUNDANT
                    narrative_segment.keep_verbatim = False
                    narrative_segment.creative_zoom = False
                    narrative_segment.reason = (
                        f"{narrative_segment.reason} Ricalcolo proposta: taglio per nuova durata/sintesi."
                    )
                    retained_duration_s = _retained_duration_s(
                        refined,
                        {
                            (clip.clip_id, segment.id): segment
                            for clip in transcriptions
                            for segment in clip.segments
                        },
                    )
                    if retained_duration_s <= target_duration_s:
                        break
                if retained_duration_s <= target_duration_s:
                    break
        refined.estimated_duration_s = retained_duration_s
        refined.editorial_strategy = (
            f"{refined.editorial_strategy} Proposta ricalcolata mock con synthesis_level={brief.synthesis_level}."
        )
        return refined
