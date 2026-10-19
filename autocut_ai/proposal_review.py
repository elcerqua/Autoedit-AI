from __future__ import annotations

from typing import Protocol

from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from autocut_ai.config import AppSettings
from autocut_ai.cuts import build_edit_decision_list
from autocut_ai.narrative import (
    NarrativeValidationError,
    repair_narrative_map_for_cut_safety,
    validate_narrative_map,
)
from autocut_ai.ollama_runtime import (
    OllamaRuntimeError,
    OllamaTimeoutError,
    structured_chat,
    validation_error_is_json_invalid,
)
from autocut_ai.schemas import (
    ClipTranscription,
    EditDecisionList,
    EditingOptions,
    EditorialBrief,
    NarrativeMap,
    NarrativeClip,
    NarrativeSegment,
    ProposalIssue,
    ProposalReview,
    ProposalReviewPatch,
    ProposalSegmentAdjustment,
    ProposalWordRemoval,
    Segment,
    SegmentImportance,
    WordRemovalRange,
)


class ProposalReviewError(RuntimeError):
    pass


_COMPACT_REVIEW_SEGMENT_THRESHOLD = 40
_COMPACT_REVIEW_DECISION_THRESHOLD = 40
_COMPACT_REVIEW_HEAD_DECISIONS = 10
_COMPACT_REVIEW_TAIL_DECISIONS = 6
_LIGHT_REVIEW_CHUNK_DECISIONS = 16
_MAX_LIGHT_REVIEW_CHUNKS = 3
_MANDATORY_REVIEW_UNDER_DURATION_S = 10 * 60


class GlobalReviewBlock(BaseModel):
    block_id: int = Field(ge=0)
    clip_id: int
    playback_position: int = Field(ge=0)
    role: str
    topic_summary: str
    segment_ids: list[int]
    duration_s: float = Field(ge=0)
    text_preview: str


class GlobalReviewRestoredSegment(BaseModel):
    clip_id: int
    segment_id: int
    importance: SegmentImportance = SegmentImportance.VALUABLE
    reason: str


class GlobalProposalReviewPlan(BaseModel):
    ordered_block_ids: list[int] = Field(default_factory=list)
    dropped_block_ids: list[int] = Field(default_factory=list)
    restored_segments: list[GlobalReviewRestoredSegment] = Field(default_factory=list)
    issues: list[ProposalIssue] = Field(default_factory=list)
    editorial_notes: str = ""


def _short_text(text: str, max_chars: int = 160) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3]}..."


def _proposal_review_uses_compact_context(
    transcriptions: list[ClipTranscription],
    edit_decision_list: EditDecisionList,
) -> bool:
    segment_count = sum(len(clip.segments) for clip in transcriptions)
    return (
        segment_count > _COMPACT_REVIEW_SEGMENT_THRESHOLD
        or len(edit_decision_list.decisions) > _COMPACT_REVIEW_DECISION_THRESHOLD
    )


def _compact_transcript_context(transcriptions: list[ClipTranscription]) -> str:
    lines: list[str] = []
    for clip in transcriptions:
        segment_ids = [segment.id for segment in clip.segments]
        if segment_ids:
            id_summary = f"{segment_ids[0]}-{segment_ids[-1]}"
        else:
            id_summary = "nessuno"
        preview = " | ".join(_short_text(segment.text, 90) for segment in clip.segments[:2])
        lines.append(
            f"CLIP {clip.clip_id} durata={clip.duration:.2f}s segmenti={len(clip.segments)} "
            f"id_range={id_summary} preview={preview}"
        )
    return "\n".join(lines)


def _valid_patch_id_context(transcriptions: list[ClipTranscription]) -> str:
    return "\n".join(
        f"clip_id={clip.clip_id} segment_ids={[segment.id for segment in clip.segments]}"
        for clip in transcriptions
    )


def _compact_narrative_context(narrative_map: NarrativeMap) -> str:
    lines = [
        f"strategy={_short_text(narrative_map.editorial_strategy, 220)}",
        f"estimated_duration_s={narrative_map.estimated_duration_s:.1f}",
    ]
    for clip in sorted(narrative_map.ordered_clips, key=lambda item: item.playback_position):
        kept_ids = [
            segment.segment_id
            for segment in clip.segments
            if segment.keep_verbatim
        ]
        cut_ids = [
            segment.segment_id
            for segment in clip.segments
            if not segment.keep_verbatim
        ]
        lines.append(
            f"clip_id={clip.clip_id} playback_position={clip.playback_position} role={clip.role} "
            f"topic={_short_text(clip.topic_summary, 100)} kept_ids={kept_ids} cut_ids={cut_ids}"
        )
    return "\n".join(lines)


def _edl_context(edit_decision_list: EditDecisionList, *, compact: bool) -> str:
    lines: list[str] = []
    max_text_chars = 110 if compact else 10000
    decisions = edit_decision_list.decisions
    if compact:
        lines.append(
            f"decision_count={len(decisions)} estimated_duration_s={edit_decision_list.estimated_duration_s:.1f}"
        )
        if len(decisions) > _COMPACT_REVIEW_HEAD_DECISIONS + _COMPACT_REVIEW_TAIL_DECISIONS:
            omitted = len(decisions) - _COMPACT_REVIEW_HEAD_DECISIONS - _COMPACT_REVIEW_TAIL_DECISIONS
            sampled_decisions = (
                decisions[:_COMPACT_REVIEW_HEAD_DECISIONS]
                + decisions[-_COMPACT_REVIEW_TAIL_DECISIONS:]
            )
            lines.append(f"campione_edl=head_tail; decisioni_omesse={omitted}")
        else:
            sampled_decisions = decisions
    else:
        sampled_decisions = decisions
    for decision in sampled_decisions:
        lines.append(
            " ".join(
                [
                    f"- output_order={decision.output_order}",
                    f"clip_id={decision.clip_id}",
                    f"source_start={decision.source_start:.2f}",
                    f"source_end={decision.source_end:.2f}",
                    f"creative_zoom={decision.creative_zoom}",
                    f"text={_short_text(decision.included_text, max_text_chars)}",
                ]
            )
        )
    return "\n".join(lines)


def _word_context_for_edl(
    transcriptions: list[ClipTranscription],
    edit_decision_list: EditDecisionList,
) -> str:
    clips_by_id = {clip.clip_id: clip for clip in transcriptions}
    lines: list[str] = []
    for decision in edit_decision_list.decisions:
        clip = clips_by_id.get(decision.clip_id)
        if clip is None:
            continue
        for segment in clip.segments:
            overlapping_words = [
                (index, word)
                for index, word in enumerate(segment.words)
                if word.end >= decision.source_start and word.start <= decision.source_end
            ]
            if not overlapping_words:
                continue
            word_parts = [f"{index}:{word.word}" for index, word in overlapping_words[:120]]
            suffix = " ..." if len(overlapping_words) > 120 else ""
            lines.append(
                f"decision_order={decision.output_order} clip_id={decision.clip_id} "
                f"segment_id={segment.id} words=[{' '.join(word_parts)}{suffix}]"
            )
    return "\n".join(lines)


def _retained_narrative_segments(narrative_clip: NarrativeClip) -> list[NarrativeSegment]:
    return [
        segment
        for segment in narrative_clip.segments
        if segment.keep_verbatim and segment.importance != SegmentImportance.REDUNDANT
    ]


def _global_review_clip_pairs(narrative_map: NarrativeMap) -> list[tuple[int, NarrativeClip]]:
    pairs: list[tuple[int, NarrativeClip]] = []
    for narrative_clip in sorted(narrative_map.ordered_clips, key=lambda item: item.playback_position):
        if _retained_narrative_segments(narrative_clip):
            pairs.append((len(pairs), narrative_clip))
    return pairs


def _global_review_blocks(
    transcriptions: list[ClipTranscription],
    narrative_map: NarrativeMap,
) -> list[GlobalReviewBlock]:
    source_segments = _segment_lookup(transcriptions)
    blocks: list[GlobalReviewBlock] = []
    for block_id, narrative_clip in _global_review_clip_pairs(narrative_map):
        retained_segments = _retained_narrative_segments(narrative_clip)
        source_items = [
            source_segments[(narrative_clip.clip_id, segment.segment_id)]
            for segment in retained_segments
            if (narrative_clip.clip_id, segment.segment_id) in source_segments
        ]
        text_preview = " ".join(_short_text(segment.text, 140) for segment in source_items)
        duration_s = sum(max(0.0, segment.end - segment.start) for segment in source_items)
        blocks.append(
            GlobalReviewBlock(
                block_id=block_id,
                clip_id=narrative_clip.clip_id,
                playback_position=narrative_clip.playback_position,
                role=narrative_clip.role,
                topic_summary=narrative_clip.topic_summary,
                segment_ids=[segment.segment_id for segment in retained_segments],
                duration_s=duration_s,
                text_preview=_short_text(text_preview, 420),
            )
        )
    return blocks


def build_global_proposal_review_prompt(
    transcriptions: list[ClipTranscription],
    narrative_map: NarrativeMap,
    edit_decision_list: EditDecisionList,
    brief: EditorialBrief,
    validation_feedback: str | None = None,
    editing_options: EditingOptions | None = None,
) -> str:
    blocks = _global_review_blocks(transcriptions, narrative_map)
    block_lines = [
        " ".join(
            [
                f"block_id={block.block_id}",
                f"current_position={block.playback_position}",
                f"clip_id={block.clip_id}",
                f"segment_ids={block.segment_ids}",
                f"role={block.role}",
                f"duration_s={block.duration_s:.1f}",
                f"topic={_short_text(block.topic_summary, 120)}",
                f"text={block.text_preview}",
            ]
        )
        for block in blocks
    ]
    feedback = f"\nCorrezione richiesta: {validation_feedback}\n" if validation_feedback else ""
    zoom_line = (
        "Zoom creativo attivo: ignora lo zoom; valuta solo struttura narrativa e contesto."
        if editing_options is not None and editing_options.enable_creative_zoom
        else "Zoom creativo non attivo."
    )
    lines = [
        "Review globale a blocchi della proposta editoriale.",
        "Sei un editor video professionista: verifica prima la logica narrativa complessiva, poi lascia alla review locale le micro-limature.",
        "Non generare una NarrativeMap completa, non generare timestamp e non riscrivere testo libero.",
        "Restituisci solo output strutturato conforme allo schema GlobalProposalReviewPlan.",
        "Usa ordered_block_ids solo quando l'ordine narrativo va corretto; se l'ordine e gia corretto lascia ordered_block_ids vuoto.",
        "Usa dropped_block_ids solo per blocchi interamente ridondanti o fuori tema.",
        "Usa restored_segments per ripristinare soggetti, antecedenti, definizioni o premesse indispensabili usando solo clip_id/segment_id esistenti.",
        "Non spezzettare il montaggio: preferisci blocchi narrativi completi e naturali.",
        "Controlla in particolare soggetti mancanti, anafore senza antecedente, riferimenti come 'la stessa norma' prima della norma, e ordine causa-effetto.",
        zoom_line,
        f"Brief utente vincolante: synthesis_level={brief.synthesis_level}; target_duration_s={brief.target_duration_s}; prompt={brief.user_prompt}",
        feedback,
        "Blocchi della proposta corrente:",
        "\n".join(block_lines),
        "Sintesi EDL corrente:",
        _edl_context(edit_decision_list, compact=True),
        "Trascrizione sorgente compatta per eventuali restored_segments:",
        _compact_transcript_context(transcriptions),
    ]
    return "\n".join(lines)


def _proposal_review_patch_num_predict_for_attempt(settings: AppSettings, attempt: int) -> int:
    initial_budget = min(settings.ollama_num_predict, 2048)
    if attempt <= 0:
        return initial_budget
    return min(settings.ollama_num_ctx, settings.ollama_num_predict, 8192, initial_budget * (2**attempt))


def _proposal_review_settings(settings: AppSettings) -> AppSettings:
    review_timeout_s = min(settings.ollama_request_timeout_s, settings.proposal_review_timeout_s)
    return settings.model_copy(update={"ollama_request_timeout_s": review_timeout_s})


def _timeout_fallback_review(exc: OllamaTimeoutError) -> ProposalReview:
    timeout_message = str(exc)
    logger.warning("Proposal review timed out; blocking unreviewed proposal")
    return ProposalReview(
        approved=False,
        issues=[
            ProposalIssue(
                severity="error",
                code="proposal_review_timeout",
                message=(
                    "La revisione LLM finale della proposta e andata in timeout: la proposta non e stata "
                    "approvata per il rendering finale."
                ),
            )
        ],
        corrected_narrative_map=None,
        editorial_notes=(
            "Revisione LLM finale non completata per timeout; proposta non approvata. "
            f"Ricalcola una proposta piu compatta o riprova con un modello piu veloce. Dettaglio: {timeout_message}"
        ),
    )


def _invalid_global_blocking_review(feedback: str) -> ProposalReview:
    return ProposalReview(
        approved=False,
        issues=[
            ProposalIssue(
                severity="error",
                code="global_review_rejected_by_validation",
                message=(
                    "La review globale a blocchi ha restituito un piano non applicabile in sicurezza. "
                    "La variante review non e stata creata."
                ),
            )
        ],
        corrected_narrative_map=None,
        editorial_notes=feedback,
    )


def _invalid_patch_blocking_review(feedback: str, rejected_narrative_map: NarrativeMap | None = None) -> ProposalReview:
    return ProposalReview(
        approved=False,
        issues=[
            ProposalIssue(
                severity="error",
                code="review_patch_rejected_by_validation",
                message=(
                    "La review LLM ha proposto una correzione strutturalmente valida ma non accettabile "
                    "per i vincoli narrativi/durata. La patch e stata scartata."
                ),
            )
        ],
        corrected_narrative_map=None,
        rejected_narrative_map=rejected_narrative_map,
        editorial_notes=feedback,
    )


def _invalid_json_blocking_review(feedback: str) -> ProposalReview:
    return ProposalReview(
        approved=False,
        issues=[
            ProposalIssue(
                severity="error",
                code="proposal_review_invalid_json",
                message=(
                    "La revisione LLM finale ha restituito JSON troncato o non conforme allo schema. "
                    "La proposta non e stata approvata per il rendering finale."
                ),
            )
        ],
        corrected_narrative_map=None,
        editorial_notes=feedback,
    )


class ProposalReviewerProtocol(Protocol):
    def review(
        self,
        transcriptions: list[ClipTranscription],
        narrative_map: NarrativeMap,
        edit_decision_list: EditDecisionList,
        brief: EditorialBrief,
        model: str | None,
        editing_options: EditingOptions | None = None,
        progress_callback: object | None = None,
    ) -> ProposalReview:
        ...


def build_proposal_review_prompt(
    transcriptions: list[ClipTranscription],
    narrative_map: NarrativeMap,
    edit_decision_list: EditDecisionList,
    brief: EditorialBrief,
    validation_feedback: str | None = None,
    editing_options: EditingOptions | None = None,
) -> str:
    compact_context = _proposal_review_uses_compact_context(transcriptions, edit_decision_list)
    transcript_lines: list[str] = []
    narrative_context = ""
    if compact_context:
        transcript_context = _compact_transcript_context(transcriptions)
        narrative_context = _compact_narrative_context(narrative_map)
    else:
        for clip in transcriptions:
            transcript_lines.append(f"CLIP {clip.clip_id} durata={clip.duration:.2f}s")
            for segment in clip.segments:
                transcript_lines.append(
                    f"- segment_id={segment.id} start={segment.start:.2f} end={segment.end:.2f} text={segment.text}"
                )
        transcript_context = "\n".join(transcript_lines)
        narrative_context = narrative_map.model_dump_json()

    edl_context = _edl_context(edit_decision_list, compact=compact_context)

    feedback = f"\nCorrezione richiesta: {validation_feedback}\n" if validation_feedback else ""
    creative_zoom_lines = []
    if editing_options is not None and editing_options.enable_creative_zoom:
        creative_zoom_lines = [
            "Zoom creativo attivo: il renderer scegliera gli stacchi con zoom fisso dalle pause mediamente piu lunghe della proposta editoriale.",
            "Se restituisci corrected_narrative_map, non serve assegnare creative_zoom=true: conserva segmenti e pause naturali.",
        ]
    else:
        creative_zoom_lines = ["Zoom creativo non attivo: lascia creative_zoom=false nella corrected_narrative_map."]
    lines = [
            "Sei il revisore finale di una proposta di montaggio video informativo in italiano.",
            "Devi validare il testo effettivo della proposta prima del rendering finale.",
            "Leggi il montaggio proposto nella sequenza esatta dell'EditDecisionList e valuta il risultato come testo parlato finale.",
            "Controlla coerenza temporale, grammaticale, sintattica e semantica: soggetti espliciti, antecedenti gia introdotti, frasi complete e ordine logico.",
            "Non approvare aperture o riprese che iniziano con frammenti, anafore senza antecedente, soggetti mancanti o riferimenti temporali fuori ordine.",
            "Limita al minimo lo spezzettamento del montaggio ed evita di spezzettare frasi che funzionano gia come blocchi narrativi completi.",
            "Devi limare i segmenti mantenendo le parti importanti ed eliminando ripetizioni, esitazioni, overlap e giri di parole sicuri.",
            "Controlla apertura con soggetto esplicito, frasi complete, ordine logico, segmenti in ordine sorgente e assenza di ripetizioni da overlap.",
            "Se la proposta e pronta, restituisci approved=true senza corrected_narrative_map.",
            "Se trovi problemi risolvibili, restituisci approved=false e una corrected_narrative_map completa.",
            "Se trovi problemi non risolvibili, restituisci approved=false senza corrected_narrative_map e issues dettagliati.",
            "Non modificare timestamp o tagli FFmpeg: correggi solo la NarrativeMap usando clip_id e segment_id esistenti.",
            "Nella corrected_narrative_map usa playback_position consecutivi 0, 1, 2... senza duplicati.",
            "Il brief utente e vincolante: verifica che la proposta lo applichi davvero, non solo che sia formalmente valida.",
            "Se il brief chiede un montaggio piu asciutto, controlla che ripetizioni, esitazioni e giri di parole sicuri siano classificati come optional o redundant.",
            "Se la proposta e identica a una proposta neutra pur avendo un brief specifico, restituisci corrected_narrative_map quando ci sono tagli sicuri.",
        ]
    if compact_context:
        lines.extend(
            [
                "Contesto grande: usa la review compatta per verificare comunque coerenza, ordine narrativo e rispetto del brief.",
                "Anche in contesto grande puoi restituire una corrected_narrative_map completa se la correzione e deducibile dagli ID e dal contesto forniti.",
                "Se il contesto compatto non basta per correggere in modo sicuro, non approvare: segnala issue dettagliati per bloccare il Via libera automatico.",
            ]
        )
    lines.extend(creative_zoom_lines)
    lines.extend(
        [
            "Restituisci solo output strutturato conforme allo schema ProposalReview.",
            f"Brief utente vincolante: synthesis_level={brief.synthesis_level}; prompt={brief.user_prompt}",
            feedback,
            "Trascrizione sorgente compatta:" if compact_context else "Trascrizione sorgente:",
            transcript_context,
            "NarrativeMap corrente compatta:" if compact_context else "NarrativeMap corrente:",
            narrative_context,
            "EditDecisionList corrente:",
            edl_context,
        ]
    )
    return "\n".join(lines)


def build_light_proposal_review_prompt(
    transcriptions: list[ClipTranscription],
    narrative_map: NarrativeMap,
    edit_decision_list: EditDecisionList,
    brief: EditorialBrief,
    validation_feedback: str | None = None,
    editing_options: EditingOptions | None = None,
) -> str:
    transcript_context = _compact_transcript_context(transcriptions)
    valid_id_context = _valid_patch_id_context(transcriptions)
    narrative_context = _compact_narrative_context(narrative_map)
    edl_context = _edl_context(
        edit_decision_list,
        compact=True,
    )
    word_context = _word_context_for_edl(transcriptions, edit_decision_list)
    feedback = f"\nCorrezione richiesta: {validation_feedback}\n" if validation_feedback else ""
    creative_zoom_line = (
        "Zoom creativo attivo: non modificare zoom; correggi solo keep/cut/order se serve."
        if editing_options is not None and editing_options.enable_creative_zoom
        else "Zoom creativo non attivo."
    )
    lines = [
        "Sei il revisore finale leggero di una proposta di montaggio video informativo in italiano.",
        "Non generare una NarrativeMap completa: restituisci solo una ProposalReviewPatch con modifiche puntuali.",
        "Leggi l'EditDecisionList nella sequenza esatta e valuta il testo parlato finale.",
        "Controlla coerenza temporale, grammaticale, sintattica e semantica.",
        "Blocca soggetti mancanti, antecedenti non introdotti, frasi monche, riferimenti fuori ordine, ripetizioni e overlap.",
        "Limita al minimo lo spezzettamento: preferisci segmenti narrativi completi e naturali.",
        "Per limare il montaggio usa segment_adjustments: marca come redundant/optional i segmenti ripetitivi o deboli, e keep_verbatim=false solo quando il taglio e sicuro.",
        "Per autoriparazioni o micro-ripetizioni interne allo stesso segmento usa word_removals con indici parola inclusivi 0-based.",
        "Esempio: se il testo dice 'ha perso, ha rimesso, ha una perdita' e il brief chiede ritmo asciutto, rimuovi solo le parole di 'ha perso, ha rimesso' preservando numeri e dato finale.",
        "Usa word_removals solo quando il testo resta grammaticale e il taglio cade su confini parola; non rimuovere numeri, date, importi o nomi propri.",
        "Per ripristinare contesto essenziale usa segment_adjustments con keep_verbatim=true su segmenti esistenti.",
        "Per correggere l'ordine globale usa clip_order con clip_id esistenti; ometti clip_order se l'ordine e gia corretto.",
        "Usa esclusivamente gli ID validi elencati sotto; se non trovi un ID valido, non proporre quella modifica.",
        "Se la proposta e coerente restituisci approved=true, issues=[], segment_adjustments=[] e clip_order=[].",
        "Se servono correzioni restituisci approved=false con segment_adjustments e/o clip_order.",
        "Mantieni il JSON corto: massimo 6 modifiche totali tra segment_adjustments e word_removals, massimo 3 issues, editorial_notes sotto 160 caratteri.",
        "Se servirebbero piu di 6 modifiche, restituisci solo le 6 piu importanti e segnala issue code='partial_review_patch'.",
        "Se il contesto non basta per correggere in sicurezza, restituisci approved=false con issues dettagliati e nessuna patch rischiosa.",
        "Il brief utente e vincolante: verifica che venga applicato davvero.",
        creative_zoom_line,
        "Restituisci solo output strutturato conforme allo schema ProposalReviewPatch.",
        f"Brief utente vincolante: synthesis_level={brief.synthesis_level}; target_duration_s={brief.target_duration_s}; prompt={brief.user_prompt}",
        feedback,
        "ID validi per segment_adjustments e clip_order:",
        valid_id_context,
        "Trascrizione sorgente compatta:",
        transcript_context,
        "NarrativeMap corrente compatta:",
        narrative_context,
        "EditDecisionList corrente:",
        edl_context,
        "Indici parola disponibili per micro-rimozioni:",
        word_context,
    ]
    return "\n".join(lines)


def _extract_message_content(response: object) -> str:
    if isinstance(response, dict):
        message = response.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
    message_object = getattr(response, "message", None)
    content_object = getattr(message_object, "content", None)
    if isinstance(content_object, str):
        return content_object
    raise ProposalReviewError("Risposta Ollama priva di message.content")


def _known_segments_by_clip(transcriptions: list[ClipTranscription]) -> dict[int, set[int]]:
    return {
        clip.clip_id: {segment.id for segment in clip.segments}
        for clip in transcriptions
    }


def _segment_lookup(transcriptions: list[ClipTranscription]) -> dict[tuple[int, int], Segment]:
    return {
        (clip.clip_id, segment.id): segment
        for clip in transcriptions
        for segment in clip.segments
    }


def _segment_source_positions(transcriptions: list[ClipTranscription]) -> dict[tuple[int, int], int]:
    positions: dict[tuple[int, int], int] = {}
    for clip in transcriptions:
        for index, segment in enumerate(clip.segments):
            positions[(clip.clip_id, segment.id)] = index
    return positions


def _word_removals_cover_entire_segment(word_count: int, removals: list[ProposalWordRemoval]) -> bool:
    if word_count <= 0:
        return False
    removed_indexes: set[int] = set()
    for removal in removals:
        removed_indexes.update(range(removal.start_word_index, removal.end_word_index + 1))
    return len(removed_indexes) >= word_count


def _sanitize_patch_references(
    patch: ProposalReviewPatch,
    transcriptions: list[ClipTranscription],
    _narrative_map: NarrativeMap | None = None,
) -> ProposalReviewPatch:
    known_segments = _known_segments_by_clip(transcriptions)
    segments = _segment_lookup(transcriptions)
    known_clips = set(known_segments)
    valid_adjustments: list[ProposalSegmentAdjustment] = []
    valid_word_removals: list[ProposalWordRemoval] = []
    valid_clip_order: list[int] = []
    issues = list(patch.issues)
    ignored_references: list[str] = []
    seen_clip_order: set[int] = set()

    for clip_id in patch.clip_order:
        if clip_id in seen_clip_order:
            ignored_references.append(f"duplicate clip_id={clip_id} in clip_order")
            continue
        seen_clip_order.add(clip_id)
        if clip_id not in known_clips:
            ignored_references.append(f"unknown clip_id={clip_id} in clip_order")
            continue
        valid_clip_order.append(clip_id)

    for adjustment in patch.segment_adjustments:
        if adjustment.clip_id not in known_segments:
            ignored_references.append(
                f"unknown clip_id={adjustment.clip_id} segment_id={adjustment.segment_id}"
            )
            continue
        if adjustment.segment_id not in known_segments[adjustment.clip_id]:
            ignored_references.append(
                f"unknown clip_id={adjustment.clip_id} segment_id={adjustment.segment_id}"
            )
            continue
        valid_adjustments.append(adjustment)

    for removal in patch.word_removals:
        if removal.clip_id not in known_segments:
            ignored_references.append(
                f"unknown word_removal clip_id={removal.clip_id} segment_id={removal.segment_id}"
            )
            continue
        if removal.segment_id not in known_segments[removal.clip_id]:
            ignored_references.append(
                f"unknown word_removal clip_id={removal.clip_id} segment_id={removal.segment_id}"
            )
            continue
        word_count = len(segments[(removal.clip_id, removal.segment_id)].words)
        if removal.end_word_index >= word_count:
            ignored_references.append(
                "out-of-range word_removal "
                f"clip_id={removal.clip_id} segment_id={removal.segment_id} "
                f"range={removal.start_word_index}-{removal.end_word_index}"
            )
            continue
        valid_word_removals.append(removal)

    full_segment_removal_keys: set[tuple[int, int]] = set()
    grouped_removals: dict[tuple[int, int], list[ProposalWordRemoval]] = {}
    for removal in valid_word_removals:
        grouped_removals.setdefault((removal.clip_id, removal.segment_id), []).append(removal)
    for key, removals in grouped_removals.items():
        clip_id, segment_id = key
        word_count = len(segments[(clip_id, segment_id)].words)
        if _word_removals_cover_entire_segment(word_count, removals):
            full_segment_removal_keys.add(key)

    if full_segment_removal_keys:
        valid_word_removals = [
            removal
            for removal in valid_word_removals
            if (removal.clip_id, removal.segment_id) not in full_segment_removal_keys
        ]
        converted_adjustments: list[ProposalSegmentAdjustment] = []
        converted_keys: set[tuple[int, int]] = set()
        for adjustment in valid_adjustments:
            key = (adjustment.clip_id, adjustment.segment_id)
            if key not in full_segment_removal_keys:
                converted_adjustments.append(adjustment)
                continue
            if key in converted_keys:
                continue
            clip_id, segment_id = key
            source_segment = segments[key]
            converted_adjustments.append(
                ProposalSegmentAdjustment(
                    clip_id=clip_id,
                    segment_id=segment_id,
                    importance=SegmentImportance.REDUNDANT,
                    keep_verbatim=False,
                    reason=(
                        "Review LLM ha richiesto la rimozione di tutte le parole del segmento; "
                        f"convertito in taglio dell'intero segmento: {_short_text(source_segment.text, 120)}"
                    ),
                )
            )
            converted_keys.add(key)
        for clip_id, segment_id in sorted(full_segment_removal_keys - converted_keys):
            source_segment = segments[(clip_id, segment_id)]
            converted_adjustments.append(
                ProposalSegmentAdjustment(
                    clip_id=clip_id,
                    segment_id=segment_id,
                    importance=SegmentImportance.REDUNDANT,
                    keep_verbatim=False,
                    reason=(
                        "Review LLM ha richiesto la rimozione di tutte le parole del segmento; "
                        f"convertito in taglio dell'intero segmento: {_short_text(source_segment.text, 120)}"
                    ),
                )
            )
        valid_adjustments = converted_adjustments
        issues.append(
            ProposalIssue(
                severity="warning",
                code="full_segment_word_removal_converted",
                message=(
                    "La review LLM ha espresso un taglio di segmento come word_removal totale; "
                    "la patch e stata convertita nel taglio segmento equivalente."
                ),
            )
        )

    if not ignored_references:
        return patch.model_copy(
            update={
                "issues": issues,
                "segment_adjustments": valid_adjustments,
                "word_removals": valid_word_removals,
                "clip_order": valid_clip_order,
            }
        )

    logger.bind(ignored_patch_references=ignored_references).warning(
        "Ignored invalid ProposalReviewPatch references"
    )
    issues.append(
        ProposalIssue(
            severity="warning",
            code="invalid_patch_reference_ignored",
            message=(
                "La review LLM ha indicato riferimenti clip/segmento non validi; "
                "le modifiche non applicabili sono state ignorate."
            ),
        )
    )
    approved = patch.approved
    if (
        not valid_adjustments
        and not valid_word_removals
        and not valid_clip_order
        and not any(issue.severity == "error" for issue in issues)
    ):
        approved = True
    return patch.model_copy(
        update={
            "approved": approved,
            "issues": issues,
            "segment_adjustments": valid_adjustments,
            "word_removals": valid_word_removals,
            "clip_order": valid_clip_order,
        }
    )


def _find_narrative_clip(narrative_map: NarrativeMap, clip_id: int) -> NarrativeClip | None:
    for clip in narrative_map.ordered_clips:
        if clip.clip_id == clip_id:
            return clip
    return None


def _ensure_narrative_clip(narrative_map: NarrativeMap, clip_id: int) -> NarrativeClip:
    narrative_clip = _find_narrative_clip(narrative_map, clip_id)
    if narrative_clip is not None:
        return narrative_clip
    narrative_clip = NarrativeClip(
        clip_id=clip_id,
        playback_position=len(narrative_map.ordered_clips),
        topic_summary="Segmenti ripristinati dalla review finale",
        role="supporting",
        segments=[],
    )
    narrative_map.ordered_clips.append(narrative_clip)
    return narrative_clip


def _apply_segment_adjustment(
    narrative_map: NarrativeMap,
    adjustment: ProposalSegmentAdjustment,
    source_positions: dict[tuple[int, int], int],
) -> bool:
    narrative_clip = _ensure_narrative_clip(narrative_map, adjustment.clip_id)
    for index, segment in enumerate(narrative_clip.segments):
        if segment.segment_id == adjustment.segment_id:
            clear_synthesis_text = adjustment.keep_verbatim and segment.synthesis_text is not None
            if (
                segment.importance == adjustment.importance
                and segment.keep_verbatim == adjustment.keep_verbatim
                and not clear_synthesis_text
            ):
                return False
            narrative_clip.segments[index] = segment.model_copy(
                update={
                    "importance": adjustment.importance,
                    "keep_verbatim": adjustment.keep_verbatim,
                    "synthesis_text": None if adjustment.keep_verbatim else segment.synthesis_text,
                    "reason": adjustment.reason,
                }
            )
            break
    else:
        narrative_clip.segments.append(
            NarrativeSegment(
                segment_id=adjustment.segment_id,
                importance=adjustment.importance,
                keep_verbatim=adjustment.keep_verbatim,
                reason=adjustment.reason,
            )
        )
    narrative_clip.segments.sort(
        key=lambda segment: source_positions.get((narrative_clip.clip_id, segment.segment_id), segment.segment_id)
    )
    return True


def _apply_word_removal(
    narrative_map: NarrativeMap,
    removal: ProposalWordRemoval,
) -> bool:
    narrative_clip = _find_narrative_clip(narrative_map, removal.clip_id)
    if narrative_clip is None:
        return False
    for index, segment in enumerate(narrative_clip.segments):
        if segment.segment_id != removal.segment_id:
            continue
        next_range = WordRemovalRange(
            start_word_index=removal.start_word_index,
            end_word_index=removal.end_word_index,
            reason=removal.reason,
        )
        existing_ranges = list(segment.remove_word_ranges)
        if any(
            existing.start_word_index == next_range.start_word_index
            and existing.end_word_index == next_range.end_word_index
            for existing in existing_ranges
        ):
            return False
        narrative_clip.segments[index] = segment.model_copy(
            update={"remove_word_ranges": existing_ranges + [next_range]}
        )
        return True
    return False


def _apply_clip_order_patch(narrative_map: NarrativeMap, clip_order: list[int]) -> bool:
    if not clip_order:
        return False
    current_by_id = {clip.clip_id: clip for clip in narrative_map.ordered_clips}
    current_order = sorted(narrative_map.ordered_clips, key=lambda clip: clip.playback_position)
    ordered: list[NarrativeClip] = [current_by_id[clip_id] for clip_id in clip_order if clip_id in current_by_id]
    ordered.extend(clip for clip in current_order if clip.clip_id not in set(clip_order))
    current_clip_order = [clip.clip_id for clip in current_order]
    new_clip_order = [clip.clip_id for clip in ordered]
    if current_clip_order == new_clip_order:
        return False
    for position, clip in enumerate(ordered):
        clip.playback_position = position
    narrative_map.ordered_clips = ordered
    return True


def _normalize_review_playback_positions(narrative_map: NarrativeMap) -> None:
    for position, clip in enumerate(narrative_map.ordered_clips):
        clip.playback_position = position


def _validate_global_review_plan_references(
    plan: GlobalProposalReviewPlan,
    narrative_map: NarrativeMap,
    transcriptions: list[ClipTranscription],
) -> None:
    block_ids = {block_id for block_id, _clip in _global_review_clip_pairs(narrative_map)}
    ordered_ids = plan.ordered_block_ids
    dropped_ids = plan.dropped_block_ids
    if len(ordered_ids) != len(set(ordered_ids)):
        raise NarrativeValidationError("GlobalProposalReviewPlan contiene ordered_block_ids duplicati")
    if len(dropped_ids) != len(set(dropped_ids)):
        raise NarrativeValidationError("GlobalProposalReviewPlan contiene dropped_block_ids duplicati")
    unknown_ordered = [block_id for block_id in ordered_ids if block_id not in block_ids]
    if unknown_ordered:
        raise NarrativeValidationError(f"GlobalProposalReviewPlan contiene block_id sconosciuti: {unknown_ordered}")
    unknown_dropped = [block_id for block_id in dropped_ids if block_id not in block_ids]
    if unknown_dropped:
        raise NarrativeValidationError(f"GlobalProposalReviewPlan contiene dropped block_id sconosciuti: {unknown_dropped}")
    overlap = sorted(set(ordered_ids).intersection(dropped_ids))
    if overlap:
        raise NarrativeValidationError(f"GlobalProposalReviewPlan ordina e scarta gli stessi block_id: {overlap}")

    known_segments = _known_segments_by_clip(transcriptions)
    for restored in plan.restored_segments:
        if restored.clip_id not in known_segments:
            raise NarrativeValidationError(
                f"GlobalProposalReviewPlan restored_segments contiene clip_id sconosciuto {restored.clip_id}"
            )
        if restored.segment_id not in known_segments[restored.clip_id]:
            raise NarrativeValidationError(
                "GlobalProposalReviewPlan restored_segments contiene segment_id sconosciuto "
                f"{restored.segment_id} per clip_id {restored.clip_id}"
            )


def _apply_global_review_plan(
    narrative_map: NarrativeMap,
    plan: GlobalProposalReviewPlan,
    transcriptions: list[ClipTranscription],
) -> NarrativeMap | None:
    _validate_global_review_plan_references(plan, narrative_map, transcriptions)
    block_pairs = _global_review_clip_pairs(narrative_map)
    if not block_pairs:
        return None
    if not plan.ordered_block_ids and not plan.dropped_block_ids and not plan.restored_segments:
        return None

    corrected_map = narrative_map.model_copy(deep=True)
    corrected_pairs = _global_review_clip_pairs(corrected_map)
    clips_by_block_id = {block_id: clip for block_id, clip in corrected_pairs}
    current_ids = [block_id for block_id, _clip in corrected_pairs]
    dropped_ids = set(plan.dropped_block_ids)
    ordered_ids = plan.ordered_block_ids or current_ids
    next_ids = [block_id for block_id in ordered_ids if block_id not in dropped_ids]
    next_ids.extend(block_id for block_id in current_ids if block_id not in set(next_ids) and block_id not in dropped_ids)
    if not next_ids:
        raise NarrativeValidationError("GlobalProposalReviewPlan non puo rimuovere tutti i blocchi narrativi")

    changed = next_ids != current_ids
    corrected_map.ordered_clips = [clips_by_block_id[block_id] for block_id in next_ids]
    _normalize_review_playback_positions(corrected_map)

    source_positions = _segment_source_positions(transcriptions)
    for restored in plan.restored_segments:
        changed = (
            _apply_segment_adjustment(
                corrected_map,
                ProposalSegmentAdjustment(
                    clip_id=restored.clip_id,
                    segment_id=restored.segment_id,
                    importance=restored.importance,
                    keep_verbatim=True,
                    reason=restored.reason,
                ),
                source_positions,
            )
            or changed
        )
    _normalize_review_playback_positions(corrected_map)
    if not changed:
        return None
    note = _short_text(plan.editorial_notes, 240)
    if note:
        corrected_map.editorial_strategy = f"{corrected_map.editorial_strategy} Review globale: {note}"
    return corrected_map


def _apply_review_patch(
    narrative_map: NarrativeMap,
    patch: ProposalReviewPatch,
    transcriptions: list[ClipTranscription],
) -> NarrativeMap | None:
    if not patch.segment_adjustments and not patch.word_removals and not patch.clip_order:
        return None
    corrected_map = narrative_map.model_copy(deep=True)
    source_positions = _segment_source_positions(transcriptions)
    changed = False
    for adjustment in patch.segment_adjustments:
        changed = _apply_segment_adjustment(corrected_map, adjustment, source_positions) or changed
    for removal in patch.word_removals:
        changed = _apply_word_removal(corrected_map, removal) or changed
    changed = _apply_clip_order_patch(corrected_map, patch.clip_order) or changed
    if not changed:
        return None
    note = _short_text(patch.editorial_notes, 240)
    if note:
        corrected_map.editorial_strategy = f"{corrected_map.editorial_strategy} Review finale: {note}"
    return corrected_map


def _light_review_chunks(edit_decision_list: EditDecisionList) -> list[EditDecisionList]:
    if len(edit_decision_list.decisions) <= _LIGHT_REVIEW_CHUNK_DECISIONS:
        return [edit_decision_list]
    chunks: list[EditDecisionList] = []
    for start in range(0, len(edit_decision_list.decisions), _LIGHT_REVIEW_CHUNK_DECISIONS):
        decisions = edit_decision_list.decisions[start : start + _LIGHT_REVIEW_CHUNK_DECISIONS]
        duration_s = sum(max(0.0, decision.source_end - decision.source_start) for decision in decisions)
        chunks.append(EditDecisionList(decisions=decisions, estimated_duration_s=duration_s))
    return chunks


def _review_reference_duration_s(
    transcriptions: list[ClipTranscription],
    edit_decision_list: EditDecisionList,
) -> float:
    source_duration_s = sum(max(0.0, clip.duration) for clip in transcriptions)
    if source_duration_s > 0:
        return source_duration_s
    if edit_decision_list.estimated_duration_s > 0:
        return edit_decision_list.estimated_duration_s
    return sum(max(0.0, decision.source_end - decision.source_start) for decision in edit_decision_list.decisions)


def _review_is_mandatory_for_duration(
    transcriptions: list[ClipTranscription],
    edit_decision_list: EditDecisionList,
) -> bool:
    return _review_reference_duration_s(transcriptions, edit_decision_list) < _MANDATORY_REVIEW_UNDER_DURATION_S


def _combine_review_patches(patches: list[ProposalReviewPatch]) -> ProposalReviewPatch:
    issues: list[ProposalIssue] = []
    segment_adjustments: list[ProposalSegmentAdjustment] = []
    word_removals: list[ProposalWordRemoval] = []
    clip_order: list[int] = []
    notes: list[str] = []
    approved = True
    for patch in patches:
        approved = approved and patch.approved
        issues.extend(patch.issues)
        segment_adjustments.extend(patch.segment_adjustments)
        word_removals.extend(patch.word_removals)
        if patch.clip_order:
            clip_order = patch.clip_order
        if patch.editorial_notes:
            notes.append(_short_text(patch.editorial_notes, 160))
    if segment_adjustments or word_removals or clip_order:
        approved = False
    return ProposalReviewPatch(
        approved=approved,
        issues=issues,
        segment_adjustments=segment_adjustments,
        word_removals=word_removals,
        clip_order=clip_order,
        editorial_notes=" ".join(notes),
    )


def _patch_has_invalid_references(patch: ProposalReviewPatch) -> bool:
    return any(issue.code == "invalid_patch_reference_ignored" for issue in patch.issues)


def _local_review_skipped_large_proposal_issue(chunk_count: int) -> ProposalIssue:
    return ProposalIssue(
        severity="warning",
        code="local_review_skipped_large_proposal",
        message=(
            "La review locale dettagliata e stata saltata per evitare blocchi LLM su una proposta grande "
            f"({chunk_count} chunk locali). La review globale a blocchi resta applicata."
        ),
    )


def _proposal_review_skipped_large_proposal_issue(chunk_count: int) -> ProposalIssue:
    return ProposalIssue(
        severity="warning",
        code="proposal_review_skipped_large_proposal",
        message=(
            "La review LLM finale e stata saltata per evitare un blocco su una proposta grande "
            f"({chunk_count} chunk locali stimati). Usa la proposta iniziale o ricalcola con una durata target piu breve."
        ),
    )


def _ultra_light_review_issue() -> ProposalIssue:
    return ProposalIssue(
        severity="warning",
        code="ultra_light_review",
        message=(
            "La review locale dettagliata non ha prodotto una patch valida; "
            "e stata usata una review globale ultra-light renderizzabile."
        ),
    )


class ProposalReviewer:
    def __init__(self, settings: AppSettings, max_retries: int = 2) -> None:
        self.settings = settings
        self.max_retries = max_retries

    def review(
        self,
        transcriptions: list[ClipTranscription],
        narrative_map: NarrativeMap,
        edit_decision_list: EditDecisionList,
        brief: EditorialBrief,
        model: str | None,
        editing_options: EditingOptions | None = None,
        progress_callback: object | None = None,
    ) -> ProposalReview:
        estimated_chunks = _light_review_chunks(edit_decision_list)
        mandatory_review = _review_is_mandatory_for_duration(transcriptions, edit_decision_list)
        if len(estimated_chunks) > _MAX_LIGHT_REVIEW_CHUNKS and not mandatory_review:
            if callable(progress_callback):
                progress_callback(
                    "Review LLM saltata: proposta grande oltre 10 minuti, variante conservativa renderizzabile"
                )
            chunk_count = len(estimated_chunks)
            return ProposalReview(
                approved=True,
                issues=[_proposal_review_skipped_large_proposal_issue(chunk_count)],
                corrected_narrative_map=None,
                editorial_notes=(
                    "Review LLM finale saltata per evitare blocchi di elaborazione su una proposta grande; "
                    "viene esposta una variante review conservativa uguale alla proposta corrente."
                ),
            )
        try:
            import ollama
        except ImportError as exc:
            raise ProposalReviewError("Il pacchetto ollama non e installato") from exc

        selected_model = model or self.settings.default_ollama_model
        review_settings = _proposal_review_settings(self.settings)
        working_map = narrative_map
        working_edl = edit_decision_list
        global_issues: list[ProposalIssue] = []
        global_notes: list[str] = []
        global_feedback: str | None = None
        invalid_global_feedback: str | None = None
        invalid_global_json_feedback: str | None = None
        invalid_global_narrative_map: NarrativeMap | None = None
        for attempt in range(self.max_retries + 1):
            try:
                if callable(progress_callback):
                    progress_callback(f"Review LLM globale blocchi tentativo {attempt + 1}/{self.max_retries + 1}")
                invalid_global_feedback = None
                invalid_global_json_feedback = None
                invalid_global_narrative_map = None
                prompt = build_global_proposal_review_prompt(
                    transcriptions,
                    working_map,
                    working_edl,
                    brief,
                    global_feedback,
                    editing_options,
                )
                response = structured_chat(
                    ollama,
                    review_settings,
                    model=selected_model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=GlobalProposalReviewPlan.model_json_schema(),
                    num_predict=_proposal_review_patch_num_predict_for_attempt(self.settings, attempt),
                )
                global_plan = GlobalProposalReviewPlan.model_validate_json(_extract_message_content(response))
                corrected_global_map = _apply_global_review_plan(working_map, global_plan, transcriptions)
                global_issues = global_plan.issues
                if global_plan.editorial_notes:
                    global_notes.append(global_plan.editorial_notes)
                if corrected_global_map is not None:
                    invalid_global_narrative_map = corrected_global_map
                    repair_narrative_map_for_cut_safety(corrected_global_map, transcriptions)
                    validate_narrative_map(corrected_global_map, transcriptions, editing_options, brief)
                    working_map = corrected_global_map
                    working_edl = build_edit_decision_list(transcriptions, working_map, editing_options or EditingOptions())
                break
            except OllamaTimeoutError as exc:
                return _timeout_fallback_review(exc)
            except OllamaRuntimeError as exc:
                raise ProposalReviewError(str(exc)) from exc
            except ValidationError as exc:
                if validation_error_is_json_invalid(exc):
                    global_feedback = (
                        "Output JSON troncato o non valido: restituisci un GlobalProposalReviewPlan completo, "
                        "usa JSON compatto e chiudi tutti gli array/oggetti."
                    )
                else:
                    global_feedback = f"Schema GlobalProposalReviewPlan non valido: {exc}"
                invalid_global_json_feedback = global_feedback
                invalid_global_feedback = None
            except NarrativeValidationError as exc:
                global_feedback = f"GlobalProposalReviewPlan non valido: {exc}"
                invalid_global_feedback = global_feedback
                invalid_global_json_feedback = None
        else:
            if invalid_global_feedback is not None:
                if invalid_global_narrative_map is not None and "excessive_synthesis" in invalid_global_feedback:
                    return _invalid_patch_blocking_review(invalid_global_feedback, invalid_global_narrative_map)
                return _invalid_global_blocking_review(invalid_global_feedback)
            if invalid_global_json_feedback is not None:
                return _invalid_json_blocking_review(invalid_global_json_feedback)
            raise ProposalReviewError(global_feedback or "GlobalProposalReviewPlan non valido")

        feedback: str | None = None
        invalid_patch_feedback: str | None = None
        invalid_patch_narrative_map: NarrativeMap | None = None
        invalid_json_feedback: str | None = None
        for attempt in range(self.max_retries + 1):
            try:
                if callable(progress_callback):
                    progress_callback(f"Review LLM locale cut e frasi tentativo {attempt + 1}/{self.max_retries + 1}")
                invalid_patch_feedback = None
                invalid_patch_narrative_map = None
                invalid_json_feedback = None
                patches: list[ProposalReviewPatch] = []
                chunks = _light_review_chunks(working_edl)
                if len(chunks) > _MAX_LIGHT_REVIEW_CHUNKS and not mandatory_review:
                    if callable(progress_callback):
                        progress_callback(
                            "Review LLM locale saltata: proposta grande, uso review globale a blocchi"
                        )
                    issues = global_issues + [_local_review_skipped_large_proposal_issue(len(chunks))]
                    notes = " ".join(
                        note
                        for note in global_notes
                        + [
                            "Review locale chunked saltata per evitare blocchi; "
                            "variante basata sulla review globale a blocchi."
                        ]
                        if note
                    )
                    if working_map is not narrative_map:
                        return ProposalReview(
                            approved=False,
                            issues=issues,
                            corrected_narrative_map=working_map,
                            editorial_notes=notes,
                        )
                    return ProposalReview(
                        approved=True,
                        issues=issues,
                        corrected_narrative_map=None,
                        editorial_notes=notes,
                    )
                for chunk_index, chunk in enumerate(chunks, start=1):
                    if callable(progress_callback):
                        progress_callback(f"Review LLM locale chunk {chunk_index}/{len(chunks)}")
                    prompt = build_light_proposal_review_prompt(
                        transcriptions,
                        working_map,
                        chunk,
                        brief,
                        feedback,
                        editing_options,
                    )
                    response = structured_chat(
                        ollama,
                        review_settings,
                        model=selected_model,
                        messages=[{"role": "user", "content": prompt}],
                        response_format=ProposalReviewPatch.model_json_schema(),
                        num_predict=_proposal_review_patch_num_predict_for_attempt(self.settings, attempt),
                    )
                    patches.append(ProposalReviewPatch.model_validate_json(_extract_message_content(response)))
                patch = _sanitize_patch_references(_combine_review_patches(patches), transcriptions, working_map)
                if _patch_has_invalid_references(patch):
                    merged_issues = global_issues + patch.issues
                    merged_notes = " ".join(note for note in global_notes + [patch.editorial_notes] if note)
                    if callable(progress_callback):
                        progress_callback(
                            "Review LLM locale con riferimenti non validi: salto patch locale parziale"
                        )
                    if working_map is not narrative_map:
                        return ProposalReview(
                            approved=False,
                            issues=merged_issues,
                            corrected_narrative_map=working_map,
                            editorial_notes=merged_notes,
                        )
                    return ProposalReview(
                        approved=True,
                        issues=merged_issues,
                        corrected_narrative_map=None,
                        editorial_notes=merged_notes,
                    )
                corrected_narrative_map = _apply_review_patch(working_map, patch, transcriptions)
                if corrected_narrative_map is not None:
                    invalid_patch_narrative_map = corrected_narrative_map
                    repair_narrative_map_for_cut_safety(corrected_narrative_map, transcriptions)
                    validate_narrative_map(corrected_narrative_map, transcriptions, editing_options, brief)
                requested_noop_patch = (
                    corrected_narrative_map is None
                    and (bool(patch.segment_adjustments) or bool(patch.word_removals) or bool(patch.clip_order))
                    and not any(issue.severity == "error" for issue in patch.issues)
                )
                merged_issues = global_issues + patch.issues
                merged_notes = " ".join(note for note in global_notes + [patch.editorial_notes] if note)
                if corrected_narrative_map is None and working_map is not narrative_map:
                    return ProposalReview(
                        approved=False,
                        issues=merged_issues,
                        corrected_narrative_map=working_map,
                        editorial_notes=merged_notes,
                    )
                return ProposalReview(
                    approved=(patch.approved or requested_noop_patch) and corrected_narrative_map is None,
                    issues=merged_issues,
                    corrected_narrative_map=corrected_narrative_map,
                    editorial_notes=merged_notes,
                )
            except OllamaTimeoutError as exc:
                return _timeout_fallback_review(exc)
            except OllamaRuntimeError as exc:
                raise ProposalReviewError(str(exc)) from exc
            except ValidationError as exc:
                if validation_error_is_json_invalid(exc):
                    feedback = (
                        "Output JSON troncato o non valido: restituisci una ProposalReviewPatch completa, "
                        "usa JSON compatto e chiudi tutti gli array/oggetti."
                    )
                else:
                    feedback = f"Schema ProposalReviewPatch non valido: {exc}"
                invalid_patch_feedback = None
                invalid_json_feedback = feedback
            except NarrativeValidationError as exc:
                available = [
                    f"clip_id={clip.clip_id}: segment_ids={[segment.id for segment in clip.segments]}"
                    for clip in transcriptions
                ]
                feedback = f"ProposalReviewPatch non valida: {exc}. Usa solo questi ID: {'; '.join(available)}"
                invalid_patch_feedback = feedback
                invalid_json_feedback = None
        if invalid_patch_feedback is not None:
            return _invalid_patch_blocking_review(invalid_patch_feedback, invalid_patch_narrative_map)
        if invalid_json_feedback is not None:
            try:
                if callable(progress_callback):
                    progress_callback("Review LLM fallback ultra-light")
                prompt = build_global_proposal_review_prompt(
                    transcriptions,
                    working_map,
                    working_edl,
                    brief,
                    (
                        f"{invalid_json_feedback} Fallback ultra-light: correggi solo ordine blocchi, "
                        "blocchi da scartare o segmenti da ripristinare. Non proporre word_removals."
                    ),
                    editing_options,
                )
                response = structured_chat(
                    ollama,
                    review_settings,
                    model=selected_model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=GlobalProposalReviewPlan.model_json_schema(),
                    num_predict=_proposal_review_patch_num_predict_for_attempt(self.settings, self.max_retries + 1),
                )
                fallback_plan = GlobalProposalReviewPlan.model_validate_json(_extract_message_content(response))
                fallback_map = _apply_global_review_plan(working_map, fallback_plan, transcriptions)
                review_map = fallback_map if fallback_map is not None else working_map.model_copy(deep=True)
                repair_narrative_map_for_cut_safety(review_map, transcriptions)
                validate_narrative_map(
                    review_map,
                    transcriptions,
                    editing_options,
                    brief,
                    enforce_duration_budget=False,
                )
                issues = global_issues + fallback_plan.issues
                if not issues:
                    issues.append(_ultra_light_review_issue())
                notes = " ".join(
                    note
                    for note in global_notes + [fallback_plan.editorial_notes, "Fallback ultra-light renderizzabile."]
                    if note
                )
                return ProposalReview(
                    approved=False,
                    issues=issues,
                    corrected_narrative_map=review_map,
                    editorial_notes=notes,
                )
            except OllamaTimeoutError as exc:
                return _timeout_fallback_review(exc)
            except OllamaRuntimeError as exc:
                raise ProposalReviewError(str(exc)) from exc
            except ValidationError as exc:
                if validation_error_is_json_invalid(exc):
                    return _invalid_json_blocking_review(
                        "Output JSON troncato o non valido anche nel fallback ultra-light: "
                        "restituisci un GlobalProposalReviewPlan completo."
                    )
                return _invalid_json_blocking_review(f"Schema GlobalProposalReviewPlan fallback non valido: {exc}")
            except NarrativeValidationError as exc:
                return _invalid_global_blocking_review(f"Fallback ultra-light non renderizzabile: {exc}")
        raise ProposalReviewError(feedback or "ProposalReviewPatch non valida")


class MockProposalReviewer:
    def review(
        self,
        transcriptions: list[ClipTranscription],
        narrative_map: NarrativeMap,
        edit_decision_list: EditDecisionList,
        brief: EditorialBrief,
        model: str | None,
        editing_options: EditingOptions | None = None,
    ) -> ProposalReview:
        return ProposalReview(
            approved=True,
            issues=[],
            corrected_narrative_map=None,
            editorial_notes="Proposta validata dal reviewer mockabile.",
        )
