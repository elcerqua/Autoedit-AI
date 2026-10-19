from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PipelineStage(str, Enum):
    QUEUED = "queued"
    INTERPRETING_PROMPT = "interpreting_prompt"
    NORMALIZING_CLIPS = "normalizing_clips"
    TRANSCRIBING = "transcribing"
    NARRATIVE_ANALYSIS = "narrative_analysis"
    REFINING = "refining"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    COMPOSING = "composing"
    MIXING_MUSIC = "mixing_music"
    DONE = "done"
    FAILED = "failed"


class EditorialBrief(BaseModel):
    user_prompt: str = ""
    target_duration_s: float | None = Field(default=None, gt=0)
    tone: Literal["energetic", "intimate", "ironic", "informative", "neutral"] = "neutral"
    synthesis_level: int = Field(default=0, ge=0, le=5)
    clip_order: Literal["upload", "ai_narrative"] = "ai_narrative"
    must_include_keywords: list[str] = Field(default_factory=list)
    must_avoid_keywords: list[str] = Field(default_factory=list)


class WordToken(BaseModel):
    word: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    probability: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_time_order(self) -> "WordToken":
        if self.end < self.start:
            raise ValueError("word end must be greater than or equal to start")
        return self


class Segment(BaseModel):
    id: int
    clip_id: int
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str
    words: list[WordToken]

    @model_validator(mode="after")
    def validate_segment_time_order(self) -> "Segment":
        if self.end < self.start:
            raise ValueError("segment end must be greater than or equal to start")
        return self


class ClipTranscription(BaseModel):
    clip_id: int
    source_path: str | None = None
    duration: float = Field(ge=0)
    segments: list[Segment]


class SegmentImportance(str, Enum):
    ESSENTIAL = "essential"
    VALUABLE = "valuable"
    OPTIONAL = "optional"
    REDUNDANT = "redundant"


class FillerRef(BaseModel):
    clip_id: int
    segment_id: int
    word_index: int = Field(ge=0)
    word_text: str


class WordRemovalRange(BaseModel):
    start_word_index: int = Field(ge=0)
    end_word_index: int = Field(ge=0)
    reason: str

    @model_validator(mode="after")
    def validate_word_range_order(self) -> "WordRemovalRange":
        if self.end_word_index < self.start_word_index:
            raise ValueError("end_word_index must be greater than or equal to start_word_index")
        return self


class NarrativeSegment(BaseModel):
    segment_id: int
    importance: SegmentImportance
    keep_verbatim: bool
    synthesis_text: str | None = None
    remove_word_ranges: list[WordRemovalRange] = Field(default_factory=list)
    creative_zoom: bool = Field(
        default=False,
        description="True only when this retained important scene should receive the optional creative zoom emphasis.",
    )
    creative_zoom_peak_scale: float = Field(
        default=1.35,
        ge=1.2,
        le=1.5,
        description="Peak zoom scale for creative_zoom, where 1.2 means +20% and 1.5 means +50%.",
    )
    reason: str


class NarrativeClip(BaseModel):
    clip_id: int
    playback_position: int = Field(ge=0)
    topic_summary: str
    role: Literal["introduction", "development", "climax", "conclusion", "supporting", "skip"]
    segments: list[NarrativeSegment]


class NarrativeMap(BaseModel):
    ordered_clips: list[NarrativeClip]
    filler_words: list[FillerRef]
    editorial_strategy: str
    estimated_duration_s: float = Field(ge=0)
    synthesis_script: str | None = None


TransitionName = Literal[
    "cut",
    "cross_dissolve",
    "fade_black",
    "zoom_cut",
    "whip_pan",
    "fade",
    "wipeleft",
    "wiperight",
    "wipeup",
    "wipedown",
    "slideleft",
    "slideright",
    "slideup",
    "slidedown",
    "circlecrop",
    "rectcrop",
    "distance",
    "fadeblack",
    "fadewhite",
    "radial",
    "smoothleft",
    "smoothright",
    "smoothup",
    "smoothdown",
    "circleopen",
    "circleclose",
    "vertopen",
    "vertclose",
    "horzopen",
    "horzclose",
    "dissolve",
    "pixelize",
    "diagtl",
    "diagtr",
    "diagbl",
    "diagbr",
    "hlslice",
    "hrslice",
    "vuslice",
    "vdslice",
    "hblur",
    "fadegrays",
    "wipetl",
    "wipetr",
    "wipebl",
    "wipebr",
    "squeezeh",
    "squeezev",
    "zoomin",
    "fadefast",
    "fadeslow",
    "hlwind",
    "hrwind",
    "vuwind",
    "vdwind",
    "coverleft",
    "coverright",
    "coverup",
    "coverdown",
    "revealleft",
    "revealright",
    "revealup",
    "revealdown",
]
AspectRatioName = Literal["source", "16:9", "9:16", "1:1"]
SubtitlePosition = Literal["center", "bottom"]
SubtitleStylePresetName = Literal["classic_white", "creator_yellow", "boxed_black", "minimal_shadow"]


class SubtitleOptions(BaseModel):
    enabled: bool = False
    position: SubtitlePosition = "bottom"
    style_preset: SubtitleStylePresetName = "classic_white"
    max_chars_per_line: int = Field(default=34, ge=12, le=80)
    max_lines: int = Field(default=2, ge=1, le=4)


class SubtitleStylePresetCatalogItem(BaseModel):
    id: SubtitleStylePresetName
    label: str
    font_family: str
    preview_data_uri: str


class EditingOptions(BaseModel):
    transition_duration_ms: int = Field(default=200, ge=0, le=3000)
    default_transition: TransitionName = "cut"
    enable_dynamic_zoom: bool = False
    enable_creative_zoom: bool = False
    min_creative_zoom_count: int = Field(default=0, ge=0, le=200)
    zoom_intensity: float = Field(default=0.04, ge=0, le=0.5)
    enable_motion_transition: bool = False
    target_aspect_ratio: AspectRatioName = "source"
    target_resolution: str | None = None
    normalize_audio: bool = True
    loudness_target_lufs: float = Field(default=-16.0, ge=-40, le=0)
    remove_silence: bool = True
    silence_threshold_db: float = Field(default=-35.0, ge=-80, le=0)
    min_silence_duration_ms: int = Field(default=500, ge=50, le=10000)
    add_background_music: bool = False
    music_prompt: str | None = None
    music_volume_db: float = Field(default=-24.0, ge=-60, le=0)
    ducking_enabled: bool = True
    add_transition_sounds: bool = False
    transition_sound: str | None = None
    min_pause_to_cut_ms: int = Field(default=300, ge=0, le=10000)
    pre_roll_ms: int = Field(default=80, ge=0, le=2000)
    post_roll_ms: int = Field(default=120, ge=0, le=2000)
    min_segment_duration_ms: int = Field(default=700, ge=0, le=60000)
    max_cut_aggressiveness: float = Field(default=0.6, ge=0, le=1)
    preserve_numbers: bool = True
    preserve_named_entities: bool = True
    avoid_hard_cut_inside_sentence: bool = True
    subtitle_options: SubtitleOptions = Field(default_factory=SubtitleOptions)

    @field_validator("target_resolution")
    @classmethod
    def validate_target_resolution(cls, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        parts = value.lower().split("x")
        if len(parts) != 2 or not all(part.isdigit() and int(part) > 0 for part in parts):
            raise ValueError("target_resolution must use WIDTHxHEIGHT, for example 1920x1080")
        return value

    @field_validator("transition_sound")
    @classmethod
    def validate_transition_sound(cls, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        path = Path(value)
        if path.name != value:
            raise ValueError("transition_sound must be a file name from the Sounds folder")
        if path.suffix.lower() != ".mp3":
            raise ValueError("transition_sound must be an mp3 file")
        return value

    @model_validator(mode="after")
    def validate_transition_sound_enabled(self) -> "EditingOptions":
        if self.add_transition_sounds and self.transition_sound is None:
            raise ValueError("transition_sound is required when add_transition_sounds is enabled")
        return self


class CreativeZoomSpan(BaseModel):
    source_start: float = Field(ge=0)
    source_end: float = Field(ge=0)
    start_scale: float = Field(default=1.0, ge=1.0, le=1.5)
    end_scale: float = Field(default=1.35, ge=1.0, le=1.5)
    text: str = ""
    reason: str | None = None

    @model_validator(mode="after")
    def validate_span_time_order(self) -> "CreativeZoomSpan":
        if self.source_end <= self.source_start:
            raise ValueError("creative zoom span source_end must be greater than source_start")
        return self


class EditDecision(BaseModel):
    clip_id: int
    source_start: float = Field(ge=0)
    source_end: float = Field(ge=0)
    output_order: int = Field(ge=0)
    transition_before: str | None = None
    transition_after: str | None = None
    creative_zoom: bool = False
    creative_zoom_start_scale: float = Field(default=1.2, ge=1.0, le=2.0)
    creative_zoom_end_scale: float = Field(default=1.5, ge=1.0, le=2.0)
    creative_zoom_spans: list[CreativeZoomSpan] = Field(default_factory=list)
    reason: str
    included_text: str
    safety_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_decision_time_order(self) -> "EditDecision":
        if self.source_end < self.source_start:
            raise ValueError("source_end must be greater than or equal to source_start")
        return self


class EditDecisionList(BaseModel):
    decisions: list[EditDecision]
    estimated_duration_s: float = Field(ge=0)


class ProposalIssue(BaseModel):
    severity: Literal["error", "warning"]
    code: str
    message: str
    clip_id: int | None = None
    segment_id: int | None = None
    decision_order: int | None = Field(default=None, ge=0)


class ProposalSegmentAdjustment(BaseModel):
    clip_id: int
    segment_id: int
    importance: SegmentImportance
    keep_verbatim: bool
    reason: str


class ProposalWordRemoval(WordRemovalRange):
    clip_id: int
    segment_id: int


class ProposalReviewPatch(BaseModel):
    approved: bool
    issues: list[ProposalIssue] = Field(default_factory=list)
    segment_adjustments: list[ProposalSegmentAdjustment] = Field(default_factory=list)
    word_removals: list[ProposalWordRemoval] = Field(default_factory=list)
    clip_order: list[int] = Field(
        default_factory=list,
        description="Optional clip_id order when the proposal needs a lighter narrative reorder.",
    )
    editorial_notes: str = ""


class ProposalReview(BaseModel):
    approved: bool
    issues: list[ProposalIssue] = Field(default_factory=list)
    corrected_narrative_map: NarrativeMap | None = None
    rejected_narrative_map: NarrativeMap | None = None
    editorial_notes: str = ""


class ProposalVariant(BaseModel):
    id: Literal["initial", "review"]
    label: str
    narrative_map: NarrativeMap
    edit_decision_list: EditDecisionList
    duration_s: float = Field(ge=0)
    selectable: bool = True
    issues: list[ProposalIssue] = Field(default_factory=list)
    rejection_reason: str | None = None


class JobStatus(BaseModel):
    job_id: str
    stage: PipelineStage
    message: str | None = None
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    elapsed_seconds: float
    stage_started_at: str | None = None
    current_stage_elapsed_seconds: float
    error: str | None = None


class ApprovalRequest(BaseModel):
    approved: bool = True
    note: str | None = None
    proposal_variant_id: Literal["initial", "review"] | None = None


class ProposalTextEdit(BaseModel):
    output_order: int = Field(ge=0)
    included_text: str = Field(min_length=1, max_length=4000)

    @field_validator("included_text")
    @classmethod
    def validate_included_text(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("included_text must not be empty")
        return normalized


class ProposalTextPatchRequest(BaseModel):
    proposal_variant_id: Literal["initial", "review"]
    edits: list[ProposalTextEdit] = Field(default_factory=list)
    deleted_output_orders: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_output_orders(self) -> "ProposalTextPatchRequest":
        edit_output_orders = [edit.output_order for edit in self.edits]
        deleted_output_orders = self.deleted_output_orders
        if len(edit_output_orders) != len(set(edit_output_orders)):
            raise ValueError("edits must not contain duplicate output_order values")
        if any(output_order < 0 for output_order in deleted_output_orders):
            raise ValueError("deleted_output_orders must contain non-negative values")
        if len(deleted_output_orders) != len(set(deleted_output_orders)):
            raise ValueError("deleted_output_orders must not contain duplicate output_order values")
        overlapping_output_orders = set(edit_output_orders) & set(deleted_output_orders)
        if overlapping_output_orders:
            raise ValueError("edits and deleted_output_orders must not reference the same output_order")
        return self


class ProposalRecalculateRequest(BaseModel):
    synthesis_level: int | None = Field(default=None, ge=0, le=5)
    target_duration_s: float | None = Field(default=None, gt=0)


class JobCreateRequest(BaseModel):
    brief: EditorialBrief = Field(default_factory=EditorialBrief)
    llm_model: str | None = None
    editing_options: EditingOptions = Field(default_factory=EditingOptions)
    clip_paths: list[str] = Field(default_factory=list)


class JobRecord(BaseModel):
    model_config = ConfigDict(use_enum_values=False)

    job_id: str
    stage: PipelineStage = PipelineStage.QUEUED
    message: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)
    started_at: str | None = None
    completed_at: str | None = None
    elapsed_seconds: float = 0.0
    stage_started_at: str | None = None
    current_stage_elapsed_seconds: float = 0.0
    error: str | None = None
    brief: EditorialBrief = Field(default_factory=EditorialBrief)
    editing_options: EditingOptions = Field(default_factory=EditingOptions)
    llm_model: str | None = None
    input_clip_paths: list[str] = Field(default_factory=list)
    transcriptions: list[ClipTranscription] = Field(default_factory=list)
    narrative_map: NarrativeMap | None = None
    edit_decision_list: EditDecisionList | None = None
    proposal_review: ProposalReview | None = None
    proposal_variants: list[ProposalVariant] = Field(default_factory=list)
    selected_proposal_variant_id: Literal["initial", "review"] = "initial"
    final_output_path: str | None = None

    def to_status(self) -> JobStatus:
        return JobStatus(
            job_id=self.job_id,
            stage=self.stage,
            message=self.message,
            created_at=self.created_at,
            started_at=self.started_at,
            completed_at=self.completed_at,
            elapsed_seconds=self.elapsed_seconds,
            stage_started_at=self.stage_started_at,
            current_stage_elapsed_seconds=self.current_stage_elapsed_seconds,
            error=self.error,
        )


class JobCreateResponse(BaseModel):
    job_id: str
    status: JobStatus
    job: JobRecord


class ProposalRecalculateResponse(BaseModel):
    job_id: str
    status: JobStatus
    job: JobRecord


class ProposalTextPatchResponse(BaseModel):
    job_id: str
    status: JobStatus
    job: JobRecord


class UploadStartResponse(BaseModel):
    upload_id: str


class UploadChunkResponse(BaseModel):
    upload_id: str
    clip_index: int
    received_bytes: int
    file_size: int
    complete: bool
    path: str


class JobEvent(BaseModel):
    sequence: int = 0
    job_id: str
    stage: PipelineStage
    message: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)
    elapsed_seconds: float = 0.0
    current_stage_elapsed_seconds: float = 0.0


class EditingOptionsCatalog(BaseModel):
    available_transitions: list[str]
    available_transition_sounds: list[str] = Field(default_factory=list)
    subtitle_style_presets: list[SubtitleStylePresetCatalogItem] = Field(default_factory=list)
    target_aspect_ratios: list[str]
    defaults: EditingOptions


class LlmModelsResponse(BaseModel):
    models: list[str]
    default_model: str | None = None
