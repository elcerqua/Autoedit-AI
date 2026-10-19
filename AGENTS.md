# Auto Cut AI — AGENTS.md

Context file for Codex. Use this file as the operating contract for building **Auto Cut AI**, a multi-clip AI video-editing application for spoken Italian informational videos.

The application must ingest one or more video clips, transcribe spoken audio, let an LLM decide the editorial structure, produce a reviewed edit proposal, wait for explicit user approval, and only then render the final social-ready video.

---

## 1. Core principle

The LLM is the primary editorial intelligence.

It receives raw transcriptions with word-level timestamps and produces a validated `NarrativeMap`. That map drives:

- clip ordering;
- segment importance;
- filler removal;
- redundant concept removal;
- synthesis level;
- final narrative flow;
- edit decisions.

All downstream stages are deterministic executors. They validate, align, and render the LLM decisions, but they must not invent new editorial choices except for clearly technical cleanup configured by the user, such as silence removal.

Never let FFmpeg, silence detection, or deterministic code override the editorial intent of the `NarrativeMap` unless required for cut safety.

---

## 2. Required stack

Use this stack unless the existing repository already defines a stricter compatible structure:

- **Backend**: FastAPI
- **Schemas**: Pydantic v2
- **Database**: SQLite
- **Progress events**: SSE, server-sent events
- **Job runner**: FastAPI `BackgroundTasks` or equivalent simple sync runner
- **Frontend**: React
- **Transcription**: `faster-whisper`, model `large-v3`, Italian, word-level timestamps
- **LLM**: Ollama through the official Python `ollama` package
- **Default LLM**: `qwen3:8b` if available
- **Video**: FFmpeg through `subprocess`
- **Logging**: loguru
- **Tests**: pytest
- **Fixtures**: `tests/fixtures/1.mp4` and `tests/fixtures/2.mp4`

Forbidden unless explicitly requested later:

- MoviePy
- LangChain
- Instructor
- LiteLLM
- free-form JSON parsing from the LLM
- regex-based validation of LLM output
- ffmpeg command strings built by concatenation
- global mutable application state

---

## 3. Multi-agent working mode for Codex

When implementing this project, work as if these specialized agents are collaborating. Keep their responsibilities separate in code and tests.

### 3.1 Orchestrator Agent

Responsibilities:

- inspect repository structure before editing;
- respect this `AGENTS.md`;
- define implementation order;
- keep backend, frontend, schemas, pipeline, FFmpeg, LLM, and tests compatible;
- avoid speculative features;
- do not move to final video rendering before the approval phase works.

### 3.2 Backend/API Agent

Responsibilities:

- implement FastAPI routes;
- manage multi-clip uploads;
- create and persist jobs;
- expose job status;
- expose SSE events;
- expose editing options;
- expose Ollama model selection;
- implement approval endpoint;
- expose final video download;
- track total elapsed pipeline time and current-stage elapsed time.

### 3.3 LLM/Narrative Agent

Responsibilities:

- call Ollama;
- let the user select from locally available Ollama models;
- default to `qwen3:8b` if present;
- use structured output with `format=NarrativeMap.model_json_schema()`;
- produce a chronological/narrative analysis before deciding cuts;
- classify every segment by editorial importance;
- preserve names, numbers, dates, amounts, and logical dependencies;
- retry only when structured output is invalid or references unknown IDs.

### 3.4 Transcription/Alignment Agent

Responsibilities:

- transcribe every clip with word-level timestamps;
- keep `clip_id`, `segment_id`, word index, start, end, and probability;
- ensure cut boundaries are word-safe;
- prevent cuts in the middle of words, numbers, dates, acronyms, and unit expressions;
- provide deterministic helpers for timestamp alignment.

### 3.5 Video/FFmpeg Agent

Responsibilities:

- normalize clips;
- trim according to the Edit Decision List;
- concatenate;
- apply selected transitions;
- normalize or mix audio;
- render final MP4;
- keep intermediate files during development.

Rules:

- use FFmpeg only through `subprocess.run([...], check=True, capture_output=True, text=True)`;
- pass arguments as lists;
- never construct shell command strings;
- no MoviePy.

### 3.6 Frontend/UI Agent

Responsibilities:

- implement a React UI for upload, model selection, synthesis level, editing options, progress, review, approval, and download;
- show real-time status through SSE;
- show elapsed time;
- show the edit proposal before composition;
- require an explicit “Via libera” action before rendering.

### 3.7 QA/Test Agent

Responsibilities:

- write pytest unit tests and integration tests;
- use mocks for heavy components when appropriate;
- include slow/e2e markers for tests using real media fixtures;
- verify cut safety;
- verify pipeline approval gating;
- verify Ollama model listing;
- verify editing option validation;
- verify timer updates.

---

## 4. Functional goal

Build an application that edits one or more spoken informational clips into a final video ready for social publishing.

The editing is performed by synchronizing text, audio, and video. The user should be able to remove or reduce:

- silences;
- hesitations;
- repeated phrases;
- redundant concepts;
- off-topic digressions;
- filler words;
- low-value segments.

The system must produce a human-readable editorial proposal first. Rendering the final video must happen only after the user gives explicit approval.

---

## 5. Synthesis level scale

Implement `synthesis_level` as an integer from 0 to 5.

| Level | Meaning | Approximate text removed |
|---:|---|---:|
| 0 | No editorial synthesis. Remove only configured obvious silences/fillers. | 0% |
| 1 | Minimal synthesis. Remove only very obvious redundancies. | ~2% |
| 2 | Light synthesis. Remove repetitions and minor digressions. | ~5% |
| 3 | Medium synthesis. Improve rhythm while preserving natural speech. | ~10% |
| 4 | Strong synthesis. Shorten less important narrative blocks. | ~15% |
| 5 | Maximum synthesis. Keep only essential content without making the edit unnatural. | ~20% |

The percentage is a target, not a blind rule. Priority order:

1. narrative coherence;
2. natural spoken flow;
3. no cut inside words or numbers;
4. preservation of essential concepts;
5. approximate duration or text-reduction target.

---

## 6. Pipeline stages

The pipeline is divided into two mandatory phases.

### Phase 1 — Recognition, transcription, analysis, edit proposal

Allowed states:

```text
queued
→ interpreting_prompt
→ normalizing_clips
→ transcribing
→ narrative_analysis
→ refining
→ waiting_for_approval
```

Outputs from Phase 1:

- full transcript;
- detected clips and durations;
- chronological/narrative order;
- segment importance map;
- segments to keep;
- segments to cut;
- detected fillers;
- estimated final duration;
- estimated final text;
- selected technical editing options;
- text preview of the Edit Decision List;
- editorial strategy generated by the LLM.

The pipeline must stop at `waiting_for_approval`.

### Phase 2 — Final composition

Only after explicit approval:

```text
waiting_for_approval
→ composing
→ mixing_music, if enabled
→ done
```

Error state:

```text
failed
```

Never enter `composing` without approval.

---

## 7. Required API endpoints

Implement these endpoints or compatible equivalents.

### Jobs

```text
POST /jobs
GET /jobs/{job_id}
GET /jobs/{job_id}/events
POST /jobs/{job_id}/approve
GET /jobs/{job_id}/download
```

Behavior:

- `POST /jobs` creates a job from one or more clips and settings.
- `GET /jobs/{job_id}` returns status, timers, proposal, and final output metadata.
- `GET /jobs/{job_id}/events` streams SSE progress events.
- `POST /jobs/{job_id}/approve` is valid only when the job is in `waiting_for_approval`.
- `GET /jobs/{job_id}/download` returns the final MP4 only after `done`.

### LLM models

```text
GET /llm/models
```

Behavior:

- query local Ollama;
- return available model names;
- mark `qwen3:8b` as default if present;
- otherwise choose the first available model;
- if Ollama is unreachable, return a clear error without breaking the entire frontend.

### Editing options

```text
GET /editing-options
```

Behavior:

- return available transitions;
- return valid target aspect ratios and resolutions;
- return default audio and cut-safety settings.

---

## 8. Editing options

Support these options through validated Pydantic models.

### Transitions

Supported values:

- `cut`
- `cross_dissolve`
- `fade_black`
- `zoom_cut`
- `whip_pan`

### Technical parameters

At minimum support:

```python
transition_duration_ms: int
default_transition: Literal["cut", "cross_dissolve", "fade_black", "zoom_cut", "whip_pan"]
enable_dynamic_zoom: bool
zoom_intensity: float
enable_motion_transition: bool
target_aspect_ratio: Literal["source", "16:9", "9:16", "1:1"]
target_resolution: str | None
normalize_audio: bool
loudness_target_lufs: float
remove_silence: bool
silence_threshold_db: float
min_silence_duration_ms: int
add_background_music: bool
music_prompt: str | None
music_volume_db: float
ducking_enabled: bool
```

### Cut-safety parameters

At minimum support:

```python
min_pause_to_cut_ms: int
pre_roll_ms: int
post_roll_ms: int
min_segment_duration_ms: int
max_cut_aggressiveness: float
preserve_numbers: bool
preserve_named_entities: bool
avoid_hard_cut_inside_sentence: bool
```

---

## 9. Central data schemas

Use Pydantic v2 for all I/O between modules. Do not pass untyped dictionaries across boundaries unless immediately validated.

Implement or adapt these models.

```python
from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field


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
    target_duration_s: float | None = None
    tone: Literal["energetic", "intimate", "ironic", "informative", "neutral"] = "neutral"
    synthesis_level: int = Field(default=0, ge=0, le=5)
    clip_order: Literal["upload", "ai_narrative"] = "ai_narrative"
    must_include_keywords: list[str] = Field(default_factory=list)
    must_avoid_keywords: list[str] = Field(default_factory=list)


class WordToken(BaseModel):
    word: str
    start: float
    end: float
    probability: float | None = None


class Segment(BaseModel):
    id: int
    clip_id: int
    start: float
    end: float
    text: str
    words: list[WordToken]


class ClipTranscription(BaseModel):
    clip_id: int
    source_path: str | None = None
    duration: float
    segments: list[Segment]


class SegmentImportance(str, Enum):
    ESSENTIAL = "essential"
    VALUABLE = "valuable"
    OPTIONAL = "optional"
    REDUNDANT = "redundant"


class FillerRef(BaseModel):
    clip_id: int
    segment_id: int
    word_index: int
    word_text: str


class NarrativeSegment(BaseModel):
    segment_id: int
    importance: SegmentImportance
    keep_verbatim: bool
    synthesis_text: str | None = None
    reason: str


class NarrativeClip(BaseModel):
    clip_id: int
    playback_position: int
    topic_summary: str
    role: Literal["introduction", "development", "climax", "conclusion", "supporting", "skip"]
    segments: list[NarrativeSegment]


class NarrativeMap(BaseModel):
    ordered_clips: list[NarrativeClip]
    filler_words: list[FillerRef]
    editorial_strategy: str
    estimated_duration_s: float
    synthesis_script: str | None = None


class EditDecision(BaseModel):
    clip_id: int
    source_start: float
    source_end: float
    output_order: int
    transition_before: str | None = None
    transition_after: str | None = None
    reason: str
    included_text: str
    safety_notes: list[str] = Field(default_factory=list)


class EditDecisionList(BaseModel):
    decisions: list[EditDecision]
    estimated_duration_s: float


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
```

You may extend these schemas, but do not weaken validation.

---

## 10. LLM behavior

The LLM prompt must instruct the model to behave as a professional video editor working on Italian informational clips.

The LLM must:

- analyze the full transcript first;
- determine chronological and narrative periods;
- infer the role of each clip;
- classify each segment as `essential`, `valuable`, `optional`, or `redundant`;
- respect `synthesis_level` 0–5;
- preserve core argumentation;
- preserve numbers, names, dates, amounts, places, and cause-effect links;
- remove only safe repetitions, fillers, redundant concepts, and low-value digressions;
- favor natural video cuts;
- output only a structure conforming to `NarrativeMap`.

Use:

```python
format=NarrativeMap.model_json_schema()
```

Never parse free-form JSON. Validate the model response through Pydantic.

Retry policy:

- invalid schema: retry with validation error feedback;
- invalid `clip_id` or `segment_id`: retry with available IDs;
- filler mismatch: log and skip invalid filler references;
- after maximum retries: fail the job with a clear error.

---

## 11. Cut safety

Implement a deterministic function that converts:

```text
NarrativeMap + ClipTranscription + EditingOptions → EditDecisionList
```

The function must:

- include only LLM-approved retained content, plus configured technical silence cleanup;
- snap `source_start` and `source_end` to word boundaries;
- apply `pre_roll_ms` and `post_roll_ms` safely;
- avoid cuts inside words;
- avoid cuts inside numbers;
- avoid cuts between a number and its unit;
- avoid cuts inside dates, prices, percentages, named entities, or acronyms where detectable;
- avoid segments shorter than `min_segment_duration_ms`;
- log boundary corrections;
- include `safety_notes` for every adjusted decision.

The code may adjust a cut boundary for safety. It must not invent unrelated editorial cuts.

---

## 12. FFmpeg rendering rules

Use FFmpeg through subprocess only.

General rules:

- always call FFmpeg with a list of arguments;
- always use `check=True`;
- capture stderr;
- do not use `shell=True`;
- do not concatenate command strings;
- keep intermediate files during development;
- store outputs under a per-job directory.

Default export:

- container: MP4
- video codec: H.264
- CRF: 18
- audio codec: AAC
- audio bitrate: 192k
- fps: normalized to source or configured target
- resolution: source resolution unless user selected another target

Transitions:

- `cut`: direct trim/concat
- `cross_dissolve`: FFmpeg `xfade`
- `fade_black`: fade out/in through black
- `zoom_cut`: subtle scale/crop effect around cut
- `whip_pan`: motion-style transition, implemented with FFmpeg filters where feasible

If transition rendering is too complex for many clips, use a safe fallback to concat demuxer and log the fallback.

---

## 13. Timer and progress tracking

Each job must track:

```python
created_at
started_at
completed_at
elapsed_seconds
stage_started_at
current_stage_elapsed_seconds
```

The frontend must show:

- total elapsed time;
- current pipeline stage;
- elapsed time in current stage;
- progress messages.

Update progress through SSE events. Persist enough state in SQLite so refreshing the frontend does not lose the job status.

---

## 14. Frontend requirements

Build a simple but complete React interface.

Main screen:

- multi-video upload;
- user brief/prompt field;
- Ollama model selector;
- synthesis level slider from 0 to 5;
- transition selector;
- audio options;
- silence removal options;
- target aspect ratio/resolution options;
- “Analizza video” button.

Pipeline screen:

- current state;
- total elapsed timer;
- current stage timer;
- progress log;
- uploaded clip list.

Review screen at `waiting_for_approval`:

- original transcript summary;
- LLM editorial strategy;
- proposed narrative order;
- kept segments;
- removed segments;
- filler removals;
- estimated final duration;
- technical editing settings;
- Edit Decision List preview;
- “Via libera” button;
- optional “Annulla” or “Modifica impostazioni” button if easy to implement.

Final screen:

- composition progress;
- final status;
- download link for MP4.

---

## 15. Testing requirements

Write pytest tests for at least these areas.

### Synthesis level mapping

Verify:

- level 0 → 0%;
- level 1 → about 2%;
- level 2 → about 5%;
- level 3 → about 10%;
- level 4 → about 15%;
- level 5 → about 20%.

### NarrativeMap validation

Verify:

- only known `clip_id` values are accepted;
- only known `segment_id` values are accepted;
- playback positions are coherent;
- duplicate playback positions are rejected or normalized deterministically.

### Cut safety

Verify:

- no cut starts or ends inside a word;
- no cut splits a number;
- pre-roll and post-roll are applied safely;
- minimum segment duration is respected;
- boundary correction notes are emitted.

### Pipeline approval

Verify:

- pipeline stops at `waiting_for_approval`;
- `composing` does not start without approval;
- `POST /jobs/{job_id}/approve` moves the job toward `composing` only from the correct state.

### Ollama model endpoint

Verify:

- available models are returned when Ollama is mocked;
- `qwen3:8b` is selected as default if present;
- unreachable Ollama is handled clearly.

### Editing options

Verify:

- `GET /editing-options` returns valid transitions;
- job creation accepts valid options;
- invalid options are rejected.

### Fixture tests

Use:

```text
tests/fixtures/1.mp4
tests/fixtures/2.mp4
```

For real media tests, use markers:

```python
@pytest.mark.e2e
@pytest.mark.slow
```

Heavy tests may mock transcription, LLM, and FFmpeg while still verifying the pipeline contract.

---

## 16. Code conventions

- Python 3.11+
- Type hints everywhere
- Pydantic v2 for all schemas
- `pathlib.Path` for filesystem paths
- FastAPI dependency injection for config where possible
- SQLite for persisted jobs
- loguru for structured logs
- `logger.bind(job_id=job_id)` for job logs
- sync pipeline modules for CPU/GPU-heavy work
- async only in FastAPI routes or network I/O where justified
- no broad `except Exception`; catch specific exceptions
- no global mutable state
- clear function names
- small modules with explicit boundaries

---

## 17. Implementation order

Proceed in this order:

1. Inspect repository structure.
2. Identify existing backend, frontend, models, routes, tests, and fixtures.
3. Add or update Pydantic schemas.
4. Implement synthesis-level mapping.
5. Implement `EditingOptions`.
6. Implement Ollama model listing.
7. Implement job persistence and pipeline stages.
8. Implement timer tracking.
9. Implement transcription adapter or mockable interface.
10. Implement LLM `NarrativeMap` generation.
11. Implement `NarrativeMap` validation.
12. Implement Edit Decision List conversion and cut safety.
13. Implement `waiting_for_approval` pause.
14. Implement approval endpoint.
15. Implement FFmpeg composition.
16. Implement SSE progress.
17. Implement React UI.
18. Implement tests.
19. Run tests.
20. Fix failures.
21. Report changed files, tests run, and known limitations.

---

## 18. Acceptance criteria

The implementation is acceptable only when:

- the app accepts one or more clips;
- the backend creates a persisted job;
- the pipeline emits state updates;
- transcription produces word-level timestamps;
- Ollama generates a validated `NarrativeMap`;
- the system creates an edit proposal before rendering;
- the pipeline stops at `waiting_for_approval`;
- the user can give “Via libera”;
- final composition starts only after approval;
- the final video is rendered as MP4;
- the frontend shows status and timers;
- the frontend lists available Ollama models;
- editing options are visible and configurable;
- cuts do not truncate words or numbers;
- tests cover the key behavior;
- fixtures `1.mp4` and `2.mp4` are considered in tests or marked slow/e2e.

---

## 19. Final Codex report format

At the end of implementation, report:

- summary of implementation;
- files created or modified;
- available endpoints;
- pipeline structure;
- how to start backend;
- how to start frontend;
- how to run tests;
- which tests passed or failed;
- known limitations;
- confirmation that `tests/fixtures/1.mp4` and `tests/fixtures/2.mp4` were considered.
