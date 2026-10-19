# Auto Cut AI

**English** | [Italiano](README.it.md)

Local, AI-assisted editing for spoken Italian informational videos. Upload one or more clips, describe the result you want, review an editorial proposal, and explicitly approve it before rendering the final MP4.

The LLM decides the narrative structure. Deterministic alignment and FFmpeg execute those decisions, with boundary adjustments to protect words and numerical expressions.

## Features

- Multi-clip uploads, including chunked uploads through the web interface.
- Italian transcription with `faster-whisper` `large-v3` and word-level timestamps.
- Local Ollama model selection, preferring `qwen3:8b` when available.
- Structured, Pydantic-validated `NarrativeMap` output for clip order, segment importance, fillers, and editorial strategy.
- Six synthesis levels, from minimal cleanup to stronger condensation.
- Reviewable proposals with transcript, retained and removed content, estimated duration, and an Edit Decision List (EDL).
- Proposal text editing and recalculation before approval.
- Explicit **Via libera** approval before final composition.
- Live progress through server-sent events (SSE), total and stage timers, and SQLite job persistence.
- Aspect ratio and resolution controls, audio normalization, transition effects and sounds, and optional burned-in subtitles.
- H.264/AAC MP4 export.

## Requirements

- Python **3.11 or newer** and Git.
- **FFmpeg and ffprobe** available on `PATH`. Burned-in subtitles require an FFmpeg build with the ASS/libass filter.
- **Ollama** installed and running locally, with at least one downloaded model.
- Enough RAM and disk space for the transcription and language models. Defaults use CPU transcription with `int8` compute; runtime depends on hardware, model, and clip length.

Model weights are not included. If the configured local Whisper snapshot is absent, the application falls back to the `large-v3` model name, which may download weights on first use. Initial model downloads require internet access; processing uses local services.

## Installation

Clone the repository and create a virtual environment:

```sh
git clone https://github.com/elcerqua/autoedit_AI.git
cd autoedit_AI
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```sh
source .venv/bin/activate
```

Install the application and download the default Ollama model:

```sh
python -m pip install -e .
ollama pull qwen3:8b
```

Ensure Ollama is running. If its desktop application or service is not already running, start it with `ollama serve` in another terminal. Check the required tools:

```sh
ollama list
ffmpeg -version
ffprobe -version
```

## Run

From the repository root, with the virtual environment active:

```sh
python -m uvicorn autocut_ai.app:app --host 127.0.0.1 --port 8000
```

Open the [web interface](http://127.0.0.1:8000/) or the [interactive API documentation](http://127.0.0.1:8000/docs).

The backend serves the included HTML/CSS/JavaScript frontend. No Node.js installation, frontend build, or separate frontend server is required.

Windows also includes `run.bat` and `stop.bat`. The launcher first tries the original development machine's bundled Python, then falls back to `python`; use the command above when you want to select your virtual environment explicitly.

## Editing workflow

1. Upload one or more videos and enter an editorial brief.
2. Select a local Ollama model, synthesis level, and technical options.
3. Click **Analizza video** and follow the progress and timers.
4. At `waiting_for_approval`, inspect the strategy, proposed order, selected content, and EDL. Edit or recalculate the proposal if needed.
5. Click **Via libera** to approve the proposal and start composition.
6. Download the MP4 when the job reaches `done`.

### Synthesis levels

| Level | Editorial intent | Approximate text removed |
| --- | --- | --- |
| 0 | No editorial synthesis; configured technical cleanup only | 0% |
| 1 | Remove very obvious redundancies | 2% |
| 2 | Remove repetitions and minor digressions | 5% |
| 3 | Improve rhythm while preserving natural speech | 10% |
| 4 | Shorten less important narrative blocks | 15% |
| 5 | Keep essential content with natural continuity | 20% |

These are editorial targets, not guaranteed reduction percentages. Coherence, essential information, natural speech, and safe boundaries take priority.

### Pipeline

```text
queued -> interpreting_prompt -> normalizing_clips -> transcribing
       -> narrative_analysis -> refining -> waiting_for_approval

Explicit approval -> composing -> [mixing_music] -> done
Errors -> failed
```

The pipeline stops at `waiting_for_approval`. Final video composition requires approval of a proposal in that state. The `mixing_music` state is currently a placeholder; see the limitations below.

## Editing options and configuration

The interface exposes transitions (`cut`, `cross_dissolve`, `fade_black`, `zoom_cut`, `whip_pan`), silence cleanup, zoom options, audio normalization, transition sounds from `Sounds/`, subtitle styles, and output sizing (`source`, `16:9`, `9:16`, `1:1`). Resolutions use `WIDTHxHEIGHT`, for example `1920x1080`.

Cut-safety settings include pre/post roll, minimum segment duration, pause thresholds, and preservation of numbers and named entities. The actual selectable defaults are returned by `GET /editing-options`.

Application settings are defined in [`autocut_ai/config.py`](autocut_ai/config.py) as `AppSettings`. Custom application instances can be constructed with `create_app(AppSettings(...))`. Settings include the Whisper model path/device, FFmpeg executable, data paths, and Ollama timeouts/context limits. The application does not automatically load these settings from a `.env` file.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/jobs` | Create a job from uploaded clips and settings |
| `GET` | `/jobs/{job_id}` | Read job status, timers, and proposal |
| `GET` | `/jobs/{job_id}/events` | Stream SSE progress |
| `POST` | `/jobs/{job_id}/approve` | Approve and queue final composition |
| `PATCH` | `/jobs/{job_id}/proposal-texts` | Update proposal texts or remove selected decisions |
| `POST` | `/jobs/{job_id}/recalculate-proposal` | Recalculate a proposal before approval |
| `GET` | `/jobs/{job_id}/download` | Download the final MP4 after completion |
| `GET` | `/llm/models` | List local Ollama models and the preferred default |
| `GET` | `/editing-options` | Read supported editing options and defaults |
| `POST` | `/uploads/start` | Start a chunked upload session |
| `PUT` | `/uploads/{upload_id}/clips/{clip_index}/chunk` | Upload a clip chunk |
| `POST` | `/uploads/{upload_id}/jobs` | Create a job from completed uploads |

See `/docs` for request and response schemas. An unreachable Ollama instance returns a clear `503` from `/llm/models`; approval outside the expected state and download before completion return `409`.

## Data and project layout

```text
autocut_ai/       FastAPI API, schemas, persistence, editorial pipeline, rendering
frontend/        Web interface served by FastAPI
Sounds/          MP3 transition effects
tests/           Unit, integration, and marked real-media tests
tests/fixtures/  1.mp4 and 2.mp4 sample videos
docs/            Development notes and plans
data/            Runtime jobs database, uploads, outputs, and temporary files
models/          Optional local model cache
```

Jobs are persisted in `data/jobs.sqlite3`. Proposal exports use `proposal.txt` and `proposal.json`; final output is normally `data/outputs/{job_id}/final.mp4`.

**Current retention behavior:** starting a new upload or job prunes other jobs' upload/output directories. Cleanup also removes intermediate render files and temporary caches, preserving the current proposal exports and completed final video. Download or copy results before starting another job. SQLite records can remain after their media files are removed.

Runtime data, downloaded models, caches, virtual environments, and private `.env` files are excluded from Git.

## Tests

Install development dependencies:

```sh
python -m pip install -e ".[dev]"
```

Run the regular suite without marked slow or end-to-end tests:

```sh
python -m pytest -m "not slow and not e2e"
```

Run the marked real-media tests, or the complete suite:

```sh
python -m pytest -m "slow and e2e"
python -m pytest
```

Both `tests/fixtures/1.mp4` and `tests/fixtures/2.mp4` are included. Real rendering tests require FFmpeg/ffprobe; most pipeline tests mock heavy services. Coverage includes narrative validation, cut safety, approval gating, model listing, upload handling, proposal review, subtitle generation, and rendering behavior.

## Current limitations

- Speech recognition and editorial prompts target Italian; the interface is currently in Italian.
- Background music controls and the `mixing_music` stage exist, but music generation, background-track mixing, and ducking are not implemented in the current renderer.
- Failed transition rendering falls back to concatenation and logs the fallback.
- Estimated duration can differ from the rendered duration because of transitions and protective boundary extensions.
- The local application has no authentication and uses a simple background runner. Its cleanup policy assumes one active workflow; it is not a concurrent multi-user service.
- Editing proposal text does not synthesize replacement speech. Final video uses the source recordings.
