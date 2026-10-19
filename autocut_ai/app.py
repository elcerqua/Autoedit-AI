from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import BinaryIO, Protocol, cast
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import ValidationError
from python_multipart.exceptions import MultipartParseError
from python_multipart.multipart import MultipartParser, parse_options_header

from autocut_ai.config import AppSettings
from autocut_ai.editing_options import EditingOptionsAssetError, editing_options_catalog, validate_editing_options_assets
from autocut_ai.events import EventPublisher
from autocut_ai.ffmpeg import FFmpegRenderer, MockRenderer
from autocut_ai.housekeeping import cleanup_generated_data, delete_job_generated_data
from autocut_ai.llm_models import MockModelProvider, OllamaModelProvider, OllamaUnavailableError
from autocut_ai.narrative import MockNarrativePlanner, NarrativePlanner
from autocut_ai.pipeline import (
    InvalidApprovalStateError,
    InvalidProposalRecalculateStateError,
    InvalidProposalTextPatchStateError,
    PipelineRunner,
    ProposalRecalculateError,
    ProposalTextPatchError,
)
from autocut_ai.proposal_review import MockProposalReviewer, ProposalReviewer
from autocut_ai.schemas import (
    ApprovalRequest,
    EditingOptions,
    JobCreateRequest,
    JobCreateResponse,
    JobRecord,
    LlmModelsResponse,
    PipelineStage,
    ProposalRecalculateRequest,
    ProposalRecalculateResponse,
    ProposalTextPatchRequest,
    ProposalTextPatchResponse,
    UploadChunkResponse,
    UploadStartResponse,
)
from autocut_ai.storage import JobNotFoundError, JobRepository
from autocut_ai.transcription import MockTranscriptionService, TranscriptionService
from autocut_ai.upload_sessions import (
    UploadSessionError,
    completed_upload_paths,
    create_upload_session,
    write_upload_chunk,
)


class UploadReader(Protocol):
    filename: str | None

    async def read(self, size: int = -1) -> bytes:
        pass

    async def close(self) -> None:
        pass


async def _write_upload_to_path(upload: UploadReader, destination: Path, chunk_size: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("wb") as output_file:
            while True:
                chunk = await upload.read(chunk_size)
                if not chunk:
                    break
                output_file.write(chunk)
    finally:
        await upload.close()


class StreamingRequest(Protocol):
    headers: object

    def stream(self) -> AsyncIterator[bytes]:
        pass


class MultipartUploadState:
    """Accumula i campi multipart senza caricare le clip intere in memoria."""

    def __init__(self, settings: AppSettings, job_id: str) -> None:
        self.settings = settings
        self.uploads_dir = settings.uploads_dir / job_id
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.fields: dict[str, str] = {}
        self.clip_paths: list[str] = []
        self.current_headers: dict[bytes, bytes] = {}
        self.current_header_name = bytearray()
        self.current_header_value = bytearray()
        self.current_field_name: str | None = None
        self.current_file: BinaryIO | None = None
        self.current_text = bytearray()

    def on_part_begin(self) -> None:
        self.current_headers = {}
        self.current_header_name = bytearray()
        self.current_header_value = bytearray()
        self.current_field_name = None
        self.current_text = bytearray()
        self._close_current_file()

    def on_header_field(self, data: bytes, start: int, end: int) -> None:
        self.current_header_name.extend(data[start:end])

    def on_header_value(self, data: bytes, start: int, end: int) -> None:
        self.current_header_value.extend(data[start:end])

    def on_header_end(self) -> None:
        header_name = bytes(self.current_header_name).lower()
        self.current_headers[header_name] = bytes(self.current_header_value)
        self.current_header_name = bytearray()
        self.current_header_value = bytearray()

    def on_headers_finished(self) -> None:
        disposition = self.current_headers.get(b"content-disposition", b"")
        _disposition_type, options = parse_options_header(disposition)
        field_name = _decode_header_option(options.get(b"name"))
        filename = _decode_header_option(options.get(b"filename"))
        self.current_field_name = field_name
        if field_name == "clips" and filename is not None:
            destination = self._unique_destination(Path(filename).name)
            self.current_file = destination.open("wb")
            self.clip_paths.append(str(destination))

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        chunk = data[start:end]
        if self.current_file is not None:
            self.current_file.write(chunk)
        elif self.current_field_name is not None:
            self.current_text.extend(chunk)

    def on_part_end(self) -> None:
        self._close_current_file()
        if self.current_field_name is not None and self.current_file is None and self.current_text:
            self.fields[self.current_field_name] = self.current_text.decode("utf-8")
        self.current_field_name = None
        self.current_text = bytearray()

    def close(self) -> None:
        self._close_current_file()

    def _close_current_file(self) -> None:
        if self.current_file is not None:
            self.current_file.close()
            self.current_file = None

    def _unique_destination(self, filename: str) -> Path:
        safe_name = filename or f"clip_{len(self.clip_paths) + 1}.mp4"
        destination = self.uploads_dir / safe_name
        if not destination.exists():
            return destination
        stem = destination.stem
        suffix = destination.suffix
        counter = 2
        while True:
            candidate = self.uploads_dir / f"{stem}_{counter}{suffix}"
            if not candidate.exists():
                return candidate
            counter += 1


def _decode_header_option(value: bytes | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _multipart_boundary(content_type: str) -> bytes:
    media_type, options = parse_options_header(content_type.encode("latin-1"))
    if media_type != b"multipart/form-data":
        raise ValueError("Content-Type multipart non valido")
    boundary = options.get(b"boundary")
    if boundary is None:
        raise ValueError("Boundary multipart mancante")
    return boundary if isinstance(boundary, bytes) else boundary.encode("latin-1")


async def _parse_multipart_streaming(request: StreamingRequest, job_id: str, settings: AppSettings) -> dict[str, object]:
    content_type = cast(dict[str, str], request.headers).get("content-type", "")
    state = MultipartUploadState(settings, job_id)
    parser = MultipartParser(
        _multipart_boundary(content_type),
        {
            "on_part_begin": state.on_part_begin,
            "on_header_field": state.on_header_field,
            "on_header_value": state.on_header_value,
            "on_header_end": state.on_header_end,
            "on_headers_finished": state.on_headers_finished,
            "on_part_data": state.on_part_data,
            "on_part_end": state.on_part_end,
        },
    )
    try:
        async for chunk in request.stream():
            if chunk:
                parser.write(chunk)
        parser.finalize()
    finally:
        state.close()
    return {
        "brief": state.fields.get("brief"),
        "editing_options": state.fields.get("editing_options"),
        "llm_model": state.fields.get("llm_model"),
        "clip_paths": state.clip_paths,
    }


async def _save_uploads(request: Request, job_id: str, settings: AppSettings) -> JobCreateRequest:
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("multipart/form-data"):
        payload = await request.json()
        return JobCreateRequest.model_validate(payload)

    stream_payload = await _parse_multipart_streaming(request, job_id, settings)
    brief_raw = stream_payload.get("brief")
    options_raw = stream_payload.get("editing_options")
    llm_model_raw = stream_payload.get("llm_model")
    clip_paths_raw = stream_payload.get("clip_paths", [])
    payload: dict[str, object] = {}
    if isinstance(brief_raw, str) and brief_raw:
        payload["brief"] = brief_raw
    if isinstance(options_raw, str) and options_raw:
        payload["editing_options"] = options_raw
    if isinstance(llm_model_raw, str) and llm_model_raw:
        payload["llm_model"] = llm_model_raw

    clip_paths = clip_paths_raw if isinstance(clip_paths_raw, list) else []
    parsed: dict[str, object] = {"clip_paths": clip_paths}
    if "brief" in payload:
        parsed["brief"] = JobCreateRequest.model_fields["brief"].annotation.model_validate_json(payload["brief"])
    if "editing_options" in payload:
        parsed["editing_options"] = EditingOptions.model_validate_json(payload["editing_options"])
    if "llm_model" in payload:
        parsed["llm_model"] = payload["llm_model"]
    return JobCreateRequest.model_validate(parsed)


def create_app(settings: AppSettings | None = None) -> FastAPI:
    settings = settings or AppSettings()
    settings.ensure_directories()
    repository = JobRepository(settings.database_path)
    publisher = EventPublisher(repository)
    if settings.use_mock_services:
        model_provider = MockModelProvider(settings)
        transcriber = MockTranscriptionService()
        narrative_planner = MockNarrativePlanner()
        proposal_reviewer = MockProposalReviewer()
        renderer = MockRenderer(settings)
    else:
        model_provider = OllamaModelProvider(settings)
        transcriber = TranscriptionService(settings)
        narrative_planner = NarrativePlanner(settings)
        proposal_reviewer = ProposalReviewer(settings)
        renderer = FFmpegRenderer(settings)
    runner = PipelineRunner(repository, publisher, transcriber, narrative_planner, proposal_reviewer, renderer, settings=settings)

    app = FastAPI(title="Auto Cut AI", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.settings = settings
    app.state.repository = repository
    app.state.publisher = publisher
    app.state.model_provider = model_provider
    app.state.runner = runner
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    static_dir = frontend_dir / "static"

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    def persist_and_start_job(
        job_id: str,
        create_request: JobCreateRequest,
        background_tasks: BackgroundTasks,
    ) -> JobCreateResponse:
        # La persistenza precede sempre l'avvio: in questo modo refresh e SSE
        # osservano un job valido anche se la pipeline fallisce immediatamente.
        job = JobRecord(
            job_id=job_id,
            brief=create_request.brief,
            editing_options=create_request.editing_options,
            llm_model=create_request.llm_model,
            input_clip_paths=create_request.clip_paths,
        )
        repository.save(job)
        publisher.publish_job(job)
        cleanup_generated_data(settings, job_id)
        if settings.use_mock_services or settings.run_pipeline_inline:
            job = runner.run_phase_one(job_id)
        else:
            background_tasks.add_task(runner.run_phase_one, job_id)
            job = repository.get(job_id)
        return JobCreateResponse(job_id=job_id, status=job.to_status(), job=job)

    @app.get("/", include_in_schema=False)
    async def serve_frontend() -> FileResponse:
        index_path = frontend_dir / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="Frontend non trovato")
        return FileResponse(index_path, media_type="text/html")

    @app.get("/editing-options")
    async def get_editing_options() -> object:
        return editing_options_catalog(settings)

    @app.get("/llm/models", response_model=LlmModelsResponse)
    async def get_llm_models() -> LlmModelsResponse:
        try:
            models = app.state.model_provider.list_models()
        except OllamaUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return LlmModelsResponse(
            models=models,
            default_model=app.state.model_provider.default_model(models),
        )

    @app.get("/sounds/{filename}", include_in_schema=False)
    async def get_transition_sound(filename: str) -> FileResponse:
        sound_path = settings.sounds_dir / filename
        if Path(filename).name != filename or sound_path.suffix.lower() != ".mp3":
            raise HTTPException(status_code=404, detail="Effetto sonoro non trovato")
        if not sound_path.exists() or not sound_path.is_file():
            raise HTTPException(status_code=404, detail="Effetto sonoro non trovato")
        return FileResponse(sound_path, media_type="audio/mpeg", filename=filename)

    @app.post("/uploads/start", response_model=UploadStartResponse)
    async def start_upload() -> UploadStartResponse:
        upload_id = uuid4().hex
        create_upload_session(settings, upload_id)
        cleanup_generated_data(settings, upload_id)
        return UploadStartResponse(upload_id=upload_id)

    @app.put("/uploads/{upload_id}/clips/{clip_index}/chunk", response_model=UploadChunkResponse)
    async def upload_clip_chunk(
        upload_id: str,
        clip_index: int,
        request: Request,
        x_file_name: str = Header(alias="X-File-Name"),
        x_chunk_offset: int = Header(alias="X-Chunk-Offset"),
        x_file_size: int = Header(alias="X-File-Size"),
    ) -> UploadChunkResponse:
        try:
            return await write_upload_chunk(
                settings=settings,
                upload_id=upload_id,
                clip_index=clip_index,
                file_name=x_file_name,
                chunk_offset=x_chunk_offset,
                file_size=x_file_size,
                stream=request.stream(),
            )
        except UploadSessionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=500, detail="Errore durante il salvataggio del chunk") from exc

    @app.post("/uploads/{upload_id}/jobs", response_model=JobCreateResponse)
    async def create_job_from_upload(
        upload_id: str,
        create_request: JobCreateRequest,
        background_tasks: BackgroundTasks,
    ) -> JobCreateResponse:
        try:
            clip_paths = completed_upload_paths(settings, upload_id)
            create_request = create_request.model_copy(update={"clip_paths": clip_paths})
            validate_editing_options_assets(create_request.editing_options, settings)
        except UploadSessionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValidationError as exc:
            delete_job_generated_data(settings, upload_id)
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        except EditingOptionsAssetError as exc:
            delete_job_generated_data(settings, upload_id)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return persist_and_start_job(upload_id, create_request, background_tasks)

    @app.post("/jobs", response_model=JobCreateResponse)
    async def create_job(request: Request, background_tasks: BackgroundTasks) -> JobCreateResponse:
        job_id = uuid4().hex
        try:
            create_request = await _save_uploads(request, job_id, settings)
            validate_editing_options_assets(create_request.editing_options, settings)
        except ValidationError as exc:
            delete_job_generated_data(settings, job_id)
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        except EditingOptionsAssetError as exc:
            delete_job_generated_data(settings, job_id)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (MultipartParseError, UnicodeDecodeError, ValueError) as exc:
            delete_job_generated_data(settings, job_id)
            raise HTTPException(status_code=400, detail="Upload multipart non valido") from exc
        except OSError as exc:
            delete_job_generated_data(settings, job_id)
            raise HTTPException(status_code=500, detail="Errore durante il salvataggio degli upload") from exc
        return persist_and_start_job(job_id, create_request, background_tasks)

    @app.get("/jobs/{job_id}", response_model=JobRecord)
    async def get_job(job_id: str) -> JobRecord:
        try:
            return repository.get(job_id)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Job non trovato") from exc

    @app.get("/jobs/{job_id}/events")
    async def get_job_events(job_id: str) -> StreamingResponse:
        try:
            repository.get(job_id)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Job non trovato") from exc

        async def event_stream() -> object:
            last_sequence = 0
            terminal = {PipelineStage.WAITING_FOR_APPROVAL, PipelineStage.DONE, PipelineStage.FAILED}
            while True:
                for event in repository.list_events_after(job_id, last_sequence):
                    last_sequence = event.sequence
                    yield f"id: {event.sequence}\nevent: job\ndata: {event.model_dump_json()}\n\n"
                job = repository.get(job_id)
                if job.stage in terminal:
                    # Lo stream termina nelle tre soste contrattuali. Dopo il
                    # Via libera il client apre una nuova connessione SSE.
                    yield f"event: status\ndata: {job.to_status().model_dump_json()}\n\n"
                    break
                await asyncio.sleep(1)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/jobs/{job_id}/approve")
    async def approve_job(job_id: str, approval: ApprovalRequest, background_tasks: BackgroundTasks) -> dict[str, object]:
        if not approval.approved:
            raise HTTPException(status_code=400, detail="ApprovalRequest.approved deve essere true per il Via libera")
        try:
            job = runner.mark_approved(job_id, approval.proposal_variant_id)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Job non trovato") from exc
        except InvalidApprovalStateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        # mark_approved applica il gate di stato; solo dopo questo controllo la
        # composizione viene accodata come task separato dalla richiesta HTTP.
        background_tasks.add_task(runner.run_phase_two, job_id)
        return {"status": job.to_status(), "job_id": job_id}

    @app.patch("/jobs/{job_id}/proposal-texts", response_model=ProposalTextPatchResponse)
    async def update_job_proposal_texts(
        job_id: str,
        request: ProposalTextPatchRequest,
    ) -> ProposalTextPatchResponse:
        try:
            job = runner.update_proposal_texts(job_id, request)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Job non trovato") from exc
        except InvalidProposalTextPatchStateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ProposalTextPatchError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return ProposalTextPatchResponse(job_id=job_id, status=job.to_status(), job=job)

    @app.post("/jobs/{job_id}/recalculate-proposal", response_model=ProposalRecalculateResponse)
    async def recalculate_job_proposal(
        job_id: str,
        request: ProposalRecalculateRequest,
    ) -> ProposalRecalculateResponse:
        try:
            job = runner.recalculate_proposal(job_id, request)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Job non trovato") from exc
        except InvalidProposalRecalculateStateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ProposalRecalculateError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return ProposalRecalculateResponse(job_id=job_id, status=job.to_status(), job=job)

    @app.get("/jobs/{job_id}/download")
    async def download_job(job_id: str) -> FileResponse:
        try:
            job = repository.get(job_id)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Job non trovato") from exc
        if job.stage != PipelineStage.DONE or job.final_output_path is None:
            raise HTTPException(status_code=409, detail="Video finale non ancora pronto")
        path = Path(job.final_output_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail="File finale non trovato")
        return FileResponse(path, media_type="video/mp4", filename=f"auto-cut-ai-{job_id}.mp4")

    return app


app = create_app()
