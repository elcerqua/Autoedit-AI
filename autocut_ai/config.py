from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class AppSettings(BaseModel):
    data_dir: Path = Field(default=Path("data"))
    database_path: Path = Field(default=Path("data") / "jobs.sqlite3")
    uploads_dir_name: str = "uploads"
    outputs_dir_name: str = "outputs"
    whisper_model_path: Path | None = Field(
        default=Path("models")
        / "hub"
        / "models--Systran--faster-whisper-large-v3"
        / "snapshots"
        / "edaa852ec7e145841d8ffdb056a99866b5f0a478"
    )
    default_ollama_model: str = "qwen3:8b"
    ffmpeg_path: str = "ffmpeg"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_run_in_subprocess: bool = True
    whisper_beam_size: int = Field(default=1, ge=1, le=8)
    whisper_vad_filter: bool = True
    use_mock_services: bool = False
    run_pipeline_inline: bool = False
    sounds_dir: Path = Field(default=Path("Sounds"))
    upload_chunk_size_bytes: int = Field(default=1024 * 1024, ge=64 * 1024, le=16 * 1024 * 1024)
    ollama_request_timeout_s: float = Field(default=240.0, gt=0)
    ollama_num_ctx: int = Field(default=49152, ge=2048)
    ollama_num_predict: int = Field(default=32768, ge=256)
    ollama_temperature: float = Field(default=0.1, ge=0, le=2)
    ollama_think: bool | Literal["low", "medium", "high"] | None = False
    ollama_keep_alive: str = "5m"
    ollama_run_in_subprocess: bool = True
    proposal_review_timeout_s: float = Field(default=45.0, gt=0)
    narrative_hierarchical_threshold_segments: int = Field(default=24, ge=1)
    narrative_chunk_max_segments: int = Field(default=20, ge=1)

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / self.uploads_dir_name

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / self.outputs_dir_name

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.sounds_dir.mkdir(parents=True, exist_ok=True)
