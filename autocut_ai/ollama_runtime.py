from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
from uuid import uuid4

import httpx

from autocut_ai.config import AppSettings


class OllamaTimeoutError(TimeoutError):
    pass


class OllamaRuntimeError(RuntimeError):
    pass


def ollama_num_predict_for_attempt(settings: AppSettings, attempt: int) -> int:
    if attempt <= 0:
        return settings.ollama_num_predict
    return min(settings.ollama_num_ctx, settings.ollama_num_predict * (2**attempt))


def ollama_generation_options(settings: AppSettings, num_predict: int | None = None) -> dict[str, int | float]:
    return {
        "num_ctx": settings.ollama_num_ctx,
        "num_predict": num_predict if num_predict is not None else settings.ollama_num_predict,
        "temperature": settings.ollama_temperature,
    }


def validation_error_is_json_invalid(exc: object) -> bool:
    if not hasattr(exc, "errors"):
        return False
    errors = exc.errors()  # type: ignore[attr-defined]
    return any(isinstance(error, dict) and error.get("type") == "json_invalid" for error in errors)


def structured_chat(
    ollama_module: Any,
    settings: AppSettings,
    *,
    model: str,
    messages: Sequence[Mapping[str, str]],
    response_format: Mapping[str, Any],
    num_predict: int | None = None,
) -> Any:
    if settings.ollama_run_in_subprocess:
        return _structured_chat_subprocess(
            settings,
            model=model,
            messages=messages,
            response_format=response_format,
            num_predict=num_predict,
        )

    client = ollama_module.Client(timeout=settings.ollama_request_timeout_s)
    try:
        return client.chat(
            model=model,
            messages=messages,
            format=response_format,
            think=settings.ollama_think,
            options=ollama_generation_options(settings, num_predict),
            keep_alive=settings.ollama_keep_alive,
        )
    except httpx.TimeoutException as exc:
        timeout_s = f"{settings.ollama_request_timeout_s:g}"
        raise OllamaTimeoutError(
            f"Timeout Ollama dopo {timeout_s}s: la richiesta LLM e troppo lunga o il modello non sta completando."
        ) from exc


def _structured_chat_subprocess(
    settings: AppSettings,
    *,
    model: str,
    messages: Sequence[Mapping[str, str]],
    response_format: Mapping[str, Any],
    num_predict: int | None,
) -> dict[str, dict[str, str]]:
    settings.ensure_directories()
    request_dir = settings.data_dir / "tmp" / "ollama"
    request_dir.mkdir(parents=True, exist_ok=True)
    request_id = uuid4().hex
    request_path = request_dir / f"{request_id}.request.json"
    response_path = request_dir / f"{request_id}.response.json"
    request_payload = {
        "timeout": settings.ollama_request_timeout_s,
        "model": model,
        "messages": list(messages),
        "format": dict(response_format),
        "think": settings.ollama_think,
        "options": ollama_generation_options(settings, num_predict),
        "keep_alive": settings.ollama_keep_alive,
    }
    request_path.write_text(json.dumps(request_payload, ensure_ascii=True), encoding="utf-8")
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "autocut_ai.ollama_worker",
                "--request-json",
                str(request_path),
                "--response-json",
                str(response_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=settings.ollama_request_timeout_s,
        )
        response_payload = json.loads(response_path.read_text(encoding="utf-8"))
        content = response_payload["content"]
        if not isinstance(content, str):
            raise OllamaRuntimeError("Ollama worker response content non testuale")
        return {"message": {"content": content}}
    except subprocess.TimeoutExpired as exc:
        timeout_s = f"{settings.ollama_request_timeout_s:g}"
        raise OllamaTimeoutError(
            f"Timeout Ollama dopo {timeout_s}s: la richiesta LLM e troppo lunga o il modello non sta completando."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise OllamaRuntimeError(f"Ollama worker fallito: {stderr or exc.returncode}") from exc
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise OllamaRuntimeError(f"Ollama worker response non valida: {exc}") from exc
    finally:
        _unlink_if_exists(request_path)
        _unlink_if_exists(response_path)


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
