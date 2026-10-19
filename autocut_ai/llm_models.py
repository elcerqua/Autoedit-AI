from __future__ import annotations

import httpx

from autocut_ai.config import AppSettings


class OllamaUnavailableError(RuntimeError):
    pass


class OllamaModelProvider:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def list_models(self) -> list[str]:
        try:
            import ollama

            response = ollama.list()
        except (httpx.HTTPError, OSError, ConnectionError) as exc:
            raise OllamaUnavailableError("Ollama non raggiungibile") from exc

        raw_models = getattr(response, "models", None)
        if raw_models is None and isinstance(response, dict):
            raw_models = response.get("models", [])
        names: list[str] = []
        for item in raw_models or []:
            name = getattr(item, "model", None) or getattr(item, "name", None)
            if name is None and isinstance(item, dict):
                name = item.get("model") or item.get("name")
            if isinstance(name, str) and name:
                names.append(name)
        return names

    def default_model(self, models: list[str]) -> str | None:
        if self.settings.default_ollama_model in models:
            return self.settings.default_ollama_model
        return models[0] if models else None


class MockModelProvider:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._models = [settings.default_ollama_model, "llama3:latest"]
        self._error: str | None = None

    def set_models(self, models: list[str]) -> None:
        self._models = models
        self._error = None

    def fail_with(self, message: str) -> None:
        self._error = message

    def list_models(self) -> list[str]:
        if self._error is not None:
            raise OllamaUnavailableError(self._error)
        return self._models

    def default_model(self, models: list[str]) -> str | None:
        if self.settings.default_ollama_model in models:
            return self.settings.default_ollama_model
        return models[0] if models else None
