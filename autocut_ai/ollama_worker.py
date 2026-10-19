from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _message_content(response: object) -> str:
    if isinstance(response, dict):
        content = response["message"]["content"]
    else:
        content = response.message.content  # type: ignore[attr-defined]
    if not isinstance(content, str):
        raise TypeError("Ollama response message.content is not a string")
    return content


def run_worker(request_path: Path, response_path: Path) -> None:
    import ollama

    payload: dict[str, Any] = json.loads(request_path.read_text(encoding="utf-8"))
    client = ollama.Client(timeout=float(payload["timeout"]))
    response = client.chat(
        model=payload["model"],
        messages=payload["messages"],
        format=payload["format"],
        think=payload.get("think"),
        options=payload["options"],
        keep_alive=payload.get("keep_alive"),
    )
    response_path.write_text(
        json.dumps({"content": _message_content(response)}, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--response-json", required=True)
    args = parser.parse_args()
    run_worker(Path(args.request_json), Path(args.response_json))


if __name__ == "__main__":
    main()
