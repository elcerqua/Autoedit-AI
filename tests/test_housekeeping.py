from __future__ import annotations

from pathlib import Path

from autocut_ai.config import AppSettings
from autocut_ai.housekeeping import cleanup_generated_data


def test_cleanup_generated_data_keeps_only_current_job_and_final_output(tmp_path: Path) -> None:
    settings = AppSettings(data_dir=tmp_path / "data", database_path=tmp_path / "data" / "jobs.sqlite3")
    settings.ensure_directories()
    old_job_id = "old-job"
    current_job_id = "current-job"
    old_upload = settings.uploads_dir / old_job_id
    current_upload = settings.uploads_dir / current_job_id
    old_output = settings.outputs_dir / old_job_id
    current_output = settings.outputs_dir / current_job_id
    for directory in [old_upload, current_upload, old_output, current_output, settings.data_dir / "cache", settings.data_dir / "tmp"]:
        directory.mkdir(parents=True)
    (old_upload / "clip.mp4").write_bytes(b"old upload")
    (current_upload / "clip.mp4").write_bytes(b"current upload")
    (old_output / "final.mp4").write_bytes(b"old output")
    final_output = current_output / "final.mp4"
    final_output.write_bytes(b"current final")
    proposal_txt = current_output / "proposal.txt"
    proposal_json = current_output / "proposal.json"
    proposal_txt.write_text("Proposta editoriale", encoding="utf-8")
    proposal_json.write_text("{}", encoding="utf-8")
    (current_output / "segment_0000.mp4").write_bytes(b"temporary segment")
    (current_output / "concat.txt").write_text("file 'segment_0000.mp4'", encoding="utf-8")
    (settings.data_dir / "cache" / "transient.bin").write_bytes(b"cache")
    (settings.data_dir / "tmp" / "transient.bin").write_bytes(b"tmp")

    cleanup_generated_data(settings, current_job_id, final_output)

    assert not old_upload.exists()
    assert not old_output.exists()
    assert current_upload.exists()
    assert final_output.exists()
    assert proposal_txt.exists()
    assert proposal_json.exists()
    assert not (current_output / "segment_0000.mp4").exists()
    assert not (current_output / "concat.txt").exists()
    assert not (settings.data_dir / "cache").exists()
    assert not (settings.data_dir / "tmp").exists()


def test_cleanup_generated_data_refuses_to_delete_outside_managed_dirs(tmp_path: Path) -> None:
    settings = AppSettings(data_dir=tmp_path / "data", database_path=tmp_path / "data" / "jobs.sqlite3")
    settings.ensure_directories()
    outside = tmp_path / "outside.txt"
    outside.write_text("do not touch", encoding="utf-8")
    current_output = settings.outputs_dir / "current-job"
    current_output.mkdir(parents=True)
    (current_output / "temp.txt").write_text("temporary", encoding="utf-8")

    cleanup_generated_data(settings, "current-job", outside)

    assert outside.exists()
    assert not (current_output / "temp.txt").exists()
