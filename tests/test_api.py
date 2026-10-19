from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from autocut_ai.app import create_app
from autocut_ai.config import AppSettings
from autocut_ai.schemas import (
    EditDecision,
    EditDecisionList,
    JobRecord,
    NarrativeClip,
    NarrativeMap,
    NarrativeSegment,
    PipelineStage,
    ProposalVariant,
    SegmentImportance,
)


def test_editing_options_endpoint_returns_supported_transitions(client: TestClient) -> None:
    response = client.get("/editing-options")

    assert response.status_code == 200
    body = response.json()
    assert "cut" in body["available_transitions"]
    assert "cross_dissolve" in body["available_transitions"]
    assert "zoomin" in body["available_transitions"]
    assert "wipeleft" in body["available_transitions"]
    assert "fadeblack" in body["available_transitions"]
    assert body["defaults"]["default_transition"] == "cut"
    assert body["defaults"]["enable_creative_zoom"] is False
    assert body["defaults"]["subtitle_options"]["enabled"] is False
    assert body["defaults"]["subtitle_options"]["position"] == "bottom"
    assert {preset["id"] for preset in body["subtitle_style_presets"]} == {
        "classic_white",
        "creator_yellow",
        "boxed_black",
        "minimal_shadow",
    }
    assert all(preset["preview_data_uri"].startswith("data:image/svg+xml") for preset in body["subtitle_style_presets"])


def test_editing_options_endpoint_returns_transition_sounds(settings: AppSettings) -> None:
    settings.sounds_dir.mkdir(parents=True, exist_ok=True)
    (settings.sounds_dir / "whoosh.mp3").write_bytes(b"mp3")
    (settings.sounds_dir / "ignore.wav").write_bytes(b"wav")
    local_client = TestClient(create_app(settings=settings))

    response = local_client.get("/editing-options")

    assert response.status_code == 200
    body = response.json()
    assert body["available_transition_sounds"] == ["whoosh.mp3"]
    assert body["defaults"]["add_transition_sounds"] is False
    assert body["defaults"]["transition_sound"] is None


def test_llm_models_endpoint_returns_mocked_models(client: TestClient) -> None:
    app = client.app
    app.state.model_provider.set_models(["llama3:latest", "qwen3:8b"])

    response = client.get("/llm/models")

    assert response.status_code == 200
    assert response.json()["default_model"] == "qwen3:8b"


def test_llm_models_endpoint_handles_ollama_error(client: TestClient) -> None:
    app = client.app
    app.state.model_provider.fail_with("Ollama non raggiungibile")

    response = client.get("/llm/models")

    assert response.status_code == 503
    assert "Ollama" in response.json()["detail"]


def test_job_accepts_valid_editing_options_and_rejects_invalid(client: TestClient) -> None:
    valid = {
        "brief": {"user_prompt": "Taglia le ripetizioni", "synthesis_level": 2},
        "llm_model": "qwen3:8b",
        "editing_options": {"default_transition": "cross_dissolve", "transition_duration_ms": 250},
    }
    response = client.post("/jobs", json=valid)
    assert response.status_code == 200
    assert response.json()["status"]["stage"] == "waiting_for_approval"

    native_transition = {
        "brief": {"user_prompt": "Transizione nativa", "synthesis_level": 2},
        "editing_options": {"default_transition": "zoomin"},
    }
    response = client.post("/jobs", json=native_transition)
    assert response.status_code == 200

    creative_zoom = {
        "brief": {"user_prompt": "Enfatizza i passaggi chiave", "synthesis_level": 2},
        "editing_options": {"enable_creative_zoom": True},
    }
    response = client.post("/jobs", json=creative_zoom)
    assert response.status_code == 200
    body = response.json()
    assert body["job"]["editing_options"]["enable_creative_zoom"] is True

    invalid = {
        "brief": {"user_prompt": "Taglia", "synthesis_level": 2},
        "editing_options": {"default_transition": "spiral"},
    }
    response = client.post("/jobs", json=invalid)
    assert response.status_code == 422

    subtitle_valid = {
        "brief": {"user_prompt": "Sottotitoli", "synthesis_level": 2},
        "editing_options": {
            "subtitle_options": {
                "enabled": True,
                "position": "center",
                "style_preset": "creator_yellow",
            }
        },
    }
    response = client.post("/jobs", json=subtitle_valid)
    assert response.status_code == 200
    assert response.json()["job"]["editing_options"]["subtitle_options"]["enabled"] is True
    assert response.json()["job"]["editing_options"]["subtitle_options"]["position"] == "center"

    subtitle_invalid = {
        "brief": {"user_prompt": "Sottotitoli", "synthesis_level": 2},
        "editing_options": {"subtitle_options": {"style_preset": "neon"}},
    }
    response = client.post("/jobs", json=subtitle_invalid)
    assert response.status_code == 422


def test_job_accepts_available_transition_sound_and_rejects_unknown(settings: AppSettings) -> None:
    settings.sounds_dir.mkdir(parents=True, exist_ok=True)
    (settings.sounds_dir / "whoosh.mp3").write_bytes(b"mp3")
    local_client = TestClient(create_app(settings=settings))
    valid = {
        "brief": {"user_prompt": "Aggiungi suoni agli stacchi", "synthesis_level": 2},
        "editing_options": {"add_transition_sounds": True, "transition_sound": "whoosh.mp3"},
    }

    response = local_client.post("/jobs", json=valid)

    assert response.status_code == 200
    assert response.json()["job"]["editing_options"]["add_transition_sounds"] is True
    assert response.json()["job"]["editing_options"]["transition_sound"] == "whoosh.mp3"

    invalid = {
        "brief": {"user_prompt": "Aggiungi suoni agli stacchi", "synthesis_level": 2},
        "editing_options": {"add_transition_sounds": True, "transition_sound": "missing.mp3"},
    }
    response = local_client.post("/jobs", json=invalid)

    assert response.status_code == 422
    assert "transition_sound" in str(response.json()["detail"])


def test_transition_sound_preview_serves_only_available_mp3(settings: AppSettings) -> None:
    settings.sounds_dir.mkdir(parents=True, exist_ok=True)
    (settings.sounds_dir / "preview.mp3").write_bytes(b"mp3 bytes")
    (settings.sounds_dir / "ignore.wav").write_bytes(b"wav bytes")
    local_client = TestClient(create_app(settings=settings))

    response = local_client.get("/sounds/preview.mp3")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/mpeg")
    assert response.content == b"mp3 bytes"

    missing_response = local_client.get("/sounds/missing.mp3")
    assert missing_response.status_code == 404

    invalid_response = local_client.get("/sounds/ignore.wav")
    assert invalid_response.status_code == 404


def test_job_accepts_multi_clip_multipart_upload(client: TestClient, fixture_paths: tuple[Path, Path]) -> None:
    first, second = fixture_paths
    with first.open("rb") as first_file, second.open("rb") as second_file:
        response = client.post(
            "/jobs",
            data={
                "brief": json.dumps({"user_prompt": "Unisci le clip", "synthesis_level": 2}),
                "editing_options": json.dumps({"default_transition": "cut"}),
                "llm_model": "qwen3:8b",
            },
            files=[
                ("clips", ("1.mp4", first_file, "video/mp4")),
                ("clips", ("2.mp4", second_file, "video/mp4")),
            ],
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"]["stage"] == "waiting_for_approval"
    assert len(body["job"]["input_clip_paths"]) == 2


def test_generated_media_dirs_keep_only_latest_job(client: TestClient, settings: AppSettings) -> None:
    first_response = client.post(
        "/jobs",
        data={"brief": json.dumps({"user_prompt": "Primo lavoro", "synthesis_level": 1})},
        files=[("clips", ("first.mp4", b"first", "video/mp4"))],
    )
    first_job_id = first_response.json()["job_id"]
    first_approve = client.post(f"/jobs/{first_job_id}/approve", json={"approved": True})
    assert first_approve.status_code == 200

    second_response = client.post(
        "/jobs",
        data={"brief": json.dumps({"user_prompt": "Secondo lavoro", "synthesis_level": 1})},
        files=[("clips", ("second.mp4", b"second", "video/mp4"))],
    )
    second_job_id = second_response.json()["job_id"]
    second_approve = client.post(f"/jobs/{second_job_id}/approve", json={"approved": True})
    assert second_approve.status_code == 200

    upload_dirs = sorted(path.name for path in settings.uploads_dir.iterdir() if path.is_dir())
    output_dirs = sorted(path.name for path in settings.outputs_dir.iterdir() if path.is_dir())

    assert upload_dirs == [second_job_id]
    assert output_dirs == [second_job_id]
    assert (settings.outputs_dir / second_job_id / "final.mp4").exists()


def test_approve_endpoint_uses_selected_proposal_variant(client: TestClient) -> None:
    repository = client.app.state.repository
    base_map = NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=1,
                playback_position=0,
                topic_summary="Test",
                role="development",
                segments=[
                    NarrativeSegment(
                        segment_id=1,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason="Proposta iniziale.",
                    )
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Iniziale",
        estimated_duration_s=10,
    )
    review_map = base_map.model_copy(update={"editorial_strategy": "Review"})
    initial_edl = EditDecisionList(
        estimated_duration_s=10,
        decisions=[
            EditDecision(
                clip_id=1,
                source_start=0,
                source_end=10,
                output_order=0,
                reason="Initial",
                included_text="Versione iniziale completa.",
            )
        ],
    )
    review_edl = EditDecisionList(
        estimated_duration_s=5,
        decisions=[
            EditDecision(
                clip_id=1,
                source_start=0,
                source_end=5,
                output_order=0,
                reason="Review",
                included_text="Versione review asciutta.",
            )
        ],
    )
    job = JobRecord(
        job_id="variant-approve-api",
        stage=PipelineStage.WAITING_FOR_APPROVAL,
        input_clip_paths=["clip.mp4"],
        narrative_map=base_map,
        edit_decision_list=initial_edl,
        proposal_variants=[
            ProposalVariant(
                id="initial",
                label="Proposta iniziale",
                narrative_map=base_map,
                edit_decision_list=initial_edl,
                duration_s=10,
                selectable=True,
            ),
            ProposalVariant(
                id="review",
                label="Review LLM",
                narrative_map=review_map,
                edit_decision_list=review_edl,
                duration_s=5,
                selectable=True,
            ),
        ],
    )
    repository.save(job)

    response = client.post(f"/jobs/{job.job_id}/approve", json={"approved": True, "proposal_variant_id": "review"})

    assert response.status_code == 200
    saved = repository.get(job.job_id)
    assert saved.selected_proposal_variant_id == "review"
    assert saved.edit_decision_list is not None
    assert saved.edit_decision_list.estimated_duration_s == 5


def test_proposal_text_patch_updates_variant_and_approval_uses_corrected_text(client: TestClient) -> None:
    repository = client.app.state.repository
    base_map = NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=1,
                playback_position=0,
                topic_summary="Test",
                role="development",
                segments=[
                    NarrativeSegment(
                        segment_id=1,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason="Proposta.",
                    )
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Review",
        estimated_duration_s=5,
    )
    review_edl = EditDecisionList(
        estimated_duration_s=5,
        decisions=[
            EditDecision(
                clip_id=1,
                source_start=0,
                source_end=5,
                output_order=0,
                reason="Review",
                included_text="Testo con refusoo.",
            )
        ],
    )
    job = JobRecord(
        job_id="proposal-text-patch-api",
        stage=PipelineStage.WAITING_FOR_APPROVAL,
        input_clip_paths=["clip.mp4"],
        narrative_map=base_map,
        edit_decision_list=review_edl,
        proposal_variants=[
            ProposalVariant(
                id="review",
                label="Review LLM",
                narrative_map=base_map,
                edit_decision_list=review_edl,
                duration_s=5,
                selectable=True,
            )
        ],
        selected_proposal_variant_id="review",
    )
    repository.save(job)

    response = client.patch(
        f"/jobs/{job.job_id}/proposal-texts",
        json={
            "proposal_variant_id": "review",
            "edits": [{"output_order": 0, "included_text": "Testo corretto."}],
        },
    )

    assert response.status_code == 200
    patched = response.json()["job"]
    assert patched["selected_proposal_variant_id"] == "review"
    assert patched["proposal_variants"][0]["edit_decision_list"]["decisions"][0]["included_text"] == "Testo corretto."

    approve_response = client.post(
        f"/jobs/{job.job_id}/approve",
        json={"approved": True, "proposal_variant_id": "review"},
    )

    assert approve_response.status_code == 200
    saved = repository.get(job.job_id)
    assert saved.edit_decision_list is not None
    assert saved.edit_decision_list.decisions[0].included_text == "Testo corretto."


def test_proposal_text_patch_deletes_segment_and_approval_uses_shortened_edl(client: TestClient) -> None:
    repository = client.app.state.repository
    base_map = NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=1,
                playback_position=0,
                topic_summary="Test",
                role="development",
                segments=[
                    NarrativeSegment(
                        segment_id=1,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason="Proposta.",
                    )
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Review",
        estimated_duration_s=9,
    )
    review_edl = EditDecisionList(
        estimated_duration_s=9,
        decisions=[
            EditDecision(
                clip_id=1,
                source_start=0,
                source_end=4,
                output_order=0,
                reason="Intro",
                included_text="Introduzione.",
            ),
            EditDecision(
                clip_id=1,
                source_start=4,
                source_end=7,
                output_order=1,
                reason="Ridondante",
                included_text="Concetto ripetuto.",
            ),
            EditDecision(
                clip_id=1,
                source_start=7,
                source_end=9,
                output_order=2,
                reason="Finale",
                included_text="Chiusura.",
            ),
        ],
    )
    job = JobRecord(
        job_id="proposal-text-delete-api",
        stage=PipelineStage.WAITING_FOR_APPROVAL,
        input_clip_paths=["clip.mp4"],
        narrative_map=base_map,
        edit_decision_list=review_edl,
        proposal_variants=[
            ProposalVariant(
                id="review",
                label="Review LLM",
                narrative_map=base_map,
                edit_decision_list=review_edl,
                duration_s=9,
                selectable=True,
            )
        ],
        selected_proposal_variant_id="review",
    )
    repository.save(job)

    response = client.patch(
        f"/jobs/{job.job_id}/proposal-texts",
        json={"proposal_variant_id": "review", "deleted_output_orders": [1]},
    )

    assert response.status_code == 200
    patched_variant = response.json()["job"]["proposal_variants"][0]
    patched_decisions = patched_variant["edit_decision_list"]["decisions"]
    assert [decision["output_order"] for decision in patched_decisions] == [0, 1]
    assert [decision["included_text"] for decision in patched_decisions] == ["Introduzione.", "Chiusura."]
    assert patched_variant["duration_s"] == 6
    assert patched_variant["edit_decision_list"]["estimated_duration_s"] == 6

    approve_response = client.post(
        f"/jobs/{job.job_id}/approve",
        json={"approved": True, "proposal_variant_id": "review"},
    )

    assert approve_response.status_code == 200
    saved = repository.get(job.job_id)
    assert saved.edit_decision_list is not None
    assert [decision.output_order for decision in saved.edit_decision_list.decisions] == [0, 1]
    assert [decision.included_text for decision in saved.edit_decision_list.decisions] == ["Introduzione.", "Chiusura."]


def test_proposal_text_patch_rejects_invalid_state_and_unknown_order(client: TestClient) -> None:
    create_response = client.post("/jobs", json={"brief": {"user_prompt": "Patch testi", "synthesis_level": 1}})
    job_id = create_response.json()["job_id"]

    unknown_order = client.patch(
        f"/jobs/{job_id}/proposal-texts",
        json={
            "proposal_variant_id": "initial",
            "edits": [{"output_order": 999, "included_text": "Testo corretto."}],
        },
    )
    assert unknown_order.status_code == 422

    duplicate_order = client.patch(
        f"/jobs/{job_id}/proposal-texts",
        json={
            "proposal_variant_id": "initial",
            "edits": [
                {"output_order": 0, "included_text": "Primo testo."},
                {"output_order": 0, "included_text": "Secondo testo."},
            ],
        },
    )
    assert duplicate_order.status_code == 422

    approve_response = client.post(f"/jobs/{job_id}/approve", json={"approved": True})
    assert approve_response.status_code == 200
    invalid_state = client.patch(
        f"/jobs/{job_id}/proposal-texts",
        json={
            "proposal_variant_id": "initial",
            "edits": [{"output_order": 0, "included_text": "Troppo tardi."}],
        },
    )
    assert invalid_state.status_code == 409


def test_proposal_text_patch_rejects_invalid_deletions(client: TestClient) -> None:
    repository = client.app.state.repository
    base_map = NarrativeMap(
        ordered_clips=[
            NarrativeClip(
                clip_id=1,
                playback_position=0,
                topic_summary="Test",
                role="development",
                segments=[
                    NarrativeSegment(
                        segment_id=1,
                        importance=SegmentImportance.ESSENTIAL,
                        keep_verbatim=True,
                        reason="Proposta.",
                    )
                ],
            )
        ],
        filler_words=[],
        editorial_strategy="Review",
        estimated_duration_s=6,
    )
    review_edl = EditDecisionList(
        estimated_duration_s=6,
        decisions=[
            EditDecision(
                clip_id=1,
                source_start=0,
                source_end=3,
                output_order=0,
                reason="Prima",
                included_text="Prima parte.",
            ),
            EditDecision(
                clip_id=1,
                source_start=3,
                source_end=6,
                output_order=1,
                reason="Seconda",
                included_text="Seconda parte.",
            ),
        ],
    )
    job = JobRecord(
        job_id="proposal-text-delete-invalid-api",
        stage=PipelineStage.WAITING_FOR_APPROVAL,
        input_clip_paths=["clip.mp4"],
        narrative_map=base_map,
        edit_decision_list=review_edl,
        proposal_variants=[
            ProposalVariant(
                id="review",
                label="Review LLM",
                narrative_map=base_map,
                edit_decision_list=review_edl,
                duration_s=6,
                selectable=True,
            )
        ],
        selected_proposal_variant_id="review",
    )
    repository.save(job)

    unknown_delete = client.patch(
        f"/jobs/{job.job_id}/proposal-texts",
        json={"proposal_variant_id": "review", "deleted_output_orders": [999]},
    )
    assert unknown_delete.status_code == 422

    duplicate_delete = client.patch(
        f"/jobs/{job.job_id}/proposal-texts",
        json={"proposal_variant_id": "review", "deleted_output_orders": [0, 0]},
    )
    assert duplicate_delete.status_code == 422

    edit_delete_overlap = client.patch(
        f"/jobs/{job.job_id}/proposal-texts",
        json={
            "proposal_variant_id": "review",
            "edits": [{"output_order": 0, "included_text": "Prima parte corretta."}],
            "deleted_output_orders": [0],
        },
    )
    assert edit_delete_overlap.status_code == 422

    delete_everything = client.patch(
        f"/jobs/{job.job_id}/proposal-texts",
        json={"proposal_variant_id": "review", "deleted_output_orders": [0, 1]},
    )
    assert delete_everything.status_code == 422


def test_chunked_upload_creates_job_without_large_multipart(client: TestClient, settings: AppSettings) -> None:
    start_response = client.post("/uploads/start")
    assert start_response.status_code == 200
    upload_id = start_response.json()["upload_id"]

    first_chunk = client.put(
        f"/uploads/{upload_id}/clips/0/chunk",
        headers={
            "X-File-Name": "big.mp4",
            "X-Chunk-Offset": "0",
            "X-File-Size": "10",
        },
        content=b"abcde",
    )
    assert first_chunk.status_code == 200
    second_chunk = client.put(
        f"/uploads/{upload_id}/clips/0/chunk",
        headers={
            "X-File-Name": "big.mp4",
            "X-Chunk-Offset": "5",
            "X-File-Size": "10",
        },
        content=b"fghij",
    )
    assert second_chunk.status_code == 200

    create_response = client.post(
        f"/uploads/{upload_id}/jobs",
        json={"brief": {"user_prompt": "Chunk upload", "synthesis_level": 2}},
    )

    assert create_response.status_code == 200
    body = create_response.json()
    assert body["job_id"] == upload_id
    assert body["status"]["stage"] == "waiting_for_approval"
    clip_path = Path(body["job"]["input_clip_paths"][0])
    assert clip_path.read_bytes() == b"abcdefghij"
    assert clip_path.is_relative_to(settings.uploads_dir / upload_id)


def test_chunked_upload_rejects_incomplete_file_on_job_creation(client: TestClient) -> None:
    upload_id = client.post("/uploads/start").json()["upload_id"]
    chunk_response = client.put(
        f"/uploads/{upload_id}/clips/0/chunk",
        headers={
            "X-File-Name": "partial.mp4",
            "X-Chunk-Offset": "0",
            "X-File-Size": "10",
        },
        content=b"abc",
    )
    assert chunk_response.status_code == 200

    create_response = client.post(
        f"/uploads/{upload_id}/jobs",
        json={"brief": {"user_prompt": "Incomplete", "synthesis_level": 2}},
    )

    assert create_response.status_code == 400
    assert "incompleto" in create_response.json()["detail"].lower()


def test_recalculate_proposal_updates_brief_without_retranscribing(client: TestClient) -> None:
    create_response = client.post(
        "/jobs",
        json={"brief": {"user_prompt": "Prima proposta", "synthesis_level": 2}},
    )
    assert create_response.status_code == 200
    job_id = create_response.json()["job_id"]
    repository = client.app.state.repository
    last_sequence = max(event.sequence for event in repository.list_events_after(job_id, 0))

    recalculate_response = client.post(
        f"/jobs/{job_id}/recalculate-proposal",
        json={"synthesis_level": 5, "target_duration_s": None},
    )

    assert recalculate_response.status_code == 200
    body = recalculate_response.json()
    assert body["status"]["stage"] == "waiting_for_approval"
    assert body["job"]["brief"]["synthesis_level"] == 5
    assert body["job"]["brief"]["target_duration_s"] is None
    assert body["job"]["transcriptions"]
    assert body["job"]["edit_decision_list"]["decisions"]
    new_events = repository.list_events_after(job_id, last_sequence)
    assert any(event.stage.value == "refining" for event in new_events)
    assert all(event.stage.value not in {"transcribing", "narrative_analysis"} for event in new_events)
