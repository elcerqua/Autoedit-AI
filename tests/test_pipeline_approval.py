from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from autocut_ai.pipeline import InvalidApprovalStateError


def test_pipeline_stops_waiting_for_approval_before_composition(client: TestClient) -> None:
    response = client.post(
        "/jobs",
        json={"brief": {"user_prompt": "Crea proposta", "synthesis_level": 3}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"]["stage"] == "waiting_for_approval"
    assert body["job"]["final_output_path"] is None
    assert body["job"]["elapsed_seconds"] >= 0
    assert body["job"]["current_stage_elapsed_seconds"] >= 0


def test_approve_only_allowed_from_waiting_for_approval(client: TestClient) -> None:
    create_response = client.post("/jobs", json={"brief": {"user_prompt": "Approva", "synthesis_level": 1}})
    job_id = create_response.json()["job_id"]

    approve_response = client.post(f"/jobs/{job_id}/approve", json={"approved": True})

    assert approve_response.status_code == 200
    assert approve_response.json()["status"]["stage"] == "composing"

    second_approve = client.post(f"/jobs/{job_id}/approve", json={"approved": True})
    assert second_approve.status_code == 409


def test_composition_does_not_start_without_via_libera(client: TestClient) -> None:
    create_response = client.post("/jobs", json={"brief": {"user_prompt": "Aspetta", "synthesis_level": 0}})
    job_id = create_response.json()["job_id"]

    status = client.get(f"/jobs/{job_id}").json()

    assert status["stage"] == "waiting_for_approval"
    assert status["final_output_path"] is None

    with pytest.raises(InvalidApprovalStateError, match="Via libera esplicito"):
        client.app.state.runner.run_phase_two(job_id)

    unchanged = client.get(f"/jobs/{job_id}").json()
    assert unchanged["stage"] == "waiting_for_approval"
    assert unchanged["final_output_path"] is None
