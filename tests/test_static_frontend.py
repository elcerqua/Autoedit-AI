from __future__ import annotations

from fastapi.testclient import TestClient


def test_root_serves_plain_html_frontend(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Auto Cut AI" in response.text
    assert 'id="app"' in response.text


def test_frontend_does_not_show_unused_tone_selector(client: TestClient) -> None:
    html_response = client.get("/")
    js_response = client.get("/static/app.js")

    assert html_response.status_code == 200
    assert js_response.status_code == 200
    assert "tone-select" not in html_response.text
    assert "tone-select" not in js_response.text
    assert ">Tono<" not in html_response.text


def test_frontend_does_not_show_unused_dynamic_zoom_toggle(client: TestClient) -> None:
    html_response = client.get("/")
    js_response = client.get("/static/app.js")

    assert html_response.status_code == 200
    assert js_response.status_code == 200
    assert "Zoom leggero" not in html_response.text
    assert "dynamic-zoom-input" not in html_response.text
    assert "dynamic-zoom-input" not in js_response.text
    assert "enable_dynamic_zoom" not in js_response.text


def test_frontend_shows_copyable_prompt_examples(client: TestClient) -> None:
    html_response = client.get("/")
    js_response = client.get("/static/app.js")

    assert html_response.status_code == 200
    assert js_response.status_code == 200
    assert "Esempi di prompt efficaci" in html_response.text
    assert "copy-prompt-button" in html_response.text
    assert "Fai un montaggio piu asciutto" in html_response.text
    assert "copyPromptToBrief" in js_response.text
    assert "navigator.clipboard.writeText" in js_response.text


def test_frontend_exposes_creative_zoom_toggle(client: TestClient) -> None:
    html_response = client.get("/")
    js_response = client.get("/static/app.js")

    assert html_response.status_code == 200
    assert js_response.status_code == 200
    assert "Zoom creativo" in html_response.text
    assert 'id="creative-zoom-input"' in html_response.text
    assert "enable_creative_zoom" in js_response.text
    assert "creative-zoom-input" in js_response.text
    assert "decision.creative_zoom" in js_response.text
    assert "creative_zoom_spans" in js_response.text
    assert "effect-chip" in js_response.text
    assert "renderDecisionText" in js_response.text
    assert "zoom-creative-marker" in js_response.text


def test_frontend_exposes_transition_sound_controls(client: TestClient) -> None:
    html_response = client.get("/")
    js_response = client.get("/static/app.js")

    assert html_response.status_code == 200
    assert js_response.status_code == 200
    assert "Aggiungi suoni" in html_response.text
    assert 'id="transition-sounds-input"' in html_response.text
    assert 'id="transition-sound-select"' in html_response.text
    assert "add_transition_sounds" in js_response.text
    assert "transition_sound" in js_response.text
    assert "available_transition_sounds" in js_response.text
    assert "renderTransitionSoundOptions" in js_response.text


def test_transition_sound_select_is_visible_and_enabled_by_checkbox(client: TestClient) -> None:
    html_response = client.get("/")
    js_response = client.get("/static/app.js")

    assert html_response.status_code == 200
    assert js_response.status_code == 200
    assert 'id="transition-sound-field" class="field"' in html_response.text
    assert 'id="transition-sound-field" class="field hidden"' not in html_response.text
    assert 'soundSelect.disabled = !enabledInput.checked' in js_response.text
    assert 'show(byId("transition-sound-field")' not in js_response.text


def test_frontend_exposes_transition_sound_preview_button(client: TestClient) -> None:
    html_response = client.get("/")
    js_response = client.get("/static/app.js")

    assert html_response.status_code == 200
    assert js_response.status_code == 200
    assert 'id="transition-sound-play-button"' in html_response.text
    assert "Ascolta effetto" in html_response.text
    assert 'id="transition-sound-preview"' in html_response.text
    assert "previewTransitionSound" in js_response.text
    assert "/sounds/" in js_response.text
    assert "encodeURIComponent" in js_response.text


def test_frontend_exposes_subtitle_setup_controls(client: TestClient) -> None:
    html_response = client.get("/")
    js_response = client.get("/static/app.js")
    css_response = client.get("/static/styles.css")

    assert html_response.status_code == 200
    assert js_response.status_code == 200
    assert css_response.status_code == 200
    assert "Sottotitoli automatici" in html_response.text
    assert 'id="subtitles-input"' in html_response.text
    assert 'id="subtitle-position-select"' in html_response.text
    assert 'id="subtitle-style-select"' in html_response.text
    assert 'id="subtitle-style-preview-list"' in html_response.text
    assert "subtitle_options" in js_response.text
    assert "subtitle_style_presets" in js_response.text
    assert "renderSubtitleStylePreviews" in js_response.text
    assert ".subtitle-style-preview-list" in css_response.text


def test_frontend_exposes_editable_proposal_texts_before_approval(client: TestClient) -> None:
    html_response = client.get("/")
    js_response = client.get("/static/app.js")
    css_response = client.get("/static/styles.css")

    assert html_response.status_code == 200
    assert js_response.status_code == 200
    assert css_response.status_code == 200
    assert 'id="proposal-text-editor"' in html_response.text
    assert 'id="save-proposal-texts-button"' in html_response.text
    assert "Salva correzioni" in html_response.text
    assert "renderProposalTextEditor" in js_response.text
    assert "proposal-text-textarea" in js_response.text
    assert "proposal-text-delete-button" in js_response.text
    assert "proposalDeletedDrafts[key] = true" in js_response.text
    assert "proposalDeletedOutputOrdersForVariant" in js_response.text
    assert "deleted_output_orders" in js_response.text
    assert "clearProposalTextDraftsForVariant" in js_response.text
    assert "/proposal-texts" in js_response.text
    assert "saveProposalTextEdits({ silent: true })" in js_response.text
    assert ".proposal-text-editor" in css_response.text
    assert ".proposal-text-delete-button" in css_response.text


def test_frontend_preserves_proposal_textarea_focus_during_refresh(client: TestClient) -> None:
    js_response = client.get("/static/app.js")

    assert js_response.status_code == 200
    assert "activeProposalTextTextareaFor" in js_response.text
    assert "document.activeElement" in js_response.text
    assert 'active.dataset.jobId !== job.job_id' in js_response.text
    assert 'data-job-id="${escapeHtml(job.job_id)}"' in js_response.text
    assert "if (activeTextarea) {" in js_response.text
    assert "proposalTextDrafts[key] = activeTextarea.value" in js_response.text


def test_frontend_exposes_target_duration_minutes_field(client: TestClient) -> None:
    html_response = client.get("/")
    js_response = client.get("/static/app.js")

    assert html_response.status_code == 200
    assert js_response.status_code == 200
    assert 'id="target-duration-minutes-input"' in html_response.text
    assert "Durata finale target" in html_response.text
    assert "target_duration_s" in js_response.text
    assert "targetDurationMinutes * 60" in js_response.text


def test_frontend_displays_selected_video_length_in_minutes(client: TestClient) -> None:
    js_response = client.get("/static/app.js")

    assert js_response.status_code == 200
    assert "selectedVideoLengthSeconds" in js_response.text
    assert "formatVideoLengthEstimate" in js_response.text
    assert "Lunghezza stimata video" in js_response.text
    assert "somma del testo selezionato" in js_response.text


def test_frontend_exposes_review_recalculate_controls(client: TestClient) -> None:
    html_response = client.get("/")
    js_response = client.get("/static/app.js")

    assert html_response.status_code == 200
    assert js_response.status_code == 200
    assert 'id="review-synthesis-input"' in html_response.text
    assert 'id="review-target-duration-minutes-input"' in html_response.text
    assert 'id="recalculate-button"' in html_response.text
    assert "Ricalcola proposta" in html_response.text
    assert "recalculateProposal" in js_response.text
    assert "/recalculate-proposal" in js_response.text
    assert "review-target-duration-minutes-input" in js_response.text


def test_frontend_does_not_reset_dirty_review_recalculate_controls(client: TestClient) -> None:
    js_response = client.get("/static/app.js")

    assert js_response.status_code == 200
    assert "reviewRecalculateControlsDirty" in js_response.text
    assert "syncReviewRecalculateControls" in js_response.text
    assert "if (reviewRecalculateControlsDirty) {" in js_response.text
    assert "reviewRecalculateControlsDirty = true" in js_response.text
    assert "reviewRecalculateControlsDirty = false" in js_response.text


def test_frontend_shows_selectable_proposal_variants(client: TestClient) -> None:
    html_response = client.get("/")
    js_response = client.get("/static/app.js")
    css_response = client.get("/static/styles.css")

    assert html_response.status_code == 200
    assert js_response.status_code == 200
    assert css_response.status_code == 200
    assert 'id="proposal-variant-list"' in html_response.text
    assert "renderProposalVariants" in js_response.text
    assert "selectedProposalVariantId" in js_response.text
    assert "Proposta iniziale" in js_response.text
    assert "Review LLM" in js_response.text
    assert ".proposal-variant-list" in css_response.text


def test_frontend_renders_side_by_side_proposal_comparison(client: TestClient) -> None:
    html_response = client.get("/")
    js_response = client.get("/static/app.js")
    css_response = client.get("/static/styles.css")

    assert html_response.status_code == 200
    assert js_response.status_code == 200
    assert css_response.status_code == 200
    assert 'id="proposal-comparison"' in html_response.text
    assert "renderProposalComparison" in js_response.text
    assert "decisionSignature" in js_response.text
    assert "proposalVariantsForJob(job).map" in js_response.text
    assert "proposal-diff-initial" in js_response.text
    assert "proposal-diff-review" in js_response.text
    assert ".proposal-comparison-grid" in css_response.text
    assert ".comparison-decision.initial-only" in css_response.text
    assert ".comparison-decision.review-only" in css_response.text


def test_frontend_via_libera_sends_selected_proposal_variant(client: TestClient) -> None:
    js_response = client.get("/static/app.js")

    assert js_response.status_code == 200
    assert "proposal_variant_id: selectedProposalVariantId" in js_response.text
    assert "variant.selectable" in js_response.text
    assert "La proposta selezionata non e renderizzabile" in js_response.text


def test_frontend_keeps_initial_selectable_when_review_is_not_approved(client: TestClient) -> None:
    js_response = client.get("/static/app.js")
    css_response = client.get("/static/styles.css")

    assert js_response.status_code == 200
    assert css_response.status_code == 200
    assert "job.proposal_variants || []" in js_response.text
    assert "fallbackInitialVariant" in js_response.text
    assert 'variant.id === "initial"' in js_response.text
    assert ".review-note.blocking" in css_response.text


def test_frontend_marks_excessive_synthesis_review_as_selectable_warning(client: TestClient) -> None:
    js_response = client.get("/static/app.js")

    assert js_response.status_code == 200
    assert "hasSelectableExcessiveSynthesisReview" in js_response.text
    assert "hasSelectableReviewVariant" in js_response.text
    assert "Review LLM selezionabile, piu aggressiva del budget" in js_response.text
    assert 'reviewBox.classList.toggle("blocking", blockingReviewNotice)' in js_response.text


def test_frontend_uses_chunked_upload_for_video_files(client: TestClient) -> None:
    js_response = client.get("/static/app.js")

    assert js_response.status_code == 200
    assert "uploadFileInChunks" in js_response.text
    assert "/uploads/start" in js_response.text
    assert "/chunk" in js_response.text
    assert "X-Chunk-Offset" in js_response.text
    assert "X-File-Size" in js_response.text
