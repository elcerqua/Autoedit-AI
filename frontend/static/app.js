const stageLabels = {
  queued: "In coda",
  interpreting_prompt: "Interpretazione brief",
  normalizing_clips: "Normalizzazione clip",
  transcribing: "Trascrizione",
  narrative_analysis: "Analisi narrativa",
  refining: "Rifinitura cut",
  waiting_for_approval: "In attesa di Via libera",
  composing: "Composizione",
  mixing_music: "Mix musica",
  done: "Pronto",
  failed: "Errore"
};

let optionsCatalog = null;
let editingOptions = null;
let currentJob = null;
let eventSource = null;
let pollingId = null;
let timerId = null;
let tick = 0;
let reviewRecalculateControlsDirty = false;
let reviewRecalculateControlsJobId = null;
let selectedProposalVariantId = "initial";
let selectedProposalVariantJobId = null;
let proposalTextDrafts = {};
let proposalDeletedDrafts = {};
let proposalTextDraftsJobId = null;
const uploadChunkSize = 8 * 1024 * 1024;
const terminalLiveStages = new Set(["waiting_for_approval", "done", "failed"]);

const byId = (id) => document.getElementById(id);
const transitionLabels = {
  cut: "Cut netto",
  cross_dissolve: "Cross dissolve",
  fade_black: "Fade al nero",
  zoom_cut: "Zoom cut",
  whip_pan: "Whip pan",
  fade: "Fade",
  fadeblack: "Fade black",
  fadewhite: "Fade white",
  dissolve: "Dissolve",
  zoomin: "Zoom in",
  hblur: "Blur orizzontale",
  pixelize: "Pixelize"
};

function show(element, visible = true) {
  element.classList.toggle("hidden", !visible);
}

function formatSeconds(value) {
  const seconds = Math.max(0, Math.floor(value || 0));
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  return `${minutes}:${String(remainder).padStart(2, "0")}`;
}

async function readJson(response) {
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof body.detail === "string" ? body.detail : "Richiesta non riuscita";
    throw new Error(detail);
  }
  return body;
}

function setError(message) {
  const box = byId("app-error");
  box.textContent = message || "";
  show(box, Boolean(message));
}

async function copyPromptToBrief(button) {
  const prompt = button.dataset.prompt || "";
  const originalLabel = button.textContent;
  byId("brief-input").value = prompt;
  try {
    await navigator.clipboard.writeText(prompt);
    button.textContent = "Copiato";
  } catch (err) {
    button.textContent = "Inserito";
  }
  window.setTimeout(() => {
    button.textContent = originalLabel;
  }, 1400);
}

function renderClipList() {
  const list = byId("clip-list");
  const files = Array.from(byId("clips-input").files || []);
  list.innerHTML = files
    .map(
      (file) => `
        <div class="clip-item">
          <span title="${escapeHtml(file.name)}">${escapeHtml(file.name)}</span>
          <strong>${(file.size / 1024 / 1024).toFixed(1)} MB</strong>
        </div>
      `
    )
    .join("");
}

function readEditingOptions() {
  const subtitleDefaults = editingOptions.subtitle_options || {};
  return {
    ...editingOptions,
    default_transition: byId("transition-select").value,
    transition_duration_ms: Number(byId("transition-duration-input").value || 0),
    target_aspect_ratio: byId("aspect-select").value,
    target_resolution: byId("resolution-input").value || null,
    normalize_audio: byId("normalize-audio-input").checked,
    remove_silence: byId("remove-silence-input").checked,
    enable_creative_zoom: byId("creative-zoom-input").checked,
    add_transition_sounds: byId("transition-sounds-input").checked,
    transition_sound: byId("transition-sound-select").value || null,
    add_background_music: byId("music-input").checked,
    music_prompt: byId("music-prompt-input").value || null,
    subtitle_options: {
      ...subtitleDefaults,
      enabled: byId("subtitles-input").checked,
      position: byId("subtitle-position-select").value,
      style_preset: byId("subtitle-style-select").value || subtitleDefaults.style_preset || "classic_white"
    }
  };
}

function renderOptionsCatalog() {
  byId("transition-select").innerHTML = optionsCatalog.available_transitions
    .map((transition) => `<option value="${escapeHtml(transition)}">${escapeHtml(transitionLabels[transition] || transition)}</option>`)
    .join("");
  byId("aspect-select").innerHTML = optionsCatalog.target_aspect_ratios
    .map((ratio) => `<option value="${escapeHtml(ratio)}">${escapeHtml(ratio)}</option>`)
    .join("");
  byId("transition-select").value = editingOptions.default_transition;
  byId("transition-duration-input").value = editingOptions.transition_duration_ms;
  byId("aspect-select").value = editingOptions.target_aspect_ratio;
  byId("resolution-input").value = editingOptions.target_resolution || "";
  byId("normalize-audio-input").checked = editingOptions.normalize_audio;
  byId("remove-silence-input").checked = editingOptions.remove_silence;
  byId("creative-zoom-input").checked = editingOptions.enable_creative_zoom;
  byId("transition-sounds-input").checked = editingOptions.add_transition_sounds;
  byId("subtitles-input").checked = Boolean(editingOptions.subtitle_options?.enabled);
  byId("subtitle-position-select").value = editingOptions.subtitle_options?.position || "bottom";
  byId("music-input").checked = editingOptions.add_background_music;
  renderTransitionSoundOptions();
  renderSubtitleStyleOptions();
  updateSubtitleOptionsVisibility();
}

function renderTransitionSoundOptions() {
  const sounds = optionsCatalog.available_transition_sounds || [];
  const soundSelect = byId("transition-sound-select");
  const enabledInput = byId("transition-sounds-input");
  const playButton = byId("transition-sound-play-button");
  const hasSounds = sounds.length > 0;
  const currentSelection = soundSelect.value || editingOptions.transition_sound || sounds[0] || "";
  soundSelect.innerHTML = hasSounds
    ? sounds
        .map((sound) => `<option value="${escapeHtml(sound)}">${escapeHtml(sound)}</option>`)
        .join("")
    : `<option value="">Nessun mp3 trovato in Sounds</option>`;
  enabledInput.disabled = !hasSounds;
  if (!hasSounds) {
    enabledInput.checked = false;
    soundSelect.disabled = true;
    playButton.disabled = true;
    return;
  }
  soundSelect.value = sounds.includes(currentSelection) ? currentSelection : sounds[0];
  soundSelect.disabled = !enabledInput.checked;
  playButton.disabled = !enabledInput.checked;
}

function renderSubtitleStyleOptions() {
  const presets = optionsCatalog.subtitle_style_presets || [];
  const styleSelect = byId("subtitle-style-select");
  const currentStyle = styleSelect.value || editingOptions.subtitle_options?.style_preset || "classic_white";
  styleSelect.innerHTML = presets
    .map((preset) => `<option value="${escapeHtml(preset.id)}">${escapeHtml(preset.label)}</option>`)
    .join("");
  if (presets.some((preset) => preset.id === currentStyle)) {
    styleSelect.value = currentStyle;
  } else if (presets.length) {
    styleSelect.value = presets[0].id;
  }
  renderSubtitleStylePreviews();
}

function renderSubtitleStylePreviews() {
  const presets = optionsCatalog.subtitle_style_presets || [];
  const selectedStyle = byId("subtitle-style-select").value;
  byId("subtitle-style-preview-list").innerHTML = presets
    .map(
      (preset) => `
        <button class="subtitle-style-preview ${preset.id === selectedStyle ? "selected" : ""}" type="button" data-style-preset="${escapeHtml(preset.id)}">
          <img alt="${escapeHtml(preset.label)}" src="${escapeHtml(preset.preview_data_uri)}" />
          <span>${escapeHtml(preset.label)}</span>
        </button>
      `
    )
    .join("");
  byId("subtitle-style-preview-list").querySelectorAll(".subtitle-style-preview").forEach((button) => {
    button.addEventListener("click", () => {
      byId("subtitle-style-select").value = button.dataset.stylePreset || "classic_white";
      renderSubtitleStylePreviews();
    });
  });
}

function updateSubtitleOptionsVisibility() {
  show(byId("subtitle-options-panel"), byId("subtitles-input").checked);
}

function readDurationMinutesInputSeconds(inputId) {
  const rawValue = byId(inputId).value.trim().replace(",", ".");
  if (!rawValue) {
    return null;
  }
  const targetDurationMinutes = Number(rawValue);
  if (!Number.isFinite(targetDurationMinutes) || targetDurationMinutes <= 0) {
    return null;
  }
  return targetDurationMinutes * 60;
}

function readTargetDurationSeconds() {
  return readDurationMinutesInputSeconds("target-duration-minutes-input");
}

async function previewTransitionSound() {
  const soundName = byId("transition-sound-select").value;
  if (!soundName) {
    return;
  }
  const audio = byId("transition-sound-preview");
  audio.pause();
  audio.currentTime = 0;
  audio.src = `/sounds/${encodeURIComponent(soundName)}`;
  try {
    setError("");
    await audio.play();
  } catch (err) {
    setError("Impossibile riprodurre l'effetto nel browser.");
  }
}

function jobPayloadFromForm() {
  const targetDurationSeconds = readTargetDurationSeconds();
  const payload = {
    brief: {
      user_prompt: byId("brief-input").value,
      synthesis_level: Number(byId("synthesis-input").value),
      target_duration_s: targetDurationSeconds
    },
    editing_options: readEditingOptions()
  };
  if (byId("model-select").value) {
    payload.llm_model = byId("model-select").value;
  }
  return payload;
}

async function startUploadSession() {
  return readJson(await fetch("/uploads/start", { method: "POST" }));
}

async function uploadFileInChunks(uploadId, clipIndex, file, totalFiles) {
  let offset = 0;
  while (offset < file.size) {
    const end = Math.min(offset + uploadChunkSize, file.size);
    const percent = Math.floor((end / Math.max(file.size, 1)) * 100);
    byId("analyze-button").textContent = `Upload ${clipIndex + 1}/${totalFiles} ${percent}%`;
    await readJson(
      await fetch(`/uploads/${uploadId}/clips/${clipIndex}/chunk`, {
        method: "PUT",
        headers: {
          "X-File-Name": encodeURIComponent(file.name),
          "X-Chunk-Offset": String(offset),
          "X-File-Size": String(file.size)
        },
        body: file.slice(offset, end)
      })
    );
    offset = end;
  }
}

async function createJobFromUpload(uploadId) {
  return readJson(
    await fetch(`/uploads/${uploadId}/jobs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(jobPayloadFromForm())
    })
  );
}

async function loadModels() {
  const select = byId("model-select");
  const error = byId("model-error");
  try {
    const body = await readJson(await fetch("/llm/models"));
    select.innerHTML = (body.models || [])
      .map((model) => `<option value="${escapeHtml(model)}">${escapeHtml(model)}</option>`)
      .join("");
    select.value = body.default_model || body.models?.[0] || "";
    select.disabled = !body.models?.length;
    show(error, false);
  } catch (err) {
    select.innerHTML = "";
    select.disabled = true;
    error.textContent = err.message;
    show(error, true);
  }
}

async function loadOptions() {
  optionsCatalog = await readJson(await fetch("/editing-options"));
  editingOptions = optionsCatalog.defaults;
  renderOptionsCatalog();
}

function updateTimers() {
  if (!currentJob) {
    return;
  }
  const running = !["done", "failed"].includes(currentJob.stage);
  const extra = running ? tick : 0;
  byId("elapsed-label").textContent = formatSeconds((currentJob.elapsed_seconds || 0) + extra);
  byId("stage-elapsed-label").textContent = formatSeconds((currentJob.current_stage_elapsed_seconds || 0) + extra);
}

function startTimers() {
  window.clearInterval(timerId);
  tick = 0;
  timerId = window.setInterval(() => {
    tick += 1;
    updateTimers();
  }, 1000);
}

function addEvent(event) {
  const log = byId("event-log");
  const row = document.createElement("div");
  const stage = document.createElement("span");
  const message = document.createElement("p");
  row.className = "log-row";
  // I messaggi possono contenere testo prodotto dall'LLM: textContent evita
  // che markup inatteso venga eseguito nel browser.
  stage.textContent = stageLabels[event.stage] || event.stage;
  message.textContent = event.message || "";
  row.append(stage, message);
  log.prepend(row);
}

function updateWorkflow(stage) {
  const order = ["setup", "analysis", "review", "export"];
  const current = stage === "waiting_for_approval"
    ? "review"
    : ["composing", "mixing_music", "done"].includes(stage)
      ? "export"
      : stage === "setup"
        ? "setup"
        : "analysis";
  const currentIndex = order.indexOf(current);
  byId("app").dataset.stage = current;
  document.querySelectorAll("[data-workflow-step]").forEach((step) => {
    const stepIndex = order.indexOf(step.dataset.workflowStep);
    step.classList.toggle("active", stepIndex === currentIndex);
    step.classList.toggle("complete", stepIndex < currentIndex || stage === "done");
  });
}

function renderJob(job) {
  currentJob = job;
  updateWorkflow(job.stage);
  show(byId("setup-view"), false);
  show(byId("pipeline-view"), true);
  byId("stage-label").textContent = stageLabels[job.stage] || job.stage;
  byId("stage-message").textContent = job.message || "";
  updateTimers();

  const reviewReady = job.stage === "waiting_for_approval" && job.narrative_map && job.edit_decision_list;
  show(byId("review-view"), reviewReady);
  if (reviewReady) {
    renderReview(job);
  }

  const finalVisible = ["composing", "mixing_music", "done"].includes(job.stage);
  show(byId("final-view"), finalVisible);
  if (finalVisible) {
    byId("final-title").textContent = job.stage === "done" ? "Video finale pronto" : "Rendering in corso";
    const link = byId("download-link");
    link.href = `/jobs/${job.job_id}/download`;
    show(link, job.stage === "done");
  }
  if (terminalLiveStages.has(job.stage)) {
    stopLiveUpdates();
  }
  if (["done", "failed"].includes(job.stage)) {
    window.clearInterval(timerId);
    timerId = null;
  }
}

function selectedVideoLengthSeconds(editDecisionList) {
  const decisions = editDecisionList?.decisions || [];
  const decisionDuration = decisions.reduce((total, decision) => {
    const start = Number(decision.source_start || 0);
    const end = Number(decision.source_end || 0);
    return total + Math.max(0, end - start);
  }, 0);
  if (decisionDuration > 0) {
    return decisionDuration;
  }
  return Number(editDecisionList?.estimated_duration_s || 0);
}

function formatVideoLengthEstimate(durationSeconds) {
  const roundedMinutes = Math.round((durationSeconds / 60) * 100) / 100;
  return `${roundedMinutes.toFixed(2)} min (${durationSeconds.toFixed(1)}s)`;
}

function syncReviewRecalculateControls(job) {
  const jobChanged = reviewRecalculateControlsJobId !== job.job_id;
  if (jobChanged) {
    reviewRecalculateControlsDirty = false;
    reviewRecalculateControlsJobId = job.job_id;
  }
  if (reviewRecalculateControlsDirty) {
    return;
  }
  byId("review-synthesis-input").value = job.brief.synthesis_level;
  byId("review-synthesis-label").textContent = job.brief.synthesis_level;
  byId("review-target-duration-minutes-input").value = job.brief.target_duration_s
    ? (job.brief.target_duration_s / 60).toFixed(2)
    : "";
}

function fallbackInitialVariant(job) {
  return {
    id: "initial",
    label: "Proposta iniziale",
    narrative_map: job.narrative_map,
    edit_decision_list: job.edit_decision_list,
    duration_s: selectedVideoLengthSeconds(job.edit_decision_list),
    selectable: true,
    issues: [],
    rejection_reason: null
  };
}

function proposalVariantsForJob(job) {
  const variants = job.proposal_variants || [];
  if (variants.length) {
    return variants;
  }
  if (job.narrative_map && job.edit_decision_list) {
    return [fallbackInitialVariant(job)];
  }
  return [];
}

function variantDisplayLabel(variant) {
  return variant.label || (variant.id === "review" ? "Review LLM" : "Proposta iniziale");
}

function variantDurationSeconds(variant) {
  return Number(variant.duration_s || selectedVideoLengthSeconds(variant.edit_decision_list));
}

function variantIssuesText(variant) {
  const issues = variant.issues || [];
  if (!issues.length) {
    return "Nessuna issue critica.";
  }
  return issues.map((issue) => `${issue.code}: ${issue.message}`).join(" | ");
}

function variantStatusLabel(job, variant) {
  if (!variant.selectable) {
    return "Non renderizzabile";
  }
  if (variant.id === "initial") {
    return "Disponibile";
  }
  if (job.proposal_review?.approved === true) {
    return "Approvata";
  }
  if ((variant.rejection_reason || "").includes("excessive_synthesis")) {
    return "Troppo aggressiva, selezionabile";
  }
  return "Review selezionabile";
}

function hasSelectableExcessiveSynthesisReview(job) {
  return proposalVariantsForJob(job).some((variant) => {
    const issueText = (variant.issues || [])
      .map((issue) => `${issue.code || ""} ${issue.message || ""}`)
      .join(" ");
    return (
      variant.id === "review" &&
      variant.selectable &&
      (`${variant.rejection_reason || ""} ${issueText}`).includes("excessive_synthesis")
    );
  });
}

function hasSelectableReviewVariant(job) {
  return proposalVariantsForJob(job).some((variant) => variant.id === "review" && variant.selectable);
}

function syncSelectedProposalVariant(job, variants) {
  if (selectedProposalVariantJobId !== job.job_id) {
    selectedProposalVariantJobId = job.job_id;
    selectedProposalVariantId = job.selected_proposal_variant_id || "initial";
  }
  if (!variants.some((variant) => variant.id === selectedProposalVariantId)) {
    const backendSelected = variants.find((variant) => variant.id === job.selected_proposal_variant_id);
    const firstSelectable = variants.find((variant) => variant.selectable);
    selectedProposalVariantId = backendSelected?.id || firstSelectable?.id || variants[0]?.id || "initial";
  }
}

function renderProposalVariants(job) {
  const variants = proposalVariantsForJob(job);
  syncSelectedProposalVariant(job, variants);
  const list = byId("proposal-variant-list");
  list.innerHTML = variants
    .map((variant) => {
      const selected = variant.id === selectedProposalVariantId;
      const displayLabel = variantDisplayLabel(variant);
      const durationSeconds = variantDurationSeconds(variant);
      const issueText = variantIssuesText(variant);
      const rejection = variant.rejection_reason ? `<p>${escapeHtml(variant.rejection_reason)}</p>` : "";
      return `
        <article class="proposal-variant-card ${selected ? "selected" : ""} ${variant.selectable ? "" : "blocking"}">
          <div class="proposal-variant-heading">
            <div>
              <strong>${escapeHtml(displayLabel)}</strong>
              <span>${escapeHtml(variantStatusLabel(job, variant))}</span>
            </div>
            <input type="radio" name="proposal-variant" value="${escapeHtml(variant.id)}" aria-label="Seleziona ${escapeHtml(displayLabel)}" ${selected ? "checked" : ""} ${variant.selectable ? "" : "disabled"} />
          </div>
          <p>Durata stimata: ${formatVideoLengthEstimate(durationSeconds)}</p>
          ${rejection}
          <small>${escapeHtml(issueText)}</small>
          <button class="variant-select-button" type="button" data-variant-id="${escapeHtml(variant.id)}" ${variant.selectable ? "" : "disabled"}>Usa questa proposta</button>
        </article>
      `;
    })
    .join("");
  list.querySelectorAll(".variant-select-button").forEach((button) => {
    button.addEventListener("click", () => {
      selectedProposalVariantId = button.dataset.variantId || "initial";
      renderReview(job);
    });
  });
  list.querySelectorAll('input[name="proposal-variant"]').forEach((input) => {
    input.addEventListener("change", () => {
      selectedProposalVariantId = input.value || "initial";
      renderReview(job);
    });
  });
  return variants.find((variant) => variant.id === selectedProposalVariantId) || variants[0] || null;
}

function decisionSignature(decision) {
  return [
    decision.clip_id,
    Number(decision.source_start || 0).toFixed(2),
    Number(decision.source_end || 0).toFixed(2),
    String(decision.included_text || "").trim()
  ].join("|");
}

function proposalDecisionSignatureCounts(variants) {
  const counts = new Map();
  variants.forEach((variant) => {
    const decisions = variant.edit_decision_list?.decisions || [];
    decisions.forEach((decision) => {
      const signature = decisionSignature(decision);
      counts.set(signature, (counts.get(signature) || 0) + 1);
    });
  });
  return counts;
}

function decisionDiffClass(variant, decision, signatureCounts, variantCount) {
  if (variantCount < 2) {
    return "shared";
  }
  if ((signatureCounts.get(decisionSignature(decision)) || 0) > 1) {
    return "shared";
  }
  return variant.id === "review" ? "review-only" : "initial-only";
}

function renderDecisionCard(decision, diffClass) {
  const safetyNotes = decision.safety_notes || [];
  const diffBadge = diffClass === "initial-only"
    ? `<strong class="proposal-diff-initial">Solo proposta iniziale</strong>`
    : diffClass === "review-only"
      ? `<strong class="proposal-diff-review">Solo Review LLM</strong>`
      : "";
  return `
    <article class="decision-card comparison-decision ${diffClass}">
      <div>
        <strong>Clip ${decision.clip_id}</strong>
        <span>${Number(decision.source_start || 0).toFixed(2)}s - ${Number(decision.source_end || 0).toFixed(2)}s</span>
      </div>
      <p>${renderDecisionText(decision)}</p>
      ${renderCreativeZoomSpans(decision)}
      ${diffBadge}
      <small>${escapeHtml(decision.reason || "")}</small>
      ${safetyNotes.length ? `<em>${safetyNotes.map((note) => escapeHtml(note)).join(" · ")}</em>` : ""}
    </article>
  `;
}

function renderProposalComparisonCard(job, variant, signatureCounts, variantCount) {
  const selected = variant.id === selectedProposalVariantId;
  const decisions = variant.edit_decision_list?.decisions || [];
  const rejection = variant.rejection_reason ? `<p>${escapeHtml(variant.rejection_reason)}</p>` : "";
  return `
    <section class="proposal-comparison-card ${selected ? "selected" : ""} ${variant.selectable ? "" : "blocking"}">
      <div class="proposal-comparison-heading">
        <div>
          <strong>${escapeHtml(variantDisplayLabel(variant))}</strong>
          <span>${escapeHtml(variantStatusLabel(job, variant))}</span>
        </div>
        <input type="radio" name="proposal-comparison-variant" value="${escapeHtml(variant.id)}" aria-label="Confronta ${escapeHtml(variantDisplayLabel(variant))}" ${selected ? "checked" : ""} ${variant.selectable ? "" : "disabled"} />
      </div>
      <p class="proposal-comparison-duration">Durata stimata: ${formatVideoLengthEstimate(variantDurationSeconds(variant))}</p>
      ${rejection}
      <small>${escapeHtml(variantIssuesText(variant))}</small>
      <button class="comparison-select-button" type="button" data-variant-id="${escapeHtml(variant.id)}" ${variant.selectable ? "" : "disabled"}>Usa questa proposta</button>
      <div class="proposal-comparison-decisions">
        ${decisions
          .map((decision) =>
            renderDecisionCard(decision, decisionDiffClass(variant, decision, signatureCounts, variantCount))
          )
          .join("")}
      </div>
    </section>
  `;
}

function renderProposalComparison(job) {
  const variants = proposalVariantsForJob(job).map((variant) => variant);
  const comparison = byId("proposal-comparison");
  if (!variants.length) {
    comparison.innerHTML = "";
    return;
  }
  const signatureCounts = proposalDecisionSignatureCounts(variants);
  comparison.innerHTML = `
    <div class="proposal-comparison-grid ${variants.length === 1 ? "single" : ""}">
      ${variants
        .map((variant) => renderProposalComparisonCard(job, variant, signatureCounts, variants.length))
        .join("")}
    </div>
  `;
  comparison.querySelectorAll(".comparison-select-button").forEach((button) => {
    button.addEventListener("click", () => {
      selectedProposalVariantId = button.dataset.variantId || "initial";
      renderReview(job);
    });
  });
  comparison.querySelectorAll('input[name="proposal-comparison-variant"]').forEach((input) => {
    input.addEventListener("change", () => {
      selectedProposalVariantId = input.value || "initial";
      renderReview(job);
    });
  });
}

function syncProposalTextDraftsForJob(job) {
  if (proposalTextDraftsJobId !== job.job_id) {
    proposalTextDrafts = {};
    proposalDeletedDrafts = {};
    proposalTextDraftsJobId = job.job_id;
  }
}

function proposalTextDraftKey(job, variantId, outputOrder) {
  return `${job.job_id}:${variantId}:${outputOrder}`;
}

function proposalTextDraftPrefix(job, variantId) {
  return `${job.job_id}:${variantId}:`;
}

function clearProposalTextDraftsForVariant(job, variantId) {
  const prefix = proposalTextDraftPrefix(job, variantId);
  Object.keys(proposalTextDrafts).forEach((key) => {
    if (key.startsWith(prefix)) {
      delete proposalTextDrafts[key];
    }
  });
  Object.keys(proposalDeletedDrafts).forEach((key) => {
    if (key.startsWith(prefix)) {
      delete proposalDeletedDrafts[key];
    }
  });
}

function normalizeProposalText(value) {
  return String(value || "").trim().replace(/\s+/g, " ");
}

function proposalTextDeletedForDecision(job, variantId, decision) {
  const key = proposalTextDraftKey(job, variantId, decision.output_order);
  return Boolean(proposalDeletedDrafts[key]);
}

function proposalTextForDecision(job, variantId, decision) {
  const key = proposalTextDraftKey(job, variantId, decision.output_order);
  if (Object.prototype.hasOwnProperty.call(proposalTextDrafts, key)) {
    return proposalTextDrafts[key];
  }
  return decision.included_text || "";
}

function proposalTextEditsForVariant(job, variant) {
  if (!job || !variant) {
    return [];
  }
  return (variant.edit_decision_list?.decisions || [])
    .map((decision) => {
      if (proposalTextDeletedForDecision(job, variant.id, decision)) {
        return null;
      }
      const originalText = normalizeProposalText(decision.included_text || "");
      const editedText = normalizeProposalText(proposalTextForDecision(job, variant.id, decision));
      if (!editedText || editedText === originalText) {
        return null;
      }
      return { output_order: decision.output_order, included_text: editedText };
    })
    .filter(Boolean);
}

function proposalDeletedOutputOrdersForVariant(job, variant) {
  if (!job || !variant) {
    return [];
  }
  return (variant.edit_decision_list?.decisions || [])
    .filter((decision) => proposalTextDeletedForDecision(job, variant.id, decision))
    .map((decision) => decision.output_order);
}

function activeProposalTextTextareaFor(job, variant) {
  const active = document.activeElement;
  if (!active?.classList?.contains("proposal-text-textarea")) {
    return null;
  }
  if (active.dataset.jobId !== job.job_id || active.dataset.variantId !== variant.id) {
    return null;
  }
  const outputOrder = Number(active.dataset.outputOrder);
  const decisions = variant.edit_decision_list?.decisions || [];
  const decision = decisions.find((candidate) => Number(candidate.output_order) === outputOrder);
  if (!decision || proposalTextDeletedForDecision(job, variant.id, decision)) {
    return null;
  }
  return active;
}

function renderProposalTextEditor(job, variant) {
  syncProposalTextDraftsForJob(job);
  const editor = byId("proposal-text-editor");
  if (!variant?.selectable) {
    editor.innerHTML = "";
    updateProposalTextSaveButton(job, variant);
    return;
  }
  const activeTextarea = activeProposalTextTextareaFor(job, variant);
  if (activeTextarea) {
    const key = proposalTextDraftKey(job, variant.id, Number(activeTextarea.dataset.outputOrder));
    proposalTextDrafts[key] = activeTextarea.value;
    updateProposalTextSaveButton(job, variant);
    return;
  }
  const decisions = variant.edit_decision_list?.decisions || [];
  const visibleDecisions = decisions.filter((decision) => !proposalTextDeletedForDecision(job, variant.id, decision));
  const canDeleteMore = visibleDecisions.length > 1;
  editor.innerHTML = `
    <div class="proposal-text-editor-heading">
      <div>
        <h3>Testi e sottotitoli</h3>
        <p>Correggi refusi o punteggiatura: questi testi saranno usati anche nei sottotitoli automatici.</p>
      </div>
      <strong>${escapeHtml(variantDisplayLabel(variant))}</strong>
    </div>
    <div class="proposal-text-editor-list">
      ${visibleDecisions
        .map(
          (decision) => `
            <article class="proposal-text-row">
              <div class="proposal-text-row-heading">
                <span>Clip ${decision.clip_id} - ${Number(decision.source_start || 0).toFixed(2)}s / ${Number(decision.source_end || 0).toFixed(2)}s</span>
                <button class="proposal-text-delete-button" type="button" data-output-order="${decision.output_order}" aria-label="Elimina segmento dal montaggio" title="Elimina segmento dal montaggio" ${canDeleteMore ? "" : "disabled"}>X</button>
              </div>
              <textarea class="proposal-text-textarea" rows="3" data-job-id="${escapeHtml(job.job_id)}" data-variant-id="${escapeHtml(variant.id)}" data-output-order="${decision.output_order}">${escapeHtml(proposalTextForDecision(job, variant.id, decision))}</textarea>
            </article>
          `
        )
        .join("")}
    </div>
  `;
  editor.querySelectorAll(".proposal-text-textarea").forEach((textarea) => {
    textarea.addEventListener("input", () => {
      const key = proposalTextDraftKey(job, variant.id, Number(textarea.dataset.outputOrder));
      proposalTextDrafts[key] = textarea.value;
      updateProposalTextSaveButton(job, variant);
    });
  });
  editor.querySelectorAll(".proposal-text-delete-button").forEach((button) => {
    button.addEventListener("click", () => {
      const key = proposalTextDraftKey(job, variant.id, Number(button.dataset.outputOrder));
      proposalDeletedDrafts[key] = true;
      delete proposalTextDrafts[key];
      renderProposalTextEditor(job, variant);
    });
  });
  updateProposalTextSaveButton(job, variant);
}

function updateProposalTextSaveButton(job = currentJob, variant = null) {
  const button = byId("save-proposal-texts-button");
  const selectedVariant = variant || proposalVariantsForJob(job || {}).find((candidate) => candidate.id === selectedProposalVariantId);
  const hasEdits = proposalTextEditsForVariant(job, selectedVariant).length > 0;
  const hasDeletions = proposalDeletedOutputOrdersForVariant(job, selectedVariant).length > 0;
  button.disabled = !hasEdits && !hasDeletions;
  button.textContent = hasEdits || hasDeletions ? "Salva correzioni" : "Correzioni salvate";
}

async function saveProposalTextEdits({ silent = false } = {}) {
  if (!currentJob) {
    return null;
  }
  const variant = proposalVariantsForJob(currentJob).find((candidate) => candidate.id === selectedProposalVariantId);
  const edits = proposalTextEditsForVariant(currentJob, variant);
  const deletedOutputOrders = proposalDeletedOutputOrdersForVariant(currentJob, variant);
  updateProposalTextSaveButton(currentJob, variant);
  if (!variant || (!edits.length && !deletedOutputOrders.length)) {
    return currentJob;
  }
  const button = byId("save-proposal-texts-button");
  button.disabled = true;
  try {
    const body = await readJson(
      await fetch(`/jobs/${currentJob.job_id}/proposal-texts`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ proposal_variant_id: variant.id, edits, deleted_output_orders: deletedOutputOrders })
      })
    );
    if (!silent) {
      addEvent(body.status);
    }
    clearProposalTextDraftsForVariant(currentJob, variant.id);
    renderJob(body.job);
    return body.job;
  } finally {
    updateProposalTextSaveButton(currentJob);
  }
}

function applyApprovalGate(job, busy = false) {
  const approveButton = byId("approve-button");
  const variant = proposalVariantsForJob(job).find((candidate) => candidate.id === selectedProposalVariantId);
  const selectable = Boolean(variant && variant.selectable);
  approveButton.disabled = !selectable || busy;
  approveButton.title = selectable ? "" : "La proposta selezionata non e renderizzabile";
}

function renderReview(job) {
  const selectedVariant = renderProposalVariants(job);
  const narrativeMap = selectedVariant?.narrative_map || job.narrative_map;
  const editDecisionList = selectedVariant?.edit_decision_list || job.edit_decision_list;
  byId("editorial-strategy").textContent = narrativeMap.editorial_strategy;
  const selectedDuration = selectedVideoLengthSeconds(editDecisionList);
  byId("estimated-duration").textContent =
    `Lunghezza stimata video: ${formatVideoLengthEstimate(selectedDuration)} - somma del testo selezionato nelle clip`;
  syncReviewRecalculateControls(job);
  const reviewBox = byId("proposal-review");
  if (job.proposal_review) {
    const issueCodes = (job.proposal_review.issues || []).map((issue) => issue.code).join(", ");
    const reviewApproved = job.proposal_review.approved !== false;
    const selectableReview = hasSelectableReviewVariant(job);
    const selectableAggressiveReview = hasSelectableExcessiveSynthesisReview(job);
    const reviewText = job.proposal_review.editorial_notes || "Proposta validata dal reviewer LLM.";
    const prefix = reviewApproved
      ? "Revisione LLM"
      : selectableAggressiveReview
        ? "Review LLM selezionabile, piu aggressiva del budget"
        : selectableReview
          ? "Review LLM selezionabile"
          : "Revisione LLM non renderizzabile";
    const suffix = reviewApproved
      ? ""
      : selectableReview
        ? " Confronta Proposta iniziale e Review LLM: puoi renderizzare la variante che preferisci."
        : " Puoi scegliere una variante renderizzabile oppure ricalcolare la proposta.";
    const blockingReviewNotice = !reviewApproved && !selectableReview;
    reviewBox.textContent = issueCodes ? `${prefix}: ${reviewText} (${issueCodes}).${suffix}` : `${prefix}: ${reviewText}.${suffix}`;
    reviewBox.classList.toggle("blocking", blockingReviewNotice);
    show(reviewBox, true);
  } else {
    reviewBox.textContent = "";
    reviewBox.classList.remove("blocking");
    show(reviewBox, false);
  }
  applyApprovalGate(job);
  renderProposalComparison(job);
  renderProposalTextEditor(job, selectedVariant);
  byId("narrative-order").innerHTML = narrativeMap.ordered_clips
    .map(
      (clip) => `
        <div class="clip-item">
          <span>${Number(clip.playback_position) + 1}. Clip ${Number(clip.clip_id)}</span>
          <strong>${escapeHtml(clip.role)}</strong>
        </div>
      `
    )
    .join("");
  byId("options-preview").textContent = JSON.stringify(job.editing_options, null, 2);
}

function renderCreativeZoomSpans(decision) {
  const spans = decision.creative_zoom_spans || [];
  if (spans.length) {
    return `
      <div class="effect-chip-list">
        ${spans
          .map(
            (span) =>
              `<strong class="effect-chip">Zoom creativo ${span.source_start.toFixed(2)}s-${span.source_end.toFixed(2)}s ${Math.round((span.end_scale - 1) * 100)}%</strong>`
          )
          .join("")}
      </div>
    `;
  }
  return decision.creative_zoom ? `<strong class="effect-chip">Zoom creativo</strong>` : "";
}

function escapeHtml(value) {
  // Tutto il testo proveniente da Ollama o dai nomi file attraversa questa
  // funzione prima di essere inserito nei template HTML.
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderDecisionText(decision) {
  let rendered = escapeHtml(decision.included_text || "");
  const spans = decision.creative_zoom_spans || [];
  spans.forEach((span) => {
    const spanText = escapeHtml(span.text || "");
    if (!spanText || !rendered.includes(spanText)) {
      return;
    }
    rendered = rendered.replace(
      spanText,
      `<mark class="zoom-creative-marker">zoom creativo</mark> ${spanText}`
    );
  });
  return rendered;
}

async function refreshJob() {
  if (!currentJob) {
    return;
  }
  const job = await readJson(await fetch(`/jobs/${currentJob.job_id}`));
  renderJob(job);
}

function connectEvents(jobId) {
  stopLiveUpdates();
  if (currentJob && terminalLiveStages.has(currentJob.stage)) {
    return;
  }
  const source = new EventSource(`/jobs/${jobId}/events`);
  eventSource = source;
  source.addEventListener("job", (event) => {
    const payload = JSON.parse(event.data);
    addEvent(payload);
    refreshJob().catch((err) => setError(err.message));
  });
  source.addEventListener("status", () => {
    refreshJob().catch((err) => setError(err.message));
  });
  source.onerror = () => {
    if (eventSource !== source) {
      return;
    }
    source.close();
    eventSource = null;
    if (currentJob && terminalLiveStages.has(currentJob.stage)) {
      return;
    }
    setError("SSE disconnesso: uso aggiornamento periodico.");
    // Un solo fallback per volta: riconnessioni o approvazioni successive non
    // devono accumulare interval che martellano l'API.
    if (pollingId === null) {
      pollingId = window.setInterval(() => refreshJob().catch((err) => setError(err.message)), 3000);
    }
  };
}

function stopLiveUpdates() {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
  if (pollingId !== null) {
    window.clearInterval(pollingId);
    pollingId = null;
  }
}

async function submitJob() {
  const files = Array.from(byId("clips-input").files || []);
  if (!files.length) {
    setError("Carica almeno una clip video.");
    return;
  }
  setError("");
  byId("analyze-button").disabled = true;
  byId("analyze-button").textContent = "Preparazione upload";

  try {
    const upload = await startUploadSession();
    for (const [index, file] of files.entries()) {
      await uploadFileInChunks(upload.upload_id, index, file, files.length);
    }
    byId("analyze-button").textContent = "Analisi in corso";
    const body = await createJobFromUpload(upload.upload_id);
    renderJob(body.job);
    addEvent(body.status);
    startTimers();
    connectEvents(body.job_id);
  } catch (err) {
    setError(err.message);
  } finally {
    byId("analyze-button").disabled = false;
    byId("analyze-button").textContent = "Analizza video";
  }
}

async function approveJob() {
  if (!currentJob) {
    return;
  }
  let selectedVariant = proposalVariantsForJob(currentJob).find((variant) => variant.id === selectedProposalVariantId);
  if (!selectedVariant?.selectable) {
    setError("La proposta selezionata non e renderizzabile. Scegli la proposta iniziale o ricalcola.");
    applyApprovalGate(currentJob);
    return;
  }
  applyApprovalGate(currentJob, true);
  try {
    await saveProposalTextEdits({ silent: true });
    selectedVariant = proposalVariantsForJob(currentJob).find((variant) => variant.id === selectedProposalVariantId);
    if (!selectedVariant?.selectable) {
      throw new Error("La proposta selezionata non e renderizzabile. Scegli la proposta iniziale o ricalcola.");
    }
    const body = await readJson(
      await fetch(`/jobs/${currentJob.job_id}/approve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ approved: true, proposal_variant_id: selectedProposalVariantId })
      })
    );
    addEvent(body.status);
    await refreshJob();
    connectEvents(currentJob.job_id);
  } catch (err) {
    setError(err.message);
  } finally {
    applyApprovalGate(currentJob);
  }
}

async function recalculateProposal() {
  if (!currentJob) {
    return;
  }
  const button = byId("recalculate-button");
  button.disabled = true;
  byId("approve-button").disabled = true;
  try {
    const body = await readJson(
      await fetch(`/jobs/${currentJob.job_id}/recalculate-proposal`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          synthesis_level: Number(byId("review-synthesis-input").value),
          target_duration_s: readDurationMinutesInputSeconds("review-target-duration-minutes-input")
        })
      })
    );
    reviewRecalculateControlsDirty = false;
    addEvent(body.status);
    renderJob(body.job);
  } catch (err) {
    setError(err.message);
  } finally {
    button.disabled = false;
    applyApprovalGate(currentJob);
  }
}

async function init() {
  updateWorkflow("setup");
  byId("clips-input").addEventListener("change", renderClipList);
  byId("synthesis-input").addEventListener("input", (event) => {
    byId("synthesis-label").textContent = event.target.value;
  });
  byId("review-synthesis-input").addEventListener("input", (event) => {
    reviewRecalculateControlsDirty = true;
    byId("review-synthesis-label").textContent = event.target.value;
  });
  byId("review-target-duration-minutes-input").addEventListener("input", () => {
    reviewRecalculateControlsDirty = true;
  });
  byId("music-input").addEventListener("change", (event) => {
    show(byId("music-prompt-field"), event.target.checked);
  });
  byId("transition-sounds-input").addEventListener("change", () => {
    renderTransitionSoundOptions();
  });
  byId("subtitles-input").addEventListener("change", updateSubtitleOptionsVisibility);
  byId("subtitle-style-select").addEventListener("change", renderSubtitleStylePreviews);
  byId("transition-sound-select").addEventListener("change", () => {
    byId("transition-sound-preview").pause();
  });
  byId("transition-sound-play-button").addEventListener("click", () => {
    previewTransitionSound().catch((err) => setError(err.message));
  });
  byId("analyze-button").addEventListener("click", submitJob);
  byId("approve-button").addEventListener("click", approveJob);
  byId("recalculate-button").addEventListener("click", recalculateProposal);
  byId("save-proposal-texts-button").addEventListener("click", () => {
    saveProposalTextEdits().catch((err) => setError(err.message));
  });
  document.querySelectorAll(".copy-prompt-button").forEach((button) => {
    button.addEventListener("click", () => {
      copyPromptToBrief(button).catch((err) => setError(err.message));
    });
  });
  try {
    await loadOptions();
    await loadModels();
  } catch (err) {
    setError(err.message);
  }
}

init();
