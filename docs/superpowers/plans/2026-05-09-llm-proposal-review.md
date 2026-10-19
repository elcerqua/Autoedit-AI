# LLM Proposal Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an LLM review pass over the actual phase-one edit proposal before the job can enter `waiting_for_approval`.

**Architecture:** Keep the LLM as editorial authority and deterministic modules as executors. The first LLM pass creates a `NarrativeMap`, deterministic code builds the `EditDecisionList`, then a second structured LLM pass validates the proposal text/EDL and may return a corrected `NarrativeMap`; the pipeline rebuilds the EDL from that map.

**Tech Stack:** FastAPI, Pydantic v2, Ollama structured output, pytest, loguru.

---

### Task 1: Schemas And Reviewer Contract

**Files:**
- Modify: `autocut_ai/schemas.py`
- Create: `autocut_ai/proposal_review.py`
- Test: `tests/test_proposal_review.py`

- [ ] Add Pydantic models `ProposalIssue` and `ProposalReview`.
- [ ] Add a reviewer interface class with `review(transcriptions, narrative_map, edit_decision_list, brief, model) -> ProposalReview`.
- [ ] Implement `MockProposalReviewer` that approves unchanged proposals.

### Task 2: Ollama Structured Review

**Files:**
- Modify: `autocut_ai/proposal_review.py`
- Test: `tests/test_proposal_review.py`

- [ ] Build a prompt containing the source transcript, current `NarrativeMap`, and current EDL preview.
- [ ] Call `ollama.chat(..., format=ProposalReview.model_json_schema())`.
- [ ] Validate any `corrected_narrative_map` with existing `validate_narrative_map`.
- [ ] Retry on schema or narrative validation failure with feedback.

### Task 3: Pipeline Integration

**Files:**
- Modify: `autocut_ai/pipeline.py`
- Modify: `autocut_ai/app.py`
- Test: `tests/test_pipeline_proposal_review.py`

- [ ] Inject the proposal reviewer into `PipelineRunner`.
- [ ] After building the initial EDL, call the reviewer.
- [ ] If approved without correction, keep current proposal.
- [ ] If corrected, validate corrected map and rebuild the EDL.
- [ ] If not approved and not corrected, fail the job before `waiting_for_approval`.

### Task 4: Verification

**Files:**
- Test: `tests/test_pipeline_proposal_review.py`
- Test: full suite

- [ ] Run the reviewer unit tests.
- [ ] Run pipeline review tests.
- [ ] Run `pytest -q`.
