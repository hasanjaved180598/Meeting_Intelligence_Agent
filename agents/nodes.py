import time
import json

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.tools import transcribe_audio
from graph.state import AgentState, ActionItem, StepMetric

llm = ChatOllama(model="llama3.1", temperature=0.2)

MAX_ITERATIONS = 10
CONFIDENCE_THRESHOLD = 0.85


class ExtractionResult(BaseModel):
    action_items: list[dict] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    key_points: list[str] = Field(default_factory=list)


def _record_metric(state: AgentState, agent: str, start_time: float, response, tool_status: str = "not_used"):
    latency = time.monotonic() - start_time
    usage = getattr(response, "usage_metadata", None) or {}
    metric: StepMetric = {
        "agent": agent,
        "latency_seconds": round(latency, 2),
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "tool_status": tool_status,
    }
    state.setdefault("step_metrics", []).append(metric)


def supervisor_node(state: AgentState) -> AgentState:

    trace = state.get("agent_trace", [])
    iterations = state.get("iterations", 0) + 1

    if iterations > MAX_ITERATIONS:
        next_agent = "end"
    elif not state.get("transcript"):
        next_agent = "transcriber"
    elif not state.get("action_items") and not state.get("decisions") and not state.get("key_points"):
        next_agent = "extractor"
    elif not state.get("follow_up_email"):
        next_agent = "drafter"
    elif "needs_review" not in state or state.get("review_reason", "") == "":
        next_agent = "reviewer"
    else:
        next_agent = "end"

    trace.append(f"Supervisor -> {next_agent}")
    return {**state, "next_agent": next_agent, "agent_trace": trace, "iterations": iterations}


def transcriber_node(state: AgentState) -> AgentState:
    start = time.monotonic()
    tool_status = "ok"
    try:
        text = transcribe_audio(state["audio_path"])
    except Exception as e:
        text = f"[TRANSCRIPTION FAILED: {e}]"
        tool_status = "failed"

    trace = state.get("agent_trace", [])
    trace.append(f"Transcriber: transcription {tool_status}")


    class _NoLLMResponse:
        usage_metadata = {}
    _record_metric(state, "transcriber", start, _NoLLMResponse(), tool_status)

    return {**state, "transcript": text, "agent_trace": trace}


def extractor_node(state: AgentState) -> AgentState:
    transcript = state.get("transcript", "")

    start = time.monotonic()
    messages = [
        SystemMessage(content=(
            "You extract structured information from a meeting transcript. "
            "Respond with ONLY valid JSON, no other text, in exactly this shape:\n"
            '{"action_items": [{"description": "...", "owner": "...", '
            '"deadline": "...", "confidence": 0.0}], "decisions": ["..."], '
            '"key_points": ["..."]}\n\n'
            "For each action item, set owner to \"unassigned\" if no person "
            "is named, and deadline to \"not specified\" if none is given.\n\n"
            "CONFIDENCE CALIBRATION — be strict, not generous:\n"
            "- 0.9-1.0: the speaker explicitly assigns a named owner AND a "
            "specific deadline with no hedging (e.g. 'Sarah will send the "
            "report by Friday').\n"
            "- 0.5-0.7: an owner or deadline is stated but incomplete, or "
            "phrased with mild uncertainty (e.g. 'someone should probably "
            "handle this soon').\n"
            "- 0.2-0.4: the speaker explicitly voices uncertainty about who "
            "owns it or whether/when it will happen (e.g. 'I'm not sure who "
            "should own that', 'we haven't really discussed timelines', "
            "'we should figure that out separately'). Phrases like these are "
            "a strong signal for LOW confidence, not moderate — do not treat "
            "'we should follow up on X' as high-confidence just because an "
            "action is named; the uncertainty is about ownership/timing, "
            "which matters just as much as the action itself.\n"
            "- Do not default to a comfortable middle number. If the "
            "transcript contains hedging language, confidence must reflect "
            "that clearly, even if it means most items in a messy real "
            "meeting score below 0.5.\n\n"
            "OWNER ATTRIBUTION — be precise, not approximate:\n"
            "Only assign a person as owner of an action item if they are "
            "named in the SAME sentence or the sentence immediately "
            "describing that specific task. Do not assign a name to an "
            "action item just because that person was mentioned earlier or "
            "later in the transcript for a different task. If a decision "
            "(like 'we decided to launch X') has no individual named as "
            "responsible for executing it, its owner is \"unassigned\" — "
            "even if someone else's name appears nearby in the transcript "
            "for an unrelated action."
        )),
        HumanMessage(content=f"Transcript:\n{transcript}"),
    ]
    response = llm.invoke(messages)
    _record_metric(state, "extractor", start, response)

    tool_status = "ok"
    action_items: list[ActionItem] = []
    decisions: list[str] = []
    key_points: list[str] = []

    try:
        raw = response.content.strip()

        if raw.startswith("```"):
            raw = raw.strip("`").replace("json", "", 1).strip()
        parsed = ExtractionResult.model_validate_json(raw)
        action_items = [
            {
                "description": item.get("description", ""),
                "owner": item.get("owner", "unassigned"),
                "deadline": item.get("deadline", "not specified"),
                "confidence": float(item.get("confidence", 0.5)),
            }
            for item in parsed.action_items
        ]
        decisions = parsed.decisions
        key_points = parsed.key_points
    except Exception:
        tool_status = "failed"

    trace = state.get("agent_trace", [])
    trace.append(f"Extractor: extraction {tool_status} ({len(action_items)} action items)")

    return {
        **state,
        "action_items": action_items,
        "decisions": decisions,
        "key_points": key_points,
        "agent_trace": trace,
    }


def drafter_node(state: AgentState) -> AgentState:
    action_items = state.get("action_items", [])
    decisions = state.get("decisions", [])
    key_points = state.get("key_points", [])

    start = time.monotonic()
    messages = [
        SystemMessage(content=(
            "You are drafting a professional follow-up email after a "
            "meeting. Summarize decisions made, list action items with "
            "owners and deadlines where known, and note key discussion "
            "points briefly. Keep it concise and easy to scan."
        )),
        HumanMessage(content=(
            f"Action items:\n{json.dumps(action_items, indent=2)}\n\n"
            f"Decisions:\n{json.dumps(decisions, indent=2)}\n\n"
            f"Key points:\n{json.dumps(key_points, indent=2)}"
        )),
    ]
    response = llm.invoke(messages)
    _record_metric(state, "drafter", start, response)

    trace = state.get("agent_trace", [])
    trace.append("Drafter: follow-up email drafted")

    return {**state, "follow_up_email": response.content, "agent_trace": trace}


def reviewer_node(state: AgentState) -> AgentState:

    action_items = state.get("action_items", [])
    low_confidence_items = [
        item for item in action_items if item.get("confidence", 1.0) < CONFIDENCE_THRESHOLD
    ]

    transcript_failed = state.get("transcript", "").startswith("[TRANSCRIPTION FAILED")
    extraction_empty = not action_items and not state.get("decisions") and not state.get("key_points")

    needs_review = bool(low_confidence_items) or transcript_failed or extraction_empty

    if transcript_failed:
        reason = "Audio transcription failed — output is unreliable."
    elif extraction_empty:
        reason = "No action items, decisions, or key points were extracted — transcript may be unclear."
    elif low_confidence_items:
        reason = (
            f"{len(low_confidence_items)} action item(s) extracted with confidence "
            f"below {CONFIDENCE_THRESHOLD} — verify before acting on them."
        )
    else:
        reason = "none"

    trace = state.get("agent_trace", [])
    trace.append(f"Reviewer: needs_review={needs_review}")

    return {**state, "needs_review": needs_review, "review_reason": reason, "agent_trace": trace}
