import os
import tempfile
import uuid

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

from graph.build_graph import run_workflow

app = FastAPI(
    title="Meeting Intelligence Agent API",
    description="Transcribes a meeting recording and extracts action items, decisions, and a follow-up email draft",
    version="1.0.0",
)

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".mp4", ".webm", ".ogg"}


class ActionItemResponse(BaseModel):
    description: str
    owner: str
    deadline: str
    confidence: float


class StepMetricResponse(BaseModel):
    agent: str
    latency_seconds: float
    prompt_tokens: int
    completion_tokens: int
    tool_status: str


class MeetingResponse(BaseModel):
    transcript: str
    action_items: list[ActionItemResponse]
    decisions: list[str]
    key_points: list[str]
    follow_up_email: str
    needs_review: bool
    review_reason: str
    agent_trace: list[str]
    iterations: int
    step_metrics: list[StepMetricResponse]


@app.get("/")
def root():
    return {"status": "ok", "service": "meeting-intelligence-agent"}


@app.post("/run", response_model=MeetingResponse)
async def run(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, f"meeting_{uuid.uuid4().hex}{ext}")

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        with open(tmp_path, "wb") as f:
            f.write(contents)

        result = run_workflow(tmp_path)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow failed: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return MeetingResponse(
        transcript=result.get("transcript", ""),
        action_items=result.get("action_items", []),
        decisions=result.get("decisions", []),
        key_points=result.get("key_points", []),
        follow_up_email=result.get("follow_up_email", ""),
        needs_review=result.get("needs_review", False),
        review_reason=result.get("review_reason", ""),
        agent_trace=result.get("agent_trace", []),
        iterations=result.get("iterations", 0),
        step_metrics=result.get("step_metrics", []),
    )
