
from typing import TypedDict


class ActionItem(TypedDict):
    description: str
    owner: str
    deadline: str
    confidence: float


class StepMetric(TypedDict):
    agent: str
    latency_seconds: float
    prompt_tokens: int
    completion_tokens: int
    tool_status: str


class AgentState(TypedDict):
    audio_path: str
    transcript: str
    action_items: list[ActionItem]
    decisions: list[str]
    key_points: list[str]
    follow_up_email: str
    needs_review: bool
    review_reason: str
    next_agent: str
    agent_trace: list[str]
    iterations: int
    step_metrics: list[StepMetric]
