
from langgraph.graph import StateGraph, END

from graph.state import AgentState
from agents.nodes import (
    supervisor_node,
    transcriber_node,
    extractor_node,
    drafter_node,
    reviewer_node,
)


def _route_from_supervisor(state: AgentState) -> str:
    return state.get("next_agent", "end")


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("transcriber", transcriber_node)
    graph.add_node("extractor", extractor_node)
    graph.add_node("drafter", drafter_node)
    graph.add_node("reviewer", reviewer_node)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        _route_from_supervisor,
        {
            "transcriber": "transcriber",
            "extractor": "extractor",
            "drafter": "drafter",
            "reviewer": "reviewer",
            "end": END,
        },
    )

    graph.add_edge("transcriber", "supervisor")
    graph.add_edge("extractor", "supervisor")
    graph.add_edge("drafter", "supervisor")
    graph.add_edge("reviewer", "supervisor")

    return graph.compile()


_compiled_graph = None


def run_workflow(audio_path: str) -> AgentState:
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()

    initial_state: AgentState = {
        "audio_path": audio_path,
        "transcript": "",
        "action_items": [],
        "decisions": [],
        "key_points": [],
        "follow_up_email": "",
        "needs_review": False,
        "review_reason": "",
        "next_agent": "",
        "agent_trace": [],
        "iterations": 0,
        "step_metrics": [],
    }

    return _compiled_graph.invoke(initial_state)
