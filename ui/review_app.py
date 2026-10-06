from __future__ import annotations

import json
from typing import Iterable, Iterator

import httpx
import streamlit as st

from app.config import DEFAULT_DRUG_QUERY


STATUS_LABELS = {
    "running": "Running",
    "pending_review": "Pending review",
    "completed": "Completed",
    "needs_manual_review": "Needs manual review",
    "rejected": "Rejected",
    "failed": "Failed",
}

NODE_LABELS = {
    "normalize_node": "Resolving drug name",
    "retrieve_node": "Retrieving evidence",
    "grade_node": "Assessing evidence sufficiency",
    "rewrite_node": "Refining search query",
    "web_fallback_node": "Searching approved sources",
    "safety_audit_node": "Auditing safety findings",
    "generator_node": "Drafting evidence brief",
    "critic_node": "Checking citations and claims",
    "human_review_node": "Waiting for human review",
    "finalize_node": "Finalizing approved brief",
    "refine_node": "Preparing requested revision",
}


def normalize_api_url(value: str) -> str:
    """Normalize a user-entered API origin for endpoint requests."""
    return value.strip().rstrip("/")


def parse_sse(lines: Iterable[str]) -> Iterator[tuple[str, str]]:
    """Yield event names and joined data payloads from Server-Sent Event lines."""
    event_name = "message"
    data_lines: list[str] = []

    for line in lines:
        if not line:
            if data_lines:
                yield event_name, "\n".join(data_lines)
            event_name = "message"
            data_lines = []
        elif line.startswith("event:"):
            event_name = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").lstrip())

    if data_lines:
        yield event_name, "\n".join(data_lines)


def _request_json(method: str, url: str, **kwargs) -> dict:
    response = httpx.request(method, url, timeout=20, **kwargs)
    response.raise_for_status()
    return response.json()


def _follow_run(api_url: str, thread_id: str) -> None:
    """Follow graph node events and render progress while the API streams."""
    stream_url = f"{api_url}/runs/{thread_id}/stream"
    try:
        with st.status("Connecting to run...", expanded=True) as progress:
            with httpx.stream("GET", stream_url, timeout=300) as response:
                response.raise_for_status()
                for event_name, data in parse_sse(response.iter_lines()):
                    if event_name == "progress":
                        updates = json.loads(data)
                        for node_name in updates:
                            st.write(NODE_LABELS.get(node_name, node_name))
                    elif event_name == "status":
                        status = json.loads(data).get("status", "running")
                        progress.update(
                            label=f"Run {STATUS_LABELS.get(status, status.lower())}",
                            state="complete" if status != "failed" else "error",
                        )
            progress.update(label="Progress stream closed", state="complete")
    except httpx.HTTPError as error:
        st.error(f"Could not follow run progress: {error}")


def _load_run(api_url: str, thread_id: str) -> dict | None:
    try:
        return _request_json("GET", f"{api_url}/runs/{thread_id}")
    except httpx.HTTPStatusError as error:
        st.error(f"Run status request failed: {error.response.text}")
    except httpx.HTTPError as error:
        st.error(f"Could not reach the API: {error}")
    return None


def _render_findings(findings: list[dict]) -> None:
    if not findings:
        st.caption("No safety findings were returned.")
        return
    st.dataframe(findings, hide_index=True, use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="Evidence Review",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Source+Serif+4:wght@500;600&display=swap');
        :root {
          --ink: #18332e;
          --muted: #65756f;
          --line: #dce5df;
          --paper: #f5f8f5;
          --accent: #167c68;
          --warning: #b35f32;
        }
        html, body, [class*="st-"] { font-family: 'DM Sans', sans-serif; }
        [data-testid="stAppViewContainer"] {
          background: linear-gradient(180deg, #f0f6f2 0, #f7f8f6 17rem, #f7f8f6 100%);
          color: var(--ink);
        }
        [data-testid="stSidebar"] {
          background: #edf3ee;
          border-right: 1px solid var(--line);
        }
        h1, h2, h3 { color: var(--ink); font-family: 'Source Serif 4', Georgia, serif; }
        h1 { font-size: 2rem; }
        [data-testid="stMetricValue"] { color: var(--ink); }
        [data-testid="stButton"] button[kind="primary"] {
          background: var(--accent); border-color: var(--accent);
        }
        [data-testid="stForm"] { border-color: var(--line); background: #fff; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "api_url" not in st.session_state:
        st.session_state.api_url = "http://127.0.0.1:8000"
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = ""

    with st.sidebar:
        st.markdown("### Review setup")
        api_url_input = st.text_input("API address", key="api_url")
        api_url = normalize_api_url(api_url_input)
        drug_query = st.text_input("Drug or ingredient", value=DEFAULT_DRUG_QUERY)
        start_review = st.button(
            "Start review", type="primary", icon=":material/play_arrow:", use_container_width=True
        )
        if st.session_state.thread_id:
            st.divider()
            st.caption("Active run")
            st.code(st.session_state.thread_id, language=None)
            if st.button("New review", icon=":material/add:", use_container_width=True):
                st.session_state.thread_id = ""
                st.rerun()

    st.title("Evidence review")
    st.caption("Literature and label evidence for human assessment")

    if start_review:
        try:
            started = _request_json(
                "POST",
                f"{api_url}/runs",
                json={"drug_query": drug_query.strip()},
            )
            st.session_state.thread_id = started["thread_id"]
            _follow_run(api_url, started["thread_id"])
            st.rerun()
        except httpx.HTTPError as error:
            st.error(f"Could not start review: {error}")

    thread_id = st.session_state.thread_id
    if not thread_id:
        st.info("Start a review to retrieve and assess evidence.")
        return

    run = _load_run(api_url, thread_id)
    if run is None:
        return

    state = run.get("state", {})
    status = run.get("status", "running")
    metric_columns = st.columns([1, 1, 2])
    metric_columns[0].metric("Run status", STATUS_LABELS.get(status, status))
    metric_columns[1].metric("Risk level", state.get("risk_level", "Not assessed"))
    metric_columns[2].metric("Canonical name", state.get("canonical_name", "Resolving"))

    if status == "running" and st.button(
        "Follow progress", icon=":material/monitoring:"
    ):
        _follow_run(api_url, thread_id)
        st.rerun()

    if state.get("regulatory_brief"):
        st.markdown("---")
        st.markdown(state["regulatory_brief"], unsafe_allow_html=False)

    with st.expander("Safety findings", expanded=status == "pending_review"):
        _render_findings(state.get("safety_findings", []))

    if status == "pending_review":
        st.markdown("---")
        st.subheader("Human review")
        with st.form("human_review"):
            decision_label = st.radio(
                "Decision",
                ["Approve", "Request edits", "Reject"],
                horizontal=True,
                label_visibility="collapsed",
            )
            feedback = st.text_area(
                "Review comment",
                placeholder="Add context for the decision or describe the changes needed.",
                height=100,
            )
            edited_brief = None
            if decision_label == "Request edits":
                edited_brief = st.text_area(
                    "Replacement brief (optional)",
                    placeholder="Provide a complete replacement for the generated brief.",
                    height=180,
                )
            submitted = st.form_submit_button(
                "Submit review", type="primary", icon=":material/send:"
            )

        if submitted:
            if decision_label == "Request edits" and not feedback.strip() and not (
                edited_brief or ""
            ).strip():
                st.error("Add a review comment or replacement brief before requesting edits.")
            else:
                payload = {
                    "decision": {
                        "Approve": "approve",
                        "Request edits": "edit",
                        "Reject": "reject",
                    }[decision_label],
                    "feedback": feedback,
                    "edited_brief": edited_brief or None,
                }
                try:
                    _request_json(
                        "POST", f"{api_url}/runs/{thread_id}/resume", json=payload
                    )
                    _follow_run(api_url, thread_id)
                    st.rerun()
                except httpx.HTTPError as error:
                    st.error(f"Could not submit review: {error}")
    elif status == "needs_manual_review":
        st.warning("The human review round limit was reached. Manual follow-up is required.")
    elif status == "rejected":
        st.warning("This brief was rejected and the run is closed.")
    elif status == "failed":
        st.error(state.get("error", "The workflow failed."))
    elif status == "completed":
        st.success("Review completed.")


if __name__ == "__main__":
    main()