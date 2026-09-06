
import requests
import streamlit as st

API_URL = "http://localhost:8000/run"

st.set_page_config(page_title="Meeting Intelligence Agent", page_icon="🎙️", layout="wide")

st.title("🎙️ Meeting Intelligence Agent")
st.caption("Upload a meeting recording → get action items, decisions, and a drafted follow-up email — fully offline.")

with st.sidebar:
    st.header("About")
    st.write(
        "A multi-agent pipeline (Transcriber → Extractor → Drafter → Reviewer) "
        "turns a raw meeting recording into structured, actionable output. "
        "If the Extractor isn't confident about something, the Reviewer flags "
        "it for human review instead of presenting it as reliable."
    )
    st.markdown("**Agents:** Supervisor · Transcriber · Extractor · Drafter · Reviewer")
    st.markdown("**Stack:** Whisper (local) · LangGraph · LLaMA3.1 (local) · FastAPI · Streamlit")
    st.markdown("**Cost:** $0 — everything runs on-device, no API keys")

uploaded_file = st.file_uploader(
    "Upload a meeting recording",
    type=["mp3", "wav", "m4a", "mp4", "webm", "ogg"],
)

run_clicked = st.button("Analyze Meeting", type="primary")

if run_clicked:
    if uploaded_file is None:
        st.warning("Please upload an audio file first.")
    else:
        with st.spinner("Transcribing and analyzing — this can take a few minutes on local hardware..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                resp = requests.post(API_URL, files=files, timeout=1800)
                resp.raise_for_status()
                data = resp.json()
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach the API: {e}")
                data = None

        if data:
            if data.get("needs_review"):
                st.warning(f"⚠️ Needs human review: {data['review_reason']}")
            else:
                st.success("✅ Processed with high confidence — no review needed")

            st.subheader("📧 Follow-Up Email Draft")
            st.markdown(data["follow_up_email"] or "_No draft generated._")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("✅ Action Items")
                if data["action_items"]:
                    for item in data["action_items"]:
                        conf = item["confidence"]
                        flag = "🔴" if conf < 0.7 else "🟢"
                        st.markdown(
                            f"{flag} **{item['description']}**  \n"
                            f"Owner: {item['owner']} · Deadline: {item['deadline']} · "
                            f"Confidence: {conf:.0%}"
                        )
                else:
                    st.write("_None extracted._")

            with col2:
                st.subheader("📌 Decisions")
                if data["decisions"]:
                    for d in data["decisions"]:
                        st.markdown(f"- {d}")
                else:
                    st.write("_None extracted._")

                st.subheader("🔑 Key Points")
                if data["key_points"]:
                    for k in data["key_points"]:
                        st.markdown(f"- {k}")
                else:
                    st.write("_None extracted._")

            with st.expander("📝 Full Transcript"):
                st.write(data["transcript"] or "_None_")

            with st.expander("🧭 Agent Trace"):
                for i, step in enumerate(data["agent_trace"], start=1):
                    st.write(f"{i}. {step}")

            with st.expander("⚙️ Performance"):
                metrics = data.get("step_metrics", [])
                if metrics:
                    total_latency = sum(m["latency_seconds"] for m in metrics)
                    st.metric("Total processing time", f"{total_latency:.1f}s")
                    st.table(metrics)
                else:
                    st.write("_No metrics recorded._")
