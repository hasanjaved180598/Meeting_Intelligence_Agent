# 🎙️ Meeting Intelligence Agent

A multi-agent pipeline that turns a raw meeting recording into structured, actionable output — action items, decisions, key discussion points, and a drafted follow-up email — entirely offline. No API keys, no per-minute transcription cost, no audio ever leaves the machine.

## 📋 Table of Contents

- [What is Meeting Intelligence?](#what-is-meeting-intelligence)
- [Why a Confidence Gate, Not Just a Summary?](#why-a-confidence-gate-not-just-a-summary)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Key Concepts](#key-concepts)
- [API Endpoints](#api-endpoints)
- [Results](#results)
- [Real Debugging Journey](#real-debugging-journey)
- [Demo](#demo)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)
- [Project Workflow](#project-workflow)
- [Limitations](#limitations)

## 🎙️ What is Meeting Intelligence?

Most "AI meeting notes" tools do one thing: transcribe, then summarize. This project goes a step further — it extracts *structured, validated* data (who owns what, by when) rather than another paragraph of prose, and it explicitly tracks how confident it is about each piece.

| Task | Typical AI note-taker | This project |
|------|----------------------|--------------|
| Output | Free-text summary | Structured JSON: action items, owners, deadlines, confidence scores |
| Uncertainty | Presented as fact either way | Flagged for human review when confidence is low |
| Where it runs | Cloud API, per-minute cost | Fully local — Whisper + Ollama, zero cost |

**Applications:** internal meeting follow-ups, consulting/client call summaries, support call review, any workflow where "who's doing what by when" needs to come out of a conversation reliably.

## 🤖 Why a Confidence Gate, Not Just a Summary?

Every "AI extracts action items" demo has the same failure mode: it presents everything with equal confidence, whether the speaker clearly assigned a task or just vaguely mentioned something might need doing. That's fine for a demo; it's a liability if a business actually trusts the output.

This pipeline's **Reviewer agent** is the difference. It doesn't just extract — it checks the Extractor's own confidence scores against a threshold, and if anything falls short, the whole result gets flagged for human review with a specific, stated reason, instead of quietly presenting a guess as a fact.

### Confidence Calibration

The first version of the Extractor's confidence scoring was too generous — an item where the speaker explicitly said *"I'm not sure who should own that"* still scored 0.8 (high confidence). The prompt was rewritten with explicit calibration examples:

```python
"CONFIDENCE CALIBRATION — be strict, not generous:\n"
"- 0.9-1.0: the speaker explicitly assigns a named owner AND a "
"specific deadline with no hedging...\n"
"- 0.2-0.4: the speaker explicitly voices uncertainty about who "
"owns it or whether/when it will happen (e.g. 'I'm not sure who "
"should own that')..."
```

After this fix, the same test case correctly dropped that item to 0.2 confidence and triggered the review flag. See [Real Debugging Journey](#real-debugging-journey) for the full before/after.

## 📁 Project Structure

```
Meeting_Intelligence_Agent/
├── agents/
│   ├── tools.py       # transcribe_audio() — local Whisper wrapper
│   └── nodes.py        # supervisor, transcriber, extractor, drafter, reviewer
├── graph/
│   ├── state.py         # AgentState TypedDict
│   └── build_graph.py   # StateGraph wiring + run_workflow(audio_path)
├── api/
│   └── main.py            # FastAPI /run endpoint (file upload)
├── app/
│   └── app.py               # Streamlit UI
├── requirements.txt
└── README.md
```

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Language | Python | Core language |
| Speech-to-text | OpenAI Whisper (local) | Fully offline audio transcription |
| Orchestration | LangGraph | Supervisor/Worker StateGraph with conditional routing |
| LLM | LangChain + ChatOllama (LLaMA 3.1, local) | Extraction, drafting, and reasoning |
| Validation | Pydantic | Structured, validated extraction output |
| API Framework | FastAPI | REST API with audio file upload |
| Dashboard | Streamlit | Upload UI + action items, decisions, email draft, review flag |

Every component runs entirely on-device. No API keys, no per-request cost, no data leaves the machine.

## 🔑 Key Concepts

### Structured Output, Not Free Text

The Extractor is prompted to return strict JSON and its response is validated against a Pydantic schema before being trusted:

```python
class ExtractionResult(BaseModel):
    action_items: list[dict] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    key_points: list[str] = Field(default_factory=list)
```

If the model's output doesn't parse or validate, that's treated as a failure and routed to the Reviewer's "needs review" flag — never silently guessed at.

### Precise Owner Attribution

An early version occasionally attributed a task to the wrong person simply because their name appeared elsewhere in the transcript. The prompt now explicitly constrains attribution:

```python
"Only assign a person as owner of an action item if they are "
"named in the SAME sentence or the sentence immediately "
"describing that specific task. Do not assign a name to an "
"action item just because that person was mentioned earlier or "
"later in the transcript for a different task."
```

### Human-in-the-Loop by Design

```python
needs_review = bool(low_confidence_items) or transcript_failed or extraction_empty
```

The Reviewer node doesn't try to fix low-confidence data — it surfaces it, with a specific stated reason, and lets a human make the final call. This is a deliberate design choice: an agent that knows what it doesn't know is more trustworthy than one that always sounds certain.

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|--------------|
| GET | `/` | Root — confirms API is running |
| POST | `/run` | Accepts an uploaded audio file, runs the full pipeline |

### Example — Run Request

```python
import requests

with open("meeting.mp3", "rb") as f:
    response = requests.post(
        "http://localhost:8000/run",
        files={"file": ("meeting.mp3", f)}
    )
print(response.json())
```

### Example Response

```json
{
  "transcript": "Alright, let's get started...",
  "action_items": [
    {"description": "Marketing email campaign", "owner": "Sara", "deadline": "this Friday", "confidence": 0.9},
    {"description": "Follow up with vendor about updated pricing", "owner": "unassigned", "deadline": "not specified", "confidence": 0.2}
  ],
  "decisions": ["Launch the new app update"],
  "key_points": ["Server costs have been creeping up"],
  "follow_up_email": "Subject: Meeting Follow-up...",
  "needs_review": true,
  "review_reason": "2 action item(s) extracted with confidence below 0.85 — verify before acting on them.",
  "iterations": 4
}
```

## 📊 Results

A confirmed end-to-end run on a real ~60-second recorded test meeting (containing one clear decision, one clear action item, one explicitly ambiguous action item, and one vague future consideration) correctly:

- Extracted all 4 items with the right classification (action item vs. decision vs. key point)
- Assigned the correct owner to the correct task after an attribution fix
- Scored the clear item at 90% confidence and both ambiguous items at 20%
- Correctly triggered the human-review flag with a specific, accurate reason

```
Supervisor → transcriber
Transcriber: transcription ok
Supervisor → extractor
Extractor: extraction ok (4 action items)
Supervisor → drafter
Drafter: follow-up email drafted
Supervisor → reviewer
Reviewer: needs_review=True
Supervisor → end
```

Total processing time: ~216 seconds on local CPU inference (Whisper "base" + LLaMA 3.1 8B via Ollama) for a 60-second recording — three sequential LLM calls plus transcription, with no GPU acceleration.

## 🐛 Real Debugging Journey

### Confidence scores were too generous

First test run: a transcript segment where the speaker said *"I'm honestly not sure who should own that yet"* still scored **0.8 confidence** — high enough to pass a 0.7 threshold silently, defeating the entire point of the review gate.

**Root cause:** the original prompt asked for confidence scoring but gave no concrete calibration — the model defaulted to a comfortable, generically high number rather than genuinely reasoning about the hedging language in the transcript.

**Fix:** rewrote the prompt with explicit confidence bands and worked examples of what hedging language should score, and raised the threshold from 0.7 to 0.85. Re-running the identical test recording after the fix correctly dropped the same item to 0.2 confidence and triggered the review flag.

**Lesson:** LLM self-reported confidence is not calibrated out of the box — it has to be explicitly taught what "low confidence" looks like with concrete examples, the same way you'd onboard a new team member on your team's specific bar for "verify before you act on this."

### Owner attribution bled across sentences

A related issue surfaced in the same test: the model assigned "Sara" as the owner of the app-launch decision (a task nobody was actually assigned to) instead of the marketing campaign she was actually asked to own two sentences later.

**Root cause:** the prompt only said *"set owner if a person is named"* without constraining *which* task that name applies to, so the model associated the nearest name in the transcript loosely rather than precisely per-sentence.

**Fix:** added an explicit instruction requiring a name to appear in the same or immediately adjacent sentence as the specific task it's being attributed to. Re-testing confirmed both the launch item (correctly unassigned) and the marketing item (correctly assigned to Sara) resolved correctly.

**Lesson:** extraction accuracy issues can hide behind "looks right at a glance" — this only surfaced by manually checking the extracted email draft, decisions, and transcript alignment against actual test data, not by trusting the first successful-looking run.

## 🎬 Demo

https://github.com/user-attachments/assets/4765233d-3285-4230-85f4-681e6295423b

📹 **[Watch the full walkthrough on Google Drive](https://drive.google.com/file/d/1BYszjtFC3HRcLqkQK9dXfP-umLc6LzfZ/view?usp=sharing)**

## ⚙️ Setup and Installation

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed locally
- [ffmpeg](https://ffmpeg.org) installed and available on your system PATH (required by Whisper to decode audio)

### Steps

```bash
git clone https://github.com/hasanjaved180598/Meeting_Intelligence_Agent.git
cd Meeting_Intelligence_Agent

python -m venv .venv
.venv\Scripts\activate          # Windows

pip install -r requirements.txt

ollama pull llama3.1
```

Whisper's model downloads automatically on first run (the "base" model, ~150MB) — no manual model download step required.

## 🚀 How to Run

**1. Start the API (terminal 1):**
```bash
uvicorn api.main:app --reload --port 8000
```

**2. Start the dashboard (terminal 2):**
```bash
streamlit run app/app.py
```

Open the Streamlit URL, upload a meeting recording (mp3/wav/m4a/mp4/webm/ogg), and click **Analyze Meeting**.

## 🔄 Project Workflow

```
Uploaded audio file (Streamlit)
      │
      ▼
FastAPI /run endpoint
      │
      ▼
Supervisor node (reads AgentState)
      │
      ▼
Transcriber (local Whisper) → transcript
      │
      ▼
Supervisor (routes again)
      │
      ▼
Extractor (structured JSON, Pydantic-validated,
calibrated confidence scoring) → action items, decisions, key points
      │
      ▼
Supervisor (routes again)
      │
      ▼
Drafter (synthesizes everything) → follow_up_email
      │
      ▼
Supervisor (routes again)
      │
      ▼
Reviewer (confidence gate) → needs_review + reason
      │
      ▼
Supervisor → END
      │
      ▼
Streamlit dashboard shows email draft, action items
(color-coded by confidence), decisions, key points, and review banner
```

## ⚠️ Limitations

- Local Whisper and LLaMA 3.1 inference speed depends entirely on hardware — CPU-only setups process longer recordings noticeably slower than GPU-backed ones (~216s for a 60-second clip in local testing). Keep test recordings short for fast iteration.
- Confidence scores are self-reported by the LLM based on prompt-level calibration, not an independently verified statistical measure — a genuinely useful heuristic after tuning, but not a mathematically calibrated probability.
- No speaker diarization (identifying *who* said what by voice) — action item owners come from what the transcript content states, not voice identification, so a recording where names aren't spoken aloud will produce more "unassigned" owners.
- Tested primarily on short, single-speaker-style recordings; longer or heavily cross-talking multi-speaker meetings have not yet been validated against the same calibration.

## 📜 License

This project is licensed under the MIT License.

## 👤 Author

**Hasan Javed** — Flutter Developer transitioning into AI/ML Engineering
[GitHub](https://github.com/hasanjaved180598) · [LinkedIn](https://linkedin.com/in/hasanjaved1/)

---

*It doesn't just take notes — it tells you which ones to double-check. 🎙️✅*
