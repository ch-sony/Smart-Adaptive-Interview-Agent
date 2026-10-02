# SMART ADAPTIVE INTERVIEW AGENT

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2+-1C3C3C?style=flat&logo=chainlink)](https://github.com/langchain-ai/langgraph)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-gemini--3.5--flash--lite-4285F4?style=flat&logo=google)](https://ai.google.dev/)
[![Render Compatible](https://img.shields.io/badge/Deploy-Render%20Compatible-46E3B7?style=flat&logo=render)](https://render.com)

An AI-powered adaptive technical interview platform. Unlike traditional question-and-answer chatbots with fixed question sequences, this agent **dynamically evaluates each response in real time, diagnoses candidate strengths and weaknesses, adjusts question difficulty up or down, and decides the next topic to explore.**

---

## 1. Problem Statement

Standard mock interview platforms and chatbots suffer from key structural flaws:
1. **Static Question Sequences**: They ask a pre-scripted list of questions regardless of whether the candidate finds them trivially easy or overwhelmingly difficult.
2. **No Adaptive Feedback Loop**: They do not dynamically pivot to probe identified technical weaknesses or escalate to deeper architectural scenarios when a candidate demonstrates mastery.
3. **Superficial Chat Interfaces**: Typical LLM wrappers merely act as conversational chat windows rather than stateful interview agents with structured evaluation criteria.

---

## 2. Solution: The Adaptive State Machine

The **Smart Adaptive Interview Agent** turns technical interviewing into a closed-loop adaptive state machine using **LangGraph StateGraph**.

- Questions are **never pre-generated**.
- After every candidate answer, the agent:
  1. Scores technical correctness, completeness, and reasoning (0–10).
  2. Updates a cumulative profile of candidate strengths and weaknesses.
  3. Applies adaptive rules to increase, maintain, or decrease difficulty.
  4. Decides what topic and angle to probe next (storing its rationale in `next_question_reason`).
  5. Dynamically generates the next question only after the evaluation loop finishes.
  6. Compiles an executive competency report with topic breakdowns and study roadmaps upon completion.

---

## 3. LangGraph Architecture & Workflow

The interview flow is orchestrated by a compiled LangGraph `StateGraph` with state persistence (`MemorySaver`) and `interrupt()` human-in-the-loop pauses.

```
       START
         │
         ▼
 ┌───────────────────────┐
 │   INTERVIEW PLANNER   │  Initializes syllabus, starting difficulty & state
 └───────────────────────┘
         │
         ▼
 ┌───────────────────────┐
 │  QUESTION GENERATOR   │  Dynamically crafts 1 question based on state
 └───────────────────────┘
         │
         ▼
 ┌───────────────────────┐
 │ WAIT FOR USER ANSWER  │  LangGraph interrupt() pauses state machine
 └───────────────────────┘
         │
   [User Submits Answer via HTTP]
         │
         ▼
 ┌───────────────────────┐
 │   ANSWER EVALUATOR    │  Scores technical correctness, depth (0-10)
 └───────────────────────┘
         │
         ▼
 ┌───────────────────────┐
 │    SKILL ANALYZER     │  Aggregates cumulative strengths & weaknesses
 └───────────────────────┘
         │
         ▼
 ┌───────────────────────┐
 │  DIFFICULTY DECIDER   │  Applies dynamic adaptation rules
 └───────────────────────┘
         │
         ▼
 ┌───────────────────────┐
 │ NEXT QUESTION DECIDER │  Selects next topic & logs next_question_reason
 └───────────────────────┘
         │
         ├────────────────────────────────────────┐
   [Questions Remaining?]                         │ [All Questions Done]
         │                                        │
         ▼ (Yes)                                  ▼ (No)
 ┌───────────────────────┐                ┌───────────────────────┐
 │  QUESTION GENERATOR   │                │    FINAL EVALUATOR    │
 └───────────────────────┘                └───────────────────────┘
         │                                        │
         ▼                                        ▼
   (Cycle Repeats)                               END
```

### Shared State Schema (`InterviewState`)

```python
class InterviewState(TypedDict):
    session_id: str
    interview_domain: str
    target_difficulty: str          # "Easy", "Medium", "Hard"
    current_difficulty: str         # Dynamically adapted
    total_questions: int            # e.g., 5
    current_question_number: int    # 1-indexed counter
    current_question: str
    current_topic: str
    question_type: str              # Conceptual, Scenario-based, Debugging, etc.
    expected_key_points: List[str]
    current_answer: str
    answer_evaluation: Dict[str, Any]
    strengths: List[str]
    weaknesses: List[str]
    topics_covered: List[str]
    topics_planned: List[str]
    questions_history: List[Dict[str, Any]]
    answers_history: List[Dict[str, Any]]
    scores: List[float]
    difficulty_history: List[str]
    next_question_reason: str
    final_score: float
    final_feedback: str
    final_report: Dict[str, Any]
    status: str
    is_completed: bool
    error_message: Optional[str]
```

---

## 4. Adaptive Difficulty Logic

Difficulty adaptation is **non-linear and dynamic**:

| Answer Score | Classification | Difficulty Action | Topic & Question Strategy |
|---|---|---|---|
| **≥ 7.5 / 10** | `CORRECT` | **Increase** (Easy → Medium → Hard) | Advance to an unexplored topic from the syllabus at higher complexity. |
| **4.0 – 7.4 / 10** | `PARTIALLY_CORRECT` | **Maintain** (Similar level) | Probe practical implementation or related scenario in the same/adjacent concept. |
| **< 4.0 / 10** | `INCORRECT` | **Decrease** (Hard → Medium → Easy) | Re-test foundational mechanics of the struggled concept with a simpler question. |

---

## 5. Technology Stack

- **Backend**: Python 3.10+, FastAPI, Uvicorn
- **Orchestration**: LangGraph (`StateGraph`, `interrupt`, `Command`, `MemorySaver`)
- **LLM Integration**: LangChain Google GenAI (`ChatGoogleGenerativeAI`)
- **PDF Generation**: ReportLab (Professional multi-section evaluation reports)
- **Model**: `gemini-2.5-flash` (configurable via `GEMINI_MODEL`)
- **Validation**: Pydantic v2
- **Frontend**: Clean Single-Page App (SPA) with Vanilla HTML5, CSS3, and JavaScript (ES6+)
- **Diagrams**: Mermaid.js for real-time visualization of the LangGraph state machine

---

## 6. Project Structure

```
project/
├── app.py                 # FastAPI server & compiled LangGraph StateGraph engine
├── requirements.txt       # Production dependencies for local & Render deployment
├── README.md              # Comprehensive documentation & architecture guide
├── .env.example           # Reference environment variables
│
├── templates/
│   └── index.html         # Modern SPA interface with Setup, Interview, & Report views
│
└── static/
    ├── style.css          # Executive dark-mode design system & animations
    └── script.js          # Client controller managing UI states & API communication
```

---

## 7. Local Setup

### 1. Clone / Navigate to Project Directory
```bash
cd "d:/Pottiii/Agentic AI Agents/Agent-1"
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Linux / macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Edit `.env` to include your Google Gemini API key:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
PORT=8000
```
> *Security Note: `GEMINI_API_KEY` is loaded exclusively from the server environment and is never exposed in the client HTML, JavaScript, API responses, or generated PDF reports.*

### 5. Run the Application
```bash
python app.py
```
Or directly with Uvicorn:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at:
```
http://localhost:8000
```

---

## 8. API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the interactive single-page application frontend. |
| `GET` | `/health` | Health check endpoint returning status and Gemini readiness. |
| `GET` | `/workflow` | Returns the LangGraph nodes, edges, and Mermaid diagram. |
| `POST` | `/start-interview` | Initializes interview session and yields Question 1. |
| `POST` | `/submit-answer` | Evaluates answer, updates skills, adjusts difficulty, and returns evaluation. |
| `GET` | `/interview/{id}` | Fetches full state history, scores, and report for a session (internal state). |
| `GET` | `/interview/{id}/report` | **Generates and downloads the official, professional PDF interview report.** |
| `POST` | `/finish-interview` | Concludes interview early and generates final assessment. |

---

## 9. Render Deployment Guide

This project is configured out-of-the-box for [Render](https://render.com) Web Services.

### Option A: Using `render.yaml` (Blueprint)
1. Push this repository to GitHub.
2. In Render, select **Blueprints** → **New Blueprint Instance**.
3. Connect your repository. Render will automatically read `render.yaml`.
4. Enter your `GEMINI_API_KEY` when prompted in the environment settings.

### Option B: Manual Web Service Setup
1. Push this repository to GitHub.
2. In the Render Dashboard, click **New +** → **Web Service**.
3. Connect your GitHub repository.
4. Configure service settings:
   - **Environment**: `Python 3`
   - **Build Command**:
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command**:
     ```bash
     uvicorn app:app --host 0.0.0.0 --port $PORT
     ```
5. In **Environment Variables**, add:
   - `GEMINI_API_KEY`: Your Google Gemini API Key
   - `GEMINI_MODEL`: `gemini-3.5-flash-lite`
6. Click **Create Web Service**. Render will automatically build, provision dependencies, and deploy your application.

---

## 10. Example Adaptive Interview Walkthrough

### Scenario: Candidate with Strong Basics but Weak Internals

1. **Setup**:
   - Domain: `Python`
   - Starting Difficulty: `Medium`
   - Questions: `5`

2. **Question 1 (Medium - Functions & Scope)**:
   - *Question*: "Explain how Python handles function argument passing (call-by-sharing). What happens when mutating an object vs reassigning it inside a function?"
   - *Candidate Answer*: "Python uses call-by-object-reference. If you pass a list and append to it, the caller sees the change. But if you rebind the variable inside, it only changes the local pointer."
   - *Evaluation*: Score `8.5/10` (`CORRECT`).
   - *Agent Decision*: `"Candidate demonstrated mastery on Functions (Score: 8.5/10). Increased difficulty to Hard and advanced to Concurrency & the GIL."`

3. **Question 2 (Hard - Concurrency & the GIL)**:
   - *Question*: "Analyze Python's Global Interpreter Lock (GIL). Why does it prevent multi-threaded speedups for CPU-bound tasks, and how does it behave during I/O operations?"
   - *Candidate Answer*: "I know Python has threads, but I'm not really sure how the lock works under the hood. It just runs one thing at a time."
   - *Evaluation*: Score `3.5/10` (`INCORRECT`).
   - *Agent Decision*: `"Candidate struggled with low-level GIL mechanics (Score: 3.5/10). Decreased difficulty to Medium and re-evaluating foundational concept in Exception Handling & Context Managers."`

4. **Question 3 (Medium - Context Managers)**:
   - *Question*: "How do Python context managers work under the hood? Explain the roles of `__enter__` and `__exit__` methods."
   - *Candidate Answer*: "The `with` statement calls `__enter__` to acquire the resource and returns it. `__exit__` is called when exiting the block, receiving any exception type, value, and traceback."
   - *Evaluation*: Score `8.0/10` (`CORRECT`).
   - *Agent Decision*: `"Strong explanation of Context Managers (Score: 8.0/10). Increased difficulty to Hard and progressing to Object-Oriented Programming (MRO & C3 Linearization)."`

5. **Final Report Generated**:
   - Overall Score: `8.0 / 10`
   - Difficulty Progression: `Medium → Hard → Medium → Hard`
   - Strong Areas: `Function scoping`, `Context managers`, `Resource management`
   - Needs Improvement: `Low-level concurrency`, `Global Interpreter Lock internals`
   - Recommendation: Study Python C-API concurrency release and `multiprocessing` architecture.

---

## 11. Verification & Testing

To run the automated test suite testing both high-performing and low-performing adaptive flows:

```bash
python test_agent.py
```
