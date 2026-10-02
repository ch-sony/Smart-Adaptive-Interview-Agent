"""
Smart Adaptive Interview Agent
A dynamic, non-linear technical interview platform built with:
- FastAPI & Uvicorn
- LangChain & LangGraph (StateGraph with interrupt/resume)
- Google Gemini (ChatGoogleGenerativeAI)
- ReportLab (PDF Generation)
- Pydantic
"""

import os
import sys
import json
import re
import uuid
import logging
import datetime
from io import BytesIO
from typing import TypedDict, List, Dict, Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator
import uvicorn

# ReportLab imports for downloadable PDF reports
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables (override ensures newly saved .env is applied)
load_dotenv(override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("smart_interview_agent")

# Configuration (Single variable for model name)
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# Supported domains & default syllabus
CURRICULUM = {
    "Python": [
        "Variables, Memory Model & Mutability",
        "Functions, Closures, Scopes & *args/**kwargs",
        "Object-Oriented Programming, MRO & Inheritance",
        "Exception Handling, Custom Exceptions & Context Managers",
        "Iterators, Generators & Comprehensions",
        "Concurrency, Multiprocessing & the GIL",
        "Decorators & Functional Programming",
        "File I/O, Serialization & Performance Optimization"
    ],
    "Java": [
        "JVM Architecture, Memory Areas & Garbage Collection",
        "OOP Principles, Abstract Classes & Interfaces",
        "Java Collections Framework & Generics",
        "Multithreading, Concurrency Utilities & Synchronization",
        "Exception Handling & Checked vs Unchecked",
        "Java 8+ Streams API & Functional Lambdas",
        "Design Patterns & Spring Boot Core Concepts"
    ],
    "C": [
        "Pointers, Pointer Arithmetic & Dereferencing",
        "Dynamic Memory Allocation (malloc, calloc, realloc, free)",
        "Structures, Unions & Memory Alignment",
        "C Preprocessor Directives, Macros & Header Inclusion",
        "File I/O & System Calls (read, write, open)",
        "Bitwise Operators & Low-Level Memory Manipulation"
    ],
    "Data Structures": [
        "Arrays, Strings & Two-Pointer / Sliding Window Techniques",
        "Linked Lists, Reversal & Cycle Detection (Floyd's)",
        "Stacks, Queues & Monotonic Data Structures",
        "Binary Trees, BSTs & Tree Traversals",
        "Graphs, BFS, DFS, Dijkstra & Topological Sort",
        "Dynamic Programming, Memoization & Tabulation",
        "Hash Tables, Hash Functions & Collision Resolution"
    ],
    "DBMS": [
        "Relational Model, Primary/Foreign Keys & Constraints",
        "SQL Query Optimization & Execution Plans",
        "Indexing Structures (B-Trees, B+ Trees & Hash Indexes)",
        "ACID Properties & Transaction Isolation Levels",
        "Database Normalization (1NF, 2NF, 3NF, BCNF)",
        "Concurrency Control, Locking & Deadlock Resolution",
        "NoSQL Data Models vs Relational Systems"
    ],
    "AI/ML": [
        "Supervised vs Unsupervised vs Reinforcement Learning",
        "Loss Functions, Cost Functions & Gradient Descent Variants",
        "Neural Networks, Activation Functions & Backpropagation",
        "Overfitting, Underfitting, Regularization (L1, L2, Dropout)",
        "Transformer Architecture & Self-Attention Mechanisms",
        "Evaluation Metrics (Precision, Recall, F1, ROC-AUC)",
        "Feature Engineering, Normalization & Dimensionality Reduction"
    ],
    "General Technical Interview": [
        "Software Architecture & SOLID Principles",
        "System Reliability, Microservices vs Monoliths",
        "REST API Design, Status Codes & Idempotency",
        "Time and Space Complexity Analysis (Big-O)",
        "Caching Strategies (Redis, LRU, CDN)",
        "Debugging, Monitoring & Incident Troubleshooting"
    ],
    "System Design": [
        "Scalability, Load Balancing & Horizontal vs Vertical Scaling",
        "Database Sharding, Replication & CAP Theorem",
        "Message Queues (Kafka, RabbitMQ) & Asynchronous Processing",
        "Consistent Hashing & Distributed Caching",
        "Rate Limiting Algorithms (Token Bucket, Leaky Bucket)",
        "Microservices Communication: gRPC, REST, and WebSockets"
    ],
    "Web Development": [
        "DOM Manipulation & Browser Event Loop",
        "Client-Side vs Server-Side Rendering (CSR vs SSR)",
        "State Management Patterns & Component Lifecycle",
        "HTTP/2, HTTP/3, WebSockets & Network Optimization",
        "Security Best Practices (CORS, CSRF, XSS, CSP)",
        "Performance Optimization (Lighthouse, Core Web Vitals)"
    ]
}

# ==============================================================================
# LangGraph State Schema (No API key exposed in state)
# ==============================================================================

class InterviewState(TypedDict):
    session_id: str
    interview_domain: str
    target_difficulty: str          # "Easy", "Medium", "Hard"
    current_difficulty: str         # Dynamically adapted
    total_questions: int            # e.g., 5
    current_question_number: int    # 1-indexed counter
    current_question: str
    current_topic: str
    question_type: str              # "Conceptual", "Scenario-based", "Debugging", "Practical", "Code-related"
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
    status: str                     # "in_progress", "completed", "error"
    is_completed: bool
    error_message: Optional[str]

# ==============================================================================
# Helper Functions: JSON Extraction & Gemini Client
# ==============================================================================

def extract_text(content: Any) -> str:
    """Extract plain string from various LLM response formats (str, list of dicts, etc.)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content) if content is not None else ""

def clean_and_parse_json(raw_input: Any) -> Dict[str, Any]:
    """Safely extract and parse JSON from an LLM response string or structured content."""
    text = extract_text(raw_input).strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        json_match = re.search(r"(\{[\s\S]*\})", text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        logger.warning(f"Failed to parse JSON from LLM: {text[:200]}...")
        return {}

def get_llm(model_override: Optional[str] = None) -> Optional[ChatGoogleGenerativeAI]:
    """Instantiate ChatGoogleGenerativeAI using server environment GEMINI_API_KEY only."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        return None
        
    model_name = model_override or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    # Graceful compatibility mapping: if deprecated gemini-2.5 models are set, auto-upgrade to gemini-3.5
    if model_name in ["gemini-2.5-flash-lite", "gemini-2.5-flash"]:
        model_name = "gemini-3.5-flash-lite"
        
    try:
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=key,
            temperature=0.7,
            max_retries=2
        )
    except Exception as e:
        logger.error(f"Error instantiating ChatGoogleGenerativeAI: {e}")
        return None

# ==============================================================================
# Fallback / Offline Adaptive Engine (Guarantees stability and offline testing)
# ==============================================================================

OFFLINE_QUESTIONS = {
    "Python": {
        "Easy": [
            {
                "question": "What is the difference between mutable and immutable data types in Python? Give two examples of each.",
                "topic": "Variables, Memory Model & Mutability",
                "question_type": "Conceptual",
                "expected_key_points": ["Mutable types can be altered in place (list, dict, set)", "Immutable types cannot change identity/value (int, str, tuple)", "Reassignment vs in-place modification"]
            },
            {
                "question": "Explain the difference between '==' and 'is' operators in Python.",
                "topic": "Variables, Memory Model & Mutability",
                "question_type": "Conceptual",
                "expected_key_points": ["'==' checks value equality", "'is' checks object identity / memory address", "id() function relation"]
            }
        ],
        "Medium": [
            {
                "question": "Explain how Python handles function argument passing (call-by-sharing). What happens when you pass a mutable object vs an immutable object?",
                "topic": "Functions, Closures, Scopes & *args/**kwargs",
                "question_type": "Conceptual",
                "expected_key_points": ["Pass-by-object-reference or call-by-sharing", "Mutating in place reflects outside the function", "Rebinding a name inside the function does not change the outer reference"]
            },
            {
                "question": "Explain the difference between method overriding and method overloading in Python. How can you emulate overloading?",
                "topic": "Object-Oriented Programming, MRO & Inheritance",
                "question_type": "Conceptual",
                "expected_key_points": ["Overriding is redefining parent class methods", "Python does not support traditional overloading natively", "Emulated via default args, *args/**kwargs, or @functools.singledispatch"]
            },
            {
                "question": "How do Python context managers work under the hood? Explain the roles of __enter__ and __exit__ methods.",
                "topic": "Exception Handling, Custom Exceptions & Context Managers",
                "question_type": "Practical",
                "expected_key_points": ["__enter__ sets up resource", "__exit__ handles teardown and exception suppression", "with statement protocol"]
            }
        ],
        "Hard": [
            {
                "question": "Analyze Python's Global Interpreter Lock (GIL). Why does it prevent multi-threaded speedups for CPU-bound tasks, and how does it behave during I/O operations or native C-extensions?",
                "topic": "Concurrency, Multiprocessing & the GIL",
                "question_type": "Scenario-based",
                "expected_key_points": ["GIL allows only one native thread to execute Python bytecode at a time", "CPU-bound tasks suffer contention", "GIL is released during blocking I/O calls", "Use multiprocessing or nogil extensions for parallelism"]
            },
            {
                "question": "Explain Python's Method Resolution Order (MRO) and the C3 Linearization algorithm. Under what conditions will Python raise a TypeError: Cannot create a consistent method resolution order?",
                "topic": "Object-Oriented Programming, MRO & Inheritance",
                "question_type": "Conceptual",
                "expected_key_points": ["C3 Linearization algorithm order", "Monotonicity and local precedence preservation", "Conflicting inheritance hierarchies cause linearization failure"]
            }
        ]
    }
}

def offline_generate_question(domain: str, difficulty: str, topic: str, q_num: int) -> Dict[str, Any]:
    """Generates a contextual adaptive question without calling external APIs."""
    domain_data = OFFLINE_QUESTIONS.get(domain, OFFLINE_QUESTIONS["Python"])
    diff_pool = domain_data.get(difficulty, domain_data["Medium"])
    selected = diff_pool[(q_num - 1) % len(diff_pool)]
    return {
        "question": selected["question"],
        "topic": topic or selected["topic"],
        "difficulty": difficulty,
        "question_type": selected["question_type"],
        "expected_key_points": selected["expected_key_points"]
    }

def offline_evaluate_answer(question: str, topic: str, difficulty: str, answer: str, expected_key_points: List[str]) -> Dict[str, Any]:
    """Heuristic rule-based answer evaluator for fallback/testing."""
    answer_clean = answer.strip().lower()
    length = len(answer_clean)
    
    matches = 0
    for kp in expected_key_points:
        kp_words = [w for w in re.findall(r"\w+", kp.lower()) if len(w) > 3]
        if any(w in answer_clean for w in kp_words):
            matches += 1
    
    match_ratio = matches / max(1, len(expected_key_points))
    
    # Check for poor or dismissive answers using boundary regexes
    dismissive_regexes = [
        r"\bi don't know\b",
        r"\bidk\b",
        r"\bno idea\b",
        r"\bnot sure\b",
        r"\bdunno\b",
        r"^\s*pass[\.\!]?\s*$",
        r"^\s*skip[\.\!]?\s*$"
    ]
    is_dismissive = any(re.search(pat, answer_clean) for pat in dismissive_regexes)
    if is_dismissive or length < 15:
        score = 2.0 if length > 8 else 1.0
        classification = "INCORRECT"
        feedback = f"The answer lacks the required technical depth for {topic}. Key principles like {expected_key_points[0]} were not addressed."
        strengths = []
        weaknesses = [f"Foundational concepts in {topic}"]
    elif match_ratio >= 0.6 or length > 180:
        score = min(10.0, round(7.5 + (match_ratio * 2.5), 1))
        classification = "CORRECT"
        feedback = f"Solid technical explanation of {topic}. Demonstrated clear understanding of core concepts."
        strengths = [f"Clear grasp of {topic}", "Accurate reasoning"]
        weaknesses = []
    elif match_ratio >= 0.3 or length > 60:
        score = min(7.0, round(5.0 + (match_ratio * 2.0), 1))
        classification = "PARTIALLY_CORRECT"
        feedback = f"Good initial attempt on {topic}. Covered basics but missed critical nuances such as {expected_key_points[-1]}."
        strengths = [f"Basic understanding of {topic}"]
        weaknesses = [f"Depth on {topic} nuances"]
    else:
        score = 3.5
        classification = "INCORRECT"
        feedback = f"The response was incomplete or contained inaccuracies regarding {topic}."
        strengths = []
        weaknesses = [f"Core mechanics of {topic}"]
    
    return {
        "score": score,
        "classification": classification,
        "feedback": feedback,
        "detected_strengths": strengths,
        "detected_weaknesses": weaknesses
    }

# ==============================================================================
# LangGraph Workflow Nodes
# ==============================================================================

def interview_planner(state: InterviewState) -> Dict[str, Any]:
    """
    START -> INTERVIEW PLANNER
    Initializes interview state, syllabus, and sets starting difficulty.
    """
    domain = state["interview_domain"]
    target_difficulty = state["target_difficulty"]
    total = state["total_questions"]
    
    syllabus = CURRICULUM.get(domain, CURRICULUM["General Technical Interview"])
    planned_topics = list(syllabus[:total]) if len(syllabus) >= total else list(syllabus)
    
    logger.info(f"Session {state['session_id']}: Planned interview for domain '{domain}', starting difficulty '{target_difficulty}', total questions: {total}")
    
    return {
        "current_difficulty": target_difficulty,
        "current_question_number": 0,
        "topics_planned": planned_topics,
        "topics_covered": [],
        "strengths": [],
        "weaknesses": [],
        "questions_history": [],
        "answers_history": [],
        "scores": [],
        "difficulty_history": [],
        "next_question_reason": f"Interview initialized. Starting with '{target_difficulty}' difficulty in '{domain}' on foundational topic '{planned_topics[0]}'.",
        "current_topic": planned_topics[0] if planned_topics else domain,
        "status": "in_progress",
        "is_completed": False
    }

def question_generator(state: InterviewState) -> Dict[str, Any]:
    """
    QUESTION GENERATOR
    Generates ONE question at a time dynamically adapting to state.
    """
    next_q_num = state["current_question_number"] + 1
    domain = state["interview_domain"]
    difficulty = state["current_difficulty"]
    topic = state.get("current_topic") or (state["topics_planned"][(next_q_num - 1) % len(state["topics_planned"])])
    
    logger.info(f"Generating question {next_q_num}/{state['total_questions']} - Domain: {domain}, Difficulty: {difficulty}, Topic: {topic}")
    
    llm = get_llm()
    question_data = None
    
    if llm:
        prompt = f"""
You are an expert technical interviewer conducting an adaptive interview in {domain}.
Generate question {next_q_num} of {state['total_questions']}.

INTERVIEW CONTEXT:
- Domain: {domain}
- Current Difficulty: {difficulty} (Must strictly match this difficulty)
- Topic Target: {topic}
- Previous Questions Asked: {[q.get('question') for q in state.get('questions_history', [])]}
- Known Strengths: {state.get('strengths', [])}
- Known Weaknesses: {state.get('weaknesses', [])}
- Adaptive Reason for this Question: {state.get('next_question_reason', '')}

REQUIREMENTS:
1. Generate ONE question that tests understanding without asking duplicate concepts.
2. Select a suitable question type: Conceptual, Scenario-based, Debugging, Practical, or Code-related.
3. List 2 to 4 expected key technical points that a strong answer should include.
4. Respond ONLY with valid JSON in this exact structure:
{{
  "question": "Question text here...",
  "topic": "{topic}",
  "difficulty": "{difficulty}",
  "question_type": "Scenario-based",
  "expected_key_points": ["Point 1", "Point 2", "Point 3"]
}}
"""
        try:
            response = llm.invoke([
                SystemMessage(content="You are a professional technical interviewer who only outputs valid JSON."),
                HumanMessage(content=prompt)
            ])
            parsed = clean_and_parse_json(response.content)
            if parsed.get("question"):
                question_data = parsed
        except Exception as e:
            err_str = str(e).lower()
            logger.error(f"Gemini API error during question generation: {e}")
            if "quota" in err_str or "rate" in err_str or "429" in err_str or "resource_exhausted" in err_str:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="AI service is temporarily unavailable. Please try again later."
                )
    
    if not question_data:
        question_data = offline_generate_question(domain, difficulty, topic, next_q_num)
    
    diff_history = list(state.get("difficulty_history", []))
    diff_history.append(difficulty)
    
    q_entry = {
        "question_number": next_q_num,
        "question": question_data["question"],
        "topic": question_data.get("topic", topic),
        "difficulty": difficulty,
        "question_type": question_data.get("question_type", "Conceptual"),
        "expected_key_points": question_data.get("expected_key_points", [])
    }
    
    q_history = list(state.get("questions_history", []))
    q_history.append(q_entry)
    
    return {
        "current_question_number": next_q_num,
        "current_question": question_data["question"],
        "current_topic": question_data.get("topic", topic),
        "question_type": question_data.get("question_type", "Conceptual"),
        "expected_key_points": question_data.get("expected_key_points", []),
        "questions_history": q_history,
        "difficulty_history": diff_history
    }

def wait_for_user_answer(state: InterviewState) -> Dict[str, Any]:
    """
    WAIT FOR USER ANSWER
    Pauses execution using LangGraph interrupt() until user provides answer.
    """
    logger.info(f"Awaiting user answer for Question {state['current_question_number']}")
    user_answer = interrupt({
        "session_id": state["session_id"],
        "question_number": state["current_question_number"],
        "total_questions": state["total_questions"],
        "question": state["current_question"],
        "topic": state["current_topic"],
        "difficulty": state["current_difficulty"],
        "question_type": state["question_type"],
        "next_question_reason": state.get("next_question_reason", "")
    })
    
    ans_history = list(state.get("answers_history", []))
    ans_entry = {
        "question_number": state["current_question_number"],
        "question": state["current_question"],
        "answer": user_answer,
        "topic": state["current_topic"],
        "difficulty": state["current_difficulty"]
    }
    ans_history.append(ans_entry)
    
    return {
        "current_answer": user_answer,
        "answers_history": ans_history
    }

def answer_evaluator(state: InterviewState) -> Dict[str, Any]:
    """
    ANSWER EVALUATOR
    Evaluates the candidate's answer on correctness, reasoning, completeness (0-10 score).
    """
    question = state["current_question"]
    answer = state["current_answer"]
    topic = state["current_topic"]
    difficulty = state["current_difficulty"]
    domain = state["interview_domain"]
    key_points = state.get("expected_key_points", [])
    
    logger.info(f"Evaluating answer for Q{state['current_question_number']}: '{answer[:60]}...'")
    
    eval_data = None
    llm = get_llm()
    
    if llm:
        prompt = f"""
You are a senior technical interviewer evaluating a candidate's answer.
Domain: {domain}
Topic: {topic}
Difficulty: {difficulty}

QUESTION:
{question}

EXPECTED KEY POINTS:
{key_points}

CANDIDATE'S ANSWER:
{answer}

EVALUATION CRITERIA:
1. Technical correctness and accuracy
2. Depth of understanding and reasoning
3. Completeness vs expected key points
4. Assign a numerical score from 0.0 to 10.0
5. Classify as exactly one of: CORRECT (>= 7.5), PARTIALLY_CORRECT (4.0 to 7.4), INCORRECT (< 4.0)
6. Write a concise feedback paragraph (2 to 3 sentences max) highlighting what was accurate and what was missing.
7. Identify 1-2 demonstrated strengths and 1-2 missing areas/weaknesses.

Respond ONLY with valid JSON in this exact structure:
{{
  "score": 8.0,
  "classification": "CORRECT",
  "feedback": "Concise feedback here...",
  "detected_strengths": ["Strengths demonstrated..."],
  "detected_weaknesses": ["Omissions or misunderstandings..."]
}}
"""
        try:
            response = llm.invoke([
                SystemMessage(content="You are a precise technical evaluator that outputs strictly valid JSON."),
                HumanMessage(content=prompt)
            ])
            parsed = clean_and_parse_json(response.content)
            if "score" in parsed and "classification" in parsed:
                eval_data = {
                    "score": float(parsed["score"]),
                    "classification": str(parsed["classification"]).upper(),
                    "feedback": str(parsed.get("feedback", "Good explanation.")),
                    "detected_strengths": list(parsed.get("detected_strengths", [])),
                    "detected_weaknesses": list(parsed.get("detected_weaknesses", []))
                }
        except Exception as e:
            err_str = str(e).lower()
            logger.error(f"Gemini API error during answer evaluation: {e}")
            if "quota" in err_str or "rate" in err_str or "429" in err_str or "resource_exhausted" in err_str:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="AI service is temporarily unavailable. Please try again later."
                )
    
    if not eval_data:
        eval_data = offline_evaluate_answer(question, topic, difficulty, answer, key_points)
    
    scores = list(state.get("scores", []))
    scores.append(eval_data["score"])
    
    return {
        "answer_evaluation": eval_data,
        "scores": scores
    }

def skill_analyzer(state: InterviewState) -> Dict[str, Any]:
    """
    SKILL ANALYZER
    Updates cumulative profile of strengths and weaknesses across questions.
    """
    eval_res = state["answer_evaluation"]
    score = eval_res["score"]
    topic = state["current_topic"]
    
    strengths = list(state.get("strengths", []))
    weaknesses = list(state.get("weaknesses", []))
    topics_covered = list(state.get("topics_covered", []))
    
    if topic not in topics_covered:
        topics_covered.append(topic)
        
    for s in eval_res.get("detected_strengths", []):
        if s and s not in strengths:
            strengths.append(s)
            
    for w in eval_res.get("detected_weaknesses", []):
        if w and w not in weaknesses:
            weaknesses.append(w)
            
    if score >= 7.5:
        weaknesses = [w for w in weaknesses if topic.lower() not in w.lower()]
        if f"Proficiency in {topic}" not in strengths:
            strengths.append(f"Proficiency in {topic}")
            
    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "topics_covered": topics_covered
    }

def difficulty_decider(state: InterviewState) -> Dict[str, Any]:
    """
    DIFFICULTY DECIDER
    Dynamic adaptation rules:
    - Strong answer (score >= 7.5) -> Increase difficulty (Easy -> Medium -> Hard)
    - Partial answer (4.0 <= score < 7.5) -> Keep similar difficulty
    - Weak answer (score < 4.0) -> Decrease difficulty (Hard -> Medium -> Easy)
    """
    score = state["answer_evaluation"]["score"]
    current_diff = state["current_difficulty"]
    
    if score >= 7.5:
        if current_diff == "Easy":
            new_diff = "Medium"
        elif current_diff == "Medium":
            new_diff = "Hard"
        else:
            new_diff = "Hard"
    elif score >= 4.0:
        new_diff = current_diff
    else:
        if current_diff == "Hard":
            new_diff = "Medium"
        elif current_diff == "Medium":
            new_diff = "Easy"
        else:
            new_diff = "Easy"
            
    logger.info(f"Difficulty decision: Score {score}/10 -> Adjusted from '{current_diff}' to '{new_diff}'")
    
    return {
        "current_difficulty": new_diff
    }

def next_question_decider(state: InterviewState) -> Dict[str, Any]:
    """
    NEXT QUESTION DECIDER
    Determines next topic and rationale based on performance, weak areas, and syllabus.
    """
    score = state["answer_evaluation"]["score"]
    prev_topic = state["current_topic"]
    current_diff = state["current_difficulty"]
    topics_covered = state.get("topics_covered", [])
    topics_planned = state.get("topics_planned", [])
    
    uncovered = [t for t in topics_planned if t not in topics_covered]
    
    if score < 4.0:
        chosen_topic = prev_topic
        reason = (
            f"Candidate struggled on '{prev_topic}' (Score: {score}/10). "
            f"Decreased difficulty to {current_diff} and re-evaluating foundational concept in {chosen_topic}."
        )
    elif score >= 7.5:
        chosen_topic = uncovered[0] if uncovered else prev_topic
        reason = (
            f"Candidate demonstrated high mastery on '{prev_topic}' (Score: {score}/10). "
            f"Increased difficulty to {current_diff} and progressing to '{chosen_topic}'."
        )
    else:
        chosen_topic = uncovered[0] if uncovered else prev_topic
        reason = (
            f"Candidate showed partial understanding on '{prev_topic}' (Score: {score}/10). "
            f"Maintaining {current_diff} difficulty to assess '{chosen_topic}'."
        )
        
    logger.info(f"Next question decision: {reason}")
    
    return {
        "current_topic": chosen_topic,
        "next_question_reason": reason
    }

def check_interview_completion(state: InterviewState) -> str:
    """Conditional router: either proceed to next question or generate final report."""
    if state["current_question_number"] >= state["total_questions"]:
        return "final_evaluator"
    return "question_generator"

def final_evaluator(state: InterviewState) -> Dict[str, Any]:
    """
    FINAL EVALUATOR
    Generates comprehensive final report with performance analytics and improvement roadmap.
    """
    scores = state.get("scores", [0.0])
    avg_score = round(sum(scores) / max(1, len(scores)), 1)
    domain = state["interview_domain"]
    diff_history = state.get("difficulty_history", [])
    diff_progression = " → ".join(diff_history) if diff_history else state["target_difficulty"]
    
    topic_scores: Dict[str, List[float]] = {}
    for ans in state.get("answers_history", []):
        t = ans.get("topic", "General")
        q_idx = ans.get("question_number", 1) - 1
        s = scores[q_idx] if q_idx < len(scores) else 5.0
        topic_scores.setdefault(t, []).append(s)
        
    topic_performance = []
    for t, sc_list in topic_scores.items():
        t_avg = round(sum(sc_list) / len(sc_list), 1)
        topic_performance.append({
            "topic": t,
            "score": t_avg,
            "rating": "Strong" if t_avg >= 7.5 else ("Moderate" if t_avg >= 4.0 else "Needs Improvement")
        })
        
    llm = get_llm()
    synthesis = None
    
    if llm:
        prompt = f"""
You are the lead technical hiring director synthesizing a final technical interview report.
Domain: {domain}
Overall Score: {avg_score}/10
Questions Answered: {len(scores)}
Difficulty Path: {diff_progression}
Strengths Detected: {state.get('strengths', [])}
Weaknesses Detected: {state.get('weaknesses', [])}
Topic Breakdown: {topic_performance}

Generate a polished final assessment:
1. Executive summary of candidate's technical profile
2. Top strengths with technical context
3. High-priority areas for improvement
4. 2-3 actionable learning/practice recommendations

Respond ONLY with valid JSON:
{{
  "executive_summary": "...",
  "strengths_summary": ["..."],
  "weaknesses_summary": ["..."],
  "recommendations": ["..."]
}}
"""
        try:
            response = llm.invoke([
                SystemMessage(content="You are an expert executive technical hiring evaluator outputting valid JSON."),
                HumanMessage(content=prompt)
            ])
            synthesis = clean_and_parse_json(response.content)
        except Exception as e:
            err_str = str(e).lower()
            logger.error(f"Gemini API error during final evaluation: {e}")
            if "quota" in err_str or "rate" in err_str or "429" in err_str or "resource_exhausted" in err_str:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="AI service is temporarily unavailable. Please try again later."
                )
            
    if not synthesis or not synthesis.get("executive_summary"):
        synthesis = {
            "executive_summary": f"Candidate demonstrated an overall score of {avg_score}/10 across {len(scores)} questions in {domain}. Performance adapted across difficulties: {diff_progression}.",
            "strengths_summary": state.get("strengths") or [f"Core {domain} fundamentals"],
            "weaknesses_summary": state.get("weaknesses") or [f"Advanced {domain} edge-cases"],
            "recommendations": [
                f"Review hands-on coding scenarios in {domain}",
                "Practice explaining architectural trade-offs under varying difficulties"
            ]
        }
        
    report = {
        "overall_score": avg_score,
        "questions_answered": len(scores),
        "difficulty_progression": diff_progression,
        "topic_performance": topic_performance,
        "strong_areas": synthesis.get("strengths_summary", state.get("strengths", [])),
        "weak_areas": synthesis.get("weaknesses_summary", state.get("weaknesses", [])),
        "final_assessment": synthesis.get("executive_summary", ""),
        "suggestions": synthesis.get("recommendations", [])
    }
    
    return {
        "final_score": avg_score,
        "final_feedback": synthesis.get("executive_summary", ""),
        "final_report": report,
        "status": "completed",
        "is_completed": True
    }

# ==============================================================================
# Build & Compile LangGraph StateGraph
# ==============================================================================

def create_interview_graph():
    builder = StateGraph(InterviewState)
    
    builder.add_node("interview_planner", interview_planner)
    builder.add_node("question_generator", question_generator)
    builder.add_node("wait_for_user_answer", wait_for_user_answer)
    builder.add_node("answer_evaluator", answer_evaluator)
    builder.add_node("skill_analyzer", skill_analyzer)
    builder.add_node("difficulty_decider", difficulty_decider)
    builder.add_node("next_question_decider", next_question_decider)
    builder.add_node("final_evaluator", final_evaluator)
    
    # Workflow edges
    builder.add_edge(START, "interview_planner")
    builder.add_edge("interview_planner", "question_generator")
    builder.add_edge("question_generator", "wait_for_user_answer")
    builder.add_edge("wait_for_user_answer", "answer_evaluator")
    builder.add_edge("answer_evaluator", "skill_analyzer")
    builder.add_edge("skill_analyzer", "difficulty_decider")
    builder.add_edge("difficulty_decider", "next_question_decider")
    
    builder.add_conditional_edges(
        "next_question_decider",
        check_interview_completion,
        {
            "question_generator": "question_generator",
            "final_evaluator": "final_evaluator"
        }
    )
    builder.add_edge("final_evaluator", END)
    
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)

# Global compiled graph instance
interview_graph = create_interview_graph()

# In-memory session registry
interviews: Dict[str, Dict[str, Any]] = {}

# ==============================================================================
# ReportLab PDF Report Generation
# ==============================================================================

def generate_interview_pdf(interview_data: Dict[str, Any]) -> bytes:
    """Generates a professional multi-page PDF evaluation report using ReportLab."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0F172A'),
        alignment=1,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor('#4F46E5'),
        alignment=1,
        spaceAfter=14
    )

    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#334155')
    )

    bold_body_style = ParagraphStyle(
        'BoldBodyDark',
        parent=body_style,
        fontName='Helvetica-Bold'
    )

    qa_question_style = ParagraphStyle(
        'QAQuestion',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#1E293B')
    )

    qa_answer_style = ParagraphStyle(
        'QAAnswer',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#0F172A')
    )

    qa_feedback_style = ParagraphStyle(
        'QAFeedback',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#4338CA')
    )

    elements = []

    # Title & Subtitle
    elements.append(Paragraph("SMART ADAPTIVE INTERVIEW REPORT", title_style))
    elements.append(Paragraph("AI-POWERED ADAPTIVE EVALUATION &bull; LANGGRAPH STATEGRAPH ENGINE", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#4F46E5"), spaceAfter=10))

    # 1. Interview Overview
    elements.append(Paragraph("1. Interview Overview", section_heading))
    overview_data = [
        [
            Paragraph("<b>Interview ID:</b>", body_style),
            Paragraph(str(interview_data.get("session_id", "N/A")), body_style),
            Paragraph("<b>Domain:</b>", body_style),
            Paragraph(str(interview_data.get("domain") or interview_data.get("interview_domain", "N/A")), body_style)
        ],
        [
            Paragraph("<b>Starting Difficulty:</b>", body_style),
            Paragraph(str(interview_data.get("target_difficulty", "Medium")), body_style),
            Paragraph("<b>Total Questions:</b>", body_style),
            Paragraph(f"{len(interview_data.get('scores', []))} of {interview_data.get('total_questions', 5)}", body_style)
        ],
        [
            Paragraph("<b>Completion Status:</b>", body_style),
            Paragraph("Completed" if interview_data.get("is_completed") else "Completed / Evaluated", body_style),
            Paragraph("<b>Generated Date:</b>", body_style),
            Paragraph(datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC"), body_style)
        ]
    ]

    overview_table = Table(overview_data, colWidths=[105, 160, 105, 160])
    overview_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('BOX', (0,0), (-1,-1), 0.75, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    elements.append(overview_table)
    elements.append(Spacer(1, 8))

    # 2. Overall Performance
    elements.append(Paragraph("2. Overall Performance", section_heading))
    scores = interview_data.get("scores", [])
    overall_score = interview_data.get("final_score") or (round(sum(scores)/len(scores), 1) if scores else 0.0)
    
    if overall_score >= 8.5:
        verdict = "Senior / Expert Mastery"
        score_color = colors.HexColor('#059669')
    elif overall_score >= 7.0:
        verdict = "Proficient Technical Competency"
        score_color = colors.HexColor('#2563EB')
    elif overall_score >= 5.0:
        verdict = "Intermediate Working Knowledge"
        score_color = colors.HexColor('#D97706')
    else:
        verdict = "Foundational / Developing"
        score_color = colors.HexColor('#DC2626')

    final_report = interview_data.get("final_report", {})
    assessment_text = final_report.get("final_assessment") or interview_data.get("final_feedback") or "Candidate completed adaptive interview."

    perf_data = [
        [
            Paragraph(f"<font size=15 color='{score_color.hexval()}'><b>{overall_score:.1f} / 10</b></font><br/><font size=8 color='#64748B'>{verdict}</font>", body_style),
            Paragraph(f"<b>Performance Summary:</b><br/>{assessment_text}", body_style)
        ]
    ]
    perf_table = Table(perf_data, colWidths=[130, 400])
    perf_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F1F5F9')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    elements.append(perf_table)
    elements.append(Spacer(1, 8))

    # 3. Question-by-Question Review
    elements.append(Paragraph("3. Question-by-Question Review", section_heading))
    q_history = interview_data.get("questions_history", [])
    a_history = interview_data.get("answers_history", [])
    
    for idx in range(len(q_history)):
        q_item = q_history[idx]
        a_item = a_history[idx] if idx < len(a_history) else {}
        q_num = q_item.get("question_number", idx + 1)
        topic = q_item.get("topic", "General")
        diff = q_item.get("difficulty", "Medium")
        q_text = q_item.get("question", "")
        cand_ans = a_item.get("answer", "No answer submitted.")
        score = scores[idx] if idx < len(scores) else 0.0

        if score >= 7.5:
            classification = "CORRECT"
            class_color = colors.HexColor('#059669')
        elif score >= 4.0:
            classification = "PARTIALLY_CORRECT"
            class_color = colors.HexColor('#D97706')
        else:
            classification = "INCORRECT"
            class_color = colors.HexColor('#DC2626')

        feedback = a_item.get("feedback") or (interview_data.get("answer_evaluation", {}).get("feedback") if idx == len(q_history)-1 else "Evaluated by LangGraph agent.")

        qa_table_data = [
            [
                Paragraph(f"<b>Question {q_num} &bull; {topic}</b>", qa_question_style),
                Paragraph(f"Difficulty: <b>{diff.upper()}</b> &nbsp;|&nbsp; Score: <b>{score:.1f}/10</b> &nbsp;|&nbsp; <font color='{class_color.hexval()}'><b>{classification}</b></font>", body_style)
            ],
            [
                Paragraph(f"<b>Question:</b> {q_text}", body_style),
                ""
            ],
            [
                Paragraph(f"<b>Candidate Answer:</b> {cand_ans}", qa_answer_style),
                ""
            ],
            [
                Paragraph(f"<b>Evaluator Feedback:</b> {feedback}", qa_feedback_style),
                ""
            ]
        ]
        qa_table = Table(qa_table_data, colWidths=[360, 170])
        qa_table.setStyle(TableStyle([
            ('SPAN', (0, 1), (1, 1)),
            ('SPAN', (0, 2), (1, 2)),
            ('SPAN', (0, 3), (1, 3)),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F8FAFC')),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#FFFFFF')),
            ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor('#CBD5E1')),
            ('LINEBELOW', (0, 0), (-1, 0), 0.5, colors.HexColor('#E2E8F0')),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(KeepTogether([qa_table, Spacer(1, 5)]))

    elements.append(Spacer(1, 6))

    # 4. Adaptive Progression
    elements.append(Paragraph("4. Adaptive Progression", section_heading))
    diff_hist = interview_data.get("difficulty_history", [])
    if diff_hist:
        prog_str = " &nbsp;&rarr;&nbsp; ".join([f"<b>Q{i+1}:</b> {d}" for i, d in enumerate(diff_hist)])
        elements.append(Paragraph(f"<b>Difficulty Path:</b> {prog_str}", body_style))
        elements.append(Spacer(1, 4))
    
    prog_explanations = []
    for i in range(len(scores) - 1):
        s_cur = scores[i]
        d_from = diff_hist[i] if i < len(diff_hist) else "Medium"
        d_to = diff_hist[i+1] if i+1 < len(diff_hist) else d_from
        if s_cur >= 7.5:
            reason = f"Strong performance ({s_cur:.1f}/10) &rarr; difficulty increased or maintained at top level ({d_from} &rarr; {d_to})."
        elif s_cur >= 4.0:
            reason = f"Partial performance ({s_cur:.1f}/10) &rarr; difficulty maintained ({d_from} &rarr; {d_to}) to assess practical depth."
        else:
            reason = f"Weak performance ({s_cur:.1f}/10) &rarr; difficulty decreased ({d_from} &rarr; {d_to}) to verify core foundations."
        prog_explanations.append(f"<b>Q{i+1} &rarr; Q{i+2}:</b> {reason}")

    if prog_explanations:
        for p_exp in prog_explanations:
            elements.append(Paragraph(f"&bull; {p_exp}", body_style))
    else:
        elements.append(Paragraph("Initial question evaluated.", body_style))

    elements.append(Spacer(1, 8))

    # 5. Strengths & 6. Areas for Improvement
    elements.append(Paragraph("5. Strengths & 6. Areas for Improvement", section_heading))
    strengths = final_report.get("strong_areas") or interview_data.get("strengths") or ["Core fundamentals demonstrated"]
    weaknesses = final_report.get("weak_areas") or interview_data.get("weaknesses") or ["No critical weaknesses recorded"]

    str_items = "<br/>".join([f"&bull; <font color='#059669'>[Strong]</font> {s}" for s in strengths])
    weak_items = "<br/>".join([f"&bull; <font color='#DC2626'>[Improve]</font> {w}" for w in weaknesses])

    sw_data = [
        [
            Paragraph("<b>Identified Strengths</b>", bold_body_style),
            Paragraph("<b>Areas for Improvement</b>", bold_body_style)
        ],
        [
            Paragraph(str_items, body_style),
            Paragraph(weak_items, body_style)
        ]
    ]
    sw_table = Table(sw_data, colWidths=[265, 265])
    sw_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F8FAFC')),
        ('BOX', (0,0), (-1,-1), 0.75, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'TOP')
    ]))
    elements.append(sw_table)
    elements.append(Spacer(1, 8))

    # 7. Topic Performance
    elements.append(Paragraph("7. Topic Performance", section_heading))
    topic_perf = final_report.get("topic_performance", [])
    if topic_perf:
        tp_rows = [[
            Paragraph("<b>Topic</b>", bold_body_style),
            Paragraph("<b>Average Score</b>", bold_body_style),
            Paragraph("<b>Rating</b>", bold_body_style)
        ]]
        for tp in topic_perf:
            tp_rows.append([
                Paragraph(tp.get("topic", "N/A"), body_style),
                Paragraph(f"{tp.get('score', 0):.1f} / 10", body_style),
                Paragraph(tp.get("rating", "N/A"), body_style)
            ])
        tp_table = Table(tp_rows, colWidths=[270, 130, 130])
        tp_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F8FAFC')),
            ('BOX', (0,0), (-1,-1), 0.75, colors.HexColor('#CBD5E1')),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        elements.append(tp_table)
    else:
        elements.append(Paragraph("Topic metrics aggregated across completed turns.", body_style))

    elements.append(Spacer(1, 8))

    # 8. Final Feedback & 9. Improvement Roadmap
    elements.append(Paragraph("8. Final Feedback & 9. Improvement Roadmap", section_heading))
    recs = final_report.get("suggestions") or [
        "Review core domain documentation and practical edge cases.",
        "Practice explaining concurrency and resource management under varying conditions."
    ]
    recs_formatted = "<br/>".join([f"&rarr; {r}" for r in recs])
    roadmap_data = [
        [
            Paragraph(f"<b>Final Assessment:</b><br/>{assessment_text}<br/><br/><b>Actionable Improvement Roadmap:</b><br/>{recs_formatted}", body_style)
        ]
    ]
    roadmap_table = Table(roadmap_data, colWidths=[530])
    roadmap_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('BOX', (0,0), (-1,-1), 0.75, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    elements.append(roadmap_table)

    doc.build(elements)
    return buffer.getvalue()

# ==============================================================================
# Pydantic Schemas for API Endpoints (No API key accepted from frontend)
# ==============================================================================

class StartInterviewRequest(BaseModel):
    domain: str = Field(..., description="Interview domain, e.g., Python, Java, C, Data Structures")
    difficulty: str = Field("Medium", description="Initial difficulty: Easy, Medium, or Hard")
    total_questions: int = Field(5, ge=1, le=15, description="Number of questions to ask")

    @field_validator("difficulty")
    def validate_difficulty(cls, v):
        formatted = v.strip().capitalize()
        if formatted not in ["Easy", "Medium", "Hard"]:
            raise ValueError("Difficulty must be 'Easy', 'Medium', or 'Hard'")
        return formatted

class SubmitAnswerRequest(BaseModel):
    session_id: str = Field(..., description="Active interview session ID")
    answer: str = Field(..., min_length=1, description="Candidate's technical answer")

    @field_validator("answer")
    def validate_answer(cls, v):
        if not v.strip():
            raise ValueError("Answer cannot be blank.")
        return v.strip()

class FinishInterviewRequest(BaseModel):
    session_id: str = Field(..., description="Interview session ID to complete")

# ==============================================================================
# FastAPI Application & Routes
# ==============================================================================

app = FastAPI(
    title="Smart Adaptive Interview Agent",
    description="Dynamic, adaptive technical interview platform powered by LangGraph & Google Gemini",
    version="1.0.0"
)

# Static and template configuration
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    """Serve the clean single-page frontend without exposing API keys or model names."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "domains": list(CURRICULUM.keys())
        }
    )

@app.get("/health")
async def health_check():
    """Health check endpoint required by Render and deployment monitors."""
    return {
        "status": "healthy",
        "gemini_model": os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        "has_system_gemini_key": bool(os.environ.get("GEMINI_API_KEY", "").strip()),
        "active_interviews": len(interviews)
    }

@app.get("/workflow")
async def get_workflow():
    """Returns the LangGraph workflow structure and Mermaid diagram."""
    try:
        mermaid_graph = interview_graph.get_graph().draw_mermaid()
    except Exception as e:
        logger.warning(f"Could not draw mermaid graph: {e}")
        mermaid_graph = "graph TD;\nSTART --> interview_planner --> question_generator --> wait_for_user_answer --> answer_evaluator --> skill_analyzer --> difficulty_decider --> next_question_decider --> question_generator;\nnext_question_decider --> final_evaluator --> END;"
        
    return {
        "name": "Smart Adaptive Interview StateGraph",
        "description": "Dynamic multi-turn evaluation loop where each question is generated only after evaluating the candidate's prior answer.",
        "nodes": [
            {"id": "interview_planner", "role": "Determines curriculum, starting difficulty, and sets up state."},
            {"id": "question_generator", "role": "Generates 1 question at a time tailored to difficulty and topic."},
            {"id": "wait_for_user_answer", "role": "Pauses state machine (interrupt) waiting for candidate input."},
            {"id": "answer_evaluator", "role": "Scores technical correctness, completeness, and reasoning (0-10)."},
            {"id": "skill_analyzer", "role": "Aggregates cumulative strengths and weaknesses across turns."},
            {"id": "difficulty_decider", "role": "Adapts difficulty dynamically (Easy ↔ Medium ↔ Hard)."},
            {"id": "next_question_decider", "role": "Selects next topic and stores next_question_reason rationale."},
            {"id": "final_evaluator", "role": "Synthesizes final comprehensive interview report and roadmap."}
        ],
        "mermaid": mermaid_graph
    }

@app.post("/start-interview")
async def start_interview(req: StartInterviewRequest):
    """
    Initialize a new adaptive interview session.
    Checks server GEMINI_API_KEY environment variable.
    Executes interview_planner -> question_generator -> pauses at wait_for_user_answer.
    """
    server_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not server_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service is not configured. Please contact the administrator."
        )

    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    
    initial_state: InterviewState = {
        "session_id": session_id,
        "interview_domain": req.domain,
        "target_difficulty": req.difficulty,
        "current_difficulty": req.difficulty,
        "total_questions": req.total_questions,
        "current_question_number": 0,
        "current_question": "",
        "current_topic": "",
        "question_type": "",
        "expected_key_points": [],
        "current_answer": "",
        "answer_evaluation": {},
        "strengths": [],
        "weaknesses": [],
        "topics_covered": [],
        "topics_planned": [],
        "questions_history": [],
        "answers_history": [],
        "scores": [],
        "difficulty_history": [],
        "next_question_reason": "",
        "final_score": 0.0,
        "final_feedback": "",
        "final_report": {},
        "status": "in_progress",
        "is_completed": False,
        "error_message": None
    }
    
    try:
        interview_graph.invoke(initial_state, config=config)
        graph_state = interview_graph.get_state(config)
        current_values = graph_state.values
        
        interviews[session_id] = {
            "session_id": session_id,
            "config": config,
            "state": current_values
        }
        
        return {
            "session_id": session_id,
            "domain": current_values["interview_domain"],
            "current_question_number": current_values["current_question_number"],
            "total_questions": current_values["total_questions"],
            "current_difficulty": current_values["current_difficulty"],
            "current_topic": current_values["current_topic"],
            "question_type": current_values.get("question_type", "Conceptual"),
            "current_question": current_values["current_question"],
            "next_question_reason": current_values.get("next_question_reason", ""),
            "topics_covered": current_values.get("topics_covered", []),
            "topics_planned": current_values.get("topics_planned", []),
            "status": current_values.get("status", "in_progress")
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting interview: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to start interview. Please try again."
        )

@app.post("/submit-answer")
async def submit_answer(req: SubmitAnswerRequest):
    """
    Submits candidate's answer for the active question.
    Resumes LangGraph:
    wait_for_user_answer -> answer_evaluator -> skill_analyzer -> difficulty_decider -> next_question_decider -> (question_generator OR final_evaluator)
    """
    session = interviews.get(req.session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Interview session not found. Please start a new interview."
        )
        
    config = session["config"]
    graph_state = interview_graph.get_state(config)
    
    if not graph_state.tasks:
        if graph_state.values.get("is_completed"):
            return {
                "session_id": req.session_id,
                "is_completed": True,
                "message": "Interview has already been completed.",
                "final_report": graph_state.values.get("final_report")
            }
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session is not currently waiting for an answer."
        )
        
    try:
        prev_diff = graph_state.values.get("current_difficulty")
        interview_graph.invoke(Command(resume=req.answer), config=config)
        
        updated_state = interview_graph.get_state(config)
        values = updated_state.values
        session["state"] = values
        
        evaluation = values.get("answer_evaluation", {})
        is_completed = values.get("is_completed", False)
        
        response_data = {
            "session_id": req.session_id,
            "evaluated_question_number": len(values.get("scores", [])),
            "total_questions": values["total_questions"],
            "evaluation": {
                "score": evaluation.get("score", 0.0),
                "classification": evaluation.get("classification", "PARTIALLY_CORRECT"),
                "feedback": evaluation.get("feedback", ""),
                "detected_strengths": evaluation.get("detected_strengths", []),
                "detected_weaknesses": evaluation.get("detected_weaknesses", [])
            },
            "previous_difficulty": prev_diff,
            "new_difficulty": values.get("current_difficulty"),
            "next_question_reason": values.get("next_question_reason", ""),
            "topics_covered": values.get("topics_covered", []),
            "strengths": values.get("strengths", []),
            "weaknesses": values.get("weaknesses", []),
            "scores": values.get("scores", []),
            "is_completed": is_completed
        }
        
        if not is_completed:
            response_data["next_question"] = {
                "question_number": values["current_question_number"],
                "topic": values["current_topic"],
                "difficulty": values["current_difficulty"],
                "question_type": values.get("question_type", "Conceptual"),
                "question": values["current_question"]
            }
        else:
            response_data["final_report"] = values.get("final_report", {})
            
        return response_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error evaluating answer in session {req.session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while evaluating your answer. Please try again."
        )

@app.get("/interview/{session_id}")
async def get_interview_state(session_id: str):
    """Retrieve full status, questions history, scores, and report for a session (internal state)."""
    session = interviews.get(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Interview session not found."
        )
        
    config = session["config"]
    graph_state = interview_graph.get_state(config)
    values = graph_state.values
    
    return {
        "session_id": session_id,
        "domain": values.get("interview_domain"),
        "target_difficulty": values.get("target_difficulty"),
        "current_difficulty": values.get("current_difficulty"),
        "total_questions": values.get("total_questions"),
        "current_question_number": values.get("current_question_number"),
        "current_question": values.get("current_question"),
        "current_topic": values.get("current_topic"),
        "question_type": values.get("question_type"),
        "scores": values.get("scores", []),
        "strengths": values.get("strengths", []),
        "weaknesses": values.get("weaknesses", []),
        "topics_covered": values.get("topics_covered", []),
        "difficulty_history": values.get("difficulty_history", []),
        "questions_history": values.get("questions_history", []),
        "answers_history": values.get("answers_history", []),
        "is_completed": values.get("is_completed", False),
        "final_report": values.get("final_report", {})
    }

@app.get("/interview/{session_id}/report")
async def get_interview_pdf_report(session_id: str):
    """
    Generate and stream the official, professional PDF interview review report.
    Returns downloadable PDF file with comprehensive performance metrics.
    """
    session = interviews.get(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Interview session not found."
        )
        
    config = session["config"]
    graph_state = interview_graph.get_state(config)
    values = dict(graph_state.values)
    
    # If final_report is missing, compute it from current collected state
    if not values.get("final_report"):
        eval_result = final_evaluator(values)
        values.update(eval_result)
        session["state"] = values
        
    try:
        pdf_bytes = generate_interview_pdf(values)
        filename = f"adaptive_interview_report_{session_id}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
    except Exception as e:
        logger.error(f"Error generating PDF report for session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate PDF report."
        )

@app.post("/finish-interview")
async def finish_interview(req: FinishInterviewRequest):
    """Manually end or finalize an interview early and compute report."""
    session = interviews.get(req.session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Interview session not found."
        )
        
    config = session["config"]
    graph_state = interview_graph.get_state(config)
    values = dict(graph_state.values)
    
    if values.get("is_completed") and values.get("final_report"):
        return {
            "session_id": req.session_id,
            "final_report": values.get("final_report"),
            "status": "already_completed"
        }
        
    eval_result = final_evaluator(values)
    values.update(eval_result)
    session["state"] = values
    
    return {
        "session_id": req.session_id,
        "final_report": values["final_report"],
        "status": "completed"
    }

# ==============================================================================
# Server Entrypoint
# ==============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting Smart Adaptive Interview Agent on 0.0.0.0:{port}...")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
