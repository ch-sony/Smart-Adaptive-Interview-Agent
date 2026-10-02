/**
 * Smart Adaptive Interview Agent - Frontend Controller
 * Communicates with FastAPI backend and LangGraph StateGraph engine
 */

document.addEventListener('DOMContentLoaded', () => {
  // App State
  let sessionState = {
    sessionId: null,
    domain: 'Python',
    startingDifficulty: 'Medium',
    currentDifficulty: 'Medium',
    totalQuestions: 5,
    currentQuestionNumber: 0,
    scores: [],
    topicsCovered: [],
    topicsPlanned: [],
    history: [],
    pendingNextQuestion: null,
    pendingFinalReport: null
  };

  // DOM Elements - Views
  const setupView = document.getElementById('setupView');
  const interviewView = document.getElementById('interviewView');
  const resultsView = document.getElementById('resultsView');

  // Setup Form Elements
  const startForm = document.getElementById('startInterviewForm');
  const domainSelect = document.getElementById('domainSelect');
  const questionCountSlider = document.getElementById('questionCount');
  const questionCountDisplay = document.getElementById('questionCountDisplay');
  const btnStart = document.getElementById('btnStartInterview');

  // Interview HUD Elements
  const hudDomain = document.getElementById('hudDomain');
  const hudCurrentQ = document.getElementById('hudCurrentQ');
  const hudTotalQ = document.getElementById('hudTotalQ');
  const hudProgressBarFill = document.getElementById('hudProgressBarFill');
  const hudDifficultyBadge = document.getElementById('hudDifficultyBadge');
  const hudAvgScore = document.getElementById('hudAvgScore');
  const topicsBreadcrumbs = document.getElementById('topicsBreadcrumbs');
  const agentReasonText = document.getElementById('agentReasonText');

  // Question & Answer Elements
  const qTopicBadge = document.getElementById('qTopicBadge');
  const qTypeBadge = document.getElementById('qTypeBadge');
  const qDifficultyPill = document.getElementById('qDifficultyPill');
  const questionText = document.getElementById('questionText');
  const answerSection = document.getElementById('answerSection');
  const answerInput = document.getElementById('answerInput');
  const charCount = document.getElementById('charCount');
  const btnSubmitAnswer = document.getElementById('btnSubmitAnswer');

  // Evaluation Card Elements
  const evaluationCard = document.getElementById('evaluationCard');
  const evalScorePill = document.getElementById('evalScorePill');
  const evalStatusBadge = document.getElementById('evalStatusBadge');
  const evalDiffChange = document.getElementById('evalDiffChange');
  const evalFeedbackText = document.getElementById('evalFeedbackText');
  const evalStrengthsList = document.getElementById('evalStrengthsList');
  const evalWeaknessesList = document.getElementById('evalWeaknessesList');
  const evalNextReasonText = document.getElementById('evalNextReasonText');
  const btnNextQuestion = document.getElementById('btnNextQuestion');
  const btnNextQuestionText = document.getElementById('btnNextQuestionText');

  // Results View Elements
  const finalOverallScore = document.getElementById('finalOverallScore');
  const finalVerdict = document.getElementById('finalVerdict');
  const finalQuestionsCount = document.getElementById('finalQuestionsCount');
  const finalDomain = document.getElementById('finalDomain');
  const progressionTimeline = document.getElementById('progressionTimeline');
  const topicBarsList = document.getElementById('topicBarsList');
  const finalStrengthsList = document.getElementById('finalStrengthsList');
  const finalWeaknessesList = document.getElementById('finalWeaknessesList');
  const finalExecutiveSummary = document.getElementById('finalExecutiveSummary');
  const finalRecommendationsList = document.getElementById('finalRecommendationsList');
  const btnToggleTranscript = document.getElementById('btnToggleTranscript');
  const transcriptContent = document.getElementById('transcriptContent');
  const transcriptCount = document.getElementById('transcriptCount');
  const btnRestartInterview = document.getElementById('btnRestartInterview');
  const btnDownloadReport = document.getElementById('btnDownloadReport');

  // Workflow Modal Elements
  const btnWorkflow = document.getElementById('btnWorkflow');
  const workflowModal = document.getElementById('workflowModal');
  const btnCloseModal = document.getElementById('btnCloseModal');
  const toastContainer = document.getElementById('toastContainer');

  // Initialize Mermaid
  if (window.mermaid) {
    mermaid.initialize({
      startOnLoad: true,
      theme: 'dark',
      flowchart: { curve: 'linear' }
    });
  }

  // ============================================================================
  // Setup View Listeners
  // ============================================================================

  // Question slider update
  questionCountSlider.addEventListener('input', (e) => {
    const val = e.target.value;
    const estTime = Math.round(val * 2);
    questionCountDisplay.textContent = `${val} Questions (~${estTime} mins)`;
  });

  // Start Interview Form Submission
  startForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const domain = domainSelect.value;
    const difficultyRadio = document.querySelector('input[name="difficulty"]:checked');
    const difficulty = difficultyRadio ? difficultyRadio.value : 'Medium';
    const totalQuestions = parseInt(questionCountSlider.value, 10);

    sessionState.domain = domain;
    sessionState.startingDifficulty = difficulty;
    sessionState.currentDifficulty = difficulty;
    sessionState.totalQuestions = totalQuestions;
    sessionState.scores = [];
    sessionState.history = [];

    btnStart.disabled = true;
    btnStart.innerHTML = `
      <span class="btn-text">Initializing Agent...</span>
      <svg class="spinner" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10" stroke-opacity="0.25"></circle>
        <path d="M12 2a10 10 0 0 1 10 10" stroke-linecap="round"></path>
      </svg>
    `;

    try {
      const response = await fetch('/start-interview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          domain: domain,
          difficulty: difficulty,
          total_questions: totalQuestions
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to initialize interview.');
      }

      const data = await response.json();
      sessionState.sessionId = data.session_id;
      sessionState.currentQuestionNumber = data.current_question_number;
      sessionState.currentDifficulty = data.current_difficulty;
      sessionState.topicsCovered = data.topics_covered || [];
      sessionState.topicsPlanned = data.topics_planned || [];

      // Switch to Interview View
      switchToView('interview');

      // Populate Initial Question
      renderQuestion({
        question_number: data.current_question_number,
        total_questions: data.total_questions,
        domain: data.domain,
        difficulty: data.current_difficulty,
        topic: data.current_topic,
        question_type: data.question_type,
        question: data.current_question,
        next_question_reason: data.next_question_reason
      });

      showToast('Interview initialized with LangGraph StateGraph engine.', 'success');

    } catch (err) {
      console.error('Error starting interview:', err);
      showToast(err.message || 'Unable to start interview.', 'error');
    } finally {
      btnStart.disabled = false;
      btnStart.innerHTML = `
        <span class="btn-text">Start Adaptive Interview</span>
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="5" y1="12" x2="19" y2="12"></line>
          <polyline points="12 5 19 12 12 19"></polyline>
        </svg>
      `;
    }
  });

  // ============================================================================
  // Active Interview View Handlers
  // ============================================================================

  // Character counter for textarea
  answerInput.addEventListener('input', () => {
    const count = answerInput.value.length;
    charCount.textContent = `${count} character${count === 1 ? '' : 's'}`;
  });

  // Submit Answer Button Handler
  btnSubmitAnswer.addEventListener('click', async () => {
    const answer = answerInput.value.trim();
    if (!answer) {
      showToast('Please type an answer before submitting.', 'error');
      answerInput.focus();
      return;
    }

    btnSubmitAnswer.disabled = true;
    answerInput.disabled = true;
    btnSubmitAnswer.innerHTML = `
      <span class="btn-text">Evaluating with LangGraph...</span>
      <svg class="spinner" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10" stroke-opacity="0.25"></circle>
        <path d="M12 2a10 10 0 0 1 10 10" stroke-linecap="round"></path>
      </svg>
    `;

    try {
      const response = await fetch('/submit-answer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionState.sessionId,
          answer: answer
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Evaluation failed.');
      }

      const result = await response.json();
      
      // Update state tracking
      sessionState.scores = result.scores || [];
      sessionState.topicsCovered = result.topics_covered || [];
      sessionState.currentDifficulty = result.new_difficulty;

      // Save question history entry
      sessionState.history.push({
        question_number: result.evaluated_question_number,
        topic: qTopicBadge.textContent,
        difficulty: result.previous_difficulty,
        question: questionText.textContent,
        answer: answer,
        score: result.evaluation.score,
        classification: result.evaluation.classification,
        feedback: result.evaluation.feedback
      });

      // Update HUD Avg Score
      updateHudAvg();

      // Store pending next action
      sessionState.pendingNextQuestion = result.next_question || null;
      sessionState.pendingFinalReport = result.final_report || null;

      // Reveal evaluation card (Requirement 13: do NOT show next question until user continues!)
      displayEvaluation({
        evaluation: result.evaluation,
        prevDifficulty: result.previous_difficulty,
        newDifficulty: result.new_difficulty,
        nextQuestionReason: result.next_question_reason,
        isCompleted: result.is_completed
      });

    } catch (err) {
      console.error('Error submitting answer:', err);
      showToast(err.message || 'Error evaluating answer.', 'error');
      btnSubmitAnswer.disabled = false;
      answerInput.disabled = false;
      btnSubmitAnswer.innerHTML = `
        <span class="btn-text">Submit Answer</span>
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="20 6 9 17 4 12"></polyline>
        </svg>
      `;
    }
  });

  // Continue to Next Question (or Final Report)
  btnNextQuestion.addEventListener('click', () => {
    if (sessionState.pendingFinalReport) {
      // Completed! Transition to Results View
      switchToView('results');
      renderFinalResults(sessionState.pendingFinalReport);
      return;
    }

    if (sessionState.pendingNextQuestion) {
      const nextQ = sessionState.pendingNextQuestion;
      sessionState.currentQuestionNumber = nextQ.question_number;
      sessionState.currentDifficulty = nextQ.difficulty;

      // Reset and render next question
      renderQuestion({
        question_number: nextQ.question_number,
        total_questions: sessionState.totalQuestions,
        domain: sessionState.domain,
        difficulty: nextQ.difficulty,
        topic: nextQ.topic,
        question_type: nextQ.question_type,
        question: nextQ.question,
        next_question_reason: evalNextReasonText.textContent
      });

      // Clear pending
      sessionState.pendingNextQuestion = null;

      // Hide evaluation card & reset answer box
      evaluationCard.classList.add('hidden');
      answerInput.disabled = false;
      answerInput.value = '';
      charCount.textContent = '0 characters';
      btnSubmitAnswer.disabled = false;
      btnSubmitAnswer.innerHTML = `
        <span class="btn-text">Submit Answer</span>
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="20 6 9 17 4 12"></polyline>
        </svg>
      `;

      answerInput.focus();
    }
  });

  // ============================================================================
  // Rendering Helpers
  // ============================================================================

  function renderQuestion(data) {
    // Update HUD
    hudDomain.textContent = data.domain;
    hudCurrentQ.textContent = data.question_number;
    hudTotalQ.textContent = data.total_questions;

    const progressPct = Math.round((data.question_number / data.total_questions) * 100);
    hudProgressBarFill.style.width = `${progressPct}%`;

    updateDifficultyBadge(hudDifficultyBadge, data.difficulty);
    updateDifficultyBadge(qDifficultyPill, data.difficulty);

    // Update Topic & Type Tags
    qTopicBadge.textContent = data.topic;
    qTypeBadge.textContent = data.question_type || 'Conceptual';

    // Agent Strategy banner
    agentReasonText.textContent = data.next_question_reason || 'Evaluating technical competency.';

    // Question Text (support simple code blocks)
    formatAndSetQuestionText(data.question);

    // Update Topics Breadcrumb
    renderBreadcrumbs(data.topic);
  }

  function formatAndSetQuestionText(text) {
    if (text.includes('```')) {
      const parts = text.split(/(```[\s\S]*?```)/g);
      questionText.innerHTML = parts.map(part => {
        if (part.startsWith('```')) {
          const code = part.replace(/^```[a-z]*\n?/, '').replace(/```$/, '');
          return `<pre><code>${escapeHtml(code)}</code></pre>`;
        }
        return `<span>${escapeHtml(part)}</span>`;
      }).join('');
    } else {
      questionText.textContent = text;
    }
  }

  function renderBreadcrumbs(currentTopic) {
    topicsBreadcrumbs.innerHTML = '';
    const allTopics = Array.from(new Set([...sessionState.topicsPlanned, ...sessionState.topicsCovered, currentTopic]));
    
    allTopics.forEach(t => {
      const span = document.createElement('span');
      span.className = 'topic-crumb';
      if (sessionState.topicsCovered.includes(t) && t !== currentTopic) {
        span.classList.add('completed');
        span.textContent = `✓ ${t}`;
      } else if (t === currentTopic) {
        span.classList.add('active');
        span.textContent = `→ ${t}`;
      } else {
        span.textContent = t;
      }
      topicsBreadcrumbs.appendChild(span);
    });
  }

  function updateDifficultyBadge(element, diff) {
    element.className = '';
    const cleanDiff = (diff || 'Medium').toLowerCase();
    if (element === hudDifficultyBadge) {
      element.className = `diff-badge diff-${cleanDiff}`;
    } else {
      element.className = `diff-badge-sm diff-${cleanDiff}`;
    }
    element.textContent = (diff || 'Medium').toUpperCase();
  }

  function updateHudAvg() {
    if (!sessionState.scores || sessionState.scores.length === 0) {
      hudAvgScore.textContent = '--';
      return;
    }
    const sum = sessionState.scores.reduce((a, b) => a + b, 0);
    const avg = (sum / sessionState.scores.length).toFixed(1);
    hudAvgScore.textContent = `${avg} / 10`;
  }

  function displayEvaluation(data) {
    const { evaluation, prevDifficulty, newDifficulty, nextQuestionReason, isCompleted } = data;

    evalScorePill.textContent = (evaluation.score || 0).toFixed(1);

    // Status Badge
    evalStatusBadge.className = 'eval-status-badge';
    if (evaluation.classification === 'CORRECT') {
      evalStatusBadge.classList.add('status-correct');
      evalStatusBadge.textContent = 'CORRECT';
    } else if (evaluation.classification === 'PARTIALLY_CORRECT') {
      evalStatusBadge.classList.add('status-partial');
      evalStatusBadge.textContent = 'PARTIALLY CORRECT';
    } else {
      evalStatusBadge.classList.add('status-incorrect');
      evalStatusBadge.textContent = 'INCORRECT';
    }

    // Difficulty Shift
    evalDiffChange.textContent = `${prevDifficulty} → ${newDifficulty}`;
    evalFeedbackText.textContent = evaluation.feedback || 'Answer evaluated.';

    // Strengths
    evalStrengthsList.innerHTML = '';
    const strengths = evaluation.detected_strengths || [];
    if (strengths.length > 0) {
      strengths.forEach(s => {
        const li = document.createElement('li');
        li.textContent = s;
        evalStrengthsList.appendChild(li);
      });
    } else {
      evalStrengthsList.innerHTML = '<li>Fundamental principles need reinforcement</li>';
    }

    // Weaknesses
    evalWeaknessesList.innerHTML = '';
    const weaknesses = evaluation.detected_weaknesses || [];
    if (weaknesses.length > 0) {
      weaknesses.forEach(w => {
        const li = document.createElement('li');
        li.textContent = w;
        evalWeaknessesList.appendChild(li);
      });
    } else {
      evalWeaknessesList.innerHTML = '<li>No significant knowledge gaps detected</li>';
    }

    // Next reason
    evalNextReasonText.textContent = nextQuestionReason || `Agent adapted difficulty to ${newDifficulty}.`;

    // Button label
    if (isCompleted) {
      btnNextQuestionText.textContent = 'View Final Interview Report';
    } else {
      btnNextQuestionText.textContent = 'Continue to Next Question';
    }

    // Reveal
    evaluationCard.classList.remove('hidden');
    evaluationCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function renderFinalResults(report) {
    const score = report.overall_score || 0.0;
    finalOverallScore.textContent = score.toFixed(1);
    finalDomain.textContent = sessionState.domain;
    finalQuestionsCount.textContent = `${report.questions_answered || sessionState.scores.length} / ${sessionState.totalQuestions}`;

    // Verdict
    if (score >= 8.5) {
      finalVerdict.textContent = 'Senior / Expert Mastery';
      finalVerdict.style.color = 'var(--color-easy)';
    } else if (score >= 7.0) {
      finalVerdict.textContent = 'Proficient Technical Competency';
      finalVerdict.style.color = 'var(--accent-blue)';
    } else if (score >= 5.0) {
      finalVerdict.textContent = 'Intermediate Working Knowledge';
      finalVerdict.style.color = 'var(--color-medium)';
    } else {
      finalVerdict.textContent = 'Foundational / Developing';
      finalVerdict.style.color = 'var(--color-hard)';
    }

    // Progression Timeline
    progressionTimeline.innerHTML = '';
    const diffHistory = report.difficulty_progression ? report.difficulty_progression.split(' → ') : [sessionState.startingDifficulty];
    
    diffHistory.forEach((diff, idx) => {
      const node = document.createElement('div');
      node.className = 'prog-node';
      const cleanDiff = diff.toLowerCase();

      node.innerHTML = `
        <div class="prog-node-box diff-${cleanDiff}">
          <span class="prog-node-q">Q${idx + 1}</span>
          <span class="prog-node-diff">${diff.toUpperCase()}</span>
        </div>
        ${idx < diffHistory.length - 1 ? '<span class="prog-arrow">→</span>' : ''}
      `;
      progressionTimeline.appendChild(node);
    });

    // Topic Competency Bars
    topicBarsList.innerHTML = '';
    const topicPerf = report.topic_performance || [];
    topicPerf.forEach(tp => {
      const pct = Math.min(100, Math.round((tp.score / 10) * 100));
      const item = document.createElement('div');
      item.className = 'topic-bar-item';
      item.innerHTML = `
        <div class="tb-info">
          <span class="tb-name">${escapeHtml(tp.topic)}</span>
          <span class="tb-score">${tp.score.toFixed(1)} / 10 (${tp.rating})</span>
        </div>
        <div class="tb-track">
          <div class="tb-fill" style="width: ${pct}%;"></div>
        </div>
      `;
      topicBarsList.appendChild(item);
    });

    // Strengths Chips
    finalStrengthsList.innerHTML = '';
    const strongAreas = report.strong_areas || [];
    strongAreas.forEach(sa => {
      const li = document.createElement('li');
      li.className = 'profile-chip chip-strength';
      li.textContent = `✓ ${sa}`;
      finalStrengthsList.appendChild(li);
    });

    // Weaknesses Chips
    finalWeaknessesList.innerHTML = '';
    const weakAreas = report.weak_areas || [];
    weakAreas.forEach(wa => {
      const li = document.createElement('li');
      li.className = 'profile-chip chip-weakness';
      li.textContent = `⚠ ${wa}`;
      finalWeaknessesList.appendChild(li);
    });

    // Executive Assessment & Suggestions
    finalExecutiveSummary.textContent = report.final_assessment || 'Comprehensive technical assessment completed.';
    
    finalRecommendationsList.innerHTML = '';
    const recs = report.suggestions || [];
    recs.forEach(r => {
      const li = document.createElement('li');
      li.textContent = r;
      finalRecommendationsList.appendChild(li);
    });

    // Transcript Accordion
    transcriptCount.textContent = sessionState.history.length;
    transcriptContent.innerHTML = '';
    sessionState.history.forEach((h, idx) => {
      const item = document.createElement('div');
      item.className = 'transcript-item';
      item.innerHTML = `
        <div class="ti-header">
          <strong>Question ${idx + 1} (${h.topic} - ${h.difficulty})</strong>
          <span class="score-pill">${h.score.toFixed(1)} / 10</span>
        </div>
        <p class="ti-q">${escapeHtml(h.question)}</p>
        <div class="ti-a"><strong>Candidate:</strong> ${escapeHtml(h.answer)}</div>
        <p class="ti-feedback"><strong>Feedback:</strong> ${escapeHtml(h.feedback)}</p>
      `;
      transcriptContent.appendChild(item);
    });
  }

  // ============================================================================
  // Navigation & Utility Handlers
  // ============================================================================

  function switchToView(viewName) {
    setupView.classList.remove('active');
    interviewView.classList.remove('active');
    resultsView.classList.remove('active');

    if (viewName === 'setup') setupView.classList.add('active');
    if (viewName === 'interview') interviewView.classList.add('active');
    if (viewName === 'results') resultsView.classList.add('active');

    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  btnRestartInterview.addEventListener('click', () => {
    switchToView('setup');
  });

  // Download Interview Report (PDF)
  btnDownloadReport.addEventListener('click', () => {
    if (!sessionState.sessionId) {
      showToast('No active interview session found.', 'error');
      return;
    }
    showToast('Preparing your official PDF interview report...', 'info');
    window.location.href = `/interview/${sessionState.sessionId}/report`;
  });

  // Toggle Transcript Accordion
  btnToggleTranscript.addEventListener('click', () => {
    btnToggleTranscript.classList.toggle('open');
    transcriptContent.classList.toggle('hidden');
  });

  // Workflow Modal
  btnWorkflow.addEventListener('click', () => {
    workflowModal.classList.remove('hidden');
  });

  btnCloseModal.addEventListener('click', () => {
    workflowModal.classList.add('hidden');
  });

  workflowModal.addEventListener('click', (e) => {
    if (e.target === workflowModal) {
      workflowModal.classList.add('hidden');
    }
  });

  // Toast Notification System
  function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      toast.style.transition = 'all 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});
