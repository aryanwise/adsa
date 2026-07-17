/* =========================================================================
   ADSA Engine Console — demo application layer
   -------------------------------------------------------------------------
   HOW TO WIRE REAL ENDPOINTS LATER:
   Everything the UI renders flows through the `api` object below.
   Replace the demo bodies with fetch() calls — the shapes match your
   workspace session_state.json exactly, so no render code needs to change.

     api.getSession(workspace)  -> session_state.json contents
     api.getLogTail(workspace)  -> array of console lines (optional)

   Example:
     getSession: (ws) => fetch(`/api/workspaces/${ws}/session`).then(r => r.json())
   ========================================================================= */

"use strict";

/* ============================== demo data ================================ */
/* Mirrors the real session_state.json schema 1:1. */
const DEMO_STATE = {
  project_name: "test_workspace",
  Phase_1_Planning: {
    status: "COMPLETED",
    prompt: "Build a high-performance classification pipeline to predict 'CUSTOMER_SEGMENT'.",
    data_profile_path: "data_info/profiler_summary.txt",
    blueprint_path: "data_info/blueprint.txt",
  },
  Phase_2_Execution: {
    status: "PENDING",
    current_step_id: "step_1",
    current_step_index: 1,
    step_results: {},
    execution_plan: {
      step_1:  { title: "**Step 1: Data Loading**", status: "PENDING", script_path: "generated_scripts/step_1.py", attempts: 0 },
      step_2:  { title: "**Step 2: Initial Profiling & Quality Checks**", status: "PENDING", script_path: "generated_scripts/step_2.py", attempts: 0 },
      step_3:  { title: "**Step 3: Feature Engineering**", status: "PENDING", script_path: "generated_scripts/step_3.py", attempts: 0 },
      step_4:  { title: "**Step 4: Outlier Treatment & Skewness Mitigation**", status: "PENDING", script_path: "generated_scripts/step_4.py", attempts: 0 },
      step_5:  { title: "**Step 5: Preprocessing Pipeline Construction**", status: "PENDING", script_path: "generated_scripts/step_5.py", attempts: 0 },
      step_6:  { title: "**Step 6: Model Candidate Specification**", status: "PENDING", script_path: "generated_scripts/step_6.py", attempts: 0 },
      step_7:  { title: "**Step 7: Stratified 5-Fold Cross-Validation Setup**", status: "PENDING", script_path: "generated_scripts/step_7.py", attempts: 0 },
      step_8:  { title: "**Step 8: Model Training & Selection**", status: "PENDING", script_path: "generated_scripts/step_8.py", attempts: 0 },
      step_9:  { title: "**Step 9: Evaluation Artifacts Generation**", status: "PENDING", script_path: "generated_scripts/step_9.py", attempts: 0 },
      step_10: { title: "**Step 10: Persistence for Production**", status: "PENDING", script_path: "generated_scripts/step_10.py", attempts: 0 },
    },
  },
  Phase_3_Reporting: { status: "PENDING", final_report_path: "", metrics: {} },
};

/* Static demo context (later: derive from profiler_summary.txt + config.py) */
const DEMO_DATASET = {
  path: "data/active/copy_test_data.csv",
  rows: 100,
  columns: ["ORDER_ID", "MONTH_KEY", "CUSTOMER_SEGMENT", "TOTAL_REVENUE_USD", "ITEM_COUNT"],
};

const DEMO_MODELS = [
  { role: "Architect", model: "openai/gpt-oss-120b", kind: "cloud" },
  { role: "Coder", model: "llama-3.3-70b-versatile", kind: "cloud" },
  { role: "Debugger", model: "deepseek-r1:8b", kind: "mac" },
  { role: "Analyst", model: "gemini-1.5-pro", kind: "cloud" },
];

/* ============================== api seam ================================= */
const clone = (obj) =>
  typeof structuredClone === "function"
    ? structuredClone(obj)
    : JSON.parse(JSON.stringify(obj));

const api = {
  async getSession(workspace) {
    // TODO: return fetch(`/api/workspaces/${workspace}/session`).then(r => r.json());
    return clone(DEMO_STATE);
  },
  async getLogTail(workspace) {
    // TODO: return fetch(`/api/workspaces/${workspace}/log`).then(r => r.json());
    return [];
  },
};

/* ============================== dom refs ================================= */
const el = {
  rail: document.getElementById("phase-rail"),
  project: document.getElementById("session-project"),
  stepList: document.getElementById("step-list"),
  stepsSummary: document.getElementById("steps-summary"),
  statGrid: document.getElementById("stat-grid"),
  modelList: document.getElementById("model-list"),
  dataCard: document.getElementById("data-card"),
  log: document.getElementById("log-stream"),
  replayBtn: document.getElementById("replay-btn"),
  liveDot: document.getElementById("live-dot"),
  themeToggle: document.getElementById("theme-toggle"),
  workspaceSelect: document.getElementById("workspace-select"),
  // Ingestion
  dropZone: document.getElementById("drop-zone"),
  fileInput: document.getElementById("file-input"),
  browseBtn: document.getElementById("browse-btn"),
  ingestPreview: document.getElementById("ingest-preview"),
  previewName: document.getElementById("preview-name"),
  previewMeta: document.getElementById("preview-meta"),
  ingestBtn: document.getElementById("ingest-btn"),
  ingestStatus: document.getElementById("ingest-status"),
  // Chat
  chatHistory: document.getElementById("chat-history"),
  chatInput: document.getElementById("chat-input"),
  chatSend: document.getElementById("chat-send"),
  chatStatus: document.getElementById("chat-status"),
};

/* ============================== theme ==================================== */
const THEME_KEY = "adsa-theme";

function storedTheme() {
  try { return localStorage.getItem(THEME_KEY); } catch { return null; }
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  const toLight = theme === "dark";
  el.themeToggle.setAttribute("aria-label", `Switch to ${toLight ? "light" : "dark"} theme`);
  el.themeToggle.setAttribute("aria-pressed", String(theme === "light"));
  try { localStorage.setItem(THEME_KEY, theme); } catch { /* private mode: session only */ }
}

function initTheme() {
  const saved = storedTheme();
  const prefersLight = window.matchMedia?.("(prefers-color-scheme: light)").matches;
  applyTheme(saved ?? (prefersLight ? "light" : "dark"));
  el.themeToggle.addEventListener("click", () => {
    const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    applyTheme(next);
  });
}

/* ============================== rendering ================================ */
const PHASES = [
  { key: "Phase_1_Planning", label: "plan" },
  { key: "Phase_2_Execution", label: "execute" },
  { key: "Phase_3_Reporting", label: "report" },
];

const STATUS_TEXT = {
  PENDING: "pending",
  RUNNING: "running",
  IN_PROGRESS: "running",
  COMPLETED: "done",
  FAILED: "failed",
  HALTED: "halted",
};

function renderRail(state) {
  el.rail.replaceChildren();
  PHASES.forEach((phase, i) => {
    const status = state[phase.key]?.status ?? "PENDING";
    const node = document.createElement("div");
    node.className = "rail-node";
    node.dataset.status = status;
    node.innerHTML = `
      <span class="rail-state" aria-hidden="true"></span>
      <span class="rail-label">${phase.label}</span>
      <span class="rail-status">${STATUS_TEXT[status] ?? status.toLowerCase()}</span>`;
    el.rail.appendChild(node);
    if (i < PHASES.length - 1) {
      const edge = document.createElement("span");
      edge.className = "rail-edge";
      edge.setAttribute("aria-hidden", "true");
      el.rail.appendChild(edge);
    }
  });
  el.project.textContent = state.project_name ?? "—";
}

function cleanTitle(raw) {
  return String(raw).replaceAll("*", "").replace(/^Step \d+:\s*/i, "").trim();
}

function attemptsDots(used, max = 3) {
  const dots = [];
  for (let i = 0; i < max; i++) {
    dots.push(i < used ? '<span class="used">●</span>' : "○");
  }
  return dots.join("");
}

function renderSteps(state) {
  const plan = state.Phase_2_Execution?.execution_plan ?? {};
  el.stepList.replaceChildren();

  let done = 0;
  const entries = Object.entries(plan);
  entries.forEach(([stepId, step], index) => {
    if (step.status === "COMPLETED") done++;
    const li = document.createElement("li");
    li.className = "step-row";
    li.dataset.status = step.status;
    li.innerHTML = `
      <span class="step-num">${String(index + 1).padStart(2, "0")}</span>
      <span class="step-title" title="${cleanTitle(step.title)}">${cleanTitle(step.title)}</span>
      <span class="step-attempts" aria-label="${step.attempts} of 3 attempts used">${attemptsDots(step.attempts)}</span>
      <span class="chip" data-status="${step.status}">${STATUS_TEXT[step.status] ?? step.status.toLowerCase()}</span>`;
    el.stepList.appendChild(li);
  });

  el.stepsSummary.textContent = `${done}/${entries.length} steps`;
}

function renderStats(state) {
  const plan = state.Phase_2_Execution?.execution_plan ?? {};
  const steps = Object.values(plan);
  const done = steps.filter(s => s.status === "COMPLETED").length;
  const repairs = steps.reduce((sum, s) => sum + Math.max(0, (s.attempts || 0) - 1), 0);
  const heals = state.__demo_heals ?? 0;
  const phase = state.Phase_2_Execution?.status ?? "PENDING";

  const stats = [
    { label: "steps done", value: `${done}<small>/ ${steps.length}</small>` },
    { label: "repairs", value: String(repairs) },
    { label: "dep heals", value: String(heals) },
    { label: "engine", value: STATUS_TEXT[phase] ?? phase.toLowerCase() },
  ];

  el.statGrid.replaceChildren();
  for (const s of stats) {
    const wrap = document.createElement("div");
    wrap.className = "stat";
    wrap.innerHTML = `<dt>${s.label}</dt><dd>${s.value}</dd>`;
    el.statGrid.appendChild(wrap);
  }
}

function renderModels() {
  el.modelList.replaceChildren();
  for (const m of DEMO_MODELS) {
    const li = document.createElement("li");
    li.className = "model-row";
    li.innerHTML = `
      <div>
        <span class="model-role">${m.role}</span>
        <span class="model-name">${m.model}</span>
      </div>
      <span class="tag" data-kind="${m.kind}">${m.kind}</span>`;
    el.modelList.appendChild(li);
  }
}

function renderDataset() {
  el.dataCard.innerHTML = `
    <div class="data-path">${DEMO_DATASET.path}</div>
    <div class="data-facts">
      <div class="data-fact">
        <span class="label-eyebrow">rows</span>
        <strong>${DEMO_DATASET.rows.toLocaleString()}</strong>
      </div>
      <div class="data-fact">
        <span class="label-eyebrow">columns</span>
        <strong>${DEMO_DATASET.columns.length}</strong>
      </div>
    </div>
    <div class="data-cols">
      ${DEMO_DATASET.columns.map(c => `<span class="col-pill">${c}</span>`).join("")}
    </div>`;
}

function renderAll(state) {
  renderRail(state);
  renderSteps(state);
  renderStats(state);
}

/* ============================== console log ============================== */
function logLine(html) {
  const line = document.createElement("span");
  line.className = "log-line";
  line.innerHTML = html;
  el.log.appendChild(line);
  el.log.scrollTop = el.log.scrollHeight;
}

const glyph = {
  branch: '<span class="t-dim">  ├─</span>',
  pipe: '<span class="t-dim">  │ </span>',
  sub: '<span class="t-dim">   ↳</span>',
  end: '<span class="t-dim">  └─</span>',
};

/* ============================== data ingestion =========================== */
function initIngestion() {
  el.browseBtn.addEventListener("click", () => el.fileInput.click());

  el.dropZone.addEventListener("click", (e) => {
    if (e.target !== el.browseBtn && !e.target.closest(".link-btn")) {
      el.fileInput.click();
    }
  });

  el.dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    el.dropZone.classList.add("drag-over");
  });

  el.dropZone.addEventListener("dragleave", () => {
    el.dropZone.classList.remove("drag-over");
  });

  el.dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    el.dropZone.classList.remove("drag-over");
    const files = e.dataTransfer.files;
    if (files.length) handleFile(files[0]);
  });

  el.fileInput.addEventListener("change", () => {
    if (el.fileInput.files.length) handleFile(el.fileInput.files[0]);
  });

  el.ingestBtn.addEventListener("click", () => {
    el.ingestBtn.disabled = true;
    el.ingestBtn.textContent = "Ingesting...";
    el.ingestStatus.textContent = "Processing...";

    setTimeout(() => {
      el.ingestBtn.textContent = "Ingested";
      el.ingestStatus.textContent = "Ready";
      el.ingestStatus.style.color = "var(--ok)";

      const fileName = el.previewName.textContent;
      logLine(`${glyph.branch} <span class="t-ok">Dataset ingested:</span> <span class="t-accent">${fileName}</span>`);

      // Simulate updated dataset profile
      DEMO_DATASET.path = `data/active/${fileName}`;
      DEMO_DATASET.rows = 1247;
      DEMO_DATASET.columns = ["ID", "FEATURE_A", "FEATURE_B", "FEATURE_C", "TARGET"];
      renderDataset();

      // Also update the demo state to show planning phase used this data
      logLine(`${glyph.pipe}   <span class="t-dim">Profiler summary written to data_info/profiler_summary.txt</span>`);
    }, 1400);
  });
}

function handleFile(file) {
  el.previewName.textContent = file.name;
  const size = file.size < 1024 * 1024
    ? `${(file.size / 1024).toFixed(1)} KB`
    : `${(file.size / (1024 * 1024)).toFixed(2)} MB`;
  const type = file.type || "text/csv";
  el.previewMeta.innerHTML = `<span>${size}</span><span>${type}</span>`;
  el.dropZone.hidden = true;
  el.ingestPreview.hidden = false;
  el.ingestStatus.textContent = "Selected";
}

/* ============================== LLM chat ================================ */
const LLM_RESPONSES = [
  { keywords: ["hello", "hi", "hey"], response: "Hello! I'm the ADSA Architect. How can I help you design your pipeline today?" },
  { keywords: ["help", "how"], response: "I can help you design ML pipelines, select models, debug failures, or explain the current execution plan. What would you like to do?" },
  { keywords: ["plan", "pipeline", "steps"], response: "The current plan has 10 steps: data loading, profiling, feature engineering, outlier treatment, preprocessing, model specification, CV setup, training, evaluation, and persistence." },
  { keywords: ["model", "algorithm", "xgboost", "rf", "random forest"], response: "I've registered 3 candidates: Logistic Regression, Random Forest, and XGBoost. Based on your dataset profile, XGBoost is likely to perform best given the skewed features." },
  { keywords: ["data", "dataset", "file", "csv"], response: "You can ingest a new dataset using the Data Ingestion panel above. Once ingested, I'll auto-profile it and suggest the optimal pipeline structure." },
  { keywords: ["error", "fail", "bug", "crash"], response: "I see. Let me check the execution graph for any failed steps. If a script crashes, I can auto-repair it up to 3 attempts. Would you like me to inspect the logs?" },
  { keywords: ["feature", "engineer", "column"], response: "Feature engineering is handled in Step 3. I automatically detect categorical encodings, datetime decompositions, and interaction terms based on the profiler output." },
];

function findResponse(text) {
  const lower = text.toLowerCase();
  for (const entry of LLM_RESPONSES) {
    if (entry.keywords.some(k => lower.includes(k))) return entry.response;
  }
  return "I understand. Let me analyze that for you. Could you provide more details about your data or the specific issue you're seeing?";
}

function addChatMessage(text, sender) {
  const msg = document.createElement("div");
  msg.className = `chat-msg ${sender}`;
  msg.textContent = text;
  el.chatHistory.appendChild(msg);
  el.chatHistory.scrollTop = el.chatHistory.scrollHeight;
}

function showTyping() {
  const typing = document.createElement("div");
  typing.className = "chat-msg assistant typing-indicator";
  typing.id = "typing-indicator";
  typing.innerHTML = "<span></span><span></span><span></span>";
  el.chatHistory.appendChild(typing);
  el.chatHistory.scrollTop = el.chatHistory.scrollHeight;
}

function hideTyping() {
  const typing = document.getElementById("typing-indicator");
  if (typing) typing.remove();
}

function sendChat() {
  const text = el.chatInput.value.trim();
  if (!text) return;

  addChatMessage(text, "user");
  el.chatInput.value = "";
  el.chatStatus.textContent = "Thinking...";
  el.chatSend.disabled = true;

  showTyping();

  const response = findResponse(text);
  const delay = 1200 + Math.random() * 800;

  setTimeout(() => {
    hideTyping();
    addChatMessage(response, "assistant");
    el.chatStatus.textContent = "Idle";
    el.chatSend.disabled = false;
  }, delay);
}

function initChat() {
  el.chatSend.addEventListener("click", sendChat);
  el.chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendChat();
  });
}

/* ============================== demo replay ==============================
   A scripted playback of a real engine run (the integration test narrative):
   step 2 crashes with a KeyError and is repaired; step 3 hits a missing
   dependency and self-heals. Each event patches state + emits log lines,
   exactly the rhythm a real /log endpoint would produce.               */

let replayTimers = [];

function clearReplay() {
  replayTimers.forEach(clearTimeout);
  replayTimers = [];
}

function buildReplayScript(state) {
  const plan = state.Phase_2_Execution.execution_plan;
  const events = [];
  const stepIds = Object.keys(plan);

  events.push(() => {
    state.Phase_2_Execution.status = "RUNNING";
    logLine(`${glyph.branch} Execution graph loaded: <span class="t-accent">${stepIds.length} steps</span>`);
  });

  for (const stepId of stepIds) {
    const step = plan[stepId];
    const title = cleanTitle(step.title);

    events.push(() => {
      step.status = "IN_PROGRESS";
      step.attempts = 1;
      state.Phase_2_Execution.current_step_id = stepId;
      logLine(`${glyph.branch} Executing ${stepId}: <span class="t-accent">${title}</span>`);
      logLine(`${glyph.pipe}${glyph.sub} generating script (attempt 1) ...`);
    });

    if (stepId === "step_3") {
      // planted failure: crash -> cloud repair
      events.push(() => {
        logLine(`${glyph.pipe}${glyph.sub} <span class="t-err">crashed (KeyError) on attempt 1</span>`);
      });
      events.push(() => {
        step.attempts = 2;
        logLine(`${glyph.pipe}${glyph.sub} repairing script (attempt 2) ...`);
      });
      events.push(() => {
        step.status = "COMPLETED";
        logLine(`${glyph.pipe}${glyph.sub} <span class="t-ok">Success!</span>`);
        logLine(`${glyph.pipe}   <span class="t-dim">MONTH_NUM engineered; ORDER_ID, MONTH_KEY dropped</span>`);
      });
    } else if (stepId === "step_8") {
      // planted heal: missing xgboost -> auto pip install -> re-run
      events.push(() => {
        state.__demo_heals = (state.__demo_heals ?? 0) + 1;
        logLine(`${glyph.pipe}    <span class="t-warn">⚕ Missing dependency detected: xgboost</span>`);
        logLine(`${glyph.pipe}    <span class="t-warn">⚕ Auto-installing from requirements.txt ...</span>`);
      });
      events.push(() => {
        logLine(`${glyph.pipe}    <span class="t-ok">⚕ Dependency installed. Re-running script.</span>`);
      });
      events.push(() => {
        step.status = "COMPLETED";
        logLine(`${glyph.pipe}${glyph.sub} <span class="t-ok">Success!</span>`);
        logLine(`${glyph.pipe}   <span class="t-dim">best model: XGBoost · mean macro F1 = 0.912</span>`);
      });
    } else {
      const notes = {
        step_1: "Loaded and validated: (100, 5)",
        step_2: "skew confirmed: TOTAL_REVENUE_USD 0.74 · ITEM_COUNT 0.95",
        step_4: "winsorized p1/p99 · log1p applied to 2 features",
        step_5: "ColumnTransformer fitted · saved to artifacts/",
        step_6: "3 candidates registered: LogReg · RF · XGB",
        step_7: "StratifiedKFold(n_splits=5, shuffle=True) ready",
        step_9: "confusion matrix saved to artifacts/confusion.png",
        step_10: "manifest + model persisted to artifacts/",
      };
      events.push(() => {
        step.status = "COMPLETED";
        logLine(`${glyph.pipe}${glyph.sub} <span class="t-ok">Success!</span>`);
        if (notes[stepId]) logLine(`${glyph.pipe}   <span class="t-dim">${notes[stepId]}</span>`);
      });
    }
  }

  events.push(() => {
    state.Phase_2_Execution.status = "COMPLETED";
    state.Phase_3_Reporting.status = "RUNNING";
    logLine(`${glyph.end} <span class="t-ok">Phase 2 Completed! All steps executed and state persisted.</span>`);
  });

  events.push(() => {
    state.Phase_3_Reporting.status = "COMPLETED";
    logLine(`${glyph.end} <span class="t-ok">Phase 3 report drafted → final_report.md</span>`);
  });

  return events;
}

async function replayDemoRun() {
  clearReplay();
  el.replayBtn.disabled = true;
  el.liveDot.hidden = false;
  el.log.replaceChildren();

  const state = await api.getSession(el.workspaceSelect.value);
  state.__demo_heals = 0;
  renderAll(state);

  const events = buildReplayScript(state);
  const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
  const stepDelay = reduced ? 0 : 520;

  events.forEach((event, i) => {
    const timer = setTimeout(() => {
      event();
      renderAll(state);
      if (i === events.length - 1) {
        el.replayBtn.disabled = false;
        el.liveDot.hidden = true;
      }
    }, i * stepDelay);
    replayTimers.push(timer);
  });
}

/* ============================== boot ===================================== */
async function boot() {
  initTheme();
  initIngestion();
  initChat();
  renderModels();
  renderDataset();

  const state = await api.getSession(el.workspaceSelect.value);
  renderAll(state);

  logLine(`${glyph.branch} <span class="t-dim">console idle — press "Replay demo run" to simulate an engine session</span>`);

  el.replayBtn.addEventListener("click", replayDemoRun);
  el.workspaceSelect.addEventListener("change", async () => {
    clearReplay();
    el.log.replaceChildren();
    renderAll(await api.getSession(el.workspaceSelect.value));
  });
}

boot();