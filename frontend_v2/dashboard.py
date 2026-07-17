"""
ADSA Console — Streamlit dashboard for the Autonomous Data Science Agent.

Run from the repo root:
    export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES && streamlit run dashboard.py

Reads everything from workspaces/<name>/ — no engine imports, no API needed.
Sections: Executive Summary, Leaderboard, Data & Blueprint, Engine Logs.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import streamlit as st

# ============================================================ configuration
def _find_workspaces_dir() -> Path:
    """Locate workspaces/ regardless of where streamlit was launched from."""
    here = Path(__file__).resolve().parent
    candidates = [Path.cwd() / "workspaces",
                  here / "workspaces",
                  here.parent / "workspaces",
                  Path.cwd().parent / "workspaces"]
    for c in candidates:
        if c.is_dir():
            return c
    if (Path.cwd() / "artifacts").is_dir():
        return Path.cwd().parent
    return Path("workspaces")

WORKSPACES_DIR = _find_workspaces_dir()

PALETTE = {
    "bg": "#0F1113", "panel": "#15181B", "ink": "#E7E5E0", "muted": "#8B9096",
    "line": "#24282C", "accent": "#53C2D4", "ok": "#57B48A",
    "warn": "#D9A03F", "err": "#E07A6E",
}

STATUS_COLOR = {
    "COMPLETED": PALETTE["ok"], "RUNNING": PALETTE["accent"],
    "IN_PROGRESS": PALETTE["accent"], "PENDING": PALETTE["muted"],
    "FAILED": PALETTE["err"], "HALTED": PALETTE["err"],
}

# ============================================================ safe data loaders (no pandas/pyarrow segfault risk)
@st.cache_data
def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def load_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""

def load_session(ws: Path) -> dict:
    return load_json(ws / "session_state.json")

def safe_read_csv_dict(path: Path, max_rows: int | None = None) -> list[dict]:
    """Read CSV using stdlib csv — zero chance of pyarrow/numpy segfault."""
    rows = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if max_rows is not None and i >= max_rows:
                    break
                rows.append(row)
    except Exception:
        pass
    return rows

def safe_read_csv_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    """Return (headers, rows) using stdlib csv."""
    headers, rows = [], []
    try:
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            headers = next(reader, [])
            rows = list(reader)
    except Exception:
        pass
    return headers, rows

def csv_to_dict_list(headers: list[str], rows: list[list[str]]) -> list[dict]:
    return [dict(zip(headers, row)) for row in rows]

def try_float(val: str) -> float | None:
    try:
        return float(val)
    except Exception:
        return None

def pipeline_summary(session: dict, ws: Path) -> dict:
    plan = session.get("Phase_2_Execution", {}).get("execution_plan", {})
    steps = list(plan.values())
    done = sum(1 for s in steps if s.get("status") == "COMPLETED")
    repairs = sum(max(0, (s.get("attempts") or 0) - 1) for s in steps)

    metrics = session.get("Phase_3_Reporting", {}).get("metrics", {}) or {}
    best_model = metrics.get("model_type")
    best_f1 = next((v for k, v in metrics.items()
                    if isinstance(v, (int, float))
                    and any(t in k.lower() for t in ("holdout", "test"))), None)

    if best_model is None or best_f1 is None:
        lb = leaderboard_data(ws)
        if lb and len(lb):
            best_model = best_model or str(lb[0]["model"])
            best_f1 = best_f1 if best_f1 is not None else float(lb[0]["score"])

    return {"total": len(steps), "done": done, "repairs": repairs,
            "best_model": best_model or "—", "holdout_f1": best_f1}

@st.cache_data
def leaderboard_data(ws: Path) -> list[dict] | None:
    path = ws / "artifacts" / "model_cv_report.csv"
    if not path.exists():
        return None
    headers, rows = safe_read_csv_rows(path)
    if not headers or not rows:
        return None
    # Find numeric columns
    numeric_cols = []
    for c in headers:
        for row in rows:
            val = try_float(row[headers.index(c)] if c in headers else "")
            if val is not None:
                numeric_cols.append(c)
                break
    if not numeric_cols:
        return None
    name_col = headers[0]
    mean_col = next((c for c in numeric_cols if "mean" in c.lower()), numeric_cols[-1])
    out = []
    for row in rows:
        d = dict(zip(headers, row))
        score = try_float(d.get(mean_col, ""))
        if score is not None:
            out.append({"model": d.get(name_col, ""), "score": score})
    out.sort(key=lambda x: x["score"], reverse=True)
    return out

@st.cache_data
def dataset_stats(ws: Path) -> dict:
    profile = load_json(ws / "artifacts" / "data_profile.json")
    if not profile:
        profile = load_json(ws / "data_info" / "profiler_summary.txt")
    d = profile.get("dataset", {}) if isinstance(profile, dict) else {}
    rows, cols = d.get("rows"), d.get("columns")
    active = ws / "data" / "active" / "copy_test_data.csv"
    if (rows is None or cols is None) and active.exists():
        headers, data_rows = safe_read_csv_rows(active)
        cols = cols or len(headers)
        rows = rows or len(data_rows)
    return {"rows": rows, "cols": cols, "profile": profile}

def infer_target(ws: Path, session: dict) -> str:
    objective = session.get("Phase_1_Planning", {}).get("prompt", "")
    m = re.search(r"['']([A-Za-z_][A-Za-z0-9_]*)['']", objective)
    if m:
        return m.group(1)
    return "—"

# ============================================================ UI helpers
def chip(label: str, status: str) -> str:
    c = STATUS_COLOR.get(status, PALETTE["muted"])
    return (f'<span class="chip" style="color:{c};border-color:{c}55">'
            f'<span class="dot" style="background:{c}"></span>'
            f'{label} · {status.lower().replace("_", " ")}</span>')

def inject_css() -> None:
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Familjen+Grotesk:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {{ font-family: 'Familjen Grotesk', system-ui, sans-serif; }}
    .stApp {{ background: {PALETTE['bg']}; color: {PALETTE['ink']}; }}
    #MainMenu, footer, header {{ visibility: hidden; }}
    section[data-testid="stSidebar"] {{ background: {PALETTE['panel']};
        border-right: 1px solid {PALETTE['line']}; }}
    .block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
    .wordmark {{ display:flex; align-items:center; gap:10px; margin-bottom:2px; }}
    .wordmark .mark {{ width:10px; height:10px; background:{PALETTE['accent']};
        transform:rotate(45deg); border-radius:2px; }}
    .wordmark .name {{ font-weight:700; letter-spacing:.14em; font-size:17px; }}
    .wordmark .sub {{ font-family:'JetBrains Mono',monospace; font-size:11px;
        color:{PALETTE['muted']}; }}
    .eyebrow {{ font-family:'JetBrains Mono',monospace; font-size:10px;
        letter-spacing:.16em; text-transform:uppercase; color:{PALETTE['muted']};
        margin: 4px 0 6px; }}
    .chip {{ display:inline-flex; align-items:center; gap:7px;
        font-family:'JetBrains Mono',monospace; font-size:10.5px;
        letter-spacing:.1em; text-transform:uppercase; padding:4px 11px;
        border:1px solid; border-radius:999px; margin-right:8px; }}
    .chip .dot {{ width:6px; height:6px; border-radius:50%; }}
    div[data-testid="stMetric"] {{ background:{PALETTE['panel']};
        border:1px solid {PALETTE['line']}; border-radius:6px; padding:12px 16px; }}
    div[data-testid="stMetric"] label {{ font-family:'JetBrains Mono',monospace;
        font-size:10px !important; letter-spacing:.14em; text-transform:uppercase;
        color:{PALETTE['muted']} !important; }}
    div[data-testid="stMetricValue"] {{ font-size:22px; }}
    .stTabs [data-baseweb="tab"] {{ font-family:'JetBrains Mono',monospace;
        font-size:14px; letter-spacing:.05em; padding-bottom: 10px; }}
    div[data-testid="stExpander"] {{ background:{PALETTE['panel']};
        border:1px solid {PALETTE['line']}; border-radius:6px; }}
    code, pre, .stCode {{ font-family:'JetBrains Mono',monospace !important; }}
    </style>""", unsafe_allow_html=True)


# ============================================================ page
def main() -> None:
    st.set_page_config(page_title="ADSA Console", layout="wide", initial_sidebar_state="expanded")
    inject_css()

    # ---------------- sidebar and workspace selection
    with st.sidebar:
        st.markdown('<div class="wordmark"><span class="mark"></span>'
                    '<span class="name">ADSA</span>'
                    '<span class="sub">engine console</span></div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="eyebrow">workspace</div>', unsafe_allow_html=True)
        options = sorted(p.name for p in WORKSPACES_DIR.iterdir()
                         if p.is_dir()) if WORKSPACES_DIR.exists() else []
        if not options:
            st.error("No workspaces found. Run `adsa init <name>` first.")
            st.stop()

        ws_name = st.selectbox("Workspace", options, label_visibility="collapsed")
        ws = WORKSPACES_DIR / ws_name

    session = load_session(ws)
    if not session:
        st.error(f"session_state.json not found in {ws}. Run Phase 1 first.")
        st.stop()

    summary = pipeline_summary(session, ws)

    # ---------------- header: phase rail + headline metrics
    phases = [("plan", "Phase_1_Planning"), ("execute", "Phase_2_Execution"),
              ("report", "Phase_3_Reporting")]
    chips = "".join(chip(lbl, session.get(key, {}).get("status", "PENDING"))
                    for lbl, key in phases)
    st.markdown(chips, unsafe_allow_html=True)
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Steps Executed", f"{summary['done']} / {summary['total']}")
    c2.metric("Script Repairs", summary["repairs"])
    c3.metric("Selected Model", summary["best_model"])
    c4.metric("Holdout F1 Score", f"{summary['holdout_f1']:.3f}" if summary["holdout_f1"] is not None else "—")
    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------- Clean Tabs Layout
    tab_report, tab_board, tab_data, tab_pipe = st.tabs([
        "Executive Summary", "Model Leaderboard", "Data & Blueprint", "Engine Logs"
    ])

    # ---------------- TAB 1: Executive Summary
    with tab_report:
        report = load_text(ws / "final_report.md")
        if report:
            cleaned = re.sub(r"!\[[^\]]*\]\([^)]*confusion_matrix\.png\)", "", report)
            st.markdown(cleaned)

            cm = ws / "artifacts" / "confusion_matrix.png"
            if "confusion_matrix" in report and cm.exists():
                st.markdown("### Confusion Matrix")
                st.image(str(cm), width=520)
        else:
            st.info("final_report.md not found — run Phase 3 to generate it.")

        st.markdown('<br><div class="eyebrow">Generated Artifact Inventory</div>', unsafe_allow_html=True)
        art = ws / "artifacts"
        if art.exists():
            rows = [{"Filename": f.name, "Size (Bytes)": f"{f.stat().st_size:,}"}
                    for f in sorted(art.iterdir()) if f.is_file()]
            if rows:
                st.table(rows)
            else:
                st.caption("No artifacts found.")

    # ---------------- TAB 2: Model Leaderboard
    with tab_board:
        lb = leaderboard_data(ws)
        left, right = st.columns([1.15, 1])
        with left:
            st.markdown('<div class="eyebrow">cross-validation score comparison</div>', unsafe_allow_html=True)
            if lb is not None:
                # Native st.bar_chart — no altair, no segfault risk
                chart_data = {row["model"]: row["score"] for row in lb}
                st.bar_chart(chart_data, use_container_width=True)

                # Full matrix as plain table
                full_headers, full_rows = safe_read_csv_rows(ws / "artifacts" / "model_cv_report.csv")
                with st.expander("View Full Cross-Validation Matrix"):
                    if full_headers:
                        st.table([dict(zip(full_headers, r)) for r in full_rows])
                    else:
                        st.caption("Could not read CSV.")
            else:
                st.caption("model_cv_report.csv not found — run Phase 2.")

        with right:
            st.markdown('<div class="eyebrow">holdout evaluation matrix</div>', unsafe_allow_html=True)
            cm = ws / "artifacts" / "confusion_matrix.png"
            if cm.exists():
                st.image(str(cm), use_column_width=True)
            else:
                st.caption("confusion_matrix.png not found.")

    # ---------------- TAB 3: Data & Blueprint
    with tab_data:
        stats = dataset_stats(ws)
        target = infer_target(ws, session)
        d1, d2, d3 = st.columns(3)
        d1.metric("Total Rows", f"{stats['rows']:,}" if stats["rows"] else "—")
        d2.metric("Total Columns", stats["cols"] or "—")
        d3.metric("Target Variable", target)

        active = ws / "data" / "active" / "copy_test_data.csv"
        if active.exists():
            st.markdown('<br><div class="eyebrow">active dataset preview</div>', unsafe_allow_html=True)
            preview = safe_read_csv_dict(active, max_rows=15)
            if preview:
                st.table(preview)
            else:
                st.warning("Could not preview active CSV.")

        left, right = st.columns(2)
        with left:
            st.markdown('<div class="eyebrow">Phase 1 Data Profiler Summary</div>', unsafe_allow_html=True)
            summary_txt = load_text(ws / "data_info" / "llm_summary.txt")
            st.markdown(summary_txt or "_llm_summary.txt not found._")
        with right:
            st.markdown('<div class="eyebrow">Raw Statistical Profile JSON</div>', unsafe_allow_html=True)
            if stats["profile"]:
                with st.expander("data_profile.json", expanded=False):
                    st.json(stats["profile"])
            else:
                st.caption("No profile JSON found.")

        st.markdown('<div class="eyebrow">Phase 1 Pipeline Contract (Blueprint)</div>', unsafe_allow_html=True)
        blueprint = load_text(ws / "data_info" / "blueprint.md")
        with st.expander("View blueprint.md", expanded=False):
            st.markdown(blueprint or "_blueprint.md not found._")

    # ---------------- TAB 4: Engine Logs
    with tab_pipe:
        plan = session.get("Phase_2_Execution", {}).get("execution_plan", {})
        results = session.get("Phase_2_Execution", {}).get("step_results", {})

        if not plan:
            st.info("No execution plan yet. Run Phase 1 to generate the blueprint.")

        for i, (step_id, step) in enumerate(plan.items(), start=1):
            title = re.sub(r"[#*]", "", step.get("title", step_id)).strip()
            status = step.get("status", "PENDING")
            attempts = step.get("attempts", 0)
            res = results.get(step_id, {})

            header = f"Step {i:02d} | {title} | {status.upper()}" + (f" ({attempts} attempts)" if attempts else "")

            with st.expander(header, expanded=(status in ("FAILED", "HALTED"))):
                if res.get("stdout_tail"):
                    st.markdown('<div class="eyebrow">Standard Output</div>', unsafe_allow_html=True)
                    st.code(res["stdout_tail"], language="text")
                if res.get("stderr_tail"):
                    st.markdown('<div class="eyebrow">Standard Error</div>', unsafe_allow_html=True)
                    st.code(res["stderr_tail"], language="text")

                script = ws / step.get("script_path", "")
                if script.exists():
                    st.markdown('<div class="eyebrow">Generated Script</div>', unsafe_allow_html=True)
                    st.code(load_text(script), language="python")

                if not res and not script.exists():
                    st.caption("No output recorded for this step yet.")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        st.error("Dashboard encountered an unexpected error.")
        st.code(traceback.format_exc())