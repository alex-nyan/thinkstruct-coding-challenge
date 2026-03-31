"""
Patent Search Engine — Web Interface
=====================================
Flask application providing a polished browser UI for searching, filtering,
and exploring patent data. Supports:
  - Natural language search (patent-level and claim-level)
  - Hybrid filtering (classification codes, keywords, title)
  - Patent detail view with claims and description
  - Similar patent discovery
  - Search history (session-based)
  - Evaluation dashboard
"""

import json, os, time
from flask import Flask, render_template_string, request, jsonify, session
from patent_engine import create_engine, PatentSearchEngine

app = Flask(__name__)
app.secret_key = os.urandom(24)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "patent_data_small")
engine: PatentSearchEngine = create_engine(DATA_DIR)
stats = engine.stats()


HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PatentLens — Intelligent Patent Search</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #0b0f14;
  --bg-card: #131920;
  --bg-card-hover: #1a2230;
  --bg-input: #0e1318;
  --border: #1e2a38;
  --border-focus: #3b82f6;
  --text: #e2e8f0;
  --text-dim: #8b9cb8;
  --text-muted: #4a5d78;
  --accent: #3b82f6;
  --accent-glow: rgba(59,130,246,0.15);
  --green: #10b981;
  --amber: #f59e0b;
  --red: #ef4444;
  --purple: #8b5cf6;
  --font: 'DM Sans', -apple-system, sans-serif;
  --mono: 'JetBrains Mono', monospace;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
  font-family: var(--font);
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
  -webkit-font-smoothing: antialiased;
}

/* ---- Header ---- */
.header {
  border-bottom: 1px solid var(--border);
  padding: 1rem 2rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  backdrop-filter: blur(12px);
  position: sticky;
  top: 0;
  z-index: 100;
  background: rgba(11,15,20,0.85);
}
.logo {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  font-size: 1.25rem;
  font-weight: 700;
  letter-spacing: -0.03em;
}
.logo svg { width: 28px; height: 28px; }
.logo span.accent { color: var(--accent); }
.header-stats {
  display: flex;
  gap: 1.5rem;
  font-size: 0.8rem;
  color: var(--text-dim);
}
.header-stats .num { color: var(--accent); font-weight: 600; font-family: var(--mono); }

.nav-tabs {
  display: flex;
  gap: 0;
  border-bottom: 1px solid var(--border);
  padding: 0 2rem;
}
.nav-tab {
  padding: 0.75rem 1.25rem;
  font-size: 0.85rem;
  font-weight: 500;
  color: var(--text-dim);
  cursor: pointer;
  border-bottom: 2px solid transparent;
  transition: all 0.2s;
}
.nav-tab:hover { color: var(--text); }
.nav-tab.active { color: var(--accent); border-bottom-color: var(--accent); }

/* ---- Layout ---- */
.main { max-width: 1200px; margin: 0 auto; padding: 2rem; }
.panel { display: none; }
.panel.active { display: block; }

/* ---- Search Panel ---- */
.search-box {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
}
.search-row {
  display: flex;
  gap: 0.75rem;
  margin-bottom: 1rem;
}
.search-input {
  flex: 1;
  background: var(--bg-input);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.75rem 1rem;
  font-size: 0.95rem;
  color: var(--text);
  font-family: var(--font);
  transition: border-color 0.2s;
  outline: none;
}
.search-input:focus { border-color: var(--border-focus); box-shadow: 0 0 0 3px var(--accent-glow); }
.search-input::placeholder { color: var(--text-muted); }

.btn {
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 8px;
  font-size: 0.85rem;
  font-weight: 600;
  cursor: pointer;
  font-family: var(--font);
  transition: all 0.15s;
}
.btn-primary {
  background: var(--accent);
  color: #fff;
}
.btn-primary:hover { background: #2563eb; transform: translateY(-1px); }
.btn-ghost {
  background: transparent;
  border: 1px solid var(--border);
  color: var(--text-dim);
}
.btn-ghost:hover { border-color: var(--text-dim); color: var(--text); }

/* Filters */
.filters {
  display: none;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 0.75rem;
  margin-top: 0.75rem;
  padding-top: 0.75rem;
  border-top: 1px solid var(--border);
}
.filters.open { display: grid; }
.filter-group label {
  display: block;
  font-size: 0.75rem;
  color: var(--text-dim);
  margin-bottom: 0.3rem;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.filter-input {
  width: 100%;
  background: var(--bg-input);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.5rem 0.75rem;
  font-size: 0.85rem;
  color: var(--text);
  font-family: var(--font);
  outline: none;
}
.filter-input:focus { border-color: var(--border-focus); }
select.filter-input { appearance: none; cursor: pointer; }

/* Options row */
.options-row {
  display: flex;
  align-items: center;
  gap: 1rem;
  flex-wrap: wrap;
}
.option-group {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.8rem;
  color: var(--text-dim);
}
.option-group select, .option-group input[type=number] {
  background: var(--bg-input);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.35rem 0.5rem;
  font-size: 0.8rem;
  color: var(--text);
  font-family: var(--font);
  outline: none;
}

/* ---- Results ---- */
.results-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}
.results-header h3 { font-size: 0.9rem; font-weight: 500; color: var(--text-dim); }
.timing-badge {
  font-family: var(--mono);
  font-size: 0.75rem;
  padding: 0.25rem 0.6rem;
  background: var(--accent-glow);
  color: var(--accent);
  border-radius: 20px;
}

.result-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1.25rem;
  margin-bottom: 0.75rem;
  transition: all 0.15s;
  cursor: pointer;
}
.result-card:hover { border-color: var(--accent); background: var(--bg-card-hover); transform: translateX(4px); }

.result-top {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 1rem;
  margin-bottom: 0.6rem;
}
.result-title {
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--text);
  line-height: 1.4;
}
.result-score {
  font-family: var(--mono);
  font-size: 0.8rem;
  font-weight: 600;
  padding: 0.2rem 0.5rem;
  border-radius: 6px;
  white-space: nowrap;
  flex-shrink: 0;
}
.score-high { background: rgba(16,185,129,0.15); color: var(--green); }
.score-mid { background: rgba(245,158,11,0.15); color: var(--amber); }
.score-low { background: rgba(239,68,68,0.1); color: var(--red); }

.result-meta {
  display: flex;
  gap: 1rem;
  font-size: 0.78rem;
  color: var(--text-dim);
  margin-bottom: 0.5rem;
  flex-wrap: wrap;
}
.meta-tag {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
}
.class-badge {
  font-family: var(--mono);
  font-size: 0.72rem;
  padding: 0.15rem 0.45rem;
  background: rgba(139,92,246,0.15);
  color: var(--purple);
  border-radius: 4px;
}
.result-snippet {
  font-size: 0.85rem;
  color: var(--text-dim);
  line-height: 1.6;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.claim-label {
  font-size: 0.72rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--amber);
  margin-bottom: 0.3rem;
}

/* ---- Patent Detail Modal ---- */
.modal-overlay {
  display: none;
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.7);
  z-index: 200;
  justify-content: center;
  align-items: flex-start;
  padding: 3rem 1rem;
  overflow-y: auto;
}
.modal-overlay.open { display: flex; }
.modal {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 14px;
  width: 100%;
  max-width: 900px;
  padding: 2rem;
  position: relative;
}
.modal-close {
  position: absolute;
  top: 1rem;
  right: 1rem;
  background: none;
  border: none;
  color: var(--text-dim);
  font-size: 1.4rem;
  cursor: pointer;
}
.modal h2 { font-size: 1.2rem; font-weight: 700; margin-bottom: 0.5rem; }
.modal-section {
  margin-top: 1.25rem;
  padding-top: 1.25rem;
  border-top: 1px solid var(--border);
}
.modal-section h4 {
  font-size: 0.8rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--accent);
  margin-bottom: 0.6rem;
}
.modal-section p, .modal-section li {
  font-size: 0.87rem;
  color: var(--text-dim);
  line-height: 1.7;
}
.modal-section ol { padding-left: 1.5rem; }
.modal-section ol li { margin-bottom: 0.5rem; }
.similar-btn {
  margin-top: 1rem;
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
}

/* ---- Eval Panel ---- */
.eval-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1rem;
}
.eval-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1.25rem;
}
.eval-card h4 {
  font-size: 0.82rem;
  font-weight: 600;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 1rem;
}
.metric-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.4rem 0;
  border-bottom: 1px solid var(--border);
}
.metric-row:last-child { border-bottom: none; }
.metric-label { font-size: 0.82rem; color: var(--text-dim); }
.metric-value { font-family: var(--mono); font-size: 0.85rem; font-weight: 600; }
.metric-good { color: var(--green); }
.metric-ok { color: var(--amber); }

/* ---- History ---- */
.history-item {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.8rem 1rem;
  margin-bottom: 0.5rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  cursor: pointer;
  transition: all 0.15s;
}
.history-item:hover { border-color: var(--accent); }
.history-query { font-size: 0.87rem; }
.history-meta { font-size: 0.75rem; color: var(--text-dim); font-family: var(--mono); }

/* ---- Loading ---- */
.spinner {
  display: none;
  width: 20px;
  height: 20px;
  border: 2px solid var(--border);
  border-top: 2px solid var(--accent);
  border-radius: 50%;
  animation: spin 0.6s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* ---- Empty state ---- */
.empty-state {
  text-align: center;
  padding: 4rem 2rem;
  color: var(--text-muted);
}
.empty-state svg { width: 48px; height: 48px; margin-bottom: 1rem; opacity: 0.4; }
.empty-state h3 { font-size: 1rem; font-weight: 500; color: var(--text-dim); margin-bottom: 0.5rem; }
.empty-state p { font-size: 0.85rem; }

/* ---- Responsive ---- */
@media (max-width: 768px) {
  .header { padding: 0.75rem 1rem; }
  .main { padding: 1rem; }
  .search-row { flex-direction: column; }
  .header-stats { display: none; }
}
</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <div class="logo">
    <svg viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="28" height="28" rx="6" fill="#3b82f6" fill-opacity="0.15"/>
      <path d="M8 9h12M8 14h8M8 19h10" stroke="#3b82f6" stroke-width="2" stroke-linecap="round"/>
      <circle cx="21" cy="19" r="3.5" stroke="#3b82f6" stroke-width="1.5"/>
      <path d="M23.5 21.5L26 24" stroke="#3b82f6" stroke-width="1.5" stroke-linecap="round"/>
    </svg>
    Patent<span class="accent">Lens</span>
  </div>
  <div class="header-stats">
    <span><span class="num">{{ total_patents }}</span> patents</span>
    <span><span class="num">{{ total_claims }}</span> claims</span>
    <span><span class="num">{{ n_classes }}</span> classifications</span>
  </div>
</div>

<!-- Nav -->
<div class="nav-tabs">
  <div class="nav-tab active" onclick="switchTab('search')">Search</div>
  <div class="nav-tab" onclick="switchTab('eval')">Evaluation</div>
  <div class="nav-tab" onclick="switchTab('history')">History</div>
</div>

<!-- Main -->
<div class="main">

  <!-- SEARCH PANEL -->
  <div id="panel-search" class="panel active">
    <div class="search-box">
      <div class="search-row">
        <input type="text" class="search-input" id="query" placeholder="Search patents — try 'non-pneumatic tire' or 'aerodynamic wheel design'..." autofocus>
        <div class="spinner" id="spinner"></div>
        <button class="btn btn-primary" onclick="doSearch()">Search</button>
        <button class="btn btn-ghost" onclick="toggleFilters()">Filters ▾</button>
      </div>
      <div class="options-row">
        <div class="option-group">
          Level:
          <select id="level">
            <option value="patent">Patent</option>
            <option value="claim">Claim</option>
          </select>
        </div>
        <div class="option-group">
          Method:
          <select id="method">
            <option value="combined">Combined (TF-IDF + BM25)</option>
            <option value="tfidf">TF-IDF Cosine</option>
            <option value="bm25">BM25</option>
          </select>
        </div>
        <div class="option-group">
          Top K:
          <input type="number" id="top_k" value="10" min="1" max="40" style="width:55px">
        </div>
      </div>
      <div class="filters" id="filters">
        <div class="filter-group">
          <label>Classification Prefix</label>
          <select class="filter-input" id="class_prefix">
            <option value="">All classifications</option>
            {% for c in classes %}
            <option value="{{ c }}">{{ c }}</option>
            {% endfor %}
          </select>
        </div>
        <div class="filter-group">
          <label>Keywords (comma-separated)</label>
          <input type="text" class="filter-input" id="keywords" placeholder="e.g. tire, pneumatic">
        </div>
        <div class="filter-group">
          <label>Title contains</label>
          <input type="text" class="filter-input" id="title_query" placeholder="e.g. wheel assembly">
        </div>
        <div class="filter-group">
          <label>Keyword fields</label>
          <select class="filter-input" id="keyword_fields">
            <option value="title,abstract">Title + Abstract</option>
            <option value="title,abstract,description">Title + Abstract + Description</option>
            <option value="title">Title only</option>
            <option value="abstract">Abstract only</option>
          </select>
        </div>
      </div>
    </div>

    <div id="results-area">
      <div class="empty-state">
        <svg viewBox="0 0 48 48" fill="none"><circle cx="20" cy="20" r="14" stroke="currentColor" stroke-width="3"/><path d="M30 30L42 42" stroke="currentColor" stroke-width="3" stroke-linecap="round"/></svg>
        <h3>Search the patent database</h3>
        <p>Enter a query above to find relevant patents, claims, or descriptions</p>
      </div>
    </div>
  </div>

  <!-- EVAL PANEL -->
  <div id="panel-eval" class="panel">
    <div style="margin-bottom:1.5rem;">
      <h3 style="font-size:1.1rem;font-weight:600;margin-bottom:0.3rem;">Evaluation Dashboard</h3>
      <p style="font-size:0.85rem;color:var(--text-dim);">Retrieval quality metrics across strategies and scoring methods</p>
      <button class="btn btn-primary" style="margin-top:1rem;" onclick="runEval()" id="eval-btn">Run Evaluation</button>
      <div class="spinner" id="eval-spinner" style="display:inline-block;margin-left:0.5rem;display:none;"></div>
    </div>
    <div id="eval-results" class="eval-grid"></div>
  </div>

  <!-- HISTORY PANEL -->
  <div id="panel-history" class="panel">
    <h3 style="font-size:1.1rem;font-weight:600;margin-bottom:1rem;">Search History</h3>
    <div id="history-list">
      <div class="empty-state">
        <p>Your search history will appear here</p>
      </div>
    </div>
  </div>
</div>

<!-- Patent Detail Modal -->
<div class="modal-overlay" id="modal" onclick="if(event.target===this)closeModal()">
  <div class="modal">
    <button class="modal-close" onclick="closeModal()">✕</button>
    <div id="modal-content"></div>
  </div>
</div>

<script>
// ---- Tab switching ----
function switchTab(name) {
  document.querySelectorAll('.nav-tab').forEach((t,i) => {
    t.classList.toggle('active', t.textContent.trim().toLowerCase() === name);
  });
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById('panel-' + name).classList.add('active');
}

// ---- Filters ----
function toggleFilters() {
  document.getElementById('filters').classList.toggle('open');
}

// ---- Search ----
async function doSearch() {
  const query = document.getElementById('query').value.trim();
  if (!query) return;
  const spinner = document.getElementById('spinner');
  spinner.style.display = 'block';

  const params = new URLSearchParams({
    query,
    level: document.getElementById('level').value,
    method: document.getElementById('method').value,
    top_k: document.getElementById('top_k').value,
    classification_prefix: document.getElementById('class_prefix').value,
    keywords: document.getElementById('keywords').value,
    title_query: document.getElementById('title_query').value,
    keyword_fields: document.getElementById('keyword_fields').value,
  });

  try {
    const res = await fetch('/api/search?' + params);
    const data = await res.json();
    renderResults(data, query);
  } catch(e) {
    document.getElementById('results-area').innerHTML = '<p style="color:var(--red)">Search failed: ' + e.message + '</p>';
  } finally {
    spinner.style.display = 'none';
  }
}

// Enter key triggers search
document.getElementById('query').addEventListener('keydown', e => {
  if (e.key === 'Enter') doSearch();
});

function scoreClass(s) {
  if (s >= 0.5) return 'score-high';
  if (s >= 0.2) return 'score-mid';
  return 'score-low';
}

function renderResults(data, query) {
  const area = document.getElementById('results-area');
  const {results, timing} = data;
  if (!results.length) {
    area.innerHTML = '<div class="empty-state"><h3>No results found</h3><p>Try different keywords or broaden your filters</p></div>';
    return;
  }
  let html = `<div class="results-header">
    <h3>${results.length} result${results.length>1?'s':''} for "${query}"${timing.filters_active ? ' (filtered)' : ''}</h3>
    <span class="timing-badge">${timing.total_ms}ms · ${timing.method}</span>
  </div>`;

  for (const r of results) {
    const snippet = r.match_level === 'claim' ? r.matched_claim_text : r.abstract;
    html += `<div class="result-card" onclick='showPatent("${r.doc_number}")'>
      <div class="result-top">
        <div class="result-title">${r.title}</div>
        <div class="result-score ${scoreClass(r.score)}">${r.score.toFixed(4)}</div>
      </div>
      <div class="result-meta">
        <span class="meta-tag">📄 ${r.doc_number}</span>
        <span class="meta-tag"><span class="class-badge">${r.classification}</span></span>
        ${r.filing_date ? '<span class="meta-tag">📅 ' + r.filing_date + '</span>' : ''}
        ${r.match_level === 'claim' ? '<span class="meta-tag">🎯 Claim #' + (r.matched_claim_idx+1) + '</span>' : ''}
      </div>
      ${r.match_level === 'claim' ? '<div class="claim-label">Matched Claim</div>' : ''}
      <div class="result-snippet">${snippet || 'No abstract available'}</div>
    </div>`;
  }
  area.innerHTML = html;
}

// ---- Patent Detail ----
async function showPatent(docNum) {
  const res = await fetch('/api/patent/' + docNum);
  const p = await res.json();
  if (p.error) { alert(p.error); return; }

  let html = `<h2>${p.title}</h2>
    <div class="result-meta" style="margin-top:0.5rem;">
      <span class="meta-tag">📄 ${p.doc_number}</span>
      <span class="meta-tag"><span class="class-badge">${p.classification}</span></span>
      ${p.filing_date ? '<span class="meta-tag">📅 ' + p.filing_date + '</span>' : ''}
    </div>
    <div class="modal-section">
      <h4>Abstract</h4>
      <p>${p.abstract || 'N/A'}</p>
    </div>
    <div class="modal-section">
      <h4>Claims (${p.claims.length})</h4>
      <ol>${p.claims.map(c => '<li>' + c + '</li>').join('')}</ol>
    </div>
    <div class="modal-section">
      <h4>Detailed Description</h4>
      ${p.detailed_description.map(d => '<p style="margin-bottom:0.5rem">' + d + '</p>').join('')}
    </div>
    <button class="btn btn-primary similar-btn" onclick="findSimilar('${p.doc_number}')">🔍 Find Similar Patents</button>
  `;
  document.getElementById('modal-content').innerHTML = html;
  document.getElementById('modal').classList.add('open');
}

function closeModal() {
  document.getElementById('modal').classList.remove('open');
}
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

async function findSimilar(docNum) {
  closeModal();
  document.getElementById('query').value = '[similar to ' + docNum + ']';
  const spinner = document.getElementById('spinner');
  spinner.style.display = 'block';
  try {
    const res = await fetch('/api/similar/' + docNum + '?top_k=10');
    const data = await res.json();
    renderResults(data, 'similar to ' + docNum);
  } finally {
    spinner.style.display = 'none';
  }
}

// ---- Evaluation ----
async function runEval() {
  const btn = document.getElementById('eval-btn');
  const spinner = document.getElementById('eval-spinner');
  btn.disabled = true;
  btn.textContent = 'Running...';
  spinner.style.display = 'inline-block';

  try {
    const res = await fetch('/api/evaluate');
    const data = await res.json();
    renderEval(data);
  } catch(e) {
    document.getElementById('eval-results').innerHTML = '<p style="color:var(--red)">Evaluation failed</p>';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run Evaluation';
    spinner.style.display = 'none';
  }
}

function metricColor(val) {
  if (val >= 0.7) return 'metric-good';
  if (val >= 0.4) return 'metric-ok';
  return '';
}

function renderEval(data) {
  const area = document.getElementById('eval-results');
  let html = '';
  for (const [key, metrics] of Object.entries(data)) {
    if (typeof metrics !== 'object' || !metrics.method) continue;
    html += `<div class="eval-card">
      <h4>${key.replace(/_/g, ' ')}</h4>`;
    for (const [mk, mv] of Object.entries(metrics)) {
      if (['method','level','total_pairs','positive_pairs','negative_pairs'].includes(mk)) continue;
      const isNumeric = typeof mv === 'number';
      html += `<div class="metric-row">
        <span class="metric-label">${mk}</span>
        <span class="metric-value ${isNumeric && mv <= 1 ? metricColor(mv) : ''}">${isNumeric ? mv.toFixed(4) : mv}</span>
      </div>`;
    }
    html += '</div>';
  }

  // Add timing comparison card
  const timings = Object.entries(data).filter(([k]) => k.startsWith('timing_'));
  if (timings.length) {
    html += '<div class="eval-card"><h4>Hybrid Search Timing</h4>';
    for (const [k, v] of timings) {
      html += `<div class="metric-row">
        <span class="metric-label">${k.replace('timing_', '')}</span>
        <span class="metric-value">${v.total_ms}ms (${v.results_returned} results)</span>
      </div>`;
    }
    html += '</div>';
  }

  area.innerHTML = html;
}

// ---- History ----
let searchHistory = [];

// Override doSearch to also record history
const _origSearch = doSearch;
doSearch = async function() {
  await _origSearch();
  const query = document.getElementById('query').value.trim();
  if (query && !query.startsWith('[similar')) {
    searchHistory.unshift({
      query,
      level: document.getElementById('level').value,
      method: document.getElementById('method').value,
      time: new Date().toLocaleTimeString(),
    });
    if (searchHistory.length > 50) searchHistory.pop();
    renderHistory();
  }
};

function renderHistory() {
  const list = document.getElementById('history-list');
  if (!searchHistory.length) {
    list.innerHTML = '<div class="empty-state"><p>Your search history will appear here</p></div>';
    return;
  }
  list.innerHTML = searchHistory.map(h =>
    `<div class="history-item" onclick="replaySearch('${h.query.replace(/'/g,"\\'")}')">
      <span class="history-query">${h.query}</span>
      <span class="history-meta">${h.level} · ${h.method} · ${h.time}</span>
    </div>`
  ).join('');
}

function replaySearch(query) {
  document.getElementById('query').value = query;
  switchTab('search');
  doSearch();
}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(
        HTML_TEMPLATE,
        total_patents=stats["total_patents"],
        total_claims=stats["total_claims"],
        n_classes=len(stats["classification_groups"]),
        classes=stats["classification_groups"],
    )


@app.route("/api/search")
def api_search():
    query = request.args.get("query", "")
    level = request.args.get("level", "patent")
    method = request.args.get("method", "combined")
    top_k = int(request.args.get("top_k", 10))
    classification_prefix = request.args.get("classification_prefix", "")
    keywords_raw = request.args.get("keywords", "")
    keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()] if keywords_raw else []
    title_query = request.args.get("title_query", "")
    keyword_fields = request.args.get("keyword_fields", "title,abstract").split(",")

    results, timing = engine.search(
        query=query, level=level, top_k=top_k, method=method,
        classification_prefix=classification_prefix,
        keywords=keywords, keyword_fields=keyword_fields,
        title_query=title_query,
    )

    return jsonify({
        "results": [
            {
                "doc_number": r.patent.doc_number,
                "title": r.patent.title,
                "abstract": r.patent.abstract[:400],
                "classification": r.patent.classification,
                "filing_date": r.patent.filing_date,
                "score": round(r.score, 6),
                "match_level": r.match_level,
                "matched_claim_idx": r.matched_claim_idx,
                "matched_claim_text": (r.matched_claim_text or "")[:500],
            }
            for r in results
        ],
        "timing": timing,
    })


@app.route("/api/patent/<doc_number>")
def api_patent(doc_number):
    p = engine.get_patent(doc_number)
    if not p:
        return jsonify({"error": f"Patent {doc_number} not found"}), 404
    return jsonify({
        "doc_number": p.doc_number,
        "title": p.title,
        "abstract": p.abstract,
        "classification": p.classification,
        "filing_date": p.filing_date,
        "claims": p.claims,
        "detailed_description": p.detailed_description,
        "bibtex": p.bibtex,
    })


@app.route("/api/similar/<doc_number>")
def api_similar(doc_number):
    top_k = int(request.args.get("top_k", 5))
    results, timing = engine.find_similar_patents(doc_number, top_k=top_k)
    return jsonify({
        "results": [
            {
                "doc_number": r.patent.doc_number,
                "title": r.patent.title,
                "abstract": r.patent.abstract[:400],
                "classification": r.patent.classification,
                "filing_date": r.patent.filing_date,
                "score": round(r.score, 6),
                "match_level": r.match_level,
                "matched_claim_idx": r.matched_claim_idx,
                "matched_claim_text": "",
            }
            for r in results
        ],
        "timing": timing,
    })


@app.route("/api/evaluate")
def api_evaluate():
    from evaluation import run_full_evaluation
    results = run_full_evaluation(engine)
    return jsonify(results)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n  PatentLens running → http://localhost:8080\n")
    app.run(host="0.0.0.0", port=8080, debug=False)
