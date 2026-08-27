/**
 * Traceback AI - Mission Control Dashboard Application Logic
 */

(function () {
  "use strict";

  const state = {
    currentView: "runs", // "runs" | "detail" | "diff"
    runs: [],
    stats: null,
    selectedRunId: null,
    selectedStepId: null,
    currentTrace: null,
    detailViewMode: "timeline", // "timeline" | "blame"
    compareMode: false,
    selectedDiffRuns: new Set(),
    diffData: null,
    filterPipeline: "all",
    filterSearch: "",
    filterLimit: 50,
  };

  // Utility formatters
  function formatTs(ts) {
    if (!ts) return "N/A";
    const d = new Date(ts * 1000);
    const pad = (n) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
  }

  function formatDuration(ms) {
    if (ms === null || ms === undefined) return "--";
    if (ms < 1000) return `${Math.round(ms)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  }

  function formatValue(val) {
    if (val === null || val === undefined) return "None";
    if (typeof val === "object") {
      try {
        return JSON.stringify(val, null, 2);
      } catch (e) {
        return String(val);
      }
    }
    return String(val);
  }

  function showToast(msg) {
    const toast = document.getElementById("toast");
    if (!toast) return;
    toast.textContent = msg;
    toast.style.display = "block";
    setTimeout(() => {
      toast.style.display = "none";
    }, 2500);
  }

  // API Client
  async function fetchJson(endpoint) {
    try {
      const res = await fetch(endpoint);
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }
      return await res.json();
    } catch (err) {
      console.error(`Fetch error on ${endpoint}:`, err);
      throw err;
    }
  }

  // Application Controller
  const app = {
    async init() {
      window.addEventListener("hashchange", () => this.handleHashChange());
      await this.loadStats();
      await this.handleHashChange();
    },

    async loadStats() {
      try {
        const stats = await fetchJson("/api/stats");
        state.stats = stats;

        const storePathEl = document.getElementById("stat-store-path");
        const totalRunsEl = document.getElementById("stat-total-runs");
        const totalStepsEl = document.getElementById("stat-total-steps");
        const pipelinesEl = document.getElementById("stat-pipelines");

        if (storePathEl) {
          const p = stats.db_path || "traces.db";
          storePathEl.textContent = p.length > 28 ? "..." + p.slice(-25) : p;
          storePathEl.title = p;
        }
        if (totalRunsEl) totalRunsEl.textContent = stats.total_runs;
        if (totalStepsEl) totalStepsEl.textContent = stats.total_steps;
        if (pipelinesEl) pipelinesEl.textContent = stats.pipelines_count;

        // Populate pipeline selector dropdown
        const selectEl = document.getElementById("filter-pipeline");
        if (selectEl && stats.pipelines) {
          const currentVal = selectEl.value;
          selectEl.innerHTML = '<option value="all">ALL PIPELINES</option>';
          stats.pipelines.forEach((p) => {
            const opt = document.createElement("option");
            opt.value = p;
            opt.textContent = p.toUpperCase();
            selectEl.appendChild(opt);
          });
          selectEl.value = currentVal || "all";
        }
      } catch (err) {
        console.warn("Could not load stats:", err);
      }
    },

    async handleHashChange() {
      const hash = window.location.hash.replace(/^#/, "");
      if (!hash || hash === "runs") {
        await this.showRunsView();
      } else if (hash.startsWith("run/")) {
        const runId = hash.replace("run/", "");
        await this.showDetailView(runId, "timeline");
      } else if (hash.startsWith("blame/")) {
        const runId = hash.replace("blame/", "");
        await this.showDetailView(runId, "blame");
      } else if (hash.startsWith("diff")) {
        const urlParams = new URLSearchParams(hash.replace("diff?", ""));
        const a = urlParams.get("a");
        const b = urlParams.get("b");
        await this.showDiffView(a, b);
      } else {
        await this.showRunsView();
      }
    },

    navigate(view, param = null) {
      if (view === "runs") {
        window.location.hash = "#runs";
      } else if (view === "detail" && param) {
        window.location.hash = `#run/${param}`;
      } else if (view === "blame" && param) {
        window.location.hash = `#blame/${param}`;
      } else if (view === "diff") {
        if (param && param.a && param.b) {
          window.location.hash = `#diff?a=${param.a}&b=${param.b}`;
        } else {
          window.location.hash = "#diff";
        }
      }
    },

    async refreshCurrentView() {
      await this.loadStats();
      await this.handleHashChange();
      showToast("TELEMETRY REFRESHED");
    },

    // =========================================================================
    // VIEW 1: RUNS LOG
    // =========================================================================
    async showRunsView() {
      state.currentView = "runs";
      this.updateActiveNav("btn-nav-runs");

      document.getElementById("view-runs").style.display = "flex";
      document.getElementById("view-detail").style.display = "none";
      document.getElementById("view-diff").style.display = "none";

      await this.loadRuns();
    },

    async loadRuns() {
      const p = state.filterPipeline === "all" ? "" : `&pipeline=${encodeURIComponent(state.filterPipeline)}`;
      const limit = state.filterLimit;
      const endpoint = `/api/runs?limit=${limit}${p}`;

      try {
        const data = await fetchJson(endpoint);
        state.runs = data.runs || [];
        this.renderRunsTable();
      } catch (err) {
        console.error("Error loading runs:", err);
      }
    },

    renderRunsTable() {
      const tbody = document.getElementById("runs-tbody");
      const zeroState = document.getElementById("runs-zero-state");
      const tableContainer = document.getElementById("runs-table-container");
      const thSelect = document.getElementById("th-select");

      if (!tbody) return;

      if (thSelect) {
        thSelect.style.display = state.compareMode ? "table-cell" : "none";
      }

      // Filter by search text in client
      const search = state.filterSearch.toLowerCase();
      const filtered = state.runs.filter((r) => {
        if (!search) return true;
        return (
          r.run_id.toLowerCase().includes(search) ||
          r.pipeline_name.toLowerCase().includes(search)
        );
      });

      if (filtered.length === 0) {
        if (state.runs.length === 0) {
          tableContainer.style.display = "none";
          zeroState.style.display = "block";
        } else {
          tableContainer.style.display = "block";
          zeroState.style.display = "none";
          tbody.innerHTML = `<tr><td colspan="${state.compareMode ? 8 : 7}" style="text-align: center; color: var(--text-muted); padding: 32px;">No matching runs found for query "${state.filterSearch}".</td></tr>`;
        }
        return;
      }

      tableContainer.style.display = "block";
      zeroState.style.display = "none";

      tbody.innerHTML = filtered
        .map((r) => {
          const isSelected = state.selectedDiffRuns.has(r.run_id);
          const durStr = formatDuration(r.duration_ms);
          const startStr = formatTs(r.start_ts);

          let statusHtml = '<span class="status-pill status-healthy"><span class="status-dot"></span> [OK] HEALTHY</span>';
          if (r.status === "error") {
            statusHtml = '<span class="status-pill status-error"><span class="status-dot"></span> [CRASH] EXCEPTION</span>';
          } else if (r.status === "blame") {
            const bScore = r.blame && r.blame.blame_score ? r.blame.blame_score.toFixed(2) : "FAULT";
            statusHtml = `<span class="status-pill status-fault"><span class="status-dot"></span> [FAULT] BLAME (${bScore})</span>`;
          }

          const selectCell = state.compareMode
            ? `<td class="col-select" onclick="event.stopPropagation();">
                 <input type="checkbox" ${isSelected ? "checked" : ""} onchange="app.toggleRunSelect('${r.run_id}')">
               </td>`
            : "";

          return `
            <tr class="${isSelected ? "row-selected" : ""}" onclick="app.navigate('detail', '${r.run_id}')">
              ${selectCell}
              <td class="col-id" style="font-weight: 500;">${r.run_id}</td>
              <td class="col-pipeline">${r.pipeline_name}</td>
              <td class="col-steps">${r.step_count}</td>
              <td class="col-start">${startStr}</td>
              <td class="col-duration">${durStr}</td>
              <td class="col-status">${statusHtml}</td>
              <td class="col-actions" onclick="event.stopPropagation();">
                <a href="#run/${r.run_id}" class="action-link" title="Inspect Trace">INSPECT</a>
                <a href="#blame/${r.run_id}" class="action-link" title="Fault Attribution">BLAME</a>
              </td>
            </tr>
          `;
        })
        .join("");
    },

    onFilterChange() {
      const pipeEl = document.getElementById("filter-pipeline");
      const limitEl = document.getElementById("filter-limit");
      if (pipeEl) state.filterPipeline = pipeEl.value;
      if (limitEl) state.filterLimit = parseInt(limitEl.value, 10);
      this.loadRuns();
    },

    onSearchInput(val) {
      state.filterSearch = val || "";
      this.renderRunsTable();
    },

    toggleCompareMode() {
      state.compareMode = !state.compareMode;
      const btn = document.getElementById("btn-toggle-compare");
      const launchBtn = document.getElementById("btn-launch-diff");
      const text = document.getElementById("compare-mode-text");

      if (state.compareMode) {
        text.textContent = "CANCEL SELECTION";
        launchBtn.style.display = "inline-flex";
      } else {
        text.textContent = "SELECT FOR DIFF";
        launchBtn.style.display = "none";
        state.selectedDiffRuns.clear();
      }
      this.renderRunsTable();
    },

    toggleRunSelect(runId) {
      if (state.selectedDiffRuns.has(runId)) {
        state.selectedDiffRuns.delete(runId);
      } else {
        if (state.selectedDiffRuns.size >= 2) {
          // Keep only most recent 2 selections
          const first = state.selectedDiffRuns.values().next().value;
          state.selectedDiffRuns.delete(first);
        }
        state.selectedDiffRuns.add(runId);
      }

      const countEl = document.getElementById("selected-diff-count");
      if (countEl) countEl.textContent = state.selectedDiffRuns.size;
      this.renderRunsTable();
    },

    launchSelectedDiff() {
      const arr = Array.from(state.selectedDiffRuns);
      if (arr.length < 2) {
        showToast("SELECT EXACTLY 2 RUNS TO DIFF");
        return;
      }
      this.navigate("diff", { a: arr[0], b: arr[1] });
    },

    // =========================================================================
    // VIEW 2 & 3: TRACE DETAIL & BLAME
    // =========================================================================
    async showDetailView(runId, mode = "timeline") {
      state.currentView = "detail";
      state.selectedRunId = runId;
      state.detailViewMode = mode;
      this.updateActiveNav("btn-nav-runs");

      document.getElementById("view-runs").style.display = "none";
      document.getElementById("view-detail").style.display = "flex";
      document.getElementById("view-diff").style.display = "none";

      this.updateDetailViewButtons();

      try {
        const trace = await fetchJson(`/api/runs/${runId}`);
        state.currentTrace = trace;
        this.renderTraceDetail(trace);
      } catch (err) {
        showToast(`ERROR: Failed to load run ${runId}`);
      }
    },

    setDetailViewMode(mode) {
      state.detailViewMode = mode;
      this.updateDetailViewButtons();
      if (state.currentTrace) {
        this.renderFlightStrip(state.currentTrace);
      }
    },

    updateDetailViewButtons() {
      const btnTimeline = document.getElementById("btn-toggle-view-trace");
      const btnBlame = document.getElementById("btn-toggle-view-blame");
      if (btnTimeline && btnBlame) {
        btnTimeline.classList.toggle("active", state.detailViewMode === "timeline");
        btnBlame.classList.toggle("active", state.detailViewMode === "blame");
      }
    },

    renderTraceDetail(trace) {
      // Banner Info
      const runIdEl = document.getElementById("detail-run-id");
      const pipeEl = document.getElementById("detail-pipeline-name");
      const startEl = document.getElementById("detail-start-ts");
      const durEl = document.getElementById("detail-duration");
      const tokEl = document.getElementById("detail-tokens");
      const costEl = document.getElementById("detail-cost");
      const statusPillEl = document.getElementById("detail-status-pill");
      const finalOutEl = document.getElementById("detail-final-output");

      if (runIdEl) runIdEl.textContent = `RUN #${trace.run_id}`;
      if (pipeEl) pipeEl.textContent = `PIPELINE: ${trace.pipeline_name.toUpperCase()}`;
      if (startEl) startEl.textContent = formatTs(trace.start_ts);
      if (durEl) durEl.textContent = formatDuration(trace.duration_ms);
      if (tokEl) tokEl.textContent = `${trace.total_tokens || 0} tok`;
      if (costEl) costEl.textContent = `$${(trace.total_cost || 0).toFixed(4)}`;
      if (finalOutEl) finalOutEl.textContent = formatValue(trace.final_output);

      if (statusPillEl) {
        if (trace.status === "error") {
          statusPillEl.className = "status-pill status-error";
          statusPillEl.innerHTML = '<span class="status-dot"></span> [CRASH] EXCEPTION';
        } else if (trace.status === "blame") {
          const score = trace.blame ? trace.blame.blame_score.toFixed(2) : "FAULT";
          statusPillEl.className = "status-pill status-fault";
          statusPillEl.innerHTML = `<span class="status-dot"></span> [FAULT] BLAME (${score})`;
        } else {
          statusPillEl.className = "status-pill status-healthy";
          statusPillEl.innerHTML = '<span class="status-dot"></span> [OK] HEALTHY';
        }
      }

      this.renderFlightStrip(trace);

      // Select first step or blamed step by default in inspector
      const defaultStep = (state.detailViewMode === "blame" && trace.blame && trace.blame.primary_step_id)
        ? trace.steps.find((s) => s.step_id === trace.blame.primary_step_id) || trace.steps[0]
        : trace.steps[0];

      if (defaultStep) {
        this.selectStep(defaultStep.step_id);
      }
    },

    renderFlightStrip(trace) {
      const container = document.getElementById("flight-strip-container");
      if (!container) return;

      const steps = trace.steps || [];
      const blame = trace.blame || {};
      const primaryFaultStepId = blame.primary_step_id;
      const coBlamedIds = new Set(blame.co_blamed_step_ids || []);

      const isBlameMode = state.detailViewMode === "blame";

      container.innerHTML = steps
        .map((step, idx) => {
          const isSelected = state.selectedStepId === step.step_id;
          const isPrimaryFault = step.step_id === primaryFaultStepId && (isBlameMode || blame.blame_score > 0.35 || !!step.error);
          const isCoBlamed = coBlamedIds.has(step.step_id);

          const stepNum = String(step.index !== null && step.index !== undefined ? step.index : idx).padStart(2, "0");
          const durStr = formatDuration(step.latency_ms);
          const tokStr = step.token_count !== null && step.token_count !== undefined ? `${step.token_count} tok` : "";
          const scoreStr = step.score !== null && step.score !== undefined ? step.score.toFixed(2) : "--";

          // Score tag
          let scoreTag = `<span class="score-indicator">${scoreStr}</span>`;
          if (step.score !== null && step.score < 0.5) {
            scoreTag = `<span class="score-indicator" style="color: var(--text-primary);">[FAIL] ${scoreStr}</span>`;
          }

          // Fault Diagnostic Annotation Box (The single moment of bold phosphor amber)
          let faultAnnotationHtml = "";
          if (isPrimaryFault) {
            const confTag = blame.confidence ? blame.confidence.toUpperCase() : "MEDIUM";
            const bScore = blame.blame_score !== undefined ? blame.blame_score.toFixed(2) : "1.00";
            const reason = blame.explanation || "Execution health degradation identified at this step.";

            let coBlameHtml = "";
            if (blame.co_blamed_step_ids && blame.co_blamed_step_ids.length > 0) {
              const coSteps = steps.filter((s) => coBlamedIds.has(s.step_id));
              const coNames = coSteps.map((s) => `[${s.index}] ${s.name}`).join(", ");
              coBlameHtml = `<div class="fault-coblame-row">CO-BLAME DETECTED: ${coNames}</div>`;
            }

            faultAnnotationHtml = `
              <div class="fault-report-annotation">
                <div class="fault-report-header">
                  <span class="fault-report-tag">// FAULT ATTRIBUTION DIAGNOSTIC</span>
                  <span class="fault-report-conf">${confTag} CONFIDENCE</span>
                </div>
                <div class="fault-report-body">
                  <div class="fault-score-row">
                    <span class="telemetry-label">BLAME SCORE</span>
                    <span class="fault-score-val">${bScore}</span>
                    ${blame.is_fallback_latency ? '<span class="status-pill status-fault">[LATENCY BOTTLENECK]</span>' : ""}
                  </div>
                  <div class="fault-explanation">${reason}</div>
                  ${coBlameHtml}
                </div>
              </div>
            `;
          }

          const hasFaultPreceding = isPrimaryFault || isCoBlamed;

          return `
            <div class="timeline-node-wrapper ${isPrimaryFault ? "is-fault-step" : ""} ${hasFaultPreceding ? "has-fault-preceding" : ""}">
              ${idx > 0 ? '<div class="timeline-connector"></div>' : ""}
              <div class="timeline-node-card ${isSelected ? "selected" : ""}" onclick="app.selectStep('${step.step_id}')">
                <div class="node-marker">${stepNum}</div>
                <div class="node-info-main">
                  <div class="node-name-row">
                    <span class="node-name">${step.name}</span>
                    <span class="node-type-tag">${step.step_type}</span>
                  </div>
                  <div class="node-meta-row">
                    <span class="node-telemetry-pill">⏱ ${durStr}</span>
                    ${tokStr ? `<span class="node-telemetry-pill">🔤 ${tokStr}</span>` : ""}
                    ${step.cost_usd ? `<span class="node-telemetry-pill">💲 $${step.cost_usd.toFixed(4)}</span>` : ""}
                  </div>
                </div>
                <div class="node-score-box">
                  <span class="telemetry-label">SCORE</span>
                  ${scoreTag}
                </div>
              </div>
              ${faultAnnotationHtml}
            </div>
          `;
        })
        .join("");
    },

    selectStep(stepId) {
      state.selectedStepId = stepId;
      if (!state.currentTrace) return;

      // Update card selected classes
      const cards = document.querySelectorAll(".timeline-node-card");
      const steps = state.currentTrace.steps || [];
      const step = steps.find((s) => s.step_id === stepId) || steps[0];

      if (!step) return;

      // Update selected class in DOM
      cards.forEach((c) => c.classList.remove("selected"));
      const activeCard = document.querySelector(`.timeline-node-card[onclick*="${stepId}"]`);
      if (activeCard) activeCard.classList.add("selected");

      // Populate Telemetry Inspector
      const titleEl = document.getElementById("inspector-step-title");
      const typeEl = document.getElementById("inspector-step-type");
      const idxEl = document.getElementById("inspector-step-index");
      const scoreEl = document.getElementById("inspector-step-score");

      const latEl = document.getElementById("insp-latency");
      const tokEl = document.getElementById("insp-tokens");
      const costEl = document.getElementById("insp-cost");

      const errSec = document.getElementById("insp-error-section");
      const errText = document.getElementById("insp-error-text");

      const inPayload = document.getElementById("insp-input-payload");
      const outPayload = document.getElementById("insp-output-payload");
      const metaPayload = document.getElementById("insp-metadata-payload");

      if (titleEl) titleEl.textContent = step.name.toUpperCase();
      if (typeEl) typeEl.textContent = `TYPE: ${step.step_type.toUpperCase()}`;
      if (idxEl) idxEl.textContent = `STEP: #${step.index !== null ? step.index : "--"}`;
      if (scoreEl) scoreEl.textContent = `SCORE: ${step.score !== null && step.score !== undefined ? step.score.toFixed(2) : "UNSCORED"}`;

      if (latEl) latEl.textContent = formatDuration(step.latency_ms);
      if (tokEl) tokEl.textContent = step.token_count !== null && step.token_count !== undefined ? `${step.token_count} tok` : "--";
      if (costEl) costEl.textContent = step.cost_usd ? `$${step.cost_usd.toFixed(4)}` : "--";

      if (errSec && errText) {
        if (step.error) {
          errSec.style.display = "flex";
          errText.textContent = step.error;
        } else {
          errSec.style.display = "none";
          errText.textContent = "";
        }
      }

      if (inPayload) inPayload.textContent = formatValue(step.input);
      if (outPayload) outPayload.textContent = formatValue(step.output);
      if (metaPayload) metaPayload.textContent = formatValue(step.metadata);
    },

    copyInspectorField(field) {
      if (!state.currentTrace || !state.selectedStepId) return;
      const step = state.currentTrace.steps.find((s) => s.step_id === state.selectedStepId);
      if (!step) return;

      const val = step[field];
      const text = typeof val === "object" ? JSON.stringify(val, null, 2) : String(val || "");
      navigator.clipboard.writeText(text).then(() => {
        showToast(`COPIED ${field.toUpperCase()}`);
      });
    },

    copyFinalOutput() {
      if (!state.currentTrace) return;
      const val = state.currentTrace.final_output;
      const text = typeof val === "object" ? JSON.stringify(val, null, 2) : String(val || "");
      navigator.clipboard.writeText(text).then(() => {
        showToast("COPIED FINAL OUTPUT");
      });
    },

    // =========================================================================
    // VIEW 4: DIFF COMPARISON
    // =========================================================================
    async showDiffView(runA = null, runB = null) {
      state.currentView = "diff";
      this.updateActiveNav("btn-nav-diff");

      document.getElementById("view-runs").style.display = "none";
      document.getElementById("view-detail").style.display = "none";
      document.getElementById("view-diff").style.display = "flex";

      await this.populateDiffSelectors(runA, runB);

      if (runA && runB) {
        await this.executeDiff(runA, runB);
      }
    },

    async populateDiffSelectors(selectedA = null, selectedB = null) {
      const selectA = document.getElementById("diff-select-a");
      const selectB = document.getElementById("diff-select-b");
      if (!selectA || !selectB) return;

      if (state.runs.length === 0) {
        await this.loadRuns();
      }

      const options = state.runs.map((r) => {
        const p = r.pipeline_name || "unnamed";
        const ts = formatTs(r.start_ts);
        return `<option value="${r.run_id}">${r.run_id} [${p}] (${ts})</option>`;
      });

      selectA.innerHTML = '<option value="">Select baseline trace (A)...</option>' + options.join("");
      selectB.innerHTML = '<option value="">Select comparison trace (B)...</option>' + options.join("");

      if (selectedA) selectA.value = selectedA;
      if (selectedB) selectB.value = selectedB;
    },

    onDiffSelectorChange() {
      const a = document.getElementById("diff-select-a").value;
      const b = document.getElementById("diff-select-b").value;
      if (a && b && a !== b) {
        window.location.hash = `#diff?a=${a}&b=${b}`;
      }
    },

    async executeDiff(runAOverride = null, runBOverride = null) {
      const selectA = document.getElementById("diff-select-a");
      const selectB = document.getElementById("diff-select-b");

      const runA = runAOverride || (selectA ? selectA.value : null);
      const runB = runBOverride || (selectB ? selectB.value : null);

      if (!runA || !runB) {
        showToast("PLEASE SELECT BOTH RUN A AND RUN B");
        return;
      }
      if (runA === runB) {
        showToast("RUN A AND RUN B MUST BE DIFFERENT");
        return;
      }

      try {
        const diffData = await fetchJson(`/api/diff?a=${encodeURIComponent(runA)}&b=${encodeURIComponent(runB)}`);
        state.diffData = diffData;
        this.renderDiffResults(diffData);
      } catch (err) {
        showToast("ERROR: Failed to compute diff between runs");
      }
    },

    renderDiffResults(diff) {
      const verdictDeck = document.getElementById("diff-verdict-deck");
      const banner = document.getElementById("verdict-banner");
      const titleEl = document.getElementById("verdict-title");
      const explEl = document.getElementById("verdict-explanation");
      const resultsContainer = document.getElementById("diff-results-container");
      const zeroState = document.getElementById("diff-zero-state");
      const tbody = document.getElementById("diff-tbody");

      if (zeroState) zeroState.style.display = "none";
      if (verdictDeck) verdictDeck.style.display = "block";
      if (resultsContainer) resultsContainer.style.display = "block";

      // Verdict Banner
      if (banner && titleEl && explEl) {
        banner.className = "verdict-banner";
        if (diff.verdict === "REGRESSION") {
          banner.classList.add("verdict-regression");
          titleEl.textContent = `VERDICT: QUALITY REGRESSION IN '${diff.primary_diverged_step || "PIPELINE"}'`;
        } else if (diff.verdict === "IMPROVEMENT") {
          titleEl.textContent = `VERDICT: QUALITY IMPROVEMENT IN '${diff.primary_diverged_step || "PIPELINE"}'`;
        } else {
          titleEl.textContent = "VERDICT: NEUTRAL TRAJECTORY";
        }
        explEl.textContent = diff.explanation || "No significant divergence detected between runs.";
      }

      // Populate Diff Table
      if (!tbody) return;

      const rows = [];

      // 1. Regressed steps (use fault visual vocabulary)
      (diff.regressed_steps || []).forEach(({ step_a, step_b, delta }) => {
        const scoreA = step_a.score !== null ? step_a.score.toFixed(2) : "N/A";
        const scoreB = step_b.score !== null ? step_b.score.toFixed(2) : "N/A";
        rows.push(`
          <tr class="diff-row-regressed">
            <td class="col-step-name">${step_b.name} (${step_b.step_type})</td>
            <td class="col-score-a">${scoreA}</td>
            <td class="col-score-b">${scoreB}</td>
            <td class="col-delta">${delta > 0 ? "+" : ""}${delta.toFixed(2)}</td>
            <td class="col-diff-status"><span class="diff-tag-regressed">[-] REGRESSION</span></td>
          </tr>
        `);
      });

      // 2. Improved steps
      (diff.improved_steps || []).forEach(({ step_a, step_b, delta }) => {
        const scoreA = step_a.score !== null ? step_a.score.toFixed(2) : "N/A";
        const scoreB = step_b.score !== null ? step_b.score.toFixed(2) : "N/A";
        rows.push(`
          <tr>
            <td class="col-step-name">${step_b.name} (${step_b.step_type})</td>
            <td class="col-score-a">${scoreA}</td>
            <td class="col-score-b">${scoreB}</td>
            <td class="col-delta">${delta > 0 ? "+" : ""}${delta.toFixed(2)}</td>
            <td class="col-diff-status"><span class="diff-tag-improved">[+] IMPROVED</span></td>
          </tr>
        `);
      });

      // 3. Stable steps
      (diff.stable_steps || []).forEach(({ step_a, step_b, delta }) => {
        const scoreA = step_a.score !== null ? step_a.score.toFixed(2) : "N/A";
        const scoreB = step_b.score !== null ? step_b.score.toFixed(2) : "N/A";
        rows.push(`
          <tr>
            <td class="col-step-name">${step_b.name} (${step_b.step_type})</td>
            <td class="col-score-a">${scoreA}</td>
            <td class="col-score-b">${scoreB}</td>
            <td class="col-delta">${delta > 0 ? "+" : ""}${delta.toFixed(2)}</td>
            <td class="col-diff-status"><span class="diff-tag-stable">[=] STABLE</span></td>
          </tr>
        `);
      });

      // 4. Added steps
      (diff.added_steps || []).forEach((step) => {
        const scoreB = step.score !== null ? step.score.toFixed(2) : "N/A";
        rows.push(`
          <tr>
            <td class="col-step-name">${step.name} (${step.step_type})</td>
            <td class="col-score-a">--</td>
            <td class="col-score-b">${scoreB}</td>
            <td class="col-delta">--</td>
            <td class="col-diff-status"><span class="diff-tag-improved">[+] ADDED IN B</span></td>
          </tr>
        `);
      });

      // 5. Removed steps
      (diff.removed_steps || []).forEach((step) => {
        const scoreA = step.score !== null ? step.score.toFixed(2) : "N/A";
        rows.push(`
          <tr>
            <td class="col-step-name">${step.name} (${step.step_type})</td>
            <td class="col-score-a">${scoreA}</td>
            <td class="col-score-b">--</td>
            <td class="col-delta">--</td>
            <td class="col-diff-status"><span class="diff-tag-stable">[-] REMOVED IN B</span></td>
          </tr>
        `);
      });

      tbody.innerHTML = rows.join("");
    },

    updateActiveNav(activeId) {
      document.querySelectorAll(".nav-btn").forEach((btn) => {
        btn.classList.remove("active");
      });
      const activeEl = document.getElementById(activeId);
      if (activeEl) activeEl.classList.add("active");
    },
  };

  // Expose to window
  window.app = app;

  // Initialize on DOM ready
  document.addEventListener("DOMContentLoaded", () => {
    app.init();
  });
})();
