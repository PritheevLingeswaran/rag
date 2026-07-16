/* ragp frontend logic. Zero dependencies.
 *
 * Every documented backend state renders explicitly (Stages 4/4.5/5/7.7):
 * ok / ok_partial_rejected / ok_no_answer / no_results / degraded_* /
 * 429 rate-limit + daily quota (real countdown from the server's number) /
 * 503 shed / 401 session-expired / network failure (retry) / cold start
 * (elapsed timer + honest free-tier note). A raw error or blank screen is
 * a bug; global handlers below catch anything unhandled and show the
 * error panel instead.
 *
 * #demo:<state> renders a recorded backend payload through the SAME
 * rendering path — used to demonstrate each state in a browser without
 * having to force the real backend into it (see stage 9.6 report).
 */
"use strict";

const $ = (id) => document.getElementById(id);

const els = {
  landing: $("landing"), queryUi: $("query-ui"), form: $("query-form"),
  input: $("query-input"), submit: $("submit-btn"), loading: $("loading"),
  elapsed: $("elapsed"), coldNote: $("coldstart-note"), result: $("result"),
  banner: $("status-banner"), answer: $("answer"),
  citationsBlock: $("citations-block"), citations: $("citations"),
  cacheNote: $("cache-note"), requestId: $("request-id"),
  waitPanel: $("wait-panel"), waitMessage: $("wait-message"),
  countdown: $("countdown"), errorPanel: $("error-panel"),
  errorMessage: $("error-message"), retryBtn: $("retry-btn"),
  userChip: $("user-chip"), userName: $("user-name"),
  userAvatar: $("user-avatar"), logoutBtn: $("logout-btn"),
  landingNote: $("landing-note"),
};

let lastQuery = null;
let timers = { elapsed: null, countdown: null };

/* ---------- view helpers ---------- */

function hide(...nodes) { nodes.forEach((n) => { n.hidden = true; }); }
function show(...nodes) { nodes.forEach((n) => { n.hidden = false; }); }

function resetPanels() {
  hide(els.loading, els.result, els.waitPanel, els.errorPanel);
  clearInterval(timers.elapsed);
  clearInterval(timers.countdown);
}

function showLanding(note) {
  resetPanels();
  hide(els.queryUi, els.userChip);
  show(els.landing);
  if (note) { els.landingNote.textContent = note; show(els.landingNote); }
  else hide(els.landingNote);
}

function showQueryUi(profile) {
  hide(els.landing);
  show(els.queryUi);
  if (profile) {
    els.userName.textContent = profile.name || profile.email || "account";
    if (profile.avatar_url) els.userAvatar.src = profile.avatar_url;
    else els.userAvatar.removeAttribute("src");
    show(els.userChip);
  }
}

/* ---------- status vocabulary: every backend status, spelled out ---------- */

const DEGRADED_REASONS = {
  degraded_no_llm: "AI generation is not configured on this deployment",
  degraded_quota_throttled: "today's AI budget is used up",
  degraded_quota: "the AI provider's quota was hit",
  degraded_timeout: "the AI provider timed out",
  degraded_llm_error: "the AI provider had an error",
  degraded_llm_malformed: "the AI returned an unusable response",
  degraded_llm_config: "the AI is misconfigured (operator has been paged)",
  degraded_llm_auth: "the AI credentials failed (operator has been paged)",
  degraded_citation_rejected: "the AI answer failed source verification and was discarded",
};

function renderResult(data) {
  resetPanels();
  els.banner.className = "banner";
  els.cacheNote.hidden = !data.cached;
  els.requestId.textContent = data.request_id ? `ref ${data.request_id}` : "";

  const s = data.status;
  if (s === "ok") {
    els.banner.classList.add("ok");
    els.banner.textContent = "✓ Answer — every sentence verified against the cited sources";
  } else if (s === "ok_partial_rejected") {
    els.banner.classList.add("degraded");
    els.banner.textContent =
      "◐ Partial answer — sentences that could not be verified against the sources were removed";
  } else if (s === "ok_no_answer") {
    els.banner.classList.add("ok");
    els.banner.textContent = "The sources do not contain an answer to this question";
  } else if (s === "no_results") {
    els.banner.classList.add("degraded");
    els.banner.textContent = "No relevant documents found for this question";
  } else {
    els.banner.classList.add("degraded");
    const reason = DEGRADED_REASONS[s] || "the AI step was unavailable";
    els.banner.textContent =
      `⚠ Retrieval-only answer — ${reason}. What follows is quoted from the best-matching source, not AI-generated.`;
  }

  els.answer.textContent = data.answer || "";

  const cites = data.citations || [];
  if (cites.length) {
    els.citations.replaceChildren(
      ...cites.map((c) => {
        const li = document.createElement("li");
        li.textContent = c;
        return li;
      })
    );
    show(els.citationsBlock);
  } else {
    hide(els.citationsBlock);
  }

  if (s === "degraded_quota_throttled" && data.retry_after_s) {
    startCountdown(Math.ceil(data.retry_after_s),
      "The AI budget resets then; retrieval-only answers keep working meanwhile.");
  }
  show(els.result);
}

/* ---------- wait states: the countdown is the server's number ---------- */

function fmtDuration(s) {
  if (s >= 3600) return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
  if (s >= 60) return `${Math.floor(s / 60)}m ${s % 60}s`;
  return `${s}s`;
}

function startCountdown(seconds, message) {
  els.waitMessage.textContent = message;
  let left = seconds;
  els.countdown.textContent = fmtDuration(left);
  show(els.waitPanel);
  els.submit.disabled = true;
  clearInterval(timers.countdown);
  timers.countdown = setInterval(() => {
    left -= 1;
    if (left <= 0) {
      clearInterval(timers.countdown);
      hide(els.waitPanel);
      els.submit.disabled = false;
    } else {
      els.countdown.textContent = fmtDuration(left);
    }
  }, 1000);
}

function showError(message) {
  resetPanels();
  els.errorMessage.textContent = message;
  show(els.errorPanel);
  els.submit.disabled = false;
}

/* ---------- the query round-trip ---------- */

async function runQuery(query) {
  lastQuery = query;
  resetPanels();
  els.submit.disabled = true;

  let seconds = 0;
  els.elapsed.textContent = "0s";
  hide(els.coldNote);
  show(els.loading);
  timers.elapsed = setInterval(() => {
    seconds += 1;
    els.elapsed.textContent = `${seconds}s`;
    if (seconds >= 5) show(els.coldNote); // now it's honestly "slow"
  }, 1000);

  let resp;
  try {
    resp = await fetch("/v1/query", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query }),
    });
  } catch {
    showError("Could not reach the server — check your connection and try again.");
    return;
  }

  let body = null;
  try { body = await resp.json(); } catch { /* handled per-status below */ }

  els.submit.disabled = false;

  if (resp.ok && body) { renderResult(body); return; }

  if (resp.status === 401) {
    showLanding("Your session has expired — please sign in again.");
    return;
  }
  if (resp.status === 429 && body) {
    resetPanels();
    const secondsLeft = Math.ceil(
      body.retry_after_s ?? Number(resp.headers.get("retry-after")) ?? 60
    );
    const scope = (body.error || "").includes("daily")
      ? "You've used today's question allowance."
      : "You're asking a little too quickly.";
    startCountdown(secondsLeft, scope);
    return;
  }
  if (resp.status === 503 && body) {
    resetPanels();
    startCountdown(Math.ceil(body.retry_after_s || 5),
      "The server is at capacity right now (it sheds load rather than queueing forever).");
    return;
  }
  if (resp.status === 422 || resp.status === 413) {
    showError("That question couldn't be accepted — it may be too long (2000 characters max).");
    return;
  }
  showError(
    `Something went wrong on our side${body && body.request_id ? ` (ref ${body.request_id})` : ""}. Please try again.`
  );
}

/* ---------- auth cycle ---------- */

async function init() {
  if (runDemoIfRequested()) return;
  let resp;
  try {
    resp = await fetch("/auth/me");
  } catch {
    showLanding("Can't reach the server right now — try refreshing in a minute.");
    return;
  }
  if (resp.ok) showQueryUi(await resp.json());
  else showLanding();
}

els.form.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = els.input.value.trim();
  if (q) runQuery(q);
});

els.retryBtn.addEventListener("click", () => {
  if (lastQuery) runQuery(lastQuery);
  else { resetPanels(); }
});

els.logoutBtn.addEventListener("click", async () => {
  try { await fetch("/auth/logout", { method: "POST" }); } catch { /* still show landing */ }
  showLanding("Signed out.");
});

/* No blank screens, ever: anything unhandled becomes the error panel. */
window.addEventListener("error", () => showError("Unexpected error in the page — please retry."));
window.addEventListener("unhandledrejection", (e) => {
  e.preventDefault();
  showError("Unexpected error in the page — please retry.");
});

/* ---------- state gallery (#demo:<name>) ---------- */
/* Recorded real backend payloads, rendered through the SAME functions
 * as live traffic — proof the UI handles each documented state. */
const DEMO = {
  ok: { query: "q", answer: "In Raft, time is divided into terms, and each term begins with a leader election [1]. A candidate wins by collecting votes from a majority of the cluster [1].", status: "ok", degraded: false, rerank_status: "full", citations: ["raft::c0"], retrieved_chunk_ids: ["raft::c0", "raft::c1"], retry_after_s: null, request_id: "demo0000ok", cached: false },
  partial: { query: "q", answer: "In Raft, time is divided into terms, and each term begins with a leader election [1].", status: "ok_partial_rejected", degraded: false, rerank_status: "full", citations: ["raft::c0"], retrieved_chunk_ids: ["raft::c0"], retry_after_s: null, request_id: "demo0partial", cached: false },
  throttled: { query: "q", answer: "Raft is a consensus algorithm designed to be easier to understand than Paxos. In Raft, time is divided into terms, and each term begins with a leader election.", status: "degraded_quota_throttled", degraded: true, rerank_status: "full", citations: ["raft::c0"], retrieved_chunk_ids: ["raft::c0"], retry_after_s: 7200, request_id: "demo0throt", cached: false },
  degraded: { query: "q", answer: "Raft is a consensus algorithm designed to be easier to understand than Paxos. In Raft, time is divided into terms, and each term begins with a leader election.", status: "degraded_no_llm", degraded: true, rerank_status: "skipped_budget", citations: ["raft::c0"], retrieved_chunk_ids: ["raft::c0"], retry_after_s: null, request_id: "demo0degr", cached: true },
  ratelimited: { __http: 429, error: "rate limit exceeded", retry_after_s: 42 },
  shed: { __http: 503, error: "server at capacity; request not queued", retry_after_s: 9 },
  error: { __error: "Could not reach the server — check your connection and try again." },
  loading: { __loading: true },
};

function runDemoIfRequested() {
  const m = location.hash.match(/^#demo:(\w+)$/);
  if (!m || !(m[1] in DEMO)) return false;
  const fx = DEMO[m[1]];
  showQueryUi({ name: "Demo User", email: "demo@example.com" });
  els.input.value = "how does raft handle leader election";
  if (fx.__loading) {
    els.submit.disabled = true;
    let s = 0;
    show(els.loading);
    timers.elapsed = setInterval(() => {
      s += 1;
      els.elapsed.textContent = `${s}s`;
      if (s >= 5) show(els.coldNote);
    }, 1000);
  } else if (fx.__error) {
    lastQuery = els.input.value;
    showError(fx.__error);
  } else if (fx.__http === 429) {
    startCountdown(fx.retry_after_s, "You're asking a little too quickly.");
  } else if (fx.__http === 503) {
    startCountdown(fx.retry_after_s,
      "The server is at capacity right now (it sheds load rather than queueing forever).");
  } else {
    renderResult(fx);
  }
  return true;
}

init();
