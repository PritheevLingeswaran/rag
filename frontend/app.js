/* ragp frontend — chat layout (Stage 9.7; UX ported from SmartQA, the
 * honesty rules are ragp's own). Zero dependencies.
 *
 * Unchanged contract from Stage 9.6: every documented backend state has
 * an explicit rendering — ok / ok_partial_rejected / ok_no_answer /
 * no_results / every degraded_* reason / 429 (rate vs daily, live
 * countdown from the server's number) / 503 shed / 401 session expiry /
 * network failure with retry / truthful cold-start loading. Blank
 * screens and raw console errors are defined as bugs; window-level
 * handlers below make them impossible.
 *
 * Honesty note ported INTO the chat idiom: each question is answered
 * independently — there is no conversational memory in the backend, and
 * the UI never implies there is (the transcript is presentation, not
 * context). True multi-turn context is a backend feature with token
 * costs, deliberately not faked here.
 *
 * #demo:<state> renders recorded real payloads through the SAME
 * rendering path — evidence for the stage report.
 */
"use strict";

const $ = (id) => document.getElementById(id);

const els = {
  landing: $("landing"), landingNote: $("landing-note"),
  chatUi: $("chat-ui"), messages: $("messages"),
  composer: $("composer"), input: $("composer-input"), send: $("send-btn"),
  quotaHint: $("quota-hint"),
  userChip: $("user-chip"), userName: $("user-name"),
  userAvatar: $("user-avatar"), logoutBtn: $("logout-btn"),
};

let lastQuery = null;
let inFlight = false;
let countdownTimer = null;

/* ---------- small DOM helpers (no innerHTML with dynamic data) ---------- */

function el(tag, className, text) {
  const n = document.createElement(tag);
  if (className) n.className = className;
  if (text !== undefined) n.textContent = text;
  return n;
}

function addMsg(className) {
  const m = el("div", `msg ${className}`);
  els.messages.appendChild(m);
  els.messages.scrollTop = els.messages.scrollHeight;
  return m;
}

function scrollDown() { els.messages.scrollTop = els.messages.scrollHeight; }

/* ---------- views ---------- */

function showLanding(note) {
  els.chatUi.hidden = true;
  els.userChip.hidden = true;
  els.landing.hidden = false;
  if (note) { els.landingNote.textContent = note; els.landingNote.hidden = false; }
  else els.landingNote.hidden = true;
}

function showChat(profile) {
  els.landing.hidden = true;
  els.chatUi.hidden = false;
  if (profile) {
    els.userName.textContent = profile.name || profile.email || "account";
    if (profile.avatar_url) els.userAvatar.src = profile.avatar_url;
    else els.userAvatar.removeAttribute("src");
    els.userChip.hidden = false;
  }
  els.input.focus();
}

/* ---------- status vocabulary (unchanged from 9.6) ---------- */

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

/* Answer text with [n] markers highlighted, built via safe DOM ops. */
function answerNode(text) {
  const p = el("p");
  const parts = String(text || "").split(/(\[\d+\])/);
  for (const part of parts) {
    if (/^\[\d+\]$/.test(part)) p.appendChild(el("span", "cite-marker", part));
    else p.appendChild(document.createTextNode(part));
  }
  return p;
}

function renderAssistant(data, container) {
  const m = container || addMsg("assistant");
  m.replaceChildren();

  const banner = el("span", "banner");
  const s = data.status;
  if (s === "ok") {
    banner.classList.add("ok");
    banner.textContent = "✓ Verified — every sentence checked against its cited sources";
  } else if (s === "ok_partial_rejected") {
    banner.classList.add("degraded");
    banner.textContent = "◐ Partial — unverifiable sentences were removed";
  } else if (s === "ok_no_answer") {
    banner.classList.add("ok");
    banner.textContent = "The sources do not contain an answer to this";
  } else if (s === "no_results") {
    banner.classList.add("degraded");
    banner.textContent = "No relevant documents found";
  } else {
    banner.classList.add("degraded");
    banner.textContent =
      `⚠ Retrieval-only — ${DEGRADED_REASONS[s] || "the AI step was unavailable"}. ` +
      "The text below is quoted from the best-matching source, not AI-generated.";
  }
  m.appendChild(banner);
  m.appendChild(answerNode(data.answer));

  const cites = data.citations || [];
  if (cites.length) {
    m.appendChild(el("div", "sources-label", "Sources"));
    const ul = el("ul", "sources");
    cites.forEach((c) => ul.appendChild(el("li", null, c)));
    m.appendChild(ul);
  }

  const metaBits = [];
  if (data.cached) metaBits.push("served from cache (≤1 h old)");
  if (data.request_id) metaBits.push(`ref ${data.request_id}`);
  if (metaBits.length) m.appendChild(el("div", "meta", metaBits.join(" · ")));

  if (s === "degraded_quota_throttled" && data.retry_after_s) {
    startCountdown(Math.ceil(data.retry_after_s),
      "The AI budget resets then; retrieval-only answers keep working meanwhile.",
      /*lockComposer=*/false);
  }
  scrollDown();
  return m;
}

/* ---------- wait / error states as chat messages ---------- */

function fmtDuration(s) {
  if (s >= 3600) return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
  if (s >= 60) return `${Math.floor(s / 60)}m ${s % 60}s`;
  return `${s}s`;
}

function startCountdown(seconds, message, lockComposer = true) {
  clearInterval(countdownTimer);
  const m = addMsg("wait");
  m.setAttribute("role", "status");
  const line = el("p", null, message);
  const cd = el("p");
  cd.append("You can ask again in ");
  const strong = el("strong", null, fmtDuration(seconds));
  cd.appendChild(strong);
  cd.append(".");
  m.append(line, cd);
  if (lockComposer) els.send.disabled = true;

  let left = seconds;
  countdownTimer = setInterval(() => {
    left -= 1;
    if (left <= 0) {
      clearInterval(countdownTimer);
      m.remove();
      els.send.disabled = false;
      els.input.focus();
    } else {
      strong.textContent = fmtDuration(left);
    }
  }, 1000);
  scrollDown();
}

function showErrorMsg(message) {
  const m = addMsg("error");
  m.setAttribute("role", "alert");
  m.appendChild(el("p", null, message));
  const retry = el("button", "ghost", "Try again");
  retry.type = "button";
  retry.addEventListener("click", () => {
    m.remove();
    if (lastQuery) runQuery(lastQuery, /*reAsk=*/true);
  });
  m.appendChild(retry);
  els.send.disabled = false;
  inFlight = false;
  scrollDown();
}

/* ---------- the query round-trip ---------- */

function pendingNode() {
  const m = addMsg("assistant");
  const p = el("p");
  const spin = el("span", "spinner");
  spin.setAttribute("aria-hidden", "true");
  p.append(spin, " Working… ");
  const elapsed = el("span", null, "0s");
  p.appendChild(elapsed);
  m.appendChild(p);
  const cold = el("p", "fine",
    "This runs on a free tier that sleeps when idle — the first request " +
    "after a quiet period can take up to ~2 minutes while it wakes.");
  cold.hidden = true;
  m.appendChild(cold);
  let s = 0;
  const timer = setInterval(() => {
    s += 1;
    elapsed.textContent = `${s}s`;
    if (s >= 5) cold.hidden = false;
  }, 1000);
  return { m, stop: () => clearInterval(timer) };
}

async function runQuery(query, reAsk = false) {
  if (inFlight) return;
  inFlight = true;
  lastQuery = query;
  els.send.disabled = true;
  if (!reAsk) addMsg("user").textContent = query;

  const pending = pendingNode();

  let resp;
  try {
    resp = await fetch("/v1/query", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query }),
    });
  } catch {
    pending.stop(); pending.m.remove();
    showErrorMsg("Could not reach the server — check your connection and try again.");
    return;
  }

  let body = null;
  try { body = await resp.json(); } catch { /* per-status below */ }

  pending.stop();
  inFlight = false;
  els.send.disabled = false;

  if (resp.ok && body) { renderAssistant(body, pending.m); return; }
  pending.m.remove();

  if (resp.status === 401) {
    showLanding("Your session has expired — please sign in again.");
    return;
  }
  if (resp.status === 429 && body) {
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
    startCountdown(Math.ceil(body.retry_after_s || 5),
      "The server is at capacity right now (it sheds load rather than queueing forever).");
    return;
  }
  if (resp.status === 422 || resp.status === 413) {
    showErrorMsg("That question couldn't be accepted — it may be too long (2000 characters max).");
    return;
  }
  showErrorMsg(
    `Something went wrong on our side${body && body.request_id ? ` (ref ${body.request_id})` : ""}. Please try again.`
  );
}

/* ---------- composer behavior (Enter sends, Shift+Enter newlines) ---------- */

els.composer.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = els.input.value.trim();
  if (!q || inFlight) return;
  els.input.value = "";
  els.input.style.height = "";
  runQuery(q);
});

els.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    els.composer.requestSubmit();
  }
});

els.input.addEventListener("input", () => {
  els.input.style.height = "";
  els.input.style.height = `${Math.min(els.input.scrollHeight, 160)}px`;
});

els.logoutBtn.addEventListener("click", async () => {
  try { await fetch("/auth/logout", { method: "POST" }); } catch { /* still leave */ }
  showLanding("Signed out.");
});

window.addEventListener("error", () => showErrorMsg("Unexpected error in the page — please retry."));
window.addEventListener("unhandledrejection", (e) => {
  e.preventDefault();
  showErrorMsg("Unexpected error in the page — please retry.");
});

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
  if (resp.ok) { showChat(await resp.json()); return; }
  // Development backends allow anonymous queries (Stage 5 policy); let
  // the chat work there without sign-in instead of dead-ending on a
  // landing page. Production stays login-gated.
  try {
    const health = await (await fetch("/health")).json();
    if (health.environment === "development") {
      showChat(null);
      addMsg("system").textContent =
        "Developer mode: not signed in — queries run anonymously.";
      return;
    }
  } catch { /* fall through to landing */ }
  showLanding();
}

/* ---------- state gallery (#demo:<name>) ---------- */

const DEMO = {
  ok: { answer: "In Raft, time is divided into terms, and each term begins with a leader election [1]. A candidate wins by collecting votes from a majority of the cluster [1].", status: "ok", citations: ["raft::c0"], request_id: "demo0000ok", cached: false },
  partial: { answer: "In Raft, time is divided into terms, and each term begins with a leader election [1].", status: "ok_partial_rejected", citations: ["raft::c0"], request_id: "demo0partial", cached: false },
  throttled: { answer: "Raft is a consensus algorithm designed to be easier to understand than Paxos. In Raft, time is divided into terms, and each term begins with a leader election.", status: "degraded_quota_throttled", citations: ["raft::c0"], retry_after_s: 7200, request_id: "demo0throt", cached: false },
  degraded: { answer: "Raft is a consensus algorithm designed to be easier to understand than Paxos. In Raft, time is divided into terms, and each term begins with a leader election.", status: "degraded_no_llm", citations: ["raft::c0"], request_id: "demo0degr", cached: true },
  ratelimited: { __http: 429, error: "rate limit exceeded", retry_after_s: 42 },
  shed: { __http: 503, error: "server at capacity; request not queued", retry_after_s: 9 },
  error: { __error: "Could not reach the server — check your connection and try again." },
  loading: { __loading: true },
};

function runDemoIfRequested() {
  const m = location.hash.match(/^#demo:(\w+)$/);
  if (!m || !(m[1] in DEMO)) return false;
  const fx = DEMO[m[1]];
  showChat({ name: "Demo User", email: "demo@example.com" });
  addMsg("user").textContent = "how does raft handle leader election";
  lastQuery = "how does raft handle leader election";
  if (fx.__loading) {
    pendingNode();
    els.send.disabled = true;
  } else if (fx.__error) {
    showErrorMsg(fx.__error);
  } else if (fx.__http === 429) {
    startCountdown(fx.retry_after_s, "You're asking a little too quickly.");
  } else if (fx.__http === 503) {
    startCountdown(fx.retry_after_s,
      "The server is at capacity right now (it sheds load rather than queueing forever).");
  } else {
    renderAssistant(fx);
  }
  return true;
}

init();
