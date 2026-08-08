/* ragp frontend — Stage 9.10 app shell. Zero dependencies, no build.
 *
 * Unchanged contract from Stage 9.6: every documented backend state has
 * an explicit rendering — ok / ok_partial_rejected / ok_no_answer /
 * no_results / every degraded_* reason / 429 (rate vs daily, live
 * countdown from the server's number) / 503 shed / 503 not-ready /
 * 401 sign-in / network failure with retry / truthful cold-start
 * loading. Blank screens and raw console errors are defined as bugs;
 * the window-level handlers below make them impossible.
 *
 * Honesty rules that survive the redesign:
 *  - Each question is answered independently. The transcript is
 *    presentation, never context: no prior turn is sent to the backend,
 *    and the UI says so under the composer. Multi-turn memory is a
 *    backend feature with token costs, deliberately not faked.
 *  - "Recent" threads live in localStorage — this browser only, never
 *    uploaded. Replaying one re-renders stored payloads through the SAME
 *    render path as live traffic.
 *  - Sign-in is offered only when the deployment can actually perform it
 *    (/auth/google/login answers 503 when unconfigured).
 *
 * #demo:<state> renders recorded real payloads through that same path.
 */
"use strict";

const $ = (id) => document.getElementById(id);

const els = {
  shell: $("shell"),
  sidebar: $("sidebar"), scrim: $("scrim"),
  sidebarOpen: $("sidebar-open"), sidebarClose: $("sidebar-close"),
  newChat: $("new-chat"),
  threadList: $("thread-list"), threadsLabel: $("threads-label"),
  threadTitle: $("thread-title"), envChip: $("env-chip"),
  emptyState: $("empty-state"), messages: $("messages"),
  suggestions: $("suggestions"),
  composer: $("composer"), input: $("composer-input"), send: $("send-btn"),
  quotaHint: $("quota-hint"),
  uploadBtn: $("upload-btn"), fileInput: $("file-input"),
  accountOut: $("account-out"), accountIn: $("account-in"),
  loginLink: $("login-link"), authNote: $("auth-note"),
  userName: $("user-name"), userEmail: $("user-email"),
  userAvatar: $("user-avatar"), logoutBtn: $("logout-btn"),
};

let lastQuery = null;
let inFlight = false;
let countdownTimer = null;
let signedIn = false;
let authRequired = false;   // production refuses anonymous queries

/* ---------- small DOM helpers (no innerHTML with dynamic data) ---------- */

function el(tag, className, text) {
  const n = document.createElement(tag);
  if (className) n.className = className;
  if (text !== undefined) n.textContent = text;
  return n;
}

function enterConversation() {
  els.emptyState.hidden = true;
  els.messages.hidden = false;
}

function addMsg(className) {
  enterConversation();
  const m = el("div", `msg ${className}`);
  els.messages.appendChild(m);
  scrollDown();
  return m;
}

function addUserMsg(text) {
  const m = addMsg("user");
  m.appendChild(el("div", "bubble", text));
  return m;
}

function scrollDown() { els.messages.scrollTop = els.messages.scrollHeight; }

/* ---------- threads: this browser only, never uploaded ---------- */

const STORE_KEY = "ragp.threads.v1";
const MAX_THREADS = 30;
let threads = [];
let activeThreadId = null;

function loadThreads() {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    threads = raw ? JSON.parse(raw) : [];
    if (!Array.isArray(threads)) threads = [];
  } catch { threads = []; }
}

function saveThreads() {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify(threads.slice(0, MAX_THREADS)));
  } catch { /* private mode / quota: history is a convenience, not a feature */ }
}

function renderThreadList() {
  els.threadList.replaceChildren();
  els.threadsLabel.hidden = threads.length === 0;
  threads.slice(0, MAX_THREADS).forEach((t) => {
    const li = el("li");
    if (t.id === activeThreadId) li.className = "active";
    const b = el("button", null, t.title);
    b.type = "button";
    b.title = t.title;
    b.addEventListener("click", () => openThread(t.id));
    li.appendChild(b);
    els.threadList.appendChild(li);
  });
}

function recordTurn(query, payload) {
  let t = threads.find((x) => x.id === activeThreadId);
  if (!t) {
    t = { id: String(Date.now()), title: query.slice(0, 60), turns: [] };
    activeThreadId = t.id;
    threads.unshift(t);
    els.threadTitle.textContent = t.title;
  }
  t.turns.push({ q: query, a: payload });
  saveThreads();
  renderThreadList();
}

function openThread(id) {
  const t = threads.find((x) => x.id === id);
  if (!t) return;
  activeThreadId = id;
  els.threadTitle.textContent = t.title;
  els.messages.replaceChildren();
  enterConversation();
  t.turns.forEach((turn) => {
    addUserMsg(turn.q);
    if (turn.a) renderAssistant(turn.a);
  });
  renderThreadList();
  closeSidebarOnMobile();
  scrollDown();
}

function newThread() {
  activeThreadId = null;
  lastQuery = null;
  els.messages.replaceChildren();
  els.messages.hidden = true;
  els.emptyState.hidden = false;
  els.threadTitle.textContent = "New question";
  renderThreadList();
  closeSidebarOnMobile();
  els.input.focus();
}

/* ---------- sidebar ---------- */

const MOBILE = () => window.matchMedia("(max-width: 860px)").matches;

function closeSidebarOnMobile() {
  if (MOBILE()) els.shell.classList.add("side-hidden");
}

els.sidebarOpen.addEventListener("click", () =>
  els.shell.classList.remove("side-hidden"));
els.sidebarClose.addEventListener("click", () =>
  els.shell.classList.add("side-hidden"));
els.scrim.addEventListener("click", () =>
  els.shell.classList.add("side-hidden"));
els.newChat.addEventListener("click", newThread);

if (MOBILE()) els.shell.classList.add("side-hidden");

/* ---------- status vocabulary (unchanged since 9.6) ---------- */

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
  m.className = "msg assistant";
  m.replaceChildren();

  const who = el("div", "who");
  who.appendChild(el("span", "dot", "r"));
  who.appendChild(el("span", null, "ragp"));
  m.appendChild(who);

  const banner = el("span", "banner");
  const s = data.status;
  if (s === "ok") {
    banner.classList.add("ok");
    banner.textContent = "Verified — every sentence checked against its cited sources";
  } else if (s === "ok_partial_rejected") {
    banner.classList.add("degraded");
    banner.textContent = "Partial — unverifiable sentences were removed";
  } else if (s === "ok_no_answer") {
    banner.classList.add("ok");
    banner.textContent = "The sources do not contain an answer to this";
  } else if (s === "no_results") {
    banner.classList.add("degraded");
    banner.textContent = "No relevant documents found";
  } else {
    banner.classList.add("degraded");
    banner.textContent =
      `Retrieval-only — ${DEGRADED_REASONS[s] || "the AI step was unavailable"}. ` +
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

  /* Retrieval trace: the ranked evidence ledger. Chunks the answer
     actually cited are highlighted; the rest show what the retriever
     considered and the reranker ordered. */
  const retrieved = data.retrieved_chunk_ids || [];
  if (retrieved.length) {
    const trace = document.createElement("details");
    trace.className = "trace";
    const sum = document.createElement("summary");
    sum.textContent = `Retrieval trace — ${retrieved.length} chunks ranked`;
    trace.appendChild(sum);
    const ol = el("ol", "trace-list");
    const citedSet = new Set(cites);
    retrieved.forEach((id) => {
      const li = el("li", citedSet.has(id) ? "cited" : null, id);
      if (citedSet.has(id)) li.appendChild(el("span", "cited-tag", "cited"));
      ol.appendChild(li);
    });
    trace.appendChild(ol);
    m.appendChild(trace);
  }

  const metaBits = [];
  if (data.rerank_status) metaBits.push(`rerank ${data.rerank_status}`);
  if (data.__ms) metaBits.push(`${(data.__ms / 1000).toFixed(1)}s round trip`);
  if (data.cached) metaBits.push("served from cache (≤1 h old)");
  if (data.request_id) metaBits.push(`ref ${data.request_id}`);
  if (metaBits.length) m.appendChild(el("div", "meta", metaBits.join(" · ")));

  if (data.answer && s !== "ok_no_answer" && s !== "no_results") {
    const copy = el("button", "copy-btn", "Copy answer");
    copy.type = "button";
    copy.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(data.answer);
        copy.textContent = "Copied";
        setTimeout(() => { copy.textContent = "Copy answer"; }, 1500);
      } catch { /* clipboard unavailable (http, permissions): best effort */ }
    });
    m.appendChild(copy);
  }

  if (s === "degraded_quota_throttled" && data.retry_after_s) {
    startCountdown(Math.ceil(data.retry_after_s),
      "The AI budget resets then; retrieval-only answers keep working meanwhile.",
      /*lockComposer=*/false);
  }
  scrollDown();
  return m;
}

/* ---------- wait / error states ---------- */

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

function showErrorMsg(message, opts = {}) {
  const m = addMsg("error");
  m.setAttribute("role", "alert");
  m.appendChild(el("p", null, message));
  if (opts.signIn) {
    const a = el("a", "copy-btn", "Sign in with Google");
    a.href = "/auth/google/login";
    m.appendChild(a);
  } else {
    const retry = el("button", "ghost", "Try again");
    retry.type = "button";
    retry.addEventListener("click", () => {
      m.remove();
      if (lastQuery) runQuery(lastQuery, /*reAsk=*/true);
    });
    m.appendChild(retry);
  }
  els.send.disabled = false;
  inFlight = false;
  scrollDown();
}

/* ---------- the query round-trip ---------- */

function pendingNode() {
  const m = addMsg("assistant");
  const who = el("div", "who");
  who.appendChild(el("span", "dot", "r"));
  who.appendChild(el("span", null, "ragp"));
  m.appendChild(who);
  const p = el("p");
  const spin = el("span", "spinner");
  spin.setAttribute("aria-hidden", "true");
  p.append(spin, " Searching the corpus… ");
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
  if (!reAsk) addUserMsg(query);

  const pending = pendingNode();
  const t0 = performance.now();

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
  if (body) body.__ms = Math.round(performance.now() - t0);

  pending.stop();
  inFlight = false;
  els.send.disabled = false;

  if (resp.ok && body) {
    renderAssistant(body, pending.m);
    recordTurn(query, body);
    return;
  }
  pending.m.remove();

  if (resp.status === 401) {
    showErrorMsg(
      signedIn
        ? "Your session has expired — please sign in again."
        : "This deployment requires you to sign in before asking a question.",
      { signIn: true },
    );
    signedIn = false;
    showSignedOut();
    return;
  }
  if (resp.status === 429 && body) {
    const headerRetry = Number(resp.headers.get("retry-after"));
    const secondsLeft = Math.ceil(
      body.retry_after_s ?? (headerRetry > 0 ? headerRetry : 60)
    );
    const scope = (body.error || "").includes("daily")
      ? "You've used today's question allowance."
      : "You're asking a little too quickly.";
    startCountdown(secondsLeft, scope);
    return;
  }
  if (resp.status === 503 && body) {
    if (body.error === "service not ready") {
      showErrorMsg("This instance is starting up and cannot answer yet. "
                   + "It will be ready shortly.");
      return;
    }
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

/* ---------- composer ---------- */

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
  els.input.style.height = `${Math.min(els.input.scrollHeight, 176)}px`;
});

els.suggestions.addEventListener("click", (e) => {
  const b = e.target.closest("button[data-q]");
  if (!b || inFlight) return;
  els.input.value = b.dataset.q;
  els.composer.requestSubmit();
});

/* ---------- document upload (dev-mode; session-scoped, and says so) ---------- */

async function uploadDocument(f) {
  const m = addMsg("system");
  m.textContent = `Indexing ${f.name}…`;
  const fd = new FormData();
  fd.append("file", f);
  let resp, body = null;
  try {
    resp = await fetch("/v1/documents", { method: "POST", body: fd });
    try { body = await resp.json(); } catch { /* handled below */ }
  } catch {
    m.remove();
    showErrorMsg("Upload failed — could not reach the server.");
    return;
  }
  if (resp.ok && body) {
    m.textContent =
      `Added ${f.name} — ${body.chunks_added} chunk${body.chunks_added === 1 ? "" : "s"} ` +
      "indexed for this session (resets on restart). Ask about it.";
  } else {
    m.remove();
    showErrorMsg(
      body && body.error
        ? `Upload rejected: ${body.error}`
        : "Upload failed — please try again."
    );
  }
}

els.uploadBtn.addEventListener("click", () => els.fileInput.click());
els.fileInput.addEventListener("change", () => {
  const f = els.fileInput.files[0];
  els.fileInput.value = "";
  if (f) uploadDocument(f);
});

/* ---------- account ---------- */

function showSignedIn(profile) {
  signedIn = true;
  els.accountOut.hidden = true;
  els.accountIn.hidden = false;
  els.userName.textContent = profile.name || "Signed in";
  els.userEmail.textContent = profile.email || "";
  if (profile.avatar_url) {
    els.userAvatar.src = profile.avatar_url;
    els.userAvatar.hidden = false;
  } else {
    els.userAvatar.removeAttribute("src");
    els.userAvatar.hidden = true;   // no src => no broken-image box
  }
}

function showSignedOut(note) {
  signedIn = false;
  els.accountIn.hidden = true;
  els.accountOut.hidden = false;
  if (note) els.authNote.textContent = note;
}

els.logoutBtn.addEventListener("click", async () => {
  try { await fetch("/auth/logout", { method: "POST" }); } catch { /* still leave */ }
  showSignedOut("Signed out.");
});

/* Offer sign-in only where it can actually complete: the flow needs
   GOOGLE_CLIENT_ID/SECRET + Redis + Postgres, and answers 503 without
   them. A button that always errors is worse than an explained absence. */
async function probeGoogleLogin() {
  try {
    const r = await fetch("/auth/google/login", { redirect: "manual" });
    // An available flow redirects to Google (opaque under manual redirect).
    if (r.type === "opaqueredirect" || r.status === 0 || r.status === 302) return true;
    return r.status !== 503;
  } catch {
    return false;
  }
}

async function setUpAccount() {
  let profile = null;
  try {
    const me = await fetch("/auth/me");
    if (me.ok) profile = await me.json();
  } catch { /* treated as signed out */ }

  if (profile) { showSignedIn(profile); return; }

  showSignedOut();
  const available = await probeGoogleLogin();
  if (!available) {
    els.loginLink.setAttribute("aria-disabled", "true");
    els.authNote.textContent = authRequired
      ? "Google sign-in is not configured on this deployment, so questions "
        + "require an API key. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET to enable it."
      : "Google sign-in is not configured here — questions run anonymously "
        + "in development.";
  } else {
    els.authNote.textContent = authRequired
      ? "Sign in to ask questions."
      : "Optional in development — questions run anonymously.";
  }
}

/* Uploads are a dev/staging feature (production answers 403). Reveal the
   composer's "+" only where it can actually work. Failure to determine
   the environment leaves it hidden -- the safe direction. */
async function readEnvironment() {
  try {
    const health = await (await fetch("/health")).json();
    const isProd = health.environment === "production";
    els.uploadBtn.hidden = isProd;
    authRequired = isProd;
    if (health.environment && health.environment !== "production") {
      els.envChip.textContent = health.environment;
      els.envChip.hidden = false;
    }
  } catch {
    els.uploadBtn.hidden = true;
  }
}

/* ---------- window-level safety net ---------- */

window.addEventListener("error", () => showErrorMsg("Unexpected error in the page — please retry."));
window.addEventListener("unhandledrejection", (e) => {
  e.preventDefault();
  showErrorMsg("Unexpected error in the page — please retry.");
});

/* ---------- boot ---------- */

async function init() {
  loadThreads();
  renderThreadList();
  if (runDemoIfRequested()) return;
  await readEnvironment();
  await setUpAccount();
  els.input.focus();
}

/* ---------- state gallery (#demo:<name>) ---------- */

const DEMO = {
  ok: { answer: "In Raft, time is divided into terms, and each term begins with a leader election [1]. A candidate wins by collecting votes from a majority of the cluster [1].", status: "ok", citations: ["raft::c0"], retrieved_chunk_ids: ["raft::c0", "raft::c1", "paxos::c1", "quorum::c0"], rerank_status: "full", request_id: "demo0000ok", cached: false, __ms: 1240 },
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
  showSignedIn({ name: "Demo User", email: "demo@example.com" });
  addUserMsg("how does raft handle leader election");
  lastQuery = "how does raft handle leader election";
  els.threadTitle.textContent = "how does raft handle leader election";
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
