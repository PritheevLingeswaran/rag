# Stage 9.7 Report — Chat UI (ported from SmartQA)

Date: 2026-07-16. The comparison review of `Projects/rag-smart-qa`
identified its chat experience as the one clear advantage over ragp's
single-shot form. This stage ports that experience onto ragp's backend
and honesty rules. Still zero dependencies, zero build step.

## What changed (frontend only; backend untouched)

- **Conversational layout**: transcript of user/assistant bubbles,
  auto-scrolling `role="log"` region, composer with auto-expanding
  textarea, Enter-to-send / Shift+Enter-newline.
- **Every honest state from 9.6, re-expressed in the chat idiom**: the
  per-message status banner (verified / partial / retrieval-only with
  plain-language reason), 429/503 as inline wait bubbles with the live
  server-derived countdown (composer locks until zero), errors as inline
  bubbles with a retry button, session expiry back to the landing page,
  and the truthful cold-start pending bubble (elapsed counter, free-tier
  note after 5 s).
- **Dark mode** via `prefers-color-scheme` (SmartQA's visual signature,
  minus its glassmorphism — contrast and `:focus-visible` kept).
- **Citation markers** `[n]` highlighted inside answers via safe DOM
  construction — no `innerHTML` anywhere near dynamic data.
- **Honesty note ported INTO the idiom**: each question is answered
  independently; the transcript is presentation, not context. The UI
  never implies conversational memory the backend doesn't have (that is
  a real backend feature with token costs — deferred explicitly, below).
- `#demo:<state>` gallery re-pointed at the chat renderers (same eight
  states, same single rendering path as live traffic).

## Deliberately NOT ported (each deserves its own gated stage)

| SmartQA feature | Why deferred, not rushed |
|---|---|
| Document upload | ragp's ingestion/versioning machinery exists but is unwired from serving; wiring it means index reload semantics + an upload-authorization model (who may write to the shared corpus?) — an architecture stage, not a UI afternoon |
| Query rewriting/planning | any retrieval change must face the eval regression gate with measured results; bolting it on untested would gamble the metric the whole project is built around |
| Conversational memory | changes prompts, token spend, and the quota math; faking it in the UI was the alternative and was rejected |
| Streaming responses | backend answers arrive whole (validation runs on the complete text before anything is shown — streaming would leak unvalidated sentences); honest non-goal |

## Verified

- 50 api tests green (backend untouched); `app.js` syntax-checked.
- Local: `/` serves the chat layout, assets 200, demo states render.
- Live verification on production follows the same deploy gate as
  every change (CI -> auto-deploy); see commit for the live check.
