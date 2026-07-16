# Privacy Policy

_Last updated: 2026-07-16 · versioned in the project's public repository_

This service is an **independently operated hobby project run by one
person**. It is not a company, has no legal team, and no dedicated
support staff. It is provided as-is, best-effort. Please do not submit
personal, sensitive, or confidential information in your questions.

## If you sign in with Google (optional)

Signing in stores exactly four things from your verified Google
identity: your Google account id, email address, display name, and
avatar URL (the URL only — we never fetch or store the image). We
receive **no other access to your Google account**: the tokens Google
sends us are discarded immediately after verifying who you are.

- Kept for: the life of your account. Delete it any time by request
  (see contact below) — the account row and any live sessions are
  removed; usage counters expire on their own within 25 hours.
- Sessions: an opaque random id in an HttpOnly cookie, stored
  server-side for at most 7 days, containing your profile snapshot and
  nothing about your queries.
- **Signing in does not change the core promise below: your queries
  are still never stored, and nothing we keep links what you asked to
  who you are.** Rate counters are keyed to an internal opaque user id
  with the same short expiries as API keys.

Sign-in is optional; the API works with API keys (or anonymously in
development) with zero profile data collected.

## What happens to your query

Your question is processed **in memory** to search the indexed document
corpus and produce an answer. **We do not store your queries in any
database, and nothing we store is tied to who you are.** There are no
accounts, no cookies, no trackers, no ads, and nothing is sold to
anyone.

## What we store, exactly, and for how long

| Data | Where | Linked to you? | Retention |
|---|---|---|---|
| Cached responses (query text + answer, keyed by a hash of the query) | Redis (Upstash) | No — the cache has no user/key identifier | ≤ 1 hour (auto-expires) |
| Rate-limit / quota counters | Redis (Upstash) | Keyed to a truncated tag of your API key (not reversible to you) | ≤ 25 hours (auto-expires) |
| Operational logs (timings, status labels, random request IDs) | Host log stream (Render) | No — logs never contain query text, answers, or IP addresses written by the app | per host policy (Render free tier: ~7 days) |
| Aggregate metrics (counts, latency histograms) | In-process /metrics | No — verified: no user data ever becomes a metric label | reset on every restart |
| Google profile: account id, email, name, avatar URL (only if you sign in) | Postgres (Neon) | Yes — it is your identity | life of your account; deleted on request |
| Session (opaque id → profile snapshot) | Redis (Upstash) | Yes (to your account, never to queries) | ≤ 7 days (auto-expires; logout deletes immediately) |
| Raw queries in a database | — | — | **none — never stored** |

## Third parties that see data

- **Render** (hosting): like any web host, its edge infrastructure logs
  requests (including your IP address) under Render's own privacy
  policy. The application itself does not write IP-bearing request logs.
- **Upstash / Neon** (cache & database hosting): hold only the items in
  the table above. The database currently stores **no user data at all**
  (only the document corpus and index metadata).
- **Google Gemini API** (when generation is enabled): your query text
  and the retrieved document snippets are sent to Google to generate the
  answer. On the API's free tier, **Google may use submitted content to
  improve its products** under Google's terms. This is the single
  biggest data consideration of using this service — if that is not
  acceptable, don't submit anything you wouldn't want processed by
  Google. When no generation key is configured, nothing leaves this
  service and answers are extractive.

## Data removal

Cached entries expire on their own within an hour. If you never signed
in, there is nothing else to delete, because nothing else is kept. If
you signed in with Google, your account (the four profile fields above)
is the one thing we store about you — request deletion via the contact
below and it is removed along with any live sessions.

## Changes & contact

Changes to this policy are made by updating this document in the
project repository (full history visible there) before the behavior
changes. Questions: open an issue on the project repository.
