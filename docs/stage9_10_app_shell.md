# Stage 9.10 Report — App shell

Date: 2026-08-08. Stage 9.8 gave the frontend a visual identity but kept
9.7's single-column layout. On a 1920×1080 display that layout put a
strip of text at the top, the composer pinned to the bottom, and roughly
600px of empty background between them. The page was not broken; it was
unconsidered at the size people actually use.

## The layout decision

Rebuilt on the conventions of ChatGPT and Claude — sidebar, centred
empty state, pill composer, full-width assistant turns. That is a
deliberate choice to be **unoriginal**: a chat surface is a form people
already know, and novelty in the furniture costs comprehension without
buying anything. The identity from 9.8 is unchanged underneath (serif
brand, monospace for anything asserted as evidence, one citation-amber
accent, status as text + shape rather than colour alone).

The void fix itself is one rule: the empty state is a flex child that
centres in the remaining space, so a wide viewport opens on a
composition instead of a stripe above a gap. Once a message exists the
empty state is hidden and the transcript scrolls normally.

| element | behaviour |
|---|---|
| sidebar | brand, New question, recent threads, account block; overlay + scrim under 860px |
| empty state | centred hero, four suggestion cards naming real corpus topics |
| composer | rounded card, upload `+` and send inside it, textarea auto-grows to 11rem |
| user turn | right-aligned bubble, max 85% of the column |
| assistant turn | full column width with a label — no bubble, because an answer with sources is a record, not a remark |

## Sign-in became visible, and honest about availability

Sign-in previously lived on a landing page that development builds never
showed, so the affordance was effectively invisible. It now sits in the
sidebar permanently.

It is also offered **only where it can complete**. `/auth/google/login`
answers 503 unless `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, Redis and
Postgres are all configured, so the frontend probes it once and, when
unavailable, disables the button and states the reason instead of
producing an error on click. Same rule as the upload `+` in 9.8: an
affordance whose only outcome is failure is worse than an explained
absence.

A 401 from a query now explains itself inline and offers sign-in, rather
than throwing the user back to a landing page.

**Known deployment gap, recorded here rather than discovered later:**
production has no `GOOGLE_CLIENT_ID`/`GOOGLE_CLIENT_SECRET` set, so the
live site cannot sign anyone in — and since production refuses anonymous
queries (Stage 5 policy), a visitor without an API key cannot ask
anything. The OAuth implementation is complete and tested; only the
credentials are missing. Setting them in the Render dashboard is the
whole fix.

## Recent threads — client-side only, and not conversational memory

The sidebar keeps recent questions in `localStorage`, capped at 30. Two
constraints made this safe to add:

- **Never uploaded.** Transcripts stay in the browser. The privacy
  policy's server-side commitment is unaffected because the server is not
  involved.
- **Not context.** Replaying a thread re-renders stored payloads through
  the *same* render path as live traffic, and no prior turn is ever sent
  with a new query. The backend has no conversational memory, so the
  composer hint says each question is answered independently. Presenting
  a thread list while faking memory was the failure mode to avoid; the
  honesty rule from 9.7 is unchanged.

## Verified

Real browser (Playwright), dark mode, under the Stage 9.9 CSP:

- 1920×1080: empty state centred, sidebar and Google button present,
  environment chip reads `development`, suggestion card → **VERIFIED**
  cited answer with a 5-row retrieval trace.
- Two-turn thread stored (`turns: 2`), replayed with both questions and
  both answers, and surviving a page reload.
- 390×844: sidebar hidden behind the toggle, no horizontal scroll.
- **Zero CSP violations** — no inline style or script anywhere, which is
  what lets 9.9's policy keep `'unsafe-inline'` out. The only console
  errors are the expected 401 from `/auth/me` and 503 from the sign-in
  probe, both handled.

## Follow-up defect (same day)

The first deploy of this stage rendered the new HTML against a **cached
copy of the old stylesheet** for returning visitors — an unstyled Google
logo filling the viewport. Root cause was missing `Cache-Control` on the
static mount, not this layout; see the fix in commit `3df26d9`, which
sets `no-cache` on `/` and `/app/*` so the existing ETag turns
revalidation into a 304.
