# Stage 8.5 Report — CI/CD (GitHub Actions, free tier)

Date: 2026-07-16. Workflow: `.github/workflows/ci.yml` (committed in
the Stage 8.5 local half). This report covers the live half: real runs
on GitHub, merge blocking on master, and the deploy gate.

## What runs on every push and PR

One `test` job on `ubuntu-latest` against **real services** (Postgres
16 + Redis 7 containers, not fakes): the full suite, then both Stage 0
eval quality gates (`--fail-on-regression` vs the committed baseline;
latency excluded by design — CI hardware varies, quality metrics are
bit-reproducible). A `deploy` job runs only on push to `master` and
only after `test` passes.

## Definition of done — evidence

| Requirement | Evidence |
|---|---|
| Actual passing workflow run | [Run 29515679168](https://github.com/PritheevLingeswaran/rag/actions/runs/29515679168) on `master` (`4206517`): **206 passed in 33.82s** vs real Postgres/Redis, both eval gates "quality gate: no regression vs baseline" (skeleton + hybrid, seed=42, 20 queries) |
| Merge to master blocked on failure | Branch protection on `master` requires the `test` check (set 2026-07-16 via API). Drill: [PR #1](https://github.com/PritheevLingeswaran/rag/pull/1) deliberately broke `/health` ("ok" → "borked"); CI result `test=FAILURE, deploy=SKIPPED`, PR `mergeStateStatus: BLOCKED`; closed unmerged, branch deleted |
| Auto-deploy to Stage 8 host on merge to master | `deploy` job gated by `needs: test` + master-only condition; triggers Render via `RENDER_DEPLOY_HOOK` repo secret. **Secret not yet configured** — job currently skips loudly ("RENDER_DEPLOY_HOOK not configured; skipping deploy.", verified in the run log) |

## Broken-PR drill timeline (2026-07-16, UTC)

| Step | Time |
|---|---|
| PR #1 opened (health deliberately broken; test failed locally first) | ~16:43Z |
| Merge state BLOCKED (checks pending) | 16:44:34Z |
| Required check `test` = FAILURE, still BLOCKED | 16:45:36Z |
| PR closed unmerged, branch deleted | 16:46Z |

## Operator action to finish the deploy leg (~2 min, dashboard-side)

Render service → Settings → Deploy Hook → copy URL → GitHub repo →
Settings → Secrets and variables → Actions → new secret
`RENDER_DEPLOY_HOOK`. Keep Render's own Auto-Deploy **off** (it is off
now, post-rollback) so the ONLY path to production is a green CI run
on master. Production is currently 3 commits behind master (still
0.1.0, pre-Gemini-fix) until the hook is set or a manual deploy runs.

## Honest notes

- Branch protection uses `enforce_admins: false`: the repo owner can
  still push directly to master and can override a red PR with an
  explicit admin bypass. Chosen deliberately — this project's workflow
  is solo direct-push, and `enforce_admins: true` would reject every
  direct push (required checks can't have run on a not-yet-pushed
  commit). The normal merge path is hard-blocked; the bypass is a
  labeled emergency hatch, not a silent hole.
- GitHub free tier: unlimited Actions minutes on public repos; branch
  protection free on public repos. Both hold because the repo is
  public.
- The `deploy` job reports "success" when it skips (secret absent by
  design pre-provisioning). Once the secret exists this becomes a real
  deploy trigger; until then green master runs do NOT mean deployed.

Suite: **206 passing** (CI, real services). Standing eval unchanged
vs baseline (both pipelines).
