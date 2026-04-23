# Contributing Guide

This project uses a simple and safe team workflow.

## Branches

- `main`
  - production-ready only
  - no direct commits

- `test`
  - integration and QA branch
  - only merged from reviewed pull requests

- `dev`
  - shared development branch
  - feature work lands here first

- `feature/*`
  - personal work branches
  - create these from `dev`

## Recommended Flow

1. Pull the latest `dev`
2. Create a feature branch
3. Make your changes
4. Open a pull request into `dev`
5. After review and CI pass, merge into `dev`
6. Promote `dev` to `test` when the team wants to validate
7. Promote `test` to `main` only when the team is comfortable releasing

Flow:

`feature/* -> dev -> test -> main`

## Commit Policy

Keep commits small and clear.

Use messages like:

- `feat: add invoice review form`
- `fix: handle missing PDF text`
- `docs: update architecture section`
- `test: add config environment test`
- `ci: add release workflow`

Good commit rules:

- one clear purpose per commit
- no large mixed commits
- do not commit secrets
- do not commit local data folders
- do not commit virtual environments

## Pull Request Policy

Every pull request should:

- have a short clear title
- explain what changed
- explain why it changed
- mention any risks
- include screenshots for UI changes when possible

## CI — Automated Tests

Every pull request runs the full test suite automatically via GitHub Actions. Two suites must pass before a PR can be merged:

| Suite | File | What it covers |
|---|---|---|
| Pipeline tests | `tests/test_pipeline.py` | Extraction, validation, deduplication, end-to-end pipeline |
| UI tests | `tests/test_ui.py` | App startup, sidebar controls, file upload flows, bulk processing |

Tests run on Python 3.11 and 3.12. A PR that fails any test on either version is blocked from merging.

### Running tests locally before pushing

```bash
pytest tests/test_pipeline.py -v
pytest tests/test_ui.py -v
```

Or run the full suite:

```bash
pytest -v
```

### Adding new tests

- Pipeline / extraction logic → `tests/test_pipeline.py`
- UI behaviour → `tests/test_ui.py`
- Place fixture PDFs in `tests/fixtures/` — they are picked up automatically by the parametrized tests

## Branch Protection Setup

The following rules must be configured in **GitHub → Settings → Branches** for `dev`, `test`, and `main`. These enforce the CI gate and prevent direct pushes.

### For `dev`

1. Go to **Settings → Branches → Add rule**
2. Branch name pattern: `dev`
3. Enable **Require a pull request before merging**
4. Enable **Require status checks to pass before merging**
5. Search for and add these required status checks:
   - `test (3.11)`
   - `test (3.12)`
6. Enable **Do not allow bypassing the above settings**

### For `test`

Same as `dev`, plus:

- Enable **Require approvals** → set to **1 required reviewer**

### For `main`

Same as `test`, plus:

- Increase **Require approvals** to **2 required reviewers**
- Enable **Require branches to be up to date before merging**

Once these rules are saved, GitHub will block any PR that has failing CI checks, and direct pushes to the protected branches will be rejected.

## Review Rules

| Branch | Approvals required | CI required |
|---|---|---|
| `dev` | 1 (recommended) | Yes — both suites must pass |
| `test` | 1 (required) | Yes — both suites must pass |
| `main` | 2 (required) | Yes — both suites must pass |

## Learning Data Safety

Learning artifacts must stay separated by environment:

- `data/dev`
- `data/test`
- `data/prod`

Rules:

- do not copy raw `dev` learning directly to `prod`
- only promote approved learning artifacts forward
- production learning should come from reviewed corrections only

## What Not To Do

- do not push directly to `main`, `test`, or `dev` — open a PR instead
- do not merge failing CI into any protected branch
- do not promote unreviewed learning artifacts
- do not overwrite another teammate's work without checking first

## Before You Open A PR

Quick checklist:

- [ ] code runs locally
- [ ] `pytest` passes with no failures
- [ ] README or docs updated if needed
- [ ] no secrets included
- [ ] no `__pycache__`, `.venv`, or local data files included
