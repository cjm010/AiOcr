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

## Review Rules

Simple and safe team rule:

- `dev`
  - at least 1 teammate review recommended

- `test`
  - at least 1 approval required

- `main`
  - at least 2 approvals required
  - CI must pass

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

- do not push directly to `main`
- do not merge failing CI into `test` or `main`
- do not promote unreviewed learning artifacts
- do not overwrite another teammate's work without checking first

## Before You Open A PR

Quick checklist:

- code runs
- tests pass
- README or docs updated if needed
- no secrets included
- no `__pycache__`, `.venv`, or local data files included
