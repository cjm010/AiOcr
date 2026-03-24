# GitHub Branch Protection Setup

Apply these settings in GitHub for a simple and safe team workflow.

## `main`

Recommended protection:

- require a pull request before merging
- require 2 approvals
- dismiss stale approvals when new commits are pushed
- require status checks to pass
- require branches to be up to date before merging
- restrict direct pushes

## `test`

Recommended protection:

- require a pull request before merging
- require 1 approval
- require status checks to pass
- restrict direct pushes

## `dev`

Recommended protection:

- require a pull request before merging
- require status checks to pass
- direct pushes optional, but safer to disable for team consistency

## Required Status Checks

Use the CI workflow checks from this repo as required checks:

- dependency install
- smoke import check
- test job

## Suggested Repository Settings

- enable auto-delete head branches after merge
- enable branch update suggestions
- use squash merge or merge commit consistently across the team

## Recommended Merge Style

For this team project, `Squash and merge` is the simplest option.

Why:

- keeps history clean
- reduces noisy commit history from feature branches
- makes release review easier
