# Branching and Learning Strategy

## Branches

Use three long-lived branches:

- `dev`
  - active feature development
  - experimental extraction logic
  - isolated development learning data only

- `test`
  - integration and QA branch
  - used for system testing, demo rehearsals, and approval checks
  - isolated test learning data only

- `main`
  - production-ready branch
  - stable release history
  - production learning artifacts only

Recommended flow:

`feature/* -> dev -> test -> main`

## CI/CD Flow

- Pull requests into `dev`, `test`, and `main` run CI checks
- `dev` is for code experiments and non-production learning artifacts
- `test` is for validation and staged promotions
- `main` is release-controlled and should use protected environments in GitHub

## Learning Policy

Do not let environments share the same learning store.

Each environment should have its own isolated data:

- `data/dev`
- `data/test`
- `data/prod`

That means:

- bad learning in `dev` stays in `dev`
- bad learning in `test` stays in `test`
- only explicitly promoted reviewed artifacts reach production

## Should Training Happen Only In Production?

No. The safest approach is:

- learning can happen in `dev` and `test`
- production should not automatically trust lower-environment learning
- promotion to production should be intentional and approved

In practice:

- `dev`
  - learn freely from experiments and reviewed examples
  - expect noise and mistakes

- `test`
  - learn from curated test cases and approved QA reviews
  - evaluate whether learning improves extraction quality

- `prod`
  - either disable automatic learning, or only learn from reviewed production approvals
  - promote only approved learning artifacts from `test` or production review queues

## Recommended Production Rule

Best practice for this project:

- allow extraction in all environments
- allow learning in `dev` and `test`
- in `prod`, do one of these:
  - `preferred`: only apply promoted templates and approved reviewed examples
  - `optional`: allow production learning, but only from explicitly human-approved corrections

## How Good Data Moves Forward

Good learning data should move through promotion, not by copying raw stores blindly.

Recommended promotion flow:

1. A document is processed in `dev` or `test`
2. A human reviews and approves the extraction
3. The reviewed result is exported as an approved learning artifact
4. CI/CD promotion merges that approved artifact into the next environment's promoted template store
5. Production consumes only promoted artifacts

## How Bad Data Is Prevented From Reaching Prod

- never share one `learned_templates.json` file across environments
- require human review before promotion
- use protected environments for `main`
- use manual approval for the promotion workflow
- keep audit logs and trace files for every promoted artifact

## Recommendation Summary

For this project, the right answer is:

- do not train only in production
- do not automatically promote all dev/test learning to production
- do use isolated environment stores
- do promote only reviewed, approved learning artifacts into production

This gives you experimentation speed in `dev`, validation in `test`, and safety in `prod`.
