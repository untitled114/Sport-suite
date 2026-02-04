# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) documenting
significant technical decisions made in the NBA Props ML system.

## What is an ADR?

An ADR captures an important architectural decision along with its context
and consequences. ADRs provide:

- **Historical context**: Why decisions were made
- **Trade-off documentation**: What alternatives were considered
- **Onboarding aid**: Help new team members understand the system

## ADR Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [001](ADR-001-two-head-stacked-model-architecture.md) | Two-Head Stacked Model Architecture | Accepted | 2025-11-06 |
| [002](ADR-002-multi-database-postgresql-design.md) | Multi-Database PostgreSQL Design | Accepted | 2025-10-01 |
| [003](ADR-003-lightgbm-over-xgboost-catboost.md) | LightGBM over XGBoost/CatBoost | Accepted | 2025-10-15 |
| [004](ADR-004-disabling-assists-threes-markets.md) | Disabling ASSISTS and THREES Markets | Accepted | 2025-11-07 |
| [005](ADR-005-isotonic-calibration-for-probability.md) | Isotonic Calibration for Probabilities | Accepted | 2025-11-06 |
| [006](ADR-006-book-disagreement-features.md) | Book Disagreement Features | Accepted | 2025-11-05 |

## Related Documents

| Document | Description |
|----------|-------------|
| [Case Study: Goblin Lines](../CASE_STUDY_GOBLIN_LINES.md) | Training data mismatch analysis (Feb 2026) |
| [Database Audit](../DATABASE_AUDIT.md) | Schema documentation |

## ADR Format

Each ADR follows the template:

1. **Title**: Short descriptive name
2. **Status**: Proposed, Accepted, Deprecated, Superseded
3. **Date**: When the decision was made
4. **Context**: What is the issue being addressed?
5. **Decision**: What is the change being proposed/implemented?
6. **Consequences**: What are the trade-offs?
7. **Alternatives Considered**: What other options were evaluated?

## Creating New ADRs

1. Copy the template from an existing ADR
2. Number sequentially (ADR-NNN)
3. Fill in all sections
4. Submit for review
5. Update the index above

## Key Decisions Summary

### Model Architecture (ADR-001)
Two-head stacked model: Regressor predicts value, Classifier predicts P(OVER).
Blended output (60% classifier + 40% residual) with isotonic calibration.

### Database Design (ADR-002)
Four PostgreSQL databases by domain (players, games, team, intelligence).
Enables independent scaling and clear ownership.

### LightGBM (ADR-003)
Chosen over XGBoost and CatBoost for speed, memory efficiency, and
native categorical handling.

### Market Selection (ADR-004)
ASSISTS and THREES disabled due to poor AUC (<0.60).
POINTS and REBOUNDS maintained (AUC >0.74).

### Calibration (ADR-005)
Isotonic regression for probability calibration.
Preserves ranking while improving reliability.

### Book Features (ADR-006)
20 book disagreement features capture line shopping opportunities.
Key predictor of edge when books disagree on lines.

---

**Last Updated:** February 2026
