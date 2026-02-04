# NBA Features dbt Project

**Status:** Production Ready
**Last Updated:** February 2026

Feature engineering transformations for the NBA player props ML pipeline using SQL-based dbt models.

---

## Overview

This dbt project transforms raw player stats and prop lines into ML-ready feature vectors. It demonstrates:

- **Staging models**: Clean and normalize raw data
- **Intermediate models**: Complex feature engineering (rolling stats, book spreads)
- **Marts models**: Final feature vectors for model training

---

## Model Lineage

```
Sources                     Staging                 Intermediate              Marts
─────────────────────────────────────────────────────────────────────────────────────
player_game_logs  ──►  stg_player_game_logs  ──►  int_rolling_stats  ──┐
                                                                        ├──►  fct_feature_vectors
nba_props_xl      ──►  stg_props             ──►  int_book_spreads   ──┘
```

---

## Key Features Computed

### Rolling Stats (`int_rolling_stats`)
- L3/L5/L10/L20 averages for points, rebounds, assists
- Momentum indicators (L3 vs L10 trend)
- Consistency scores (std dev normalized)
- Back-to-back game flags
- Rest days calculation

### Book Spreads (`int_book_spreads`)
- Line spread across 10 sources (7 sportsbooks + 3 DFS)
- Per-book deviations from consensus
- Softest/sharpest book identification
- Book agreement flags
- Line movement tracking

---

## Quick Start

```bash
# Install dbt
pip install dbt-postgres

# Copy and configure profile
cp profiles.yml.example ~/.dbt/profiles.yml
# Edit with your database credentials

# Run models
dbt run

# Run tests
dbt test

# Generate documentation
dbt docs generate && dbt docs serve
```

---

## Project Structure

```
dbt/
├── models/
│   ├── staging/               # Raw data cleaning
│   │   ├── stg_player_game_logs.sql
│   │   ├── stg_props.sql
│   │   └── sources.yml
│   ├── intermediate/          # Feature engineering
│   │   ├── int_rolling_stats.sql
│   │   └── int_book_spreads.sql
│   └── marts/                 # ML-ready outputs
│       └── fct_feature_vectors.sql
├── macros/
│   └── map_book_name.sql
├── dbt_project.yml
└── profiles.yml.example
```

---

## Testing

Models include dbt tests for data quality:
- `not_null` on key identifiers
- `unique` on primary keys
- `accepted_values` for categorical fields
- `accepted_range` for numeric bounds

```bash
# Run all tests
dbt test

# Run specific model tests
dbt test --select int_rolling_stats
```

---

## Integration with Python Pipeline

The Python feature extraction (`nba/features/extract_live_features_xl.py`) mirrors these transformations for real-time inference:

1. **Training data source** - `fct_feature_vectors` provides historical features
2. **Validation baseline** - Compare Python outputs against dbt for consistency
3. **Documentation** - SQL makes feature logic explicit and reviewable

---

## Database Connection

Requires PostgreSQL databases running:

```yaml
# ~/.dbt/profiles.yml
nba_features:
  target: dev
  outputs:
    dev:
      type: postgres
      host: localhost
      port: 5536
      user: mlb_user
      password: your_password
      dbname: nba_players
      schema: dbt_features
```

---

## Related

- [Main README](../README.md) - Project overview
- [features README](../nba/features/README.md) - Python feature extraction
- [docker README](../docker/README.md) - Database setup
