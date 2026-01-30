# ADR-006: Book Disagreement Features

## Status

Accepted

## Date

2025-11-05

## Context

Different sportsbooks often set different lines for the same prop.
This disagreement signals:

1. **Market inefficiency**: One book may be mispriced
2. **Sharp vs public**: Some books react faster to information
3. **Betting opportunity**: Softest line offers better EV

Initial models used only player statistics, ignoring valuable
market-derived signals available at prediction time.

## Decision

Add **23 book disagreement features** to the feature set:

### Feature Groups

#### 1. Line Variance (5 features)
```python
"line_spread"        # max(lines) - min(lines)
"consensus_line"     # mean of all books
"line_std_dev"       # standard deviation
"num_books_offering" # count of books
"line_coef_variation"  # std_dev / mean
```

#### 2. Book Deviations (8 features)
```python
"{book}_deviation"   # book_line - consensus
# For: draftkings, fanduel, betmgm, caesars,
#      bet365, betrivers, espnbet, fanatics
```

#### 3. Line Metadata (7 features)
```python
"softest_book_id"        # Encoded book with lowest line
"hardest_book_id"        # Encoded book with highest line
"line_spread_percentile" # Spread vs historical distribution
"books_agree"            # 1 if spread < 0.5
"books_disagree"         # 1 if spread >= 2.0
"softest_vs_consensus"   # Softest line - consensus
"hardest_vs_consensus"   # Hardest line - consensus
```

#### 4. Additional (3 features)
```python
"min_line"  # Minimum line across books
"max_line"  # Maximum line across books
"line_std"  # Same as line_std_dev
```

### Feature Extraction
- Query `nba_props_xl` table for all book lines
- Calculate deviations in real-time
- Cache per (player, game_date, stat_type)

## Consequences

### Positive
- **Improved AUC**: 0.767 vs 0.745 without book features
- **Line shopping**: Identifies best book to bet
- **Market signal**: Captures information in line movements
- **Edge detection**: High spread = higher potential edge

### Negative
- **Data dependency**: Requires multiple book lines available
- **Timeliness**: Lines change, features may stale
- **Feature count**: 23 new features, risk of overfitting

### Neutral
- Feature importance shows book features rank mid-tier
- Works with single-book data (deviations = 0)

## Feature Importance Rankings

From SHAP analysis:

| Rank | Feature | Importance |
|------|---------|------------|
| 3 | `line_spread` | 0.089 |
| 7 | `consensus_line` | 0.052 |
| 12 | `softest_vs_consensus` | 0.038 |
| 15 | `num_books_offering` | 0.031 |
| 18 | `draftkings_deviation` | 0.024 |

## Validation Results

### High-Spread Performance (line_spread >= 2.5)
- Win Rate: 70.6%
- ROI: +34.82%
- Sample: 17 bets

### Books Agreement Performance (spread < 0.5)
- Win Rate: 52.3%
- ROI: +0.8%
- Conclusion: Low edge when books agree

## Alternatives Considered

### 1. No Book Features
- Pros: Simpler, no data dependency
- Cons: Missing valuable signal
- Rejected because: 2.2% AUC improvement significant

### 2. Single "Softest Line" Feature
- Pros: Simple, captures key information
- Cons: Loses disagreement signal
- Rejected because: Spread is highly predictive

### 3. Book Intelligence Model (Third Head)
- Pros: Dedicated model for book signals
- Cons: Additional complexity
- Status: Implemented as optional enhancement

### 4. Time-Series Book Features
- Pros: Capture line movements
- Cons: Requires real-time updates
- Status: Future enhancement

## Data Sources

1. **BettingPros API**: 7 sportsbooks (primary)
2. **Underdog Fantasy**: DFS lines (softer)
3. **MongoDB**: Pre-computed line_shopping subdocument
