# Prediction Output Schema

This document explains every field in the prediction output files (`nba/betting_xl/predictions/xl_picks_*.json`).

## File Location

```
nba/betting_xl/predictions/
├── xl_picks_2026-01-07.json   # Model inference output
├── xl_picks_2026-01-08.json
└── ...
```

## Full Schema

```json
{
  "generated_at": "2026-01-15T16:53:34.469659",
  "date": "2026-01-07",
  "strategy": "XL Line Shopping (Softest Line)",
  "markets_enabled": ["POINTS", "REBOUNDS"],
  "total_picks": 6,
  "picks": [...],
  "summary": {...},
  "expected_performance": {...}
}
```

---

## Top-Level Fields

| Field | Type | Description |
|-------|------|-------------|
| `generated_at` | ISO timestamp | When predictions were generated |
| `date` | YYYY-MM-DD | Game date these predictions are for |
| `strategy` | string | Filter strategy used (e.g., "XL Line Shopping") |
| `markets_enabled` | array | Which markets were active (`["POINTS", "REBOUNDS"]`) |
| `total_picks` | int | Number of picks that passed filters |
| `picks` | array | List of individual bet recommendations |
| `summary` | object | Aggregate statistics |
| `expected_performance` | object | Backtest-based expected metrics |

---

## Pick Object

Each pick represents a single bet recommendation:

```json
{
  "player_name": "Stephen Curry",
  "stat_type": "POINTS",
  "side": "OVER",
  "prediction": 31.46,
  "p_over": 0.740,
  "confidence": "HIGH",
  "filter_tier": "STAR_V3",
  "consensus_line": 28.5,
  "consensus_offset": -1.0,
  "line_spread": 2.0,
  "num_books": 7,
  "opponent_team": "MIL",
  "is_home": true,
  "top_3_lines": [...],
  "line_distribution": [...],
  "best_book": "underdog",
  "best_line": 27.5,
  "edge": 3.96,
  "edge_pct": 14.4,
  "p_under": 0.260,
  "expected_wr": 0.833,
  "model_version": "v3",
  "reasoning": "Model predicts 31.5 vs softest line 27.5."
}
```

### Field Definitions

#### Core Prediction Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `player_name` | string | Player's full name | "Stephen Curry" |
| `stat_type` | string | Prop market | "POINTS", "REBOUNDS", "ASSISTS", "THREES" |
| `side` | string | Bet direction | "OVER" or "UNDER" |
| `prediction` | float | Model's predicted stat value | 31.46 (predicts 31.46 points) |
| `p_over` | float | Calibrated P(actual > line) | 0.740 (74% chance of going over) |
| `p_under` | float | Calibrated P(actual < line) | 0.260 (26% chance) |

#### Confidence and Filtering

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `confidence` | string | Confidence bucket | "HIGH", "MEDIUM", "STANDARD" |
| `filter_tier` | string | Which filter rule this passed | "STAR_V3", "star_tier", "V3_STANDARD_UNDER" |
| `expected_wr` | float | Expected win rate from backtest | 0.833 (83.3% historical WR for this tier) |
| `model_version` | string | Model version used | "v3", "xl" |

#### Line Shopping Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `best_book` | string | Sportsbook with optimal line | "underdog" |
| `best_line` | float | The line to bet at | 27.5 |
| `consensus_line` | float | Average line across all books | 28.5 |
| `consensus_offset` | float | best_line - consensus_line | -1.0 (1 point softer than consensus) |
| `line_spread` | float | max_line - min_line | 2.0 (books range from 27.5 to 29.5) |
| `num_books` | int | Number of books with this prop | 7 |

#### Edge Metrics

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `edge` | float | prediction - best_line (absolute) | 3.96 (model sees 3.96 pts of edge) |
| `edge_pct` | float | edge / best_line * 100 | 14.4% edge |

#### Game Context

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `opponent_team` | string | Opponent team code | "MIL" (Milwaukee Bucks) |
| `is_home` | bool | Home game flag | true |

#### Multi-Book Data

**`top_3_lines`**: Best 3 books for this bet

```json
[
  {"book": "underdog", "line": 27.5, "edge": 3.96},
  {"book": "betrivers", "line": 28.5, "edge": 2.96},
  {"book": "caesars", "line": 28.5, "edge": 2.96}
]
```

**`line_distribution`**: Full breakdown by line value

```json
[
  {
    "line": 27.5,
    "books": ["underdog"],
    "count": 1,
    "edge": 3.96,
    "edge_pct": 14.4
  },
  {
    "line": 28.5,
    "books": ["betrivers", "caesars", "espnbet", "fanduel", "draftkings"],
    "count": 5,
    "edge": 2.96,
    "edge_pct": 10.4
  },
  {
    "line": 29.5,
    "books": ["betmgm"],
    "count": 1,
    "edge": 1.96,
    "edge_pct": 6.7
  }
]
```

#### Reasoning

| Field | Type | Description |
|-------|------|-------------|
| `reasoning` | string | Human-readable explanation of the pick |

Example: "Model predicts 31.5 vs softest line 27.5."

---

## Summary Object

Aggregate statistics for all picks:

```json
{
  "total": 6,
  "by_market": {
    "POINTS": 2,
    "REBOUNDS": 4
  },
  "high_confidence": 1,
  "avg_edge": 2.34,
  "avg_line_spread": 1.17,
  "by_tier": {
    "V3_STANDARD_UNDER": 1,
    "STAR_V3": 1,
    "star_tier": 4
  },
  "star_tier_picks": 4,
  "star_player_picks_total": 5,
  "star_players": ["Stephen Curry", "Cade Cunningham", ...]
}
```

---

## Expected Performance Object

Backtest-based expectations (not guarantees):

```json
{
  "POINTS": {
    "win_rate": 56.7,
    "roi": 8.27
  },
  "REBOUNDS": {
    "win_rate": 61.2,
    "roi": 16.96
  },
  "overall_line_shopping": {
    "win_rate": 54.5,
    "roi": 4.16
  },
  "high_spread_goldmine": {
    "win_rate": 70.6,
    "roi": 34.82
  }
}
```

**Caveat**: These are historical backtest results. Live performance may differ due to market efficiency, sample variance, or model degradation.

---

## How to Read a Pick

### Example Pick

```json
{
  "player_name": "Luka Doncic",
  "stat_type": "POINTS",
  "side": "UNDER",
  "prediction": 27.19,
  "p_over": 0.192,
  "best_book": "betrivers",
  "best_line": 35.5,
  "edge": 8.31,
  "reasoning": "Model predicts 27.2 vs hardest line 35.5 (UNDER). Strong edge (8.3 pts)."
}
```

### Interpretation

1. **Player**: Luka Doncic
2. **Market**: POINTS
3. **Model says**: He'll score ~27.2 points
4. **Books say**: BetRivers has him at 35.5
5. **Gap**: 8.3 points between model and line
6. **Direction**: UNDER (model thinks line is too high)
7. **Probability**: 80.8% chance he goes under 35.5 (p_under = 1 - p_over)
8. **Action**: Bet UNDER 35.5 at BetRivers

### Why UNDER?

When `prediction < line`, the model recommends UNDER:
- Model prediction: 27.2
- Line: 35.5
- 27.2 < 35.5 → UNDER

The `side` field handles this logic:
- If `prediction > line` → `side = "OVER"`
- If `prediction < line` → `side = "UNDER"`

---

## Confidence Levels

| Level | Criteria | Historical WR |
|-------|----------|---------------|
| HIGH | edge > 5.0 AND p_over > 0.75 | ~85% |
| MEDIUM | edge > 3.0 OR p_over > 0.70 | ~70% |
| STANDARD | Meets minimum filter threshold | ~55-65% |

---

## Filter Tiers

Different tiers have different filtering logic:

| Tier | Description | Typical WR |
|------|-------------|------------|
| `STAR_V3` | Star player + high edge + v3 model | Highest |
| `V3_STANDARD_UNDER` | Standard filter, UNDER side | Medium |
| `star_tier` | Star player threshold met | Medium-High |
| `TIER_X` | Experimental tier | Variable |

---

## Sportsbook Codes

| Code | Sportsbook |
|------|------------|
| `draftkings` | DraftKings |
| `fanduel` | FanDuel |
| `betmgm` | BetMGM |
| `caesars` | Caesars |
| `betrivers` | BetRivers |
| `espnbet` | ESPN Bet |
| `underdog` | Underdog Fantasy (DFS) |

---

## Team Codes

Standard 3-letter NBA codes:
- ATL, BOS, BKN, CHA, CHI, CLE, DAL, DEN, DET, GSW
- HOU, IND, LAC, LAL, MEM, MIA, MIL, MIN, NOP, NYK
- OKC, ORL, PHI, PHX, POR, SAC, SAS, TOR, UTA, WAS

---

## Validation: Checking a Pick's Result

After the game:

1. Look up actual stat in box score
2. Compare to `best_line`
3. Determine win/loss:
   - If `side = "OVER"` and `actual > best_line` → WIN
   - If `side = "UNDER"` and `actual < best_line` → WIN
   - Pushes (actual = line) typically refunded

### Example

```
Pick: Curry OVER 27.5
Actual: 32 points
Result: WIN (32 > 27.5)
```

---

## Common Questions

**Q: Why is p_under sometimes null?**
A: Some model versions only compute p_over. Calculate as `p_under = 1 - p_over`.

**Q: Why would edge be negative in line_distribution?**
A: Negative edge means the model prediction is on the OPPOSITE side of that line. For OVER picks, a negative edge line is one where the line is higher than the prediction.

**Q: What's the difference between edge and edge_pct?**
A: `edge` is absolute (3.96 points), `edge_pct` is relative (14.4% of the line). Edge_pct is useful for comparing across markets (a 2-point edge on REBOUNDS is bigger than on POINTS).

**Q: How fresh are these predictions?**
A: Check `generated_at`. Lines move, so predictions are most valid close to generation time. Morning lines can shift significantly by game time.
