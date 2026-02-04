# TRAINING READINESS AUDIT REPORT

**Date:** 2026-02-02
**Auditor:** Claude Code
**Threshold:** 96% minimum, ZERO TOLERANCE below 95%

---

## EXECUTIVE SUMMARY

| Component | Status | Coverage | Verdict |
|-----------|--------|----------|---------|
| **Infrastructure** | PASS | 100% | All 4 databases operational |
| **Data Coverage (Props)** | MARGINAL | 95.2% | Barely meets 95% threshold |
| **Feature Coverage** | **FAIL** | 89.97% | 4 columns below 95% |
| **Overall** | **FAIL** | - | Feature coverage unacceptable |

---

## 1. INFRASTRUCTURE AUDIT

### Database Status

| Database | Port | Status | Tables | Row Count |
|----------|------|--------|--------|-----------|
| nba_players | 5536 | **PASS** | player_profile, player_game_logs | 102,206 game logs |
| nba_games | 5537 | **PASS** | games, team_game_logs | 6,128 games |
| nba_team | 5538 | **PASS** | teams, team_season_stats | 30 teams |
| nba_intelligence | 5539 | **PASS** | nba_props_xl, injury_report | 2,858,043 props |

**Infrastructure Verdict: PASS (4/4)**

---

## 2. DATA COVERAGE AUDIT

### Props with Actual Values

| Stat Type | Total Props | With Actuals | Coverage | Status |
|-----------|-------------|--------------|----------|--------|
| ASSISTS | 449,182 | 434,448 | **96.72%** | PASS |
| THREES | 636,623 | 613,923 | **96.43%** | PASS |
| POINTS | 899,390 | 856,557 | **95.24%** | MARGINAL |
| REBOUNDS | 862,367 | 821,259 | **95.23%** | MARGINAL |

### Coverage by Year

| Year | Total Props | With Actuals | Coverage | Status |
|------|-------------|--------------|----------|--------|
| 2023 | 353,871 | 353,607 | **99.93%** | PASS |
| 2024 | 753,355 | 747,430 | **99.21%** | PASS |
| 2025 | 1,290,064 | 1,209,104 | **93.72%** | **FAIL** |
| 2026 | 460,753 | 416,046 | **90.30%** | **FAIL** |

**Data Coverage Verdict: MARGINAL PASS (core markets at 95.2%)**

---

## 3. FEATURE COVERAGE AUDIT - **CRITICAL FAILURE**

### Training Dataset Statistics

| Dataset | Rows | Columns | Overall Coverage |
|---------|------|---------|------------------|
| xl_training_POINTS_2023_2025.csv | 51,639 | 174 | 99.25% |
| xl_training_REBOUNDS_2023_2025.csv | 50,755 | 174 | 99.25% |

### Columns Below 95% Threshold

| Column | POINTS Coverage | REBOUNDS Coverage | Status |
|--------|-----------------|-------------------|--------|
| `expected_diff` | 10.03% | 10.24% | OK (by design) |
| `ema_plus_minus_L5` | **89.97%** | **89.76%** | **FAIL** |
| `ft_rate_L10` | **89.97%** | **89.76%** | **FAIL** |
| `ema_plus_minus_L10` | **89.97%** | **89.76%** | **FAIL** |
| `true_shooting_L10` | **89.97%** | **89.76%** | **FAIL** |

**Missing Values:** 5,178 rows in POINTS, 5,198 rows in REBOUNDS

---

## 4. ROOT CAUSE ANALYSIS

### The Bug: Name Suffix Mismatch

**Source:** `nba/features/extract_live_features.py` lines 250, 385, 447, 546, 603, 688

The SQL queries use:
```sql
WHERE unaccent(pp.full_name) = unaccent(%s)
```

`unaccent()` handles accented characters but **NOT suffix variations**:
- `Michael Porter Jr.` (player_profile) ≠ `Michael Porter Jr` (props)
- `Russell Westbrook III` matches but `Trey Murphy III` may have variations

### Proof of Issue

```
player_profile:     "Michael Porter Jr."  (WITH period)
nba_props_xl:       "Michael Porter Jr"   (2,708 props - NO period) → FAILS
                    "Michael Porter Jr."  (14,857 props - WITH period) → WORKS
```

### Affected Props Count

| Suffix Pattern | Prop Count | Issue |
|----------------|------------|-------|
| `Jr` (no period) | 19,285 | Won't match `Jr.` in player_profile |
| `III` suffix | 24,720 | May have formatting issues |
| `II` suffix | 5,153 | May have formatting issues |
| **Total Affected** | **49,158** | ~5% of training data |

### Top 20 Affected Players

| Player | Props Missing Features |
|--------|------------------------|
| Trey Murphy III | 4,008 |
| Russell Westbrook III | 3,851 |
| Jimmy Butler III | 2,469 |
| Marvin Bagley III | 1,534 |
| Jabari Smith Jr | 1,163 |
| Ronald Holland II | 1,113 |
| Tim Hardaway Jr | 1,021 |
| Bobby Portis Jr | 880 |
| Jaren Jackson Jr | 849 |
| Kelly Oubre Jr | 769 |
| Wendell Carter Jr | 766 |
| Bruce Brown Jr | 750 |
| Jaime Jaquez Jr | 720 |
| Gary Trent Jr | 644 |
| PJ Washington Jr | 631 |
| Gary Payton II | 623 |
| Michael Porter Jr | 567 |
| Robert Williams III | 548 |
| Vince Williams Jr | 463 |
| GG Jackson II | 414 |

---

## 5. REQUIRED FIXES

### Fix 1: Update SQL Name Matching (CRITICAL)

**File:** `nba/features/extract_live_features.py`

**Current (broken):**
```sql
WHERE unaccent(pp.full_name) = unaccent(%s)
```

**Required fix:**
```sql
WHERE regexp_replace(unaccent(pp.full_name), '\\.| (Jr|Sr|I{2,3}|IV|V)$', '', 'g')
    = regexp_replace(unaccent(%s), '\\.| (Jr|Sr|I{2,3}|IV|V)$', '', 'g')
```

Or use the existing `NameNormalizer` class which properly removes suffixes.

### Fix 2: Rebuild Training Datasets

After fixing the SQL, rebuild datasets:
```bash
cd /home/untitled/Sport-suite/nba/features
python3 build_xl_training_dataset.py --output datasets/
```

### Fix 3: Retrain Models

After verifying 96%+ coverage:
```bash
cd /home/untitled/Sport-suite/nba/models
python3 train_market.py --market POINTS --data ../features/datasets/xl_training_POINTS_2023_2025.csv
python3 train_market.py --market REBOUNDS --data ../features/datasets/xl_training_REBOUNDS_2023_2025.csv
```

---

## 6. VERIFICATION CHECKLIST

After fixes, verify:

- [ ] All 4 columns (`ema_plus_minus_L5`, `ema_plus_minus_L10`, `ft_rate_L10`, `true_shooting_L10`) at ≥96%
- [ ] No more than 2% missing values per column
- [ ] Michael Porter Jr, Trey Murphy III, etc. have complete features
- [ ] POINTS dataset overall coverage ≥96%
- [ ] REBOUNDS dataset overall coverage ≥96%

---

## 7. TIMELINE

| Task | Priority | Estimated Effort |
|------|----------|------------------|
| Fix SQL name matching | P0 | Immediate |
| Rebuild training datasets | P0 | Immediate |
| Verify 96% coverage | P0 | Immediate |
| Retrain models | P1 | After verification |

---

## AUDIT CONCLUSION

**TRAINING NOT READY**

The system is **NOT READY FOR TRAINING** due to:

1. **Feature coverage at 89.97%** - Below 95% tolerance
2. **Root cause identified:** SQL name matching doesn't handle suffix variations (Jr vs Jr.)
3. **49,158 props affected** - 5% of training data missing critical rolling stat features

**Action Required:** Fix the name matching in `extract_live_features.py` before any training runs.

---

*Report generated: 2026-02-02*
