# POINTS Walk-Forward Validation

**3 folds** | Mean AUC: **0.730** (std 0.017) | Win Rate: **62.6%** | Edge WR: **68.7%** | ROI: **+25.3%**

| Fold | Train Period | Test Period | AUC | Win Rate | Edge WR | ROI |
|------|-------------|-------------|-----|----------|---------|-----|
| 1 | 2024-10-22 to 2025-04-13 | 2025-10-21 to 2025-10-21 | 0.753 | 60.0% | 70.6% | +22.2% |
| 2 | 2024-10-22 to 2025-10-21 | 2025-10-22 to 2025-12-21 | 0.712 | 65.1% | 67.7% | +25.4% |
| 3 | 2024-10-22 to 2025-12-21 | 2025-12-22 to 2026-02-21 | 0.724 | 62.6% | 67.8% | +25.3% |

![AUC](images/POINTS_walkforward_auc.png)
![Win Rate](images/POINTS_walkforward_winrate.png)
![ROI](images/POINTS_walkforward_roi.png)
