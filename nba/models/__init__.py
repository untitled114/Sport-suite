"""
Model Training and Inference
============================
Stacked two-head LightGBM models for NBA player prop predictions.

Architecture:
    - HEAD 1 (Regressor): Predicts absolute stat value
    - HEAD 2 (Classifier): Predicts P(actual > line)
    - Isotonic calibration for probability reliability
    - Ensemble blending (60% classifier, 40% residual-based)
"""
