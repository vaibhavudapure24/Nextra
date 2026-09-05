"""
Predictive Health Analytics Model Trainer.
Trains a model to estimate animal vitality and health index from historical activity logs.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from utils.logger import get_logger

logger = get_logger(__name__)


def train_health_predictor(output_path: str = "models/health/health_predictor.joblib"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(42)

    # Features: [walking_dist_km, active_hours, rest_hours, speed_var, stride_asymmetry]
    num_samples = 800
    walk = np.random.uniform(1.0, 15.0, num_samples)
    active = np.random.uniform(4.0, 14.0, num_samples)
    rest = 24.0 - active
    speed_var = np.random.uniform(0.1, 1.5, num_samples)
    asymmetry = np.random.uniform(0.0, 0.4, num_samples)

    X = np.column_stack([walk, active, rest, speed_var, asymmetry])
    # Target: Vitality score 0.0 - 100.0
    y = (
        (active * 4.0)
        + (walk * 2.5)
        - (asymmetry * 100.0)
        - (speed_var * 10.0)
        + np.random.normal(0, 3, num_samples)
    )
    y = np.clip(y, 10.0, 100.0)

    logger.info("Fitting Gradient Boosting Regressor for predictive health scoring...")
    reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
    reg.fit(X, y)

    joblib.dump(reg, output_path)
    logger.info(f"Saved trained predictive health model to: {output_path}")
    return output_path


if __name__ == "__main__":
    train_health_predictor()
