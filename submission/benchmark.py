"""LightGBM benchmark for the Kaggle Credit Card Fraud dataset.

Run on the GCP Compute Engine CPU node after downloading creditcard.csv.
The script writes benchmark_result.json in the current directory.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42


def seconds_since(start: float) -> float:
    return round(time.perf_counter() - start, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and benchmark LightGBM on creditcard.csv")
    parser.add_argument("--data", type=Path, default=Path("creditcard.csv"))
    parser.add_argument("--output", type=Path, default=Path("benchmark_result.json"))
    args = parser.parse_args()

    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.data.resolve()}")

    load_start = time.perf_counter()
    data = pd.read_csv(args.data)
    load_data_seconds = seconds_since(load_start)

    if "Class" not in data.columns:
        raise ValueError("Expected a 'Class' target column in the dataset.")

    features = data.drop(columns=["Class"])
    target = data["Class"]

    x_train_full, x_test, y_train_full, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )

    model = LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=2,
        verbosity=-1,
    )

    training_start = time.perf_counter()
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    training_seconds = seconds_since(training_start)

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    one_row = x_test.iloc[[0]]
    batch = x_test.iloc[: min(1000, len(x_test))]
    model.predict_proba(one_row)  # Warm-up.
    model.predict_proba(batch)

    repeats = 1000
    latency_start = time.perf_counter()
    for _ in range(repeats):
        model.predict_proba(one_row)
    single_row_latency_ms = (time.perf_counter() - latency_start) * 1000 / repeats

    throughput_start = time.perf_counter()
    model.predict_proba(batch)
    batch_elapsed_seconds = time.perf_counter() - throughput_start
    throughput_rows_per_second = len(batch) / batch_elapsed_seconds

    results = {
        "dataset": str(args.data),
        "rows": int(len(data)),
        "features": int(features.shape[1]),
        "train_rows": int(len(x_train)),
        "validation_rows": int(len(x_valid)),
        "test_rows": int(len(x_test)),
        "load_data_seconds": load_data_seconds,
        "training_seconds": training_seconds,
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
        "auc_roc": round(float(roc_auc_score(y_test, probabilities)), 6),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 6),
        "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 6),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 6),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 6),
        "single_row_inference_latency_ms": round(single_row_latency_ms, 4),
        "batch_size": int(len(batch)),
        "batch_inference_seconds": round(batch_elapsed_seconds, 6),
        "inference_throughput_rows_per_second": round(throughput_rows_per_second, 2),
    }

    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nSaved results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
