from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.evaluation.metrics import evaluate_macro_f05
from src.features.similarity_v3 import FEATURE_NAMES, extract_pair_features


class EntityResolutionModelV3:
    """
    Supervised model wrapper for V3 entity resolution with macro F0.5 threshold tuning.
    """

    def __init__(self, model_type: str = "hist_gb"):
        self.model_type = model_type
        if model_type == "hist_gb":
            self.model = HistGradientBoostingClassifier(
                random_state=42,
                max_iter=150,
                learning_rate=0.1,
            )
        elif model_type == "rf":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                random_state=42,
                n_jobs=-1,
            )
        else:
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
            )
        self.selected_threshold: float = 0.80
        self.feature_names = FEATURE_NAMES

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fit supervised classifier on candidate feature matrix."""
        self.model.fit(X_train, y_train)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of match (class 1)."""
        return self.model.predict_proba(X)[:, 1]

    def tune_threshold(
        self,
        val_candidate_df: pd.DataFrame,
        val_X: np.ndarray,
        val_ground_truth_map: dict[str, set[str]],
        val_s1_ids: list[str],
        thresholds: list[float] = None,
    ) -> dict[str, Any]:
        """
        Evaluate candidate match probabilities across candidate thresholds on validation set
        and select the threshold maximizing Macro F0.5.
        """
        if thresholds is None:
            thresholds = [
                0.50,
                0.55,
                0.60,
                0.65,
                0.70,
                0.75,
                0.80,
                0.85,
                0.90,
                0.95,
            ]

        probs = self.predict_proba(val_X)
        val_df = val_candidate_df.copy()
        val_df["prob"] = probs

        best_f05 = -1.0
        best_threshold = 0.80
        best_metrics = {}
        threshold_results = []

        for thresh in thresholds:
            # Filter predicted candidates above threshold
            retained = val_df[val_df["prob"] >= thresh]

            preds_map = (
                retained.groupby("source1_entity_id")["candidate_entity_id"]
                .apply(set)
                .to_dict()
            )

            p, r, f05 = evaluate_macro_f05(
                val_ground_truth_map, preds_map, val_s1_ids
            )

            res = {
                "threshold": thresh,
                "precision": p,
                "recall": r,
                "macro_f05": f05,
                "num_candidate_pairs": len(val_df),
                "num_predicted_matches": len(retained),
            }
            threshold_results.append(res)

            if f05 > best_f05:
                best_f05 = f05
                best_threshold = thresh
                best_metrics = res

        self.selected_threshold = best_threshold
        return {
            "best_threshold": best_threshold,
            "best_metrics": best_metrics,
            "threshold_results": threshold_results,
        }
