"""
Logic for running the automated feature selection and model optimization experiments.
"""
from typing import Dict, List, Optional, Tuple, Callable
import time
import pandas as pd
import numpy as np
import optuna
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# Reuse model training logic
from src.model_training import (
    build_model,
    far_and_sensitivity,
    get_model_scores,
    select_threshold_for_far,
)
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

class ExperimentsRunner:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def calculate_feature_importance(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "target",
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Calculates feature importance using a Random Forest.
        Returns a DataFrame with 'variable' and 'importance' columns, sorted by importance.
        """
        X = df[feature_cols].fillna(0)
        y = df[target_col].astype(int)
        
        if y.nunique() < 2:
            raise ValueError("Target must have at least 2 classes for feature importance.")
            
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            class_weight="balanced",
            n_jobs=-1
        )
        model.fit(X, y)
        
        importance_df = pd.DataFrame({
            "variable": feature_cols,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        
        return importance_df

    def run_optimization_loop(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: List[str],
        model_choice: str,
        n_trials: int,
        timeout: int,
        far_target: float,
        search_space_config: Dict[str, Dict[str, float]],
        objective_key: str = "far_sens",
        objective_direction: str = "minimize",
        threshold_strategy: str = "optuna",
        progress_callback: Optional[Callable[[str], None]] = None,
        optuna_callbacks: Optional[List[Callable]] = None,
        return_model: bool = False,
    ) -> Dict[str, object]:
        """
        Runs Optuna optimization for a specific set of features and model.
        """
        
        X_train = train_df[feature_cols].fillna(0).astype("float32")
        y_train = train_df["target"].astype(int)
        X_val = val_df[feature_cols].fillna(0).astype("float32")
        y_val = val_df["target"].astype(int)
        X_test = test_df[feature_cols].fillna(0).astype("float32")
        y_test = test_df["target"].astype(int)
        
        # Extract search space bounds
        smote_cfg = search_space_config.get("smote", {})
        model_cfg = search_space_config.get("model", {})
        
        objective_direction = str(objective_direction).lower()
        if objective_direction not in {"minimize", "maximize"}:
            objective_direction = "minimize"
        threshold_strategy = str(threshold_strategy).lower()
        if threshold_strategy in {"far", "calibrate", "calibrar"}:
            threshold_strategy = "far"
        elif threshold_strategy not in {"optuna", "optimize", "optimizar"}:
            threshold_strategy = "optuna"

        def _objective_value(
            y_true: np.ndarray,
            preds: np.ndarray,
            scores: np.ndarray,
        ) -> float:
            key = str(objective_key).lower()
            if key in {"best_f1", "f1"}:
                return float(f1_score(y_true, preds, zero_division=0))
            if key == "roc_auc":
                try:
                    return float(roc_auc_score(y_true, scores))
                except ValueError:
                    return 0.5
            if key == "accuracy":
                return float(accuracy_score(y_true, preds))
            if key == "recall":
                return float(recall_score(y_true, preds, zero_division=0))
            if key == "precision":
                return float(precision_score(y_true, preds, zero_division=0))
            if key == "fnr":
                tn, fp, fn, tp = confusion_matrix(
                    y_true, preds, labels=[0, 1]
                ).ravel()
                return float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
            if key in {"far_sens", "far_sensitivity", "far_minus_sens"}:
                far_val, sens_val = far_and_sensitivity(y_true, preds)
                return float(far_val) - (float(sens_val) * 1e-3)
            far_val, sens_val = far_and_sensitivity(y_true, preds)
            return float(far_val) - (float(sens_val) * 1e-3)

        def objective(trial: optuna.Trial) -> float:
            # SMOTE Params
            k_neighbors = trial.suggest_int(
                "smote_k_neighbors", 
                int(smote_cfg.get("k_neighbors", {}).get("min", 1)),
                int(smote_cfg.get("k_neighbors", {}).get("max", 5))
            )
            sampling_strategy = trial.suggest_float(
                "smote_sampling_strategy",
                float(smote_cfg.get("sampling_strategy", {}).get("min", 0.1)),
                float(smote_cfg.get("sampling_strategy", {}).get("max", 1.0))
            )
            
            try:
                smote = SMOTE(
                    k_neighbors=k_neighbors,
                    sampling_strategy=sampling_strategy,
                    random_state=self.random_state
                )
                X_res, y_res = smote.fit_resample(X_train, y_train)
            except ValueError as exc:
                raise optuna.TrialPruned(f"SMOTE failed: {exc}")

            # Model Params
            params = {}
            if model_choice == "Random Forest":
                params["n_estimators"] = trial.suggest_int(
                    "n_estimators",
                    int(model_cfg.get("n_estimators", {}).get("min", 50)),
                    int(model_cfg.get("n_estimators", {}).get("max", 200))
                )
                d_min = int(model_cfg.get("max_depth", {}).get("min", 0))
                d_max = int(model_cfg.get("max_depth", {}).get("max", 20))
                if d_max > 0:
                    depth = trial.suggest_int("max_depth", d_min, d_max)
                    params["max_depth"] = depth if depth > 0 else None
                else:
                    params["max_depth"] = None
            
            elif model_choice == "XGBoost":
                params["n_estimators"] = trial.suggest_int(
                    "n_estimators",
                    int(model_cfg.get("n_estimators", {}).get("min", 50)),
                    int(model_cfg.get("n_estimators", {}).get("max", 200))
                )
                params["max_depth"] = trial.suggest_int(
                    "max_depth",
                    int(model_cfg.get("max_depth", {}).get("min", 2)),
                    int(model_cfg.get("max_depth", {}).get("max", 10))
                )
                params["learning_rate"] = trial.suggest_float(
                    "learning_rate",
                    float(model_cfg.get("learning_rate", {}).get("min", 0.01)),
                    float(model_cfg.get("learning_rate", {}).get("max", 0.3))
                )
                params["subsample"] = trial.suggest_float(
                    "subsample",
                    float(model_cfg.get("subsample", {}).get("min", 0.5)),
                    float(model_cfg.get("subsample", {}).get("max", 1.0))
                )
                params["colsample_bytree"] = trial.suggest_float(
                    "colsample_bytree",
                    float(model_cfg.get("colsample_bytree", {}).get("min", 0.5)),
                    float(model_cfg.get("colsample_bytree", {}).get("max", 1.0))
                )
            elif model_choice == "SVM":
                c_min = float(model_cfg.get("C", {}).get("min", 0.1))
                c_max = float(model_cfg.get("C", {}).get("max", 10.0))
                params["C"] = trial.suggest_float("C", c_min, c_max)
                params["kernel"] = str(model_cfg.get("kernel", "rbf"))
            
            # Train and Eval
            try:
                model = build_model(model_choice, params, self.random_state)
                model.fit(X_res, y_res)
                
                scores_val = get_model_scores(model, X_val)
                if threshold_strategy == "far":
                    thr_info = select_threshold_for_far(
                        y_val, scores_val, far_target=far_target
                    )
                    threshold = float(thr_info["threshold"])
                else:
                    threshold = trial.suggest_float("threshold", 0.0, 1.0)
                preds_val = (scores_val >= threshold).astype(int)
                score = _objective_value(y_val, preds_val, scores_val)
            except Exception as exc:
                raise optuna.TrialPruned(f"Training failed: {exc}")
                
            return float(score)

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(
            direction=objective_direction, sampler=sampler
        )
        
        if progress_callback:
            progress_callback(f"Starting Optuna for {len(feature_cols)} features...")
            
        study.optimize(
            objective, 
            n_trials=n_trials, 
            timeout=timeout, 
            callbacks=optuna_callbacks
        )
        
        best_trial = study.best_trial
        best_params = dict(best_trial.params)
        if model_choice == "SVM":
            best_params["kernel"] = str(model_cfg.get("kernel", "rbf"))
        
        # --- Retrain Best Model on Full Training Set for Final Metrics ---
        # 1. Re-apply best SMOTE
        smote_best = SMOTE(
            k_neighbors=best_params.get("smote_k_neighbors", 5),
            sampling_strategy=best_params.get("smote_sampling_strategy", 1.0),
            random_state=self.random_state
        )
        try:
            X_train_res, y_train_res = smote_best.fit_resample(X_train, y_train)
        except Exception:
             # Fallback if SMOTE fails (rare)
             X_train_res, y_train_res = X_train, y_train

        # 2. Extract model params (filter out smote_)
        model_params_final = {
            k: v
            for k, v in best_params.items()
            if not k.startswith("smote_") and k != "threshold"
        }
        # Add fixed params if needed or rely on build_model defaults + overrides
        
        # 3. Train
        final_model = build_model(model_choice, model_params_final, self.random_state)
        final_model.fit(X_train_res, y_train_res)
        
        # 4. Threshold on Val
        scores_val = get_model_scores(final_model, X_val)
        final_threshold = float(best_params.get("threshold", 0.5))
        if "threshold" not in best_params:
            thr_info = select_threshold_for_far(
                y_val, scores_val, far_target=far_target
            )
            final_threshold = float(thr_info["threshold"])
        
        # 5. Predict on Test
        scores_test = get_model_scores(final_model, X_test)
        preds_test = (scores_test >= final_threshold).astype(int)
        
        # 6. Metrics
        final_f1 = f1_score(y_test, preds_test, zero_division=0)
        final_acc = accuracy_score(y_test, preds_test)
        final_rec = recall_score(y_test, preds_test, zero_division=0)
        final_prec = precision_score(y_test, preds_test, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(
            y_test, preds_test, labels=[0, 1]
        ).ravel()
        
        # New Metrics
        try:
            final_roc_auc = roc_auc_score(y_test, scores_test)
        except ValueError:
            final_roc_auc = 0.5 # Fallback if only one class present
            
        final_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        final_far, final_sens = far_and_sensitivity(y_test, preds_test)
        
        result = {
            "n_features": len(feature_cols),
            "best_f1": final_f1, # Use the re-verified F1
            "best_params": best_params,
            "feature_cols": feature_cols,
            "accuracy": final_acc,
            "recall": final_rec,
            "precision": final_prec,
            "roc_auc": final_roc_auc,
            "fnr": final_fnr,
            "far": final_far,
            "sensitivity": final_sens,
            "confusion_matrix": [int(tn), int(fp), int(fn), int(tp)], # JSON serializable
            "threshold": float(final_threshold),
            "threshold_strategy": threshold_strategy,
            "dataset_rows": {
                "train": len(train_df),
                "val": len(val_df),
                "test": len(test_df)
            },
        }
        if return_model:
            result["model"] = final_model
        return result

    def run_iterative_experiment(
        self,
        base_df: pd.DataFrame,
        base_features_ordered: List[str],
        cluster_features: List[str], # These might be added in block or individually? Requirements say "features selected in order of importance", assuming global ranking.
        model_choice: str,
        n_trials: int,
        timeout: int,
        far_target: float,
        search_space_config: Dict[str, Dict[str, float]],
        step_size: int = 5,
        test_size: float = 0.2,
        val_size: float = 0.2,
        objective_key: str = "far_sens",
        objective_direction: str = "minimize",
        objective_label: Optional[str] = None,
        cluster_feature_names: Optional[List[str]] = None,
        threshold_strategy: str = "optuna",

        progress_bar = None, # Streamlit progress object
        dataset_name: str = "unknown",
        features_name: str = "unknown",
        max_k_limit: int = 1000,
        result_callback: Optional[Callable[[Dict[str, object]], None]] = None,
    ) -> List[Dict[str, object]]:
        """
        Runs the iterative experiment:
        1. Base only (Flow) - Incremental K
        2. Base + Cluster - Incremental K
        
        Returns list of results dicts.
        """
        results = []
        
        from src.cluster_accident_app import _temporal_train_test_split, _split_train_val_for_threshold
        
        # Split Data (Temporal)
        train_df, test_df = _temporal_train_test_split(base_df, test_size=test_size)
        train_full_df = train_df.copy()
        
        # Train/Val Split for Optimization
        train_opt_df, val_opt_df = _temporal_train_test_split(train_full_df, test_size=val_size / (1 - test_size))
        
        combined_features_ordered = cluster_features
        base_limit = min(len(base_features_ordered), max_k_limit)
        combined_limit = min(len(combined_features_ordered), max_k_limit)
        use_combined = combined_limit > 0

        # Calculate range based on total K (Base + Cluster when available).
        limit_total = combined_limit if use_combined else base_limit
        k_values = list(range(step_size, limit_total + 1, step_size))
        if not k_values or k_values[-1] < limit_total:
            if limit_total > 0:
                k_values.append(limit_total)

        cluster_set = set(cluster_feature_names or [])
        
        step_counter = 0
        total_steps = len(k_values)
        
        if progress_bar:
             progress_bar.progress(0, text="Starting optimization loop...")

        for k in k_values:
            step_counter += 1
            if progress_bar:
                progress_bar.progress(int(step_counter / total_steps * 100), text=f"Optimizing K={k}...")

            # --- 1. Base Strategy ---
            base_k = k
            if use_combined and combined_features_ordered:
                top_combined = combined_features_ordered[:k]
                cluster_in_top = sum(
                    1 for col in top_combined if col in cluster_set
                )
                base_k = k - cluster_in_top
            if base_k > 0:
                features_k = base_features_ordered[:base_k]

                try:
                    res = self.run_optimization_loop(
                        train_df=train_opt_df,
                        val_df=val_opt_df,
                        test_df=test_df,
                        feature_cols=features_k,
                        model_choice=model_choice,
                        n_trials=n_trials,
                        timeout=timeout,
                        far_target=far_target,
                        search_space_config=search_space_config,
                        objective_key=objective_key,
                        objective_direction=objective_direction,
                        threshold_strategy=threshold_strategy,
                    )
                    res["type"] = "Base"
                    res["k"] = k
                    res["dataset_name"] = dataset_name
                    res["features_name"] = features_name
                    res["objective_metric"] = objective_key
                    res["objective_direction"] = objective_direction
                    res["threshold_strategy"] = threshold_strategy
                    if objective_label:
                        res["objective_label"] = objective_label
                    results.append(res)
                    if result_callback:
                        result_callback(dict(res))
                except Exception as e:
                    print(f"Error in Base K={k}: {e}")
            else:
                res = {
                    "type": "Base",
                    "k": k,
                    "dataset_name": dataset_name,
                    "features_name": features_name,
                    "objective_metric": objective_key,
                    "objective_direction": objective_direction,
                    "error": "K total sin variables base disponibles.",
                }
                if objective_label:
                    res["objective_label"] = objective_label
                results.append(res)
                if result_callback:
                    result_callback(dict(res))
            
            # --- 2. Base + Cluster Strategy ---
            if use_combined:
                features_k_comb = combined_features_ordered[:k]

                try:
                    res_c = self.run_optimization_loop(
                        train_df=train_opt_df,
                        val_df=val_opt_df,
                        test_df=test_df,
                        feature_cols=features_k_comb,
                        model_choice=model_choice,
                        n_trials=n_trials,
                        timeout=timeout,
                        far_target=far_target,
                        search_space_config=search_space_config,
                        objective_key=objective_key,
                        objective_direction=objective_direction,
                        threshold_strategy=threshold_strategy,
                    )
                    res_c["type"] = "Base+Cluster"
                    res_c["k"] = k
                    res_c["dataset_name"] = dataset_name
                    res_c["features_name"] = features_name
                    res_c["objective_metric"] = objective_key
                    res_c["objective_direction"] = objective_direction
                    res_c["threshold_strategy"] = threshold_strategy
                    if objective_label:
                        res_c["objective_label"] = objective_label
                    results.append(res_c)
                    if result_callback:
                        result_callback(dict(res_c))
                except Exception as e:
                    print(f"Error in Combined K={k}: {e}")
                    
        return results
            
