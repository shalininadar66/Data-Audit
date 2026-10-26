from typing import Any

import pandas as pd


# =========================================================
# MODEL ENGINE
# =========================================================

MODEL_TASKS = {

    # Classification
    "logistic regression": "classification",
    "decision tree": "classification",
    "random forest": "classification",
    "gradient boosting": "classification",
    "xgboost": "classification",
    "support vector machine (svm)": "classification",
    "k-nearest neighbors (knn)": "classification",
    "naive bayes": "classification",
    "adaboost": "classification",
    "extra trees": "classification",

    # Regression
    "linear regression": "regression",
    "ridge regression": "regression",
    "lasso regression": "regression",
    "elastic net": "regression",
    "decision tree regressor": "regression",
    "random forest regressor": "regression",
    "gradient boosting regressor": "regression",
    "xgboost regressor": "regression",
    "support vector regression (svr)": "regression",
    "k-nearest neighbors regressor": "regression",

    # Clustering
    "k-means": "clustering",
    "hierarchical clustering": "clustering",
    "dbscan": "clustering",
    "gaussian mixture model": "clustering",
    "mean shift": "clustering",
    "spectral clustering": "clustering",

    # Time series
    "arima": "time-series",
    "sarima": "time-series",
    "exponential smoothing": "time-series",
    "prophet": "time-series",
    "lstm": "time-series",

    # Anomaly
    "isolation forest": "anomaly",
    "local outlier factor": "anomaly",
    "one-class svm": "anomaly",
    "autoencoder": "anomaly",

    # Dimensionality reduction
    "pca": "dimensionality-reduction",
    "kernel pca": "dimensionality-reduction",
    "truncated svd": "dimensionality-reduction",
    "t-sne": "dimensionality-reduction",
    "umap": "dimensionality-reduction",

    # Recommendation
    "collaborative filtering": "recommendation",
    "content-based filtering": "recommendation",
    "matrix factorization": "recommendation",
    "hybrid recommendation": "recommendation",

    # Ranking
    "learning to rank": "ranking",
    "ranknet": "ranking",
    "lambdamart": "ranking",
    "xgboost ranking": "ranking",
}


UNSUPERVISED_TASKS = {
    "clustering",
    "anomaly",
    "dimensionality-reduction",
}

SUPERVISED_TASKS = {
    "classification",
    "regression",
    "ranking",
    "recommendation",
    "time-series",
}


def _normalize_model(model: str) -> str:
    return model.strip().lower()


def run_model_audit(
    df: pd.DataFrame,
    model: str = "",
    task: str = "",
    target: str = "",
) -> dict[str, Any]:

    model = model.strip()
    task = task.strip().lower()
    target = target.strip()

    if not model:

        return {
            "provided": False,
            "model": None,
            "inferred_task": None,
            "status": "Model not provided",
            "summary": (
                "No specific machine-learning model "
                "was selected."
            ),
            "target_required": None,
            "target_available": bool(
                target and target in df.columns
            ),
            "findings": [],
            "recommendations": [],
        }

    normalized_model = _normalize_model(model)

    inferred_task = MODEL_TASKS.get(
        normalized_model
    )

    # "No specific model" means no model-specific audit.
    if normalized_model == "no specific model":

        return {
            "provided": True,
            "model": model,
            "inferred_task": task or None,
            "status": "No specific model selected",
            "summary": (
                "A specific model was not selected, "
                "so model-specific suitability checks "
                "were not performed."
            ),
            "target_required": None,
            "target_available": bool(
                target and target in df.columns
            ),
            "findings": [],
            "recommendations": [],
        }

    findings = []
    recommendations = []

    numeric_columns = (
        df.select_dtypes(
            include="number"
        ).columns.tolist()
    )

    categorical_columns = (
        df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()
    )

    rows = len(df)

    target_exists = bool(
        target and target in df.columns
    )

    # =====================================================
    # TASK / MODEL CONSISTENCY
    # =====================================================

    if inferred_task and task:

        if inferred_task != task:

            findings.append({
                "type": "model_task_mismatch",
                "severity": "high",
                "message": (
                    f'"{model}" is associated with '
                    f'{inferred_task.replace("-", " ")} '
                    f'but the selected task is '
                    f'{task.replace("-", " ")}.'
                ),
            })

    # If task wasn't explicitly supplied,
    # infer it from the model.
    effective_task = task or inferred_task

    # =====================================================
    # TARGET REQUIREMENT
    # =====================================================

    target_required = (
        effective_task in SUPERVISED_TASKS
    )

    if target_required and not target_exists:

        findings.append({
            "type": "target_missing",
            "severity": "info",
            "message": (
                f'"{model}" can be assessed at the '
                "dataset/model level, but no target "
                "column was provided. Target-specific "
                "model checks cannot be completed."
            ),
        })

        recommendations.append(
            "Provide a target column if you want "
            "target compatibility and target-specific "
            "model checks."
        )

    # =====================================================
    # CLASSIFICATION MODELS
    # =====================================================

    if effective_task == "classification":

        if not numeric_columns:

            findings.append({
                "type": "no_numeric_features",
                "severity": "medium",
                "message": (
                    f'"{model}" will require appropriate '
                    "encoding or transformation because "
                    "no numeric feature columns were detected."
                ),
            })

        if categorical_columns:

            recommendations.append(
                f"{len(categorical_columns)} categorical "
                "feature column(s) may require encoding "
                "before training."
            )

        if normalized_model in {
            "k-nearest neighbors (knn)",
            "support vector machine (svm)",
            "logistic regression",
        }:

            recommendations.append(
                f'"{model}" may benefit from feature scaling '
                "when feature magnitudes differ substantially."
            )

    # =====================================================
    # REGRESSION MODELS
    # =====================================================

    elif effective_task == "regression":

        if target_exists:

            target_series = df[target]

            if not pd.api.types.is_numeric_dtype(
                target_series
            ):

                findings.append({
                    "type": "non_numeric_target",
                    "severity": "high",
                    "message": (
                        f'The selected regression target '
                        f'"{target}" is not numeric.'
                    ),
                })

        if categorical_columns:

            recommendations.append(
                "Categorical features will require "
                "appropriate encoding."
            )

        if normalized_model in {
            "linear regression",
            "ridge regression",
            "lasso regression",
            "elastic net",
        }:

            recommendations.append(
                f'"{model}" is sensitive to feature '
                "representation and scaling, so preprocessing "
                "should be reviewed before training."
            )

    # =====================================================
    # TREE MODELS
    # =====================================================

    if normalized_model in {
        "decision tree",
        "random forest",
        "gradient boosting",
        "xgboost",
        "adaboost",
        "extra trees",
        "decision tree regressor",
        "random forest regressor",
        "gradient boosting regressor",
        "xgboost regressor",
    }:

        if categorical_columns:

            recommendations.append(
                "Tree-based models still require categorical "
                "variables to be represented in a model-compatible "
                "form unless the implementation explicitly "
                "supports categorical features."
            )

    # =====================================================
    # K-MEANS
    # =====================================================

    if normalized_model == "k-means":

        if len(numeric_columns) < 2:

            findings.append({
                "type": "insufficient_numeric_features",
                "severity": "high",
                "message": (
                    "K-Means requires suitable numeric "
                    "features, but fewer than two numeric "
                    "columns were detected."
                ),
            })

        recommendations.append(
            "Feature scaling should be considered because "
            "K-Means is distance-based."
        )

        recommendations.append(
            "A target column is not required for K-Means."
        )

    # =====================================================
    # DBSCAN
    # =====================================================

    if normalized_model == "dbscan":

        if len(numeric_columns) < 2:

            findings.append({
                "type": "insufficient_numeric_features",
                "severity": "high",
                "message": (
                    "DBSCAN requires suitable feature "
                    "representations, but fewer than two "
                    "numeric columns were detected."
                ),
            })

        recommendations.append(
            "Feature scaling is important because DBSCAN "
            "uses distances when identifying neighborhoods."
        )

        recommendations.append(
            "A target column is not required for DBSCAN."
        )

    # =====================================================
    # DIMENSIONALITY REDUCTION
    # =====================================================

    if effective_task == "dimensionality-reduction":

        if len(numeric_columns) < 2:

            findings.append({
                "type": "insufficient_numeric_features",
                "severity": "high",
                "message": (
                    f'"{model}" generally requires '
                    "multiple usable feature dimensions."
                ),
            })

        recommendations.append(
            "Review scaling and missing-value handling "
            "before dimensionality reduction."
        )

        recommendations.append(
            "A target column is not required."
        )

    # =====================================================
    # ANOMALY DETECTION
    # =====================================================

    if effective_task == "anomaly":

        if len(numeric_columns) == 0:

            findings.append({
                "type": "no_numeric_features",
                "severity": "high",
                "message": (
                    f'"{model}" requires suitable numerical '
                    "or transformed feature representations."
                ),
            })

        recommendations.append(
            "A target column is not required for this "
            "unsupervised anomaly-detection workflow."
        )

    # =====================================================
    # TIME SERIES
    # =====================================================

    if effective_task == "time-series":

        date_like_columns = []

        for column in df.columns:

            if df[column].dtype == "object":

                sample = (
                    df[column]
                    .dropna()
                    .head(100)
                )

                if len(sample) > 0:

                    converted = pd.to_datetime(
                        sample,
                        errors="coerce",
                    )

                    if converted.notna().mean() >= 0.8:
                        date_like_columns.append(
                            column
                        )

        if not date_like_columns:

            findings.append({
                "type": "no_date_column",
                "severity": "high",
                "message": (
                    f'"{model}" requires a temporal '
                    "structure, but no date-like column "
                    "was detected."
                ),
            })

        else:

            recommendations.append(
                "Review the detected date-like columns "
                "to confirm that they represent the "
                "correct forecasting time index."
            )

    # =====================================================
    # DATASET SIZE
    # =====================================================

    if rows < 100:

        findings.append({
            "type": "small_dataset",
            "severity": "medium",
            "message": (
                f"The dataset contains only {rows:,} rows. "
                "Model stability and generalization should "
                "be evaluated carefully."
            ),
        })

    # =====================================================
    # FINAL STATUS
    # =====================================================

    high_count = sum(
        1
        for finding in findings
        if finding["severity"] == "high"
    )

    medium_count = sum(
        1
        for finding in findings
        if finding["severity"] == "medium"
    )

    if high_count > 0:
        status = "Model requires attention"

    elif medium_count > 0:
        status = "Model review recommended"

    else:
        status = "No major model-level issues detected"

    if inferred_task and not task:

        recommendations.insert(
            0,
            (
                f'The selected model "{model}" was '
                f'associated with the '
                f'{inferred_task.replace("-", " ")} '
                "task for this assessment."
            ),
        )

    return {
        "provided": True,
        "model": model,
        "inferred_task": inferred_task,
        "effective_task": effective_task,
        "status": status,

        "summary": (
            f'Model-specific checks were performed '
            f'for "{model}".'
        ),

        "target_required": target_required,
        "target_available": target_exists,

        "findings": findings,

        "recommendations": recommendations,
    }