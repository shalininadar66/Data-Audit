from typing import Any

import pandas as pd


# =========================================================
# TASK ENGINE
# =========================================================
#
# This engine evaluates whether the dataset is structurally
# suitable for the selected ML task.
#
# IMPORTANT:
#
# A target is NOT required for every task.
#
# Classification / regression:
#     Can assess dataset readiness without target,
#     but target-specific checks remain unavailable.
#
# Clustering / anomaly detection /
# dimensionality reduction:
#     Can be assessed without a target.
#
# Time series:
#     Looks for date-like columns and temporal structure.
#
# =========================================================


def run_task_audit(
    df: pd.DataFrame,
    task: str = "",
    target: str = "",
) -> dict[str, Any]:

    task = task.strip().lower()
    target = target.strip()

    if not task:

        return {
            "provided": False,
            "task": None,
            "status": "Task not provided",
            "summary": (
                "No machine-learning task was provided. "
                "General dataset quality and structural "
                "analysis were performed."
            ),
            "target_required": None,
            "target_available": bool(
                target and target in df.columns
            ),
            "checks": [],
            "findings": [],
            "recommendations": [],
        }

    findings = []
    recommendations = []
    checks = []

    rows = len(df)

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

    target_exists = bool(
        target and target in df.columns
    )

    # =====================================================
    # CLASSIFICATION
    # =====================================================

    if task == "classification":

        checks.append(
            "Classification dataset structure"
        )

        if not target_exists:

            findings.append({
                "type": "target_missing",
                "severity": "info",
                "message": (
                    "No target column was provided, "
                    "so target-specific classification "
                    "checks could not be performed."
                ),
            })

            recommendations.append(
                "Provide a target column if you want "
                "class distribution, target missing-value, "
                "and target-specific classification checks."
            )

        else:

            target_series = df[target]

            unique_target = int(
                target_series.nunique(
                    dropna=True
                )
            )

            target_missing = int(
                target_series.isna().sum()
            )

            if unique_target < 2:

                findings.append({
                    "type": "invalid_target",
                    "severity": "high",
                    "message": (
                        "The selected classification "
                        "target contains fewer than "
                        "two unique non-missing classes."
                    ),
                })

            elif unique_target > 50:

                findings.append({
                    "type": "high_target_cardinality",
                    "severity": "medium",
                    "message": (
                        f'The classification target "{target}" '
                        f'contains {unique_target:,} unique values, '
                        f'which is unusually high for a standard '
                        f'classification problem.'
                    ),
                })

            if target_missing > 0:

                findings.append({
                    "type": "target_missing_values",
                    "severity": "medium",
                    "message": (
                        f'The target "{target}" contains '
                        f'{target_missing:,} missing values.'
                    ),
                })

        if not numeric_columns:

            findings.append({
                "type": "no_numeric_features",
                "severity": "medium",
                "message": (
                    "No numeric feature columns were detected. "
                    "Categorical variables will require encoding "
                    "before most classification models can be trained."
                ),
            })

        if categorical_columns:

            recommendations.append(
                f"{len(categorical_columns)} categorical "
                "feature column(s) may require encoding."
            )

    # =====================================================
    # REGRESSION
    # =====================================================

    elif task == "regression":

        checks.append(
            "Regression dataset structure"
        )

        if not target_exists:

            findings.append({
                "type": "target_missing",
                "severity": "info",
                "message": (
                    "No target column was provided, "
                    "so target-specific regression checks "
                    "could not be performed."
                ),
            })

            recommendations.append(
                "Provide a target column if you want "
                "target type, distribution, missing-value, "
                "and regression-specific checks."
            )

        else:

            target_series = df[target]

            if not pd.api.types.is_numeric_dtype(
                target_series
            ):

                findings.append({
                    "type": "non_numeric_target",
                    "severity": "high",
                    "message": (
                        f'The selected regression target '
                        f'"{target}" is not stored as a '
                        f'numeric column.'
                    ),
                })

            target_missing = int(
                target_series.isna().sum()
            )

            if target_missing > 0:

                findings.append({
                    "type": "target_missing_values",
                    "severity": "medium",
                    "message": (
                        f'The target "{target}" contains '
                        f'{target_missing:,} missing values.'
                    ),
                })

        if categorical_columns:

            recommendations.append(
                f"{len(categorical_columns)} categorical "
                "feature column(s) may require encoding."
            )

    # =====================================================
    # CLUSTERING
    # =====================================================

    elif task == "clustering":

        checks.append(
            "Unsupervised clustering readiness"
        )

        if len(numeric_columns) < 2:

            findings.append({
                "type": "insufficient_numeric_features",
                "severity": "high",
                "message": (
                    "Fewer than two numeric feature columns "
                    "were detected. Most clustering workflows "
                    "require multiple numeric features."
                ),
            })

        else:

            findings.append({
                "type": "numeric_features_available",
                "severity": "info",
                "message": (
                    f"{len(numeric_columns)} numeric feature "
                    "columns are available for clustering."
                ),
            })

        if categorical_columns:

            recommendations.append(
                "Categorical features will need suitable "
                "encoding or transformation before many "
                "clustering algorithms can use them."
            )

        if target:

            recommendations.append(
                "The supplied target will not be required "
                "for the clustering task."
            )

    # =====================================================
    # TIME SERIES
    # =====================================================

    elif task == "time-series":

        checks.append(
            "Time-series structure"
        )

        if not date_like_columns:

            findings.append({
                "type": "no_date_column",
                "severity": "high",
                "message": (
                    "No date-like column was detected. "
                    "A temporal index or timestamp is normally "
                    "required for time-series forecasting."
                ),
            })

        else:

            findings.append({
                "type": "date_column_available",
                "severity": "info",
                "message": (
                    f"Detected {len(date_like_columns)} "
                    f"date-like column(s): "
                    f"{', '.join(date_like_columns[:5])}."
                ),
            })

        if not target:

            recommendations.append(
                "Provide the forecasting target if you "
                "want target-specific forecasting checks."
            )

    # =====================================================
    # ANOMALY DETECTION
    # =====================================================

    elif task == "anomaly":

        checks.append(
            "Anomaly-detection readiness"
        )

        if len(numeric_columns) == 0:

            findings.append({
                "type": "no_numeric_features",
                "severity": "high",
                "message": (
                    "No numeric features were detected "
                    "for anomaly analysis."
                ),
            })

        else:

            findings.append({
                "type": "numeric_features_available",
                "severity": "info",
                "message": (
                    f"{len(numeric_columns)} numeric feature "
                    "column(s) are available for anomaly analysis."
                ),
            })

        if target:

            recommendations.append(
                "A target is not required for unsupervised "
                "anomaly detection."
            )

    # =====================================================
    # DIMENSIONALITY REDUCTION
    # =====================================================

    elif task == "dimensionality-reduction":

        checks.append(
            "Dimensionality-reduction readiness"
        )

        if len(numeric_columns) < 2:

            findings.append({
                "type": "insufficient_features",
                "severity": "high",
                "message": (
                    "Fewer than two numeric features "
                    "were detected."
                ),
            })

        else:

            findings.append({
                "type": "numeric_features_available",
                "severity": "info",
                "message": (
                    f"{len(numeric_columns)} numeric feature "
                    f"columns are available."
                ),
            })

        if target:

            recommendations.append(
                "A target is not required for most "
                "dimensionality-reduction workflows."
            )

    # =====================================================
    # RECOMMENDATION
    # =====================================================

    elif task == "recommendation":

        checks.append(
            "Recommendation-system readiness"
        )

        if rows < 50:

            findings.append({
                "type": "small_dataset",
                "severity": "medium",
                "message": (
                    f"The dataset contains only {rows:,} rows. "
                    "Recommendation systems often benefit "
                    "from substantially larger interaction datasets."
                ),
            })

        if len(categorical_columns) == 0:

            recommendations.append(
                "Review whether the dataset contains "
                "user, item, interaction, preference, "
                "or transaction information required "
                "for the intended recommendation approach."
            )

        if not target:

            recommendations.append(
                "A conventional prediction target is not "
                "required for many recommendation approaches."
            )

    # =====================================================
    # RANKING
    # =====================================================

    elif task == "ranking":

        checks.append(
            "Ranking-task readiness"
        )

        if not target:

            findings.append({
                "type": "target_missing",
                "severity": "info",
                "message": (
                    "No target or relevance column was "
                    "provided, so ranking-specific target "
                    "checks could not be performed."
                ),
            })

            recommendations.append(
                "Provide a target or relevance field when "
                "available so ranking suitability can be "
                "assessed more specifically."
            )

    # =====================================================
    # UNKNOWN TASK
    # =====================================================

    else:

        findings.append({
            "type": "unsupported_task",
            "severity": "high",
            "message": (
                f'The task "{task}" is not supported '
                "by the current audit engine."
            ),
        })

    # =====================================================
    # STATUS
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
        status = "Needs attention"

    elif medium_count > 0:
        status = "Review recommended"

    else:
        status = "No major task-level issues detected"

    return {
        "provided": True,
        "task": task,
        "status": status,
        "summary": (
            f"Task-specific checks were performed "
            f"for {task.replace('-', ' ')}."
        ),
        "target_required": task in {
            "classification",
            "regression",
        },
        "target_available": target_exists,
        "checks": checks,
        "findings": findings,
        "recommendations": recommendations,
    }