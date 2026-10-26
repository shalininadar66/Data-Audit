import math
import re
from typing import Any

import pandas as pd


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def _normalize_value(value: Any) -> str:
    """Convert a value to a normalized string."""
    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def _normalize_column_name(column: str) -> str:
    """Normalize a column name for comparison."""
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(column).lower(),
    ).strip("_")


def _target_name_tokens(target: str) -> set[str]:
    """Return meaningful tokens from the target name."""
    normalized = _normalize_column_name(target)

    if not normalized:
        return set()

    return {
        token
        for token in normalized.split("_")
        if len(token) >= 3
    }


# =========================================================
# DIRECT TARGET COPY
# =========================================================

def _check_direct_target_copy(
    df: pd.DataFrame,
    target: str,
) -> list[dict[str, Any]]:

    findings = []

    target_series = df[target].map(
        _normalize_value
    )

    for column in df.columns:

        if column == target:
            continue

        feature_series = df[column].map(
            _normalize_value
        )

        comparable = (
            (target_series != "")
            & (feature_series != "")
        )

        comparable_count = int(
            comparable.sum()
        )

        if comparable_count == 0:
            continue

        match_rate = float(
            (
                target_series[comparable]
                == feature_series[comparable]
            ).mean()
        )

        if match_rate >= 0.99:

            findings.append({
                "type": "direct_target_copy",
                "column": column,
                "severity": "high",

                "metric": {
                    "matching_rate": round(
                        match_rate * 100,
                        2,
                    )
                },

                "message": (
                    f'The column "{column}" matches '
                    f'the target "{target}" for '
                    f'approximately '
                    f'{match_rate * 100:.2f}% of '
                    f'comparable rows.'
                ),

                "recommendation": (
                    f'Investigate whether "{column}" '
                    f'is a direct copy or transformation '
                    f'of "{target}". Confirm that the '
                    f'column would genuinely be available '
                    f'at prediction time.'
                ),
            })

    return findings


# =========================================================
# TARGET-RELATED COLUMN NAMES
# =========================================================

def _check_target_name_signals(
    df: pd.DataFrame,
    target: str,
) -> list[dict[str, Any]]:

    findings = []

    target_tokens = _target_name_tokens(target)

    if not target_tokens:
        return findings

    suspicious_words = {
        "target",
        "label",
        "prediction",
        "predicted",
        "outcome",
        "result",
        "score",
        "rate",
        "rating",
        "class",
    }

    for column in df.columns:

        if column == target:
            continue

        normalized_column = (
            _normalize_column_name(column)
        )

        column_tokens = set(
            normalized_column.split("_")
        )

        shared_tokens = (
            target_tokens
            .intersection(column_tokens)
        )

        suspicious_match = (
            bool(
                column_tokens.intersection(
                    suspicious_words
                )
            )
        )

        if shared_tokens and suspicious_match:

            findings.append({
                "type": "target_related_name",
                "column": column,
                "severity": "medium",

                "metric": {
                    "shared_tokens": sorted(
                        shared_tokens
                    )
                },

                "message": (
                    f'The column "{column}" has a '
                    f'name that is closely related '
                    f'to the target "{target}".'
                ),

                "recommendation": (
                    f'Review how "{column}" was created '
                    f'and confirm that it does not contain '
                    f'information derived from "{target}".'
                ),
            })

    return findings


# =========================================================
# CATEGORICAL TARGET ASSOCIATION
# =========================================================

def _check_categorical_association(
    df: pd.DataFrame,
    target: str,
) -> list[dict[str, Any]]:

    findings = []

    target_series = df[target]

    if not (
        pd.api.types.is_object_dtype(target_series)
        or pd.api.types.is_categorical_dtype(target_series)
        or pd.api.types.is_bool_dtype(target_series)
    ):
        return findings

    target_unique = int(
        target_series.nunique(
            dropna=True
        )
    )

    if target_unique < 2:
        return findings

    # Avoid treating extremely high-cardinality
    # columns as normal categorical targets.
    if target_unique > 100:
        return findings

    target_clean = (
        target_series
        .fillna("__MISSING_TARGET__")
        .astype(str)
    )

    # Target entropy
    target_probabilities = (
        target_clean.value_counts(
            normalize=True
        )
    )

    target_entropy = 0.0

    for probability in target_probabilities:

        if probability > 0:
            target_entropy -= (
                probability
                * math.log2(probability)
            )

    if target_entropy == 0:
        return findings

    for column in df.columns:

        if column == target:
            continue

        feature = df[column]

        if feature.notna().sum() == 0:
            continue

        feature_unique = int(
            feature.nunique(
                dropna=True
            )
        )

        if feature_unique <= 1:
            continue

        # Skip extremely high-cardinality features.
        if feature_unique > 1000:
            continue

        feature_clean = (
            feature
            .fillna("__MISSING_FEATURE__")
            .astype(str)
        )

        contingency = pd.crosstab(
            feature_clean,
            target_clean,
        )

        if contingency.empty:
            continue

        total = float(
            contingency.to_numpy().sum()
        )

        if total == 0:
            continue

        row_totals = contingency.sum(
            axis=1
        )

        conditional_entropy = 0.0

        for row_name in contingency.index:

            row = contingency.loc[row_name]

            row_total = float(
                row_totals[row_name]
            )

            if row_total == 0:
                continue

            probabilities = (
                row / row_total
            )

            row_entropy = 0.0

            for probability in probabilities:

                if probability > 0:
                    row_entropy -= (
                        probability
                        * math.log2(probability)
                    )

            conditional_entropy += (
                row_total / total
            ) * row_entropy

        uncertainty_reduction = (
            1
            - (
                conditional_entropy
                / target_entropy
            )
        )

        if uncertainty_reduction >= 0.95:

            findings.append({
                "type": "strong_target_association",
                "column": column,
                "severity": "high",

                "metric": {
                    "uncertainty_reduction": round(
                        uncertainty_reduction * 100,
                        2,
                    )
                },

                "message": (
                    f'The column "{column}" has a '
                    f'very strong association with '
                    f'the target "{target}".'
                ),

                "recommendation": (
                    f'Investigate whether "{column}" '
                    f'contains target-derived or '
                    f'future information that would '
                    f'not be available at prediction time.'
                ),
            })

    return findings


# =========================================================
# NUMERIC TARGET ASSOCIATION
# =========================================================

def _check_numeric_association(
    df: pd.DataFrame,
    target: str,
) -> list[dict[str, Any]]:

    findings = []

    target_series = df[target]

    if not pd.api.types.is_numeric_dtype(
        target_series
    ):
        return findings

    numeric_columns = (
        df.select_dtypes(
            include="number"
        ).columns
    )

    for column in numeric_columns:

        if column == target:
            continue

        pair = df[
            [column, target]
        ].dropna()

        if len(pair) < 20:
            continue

        if pair[column].nunique() <= 1:
            continue

        if pair[target].nunique() <= 1:
            continue

        correlation = pair[column].corr(
            pair[target],
            method="spearman",
        )

        if pd.isna(correlation):
            continue

        correlation = float(correlation)

        if abs(correlation) >= 0.995:

            findings.append({
                "type": "strong_numeric_association",
                "column": column,
                "severity": "high",

                "metric": {
                    "spearman_correlation": round(
                        correlation,
                        4,
                    )
                },

                "message": (
                    f'The numeric column "{column}" '
                    f'has a very strong Spearman '
                    f'correlation with the target '
                    f'"{target}" '
                    f'({correlation:.4f}).'
                ),

                "recommendation": (
                    f'Investigate how "{column}" '
                    f'is generated and confirm that '
                    f'it does not contain future or '
                    f'target-derived information.'
                ),
            })

    return findings


# =========================================================
# TARGET-INDEPENDENT CHECKS
# =========================================================

def _check_target_independent(
    df: pd.DataFrame,
) -> list[dict[str, Any]]:

    findings = []

    rows = len(df)

    if rows == 0:
        return findings

    # -----------------------------------------------------
    # ID-like columns
    # -----------------------------------------------------

    for column in df.columns:

        unique_ratio = (
            df[column].nunique(
                dropna=True
            ) / rows
        )

        column_name = (
            str(column).lower()
        )

        id_name_signal = any(
            keyword in column_name
            for keyword in [
                "id",
                "uuid",
                "identifier",
                "customer_id",
                "user_id",
                "transaction_id",
            ]
        )

        if (
            unique_ratio >= 0.95
            or (
                id_name_signal
                and unique_ratio >= 0.50
            )
        ):

            findings.append({
                "type": "id_like_column",
                "column": column,
                "severity": "low",

                "metric": {
                    "unique_ratio": round(
                        unique_ratio * 100,
                        2,
                    )
                },

                "message": (
                    f'The column "{column}" appears '
                    f'to be ID-like because most of '
                    f'its values are unique.'
                ),

                "recommendation": (
                    f'Review whether "{column}" '
                    f'contains meaningful predictive '
                    f'information before including it '
                    f'in a model.'
                ),
            })

    # -----------------------------------------------------
    # Duplicate records
    # -----------------------------------------------------

    duplicate_rows = int(
        df.duplicated().sum()
    )

    if duplicate_rows > 0:

        duplicate_percentage = (
            duplicate_rows / rows * 100
        )

        findings.append({
            "type": "duplicate_records",

            "severity": (
                "high"
                if duplicate_percentage >= 20
                else "medium"
            ),

            "metric": {
                "duplicate_rows": duplicate_rows,
                "duplicate_percentage": round(
                    duplicate_percentage,
                    2,
                ),
            },

            "message": (
                f'The dataset contains '
                f'{duplicate_rows:,} duplicate rows, '
                f'representing '
                f'{duplicate_percentage:.2f}% '
                f'of the dataset.'
            ),

            "recommendation": (
                "Investigate whether these repeated "
                "records are legitimate observations "
                "or accidental duplicates before "
                "model training."
            ),
        })

    return findings


# =========================================================
# MAIN LEAKAGE AUDIT
# =========================================================

def run_leakage_audit(
    df: pd.DataFrame,
    target: str = "",
) -> dict[str, Any]:

    target = target.strip()

    target_provided = bool(target)

    target_exists = (
        target_provided
        and target in df.columns
    )

    # -----------------------------------------------------
    # Always perform target-independent checks
    # -----------------------------------------------------

    independent_findings = (
        _check_target_independent(df)
    )

    # -----------------------------------------------------
    # Invalid target
    # -----------------------------------------------------

    if target_provided and not target_exists:

        return {
            "target_provided": True,
            "target_available": False,
            "target_exists": False,
            "target": target,

            "target_independent_checks": True,
            "target_dependent_checks": False,

            "status": "Target not found",

            "summary": (
                f'The target column "{target}" '
                f'was not found in the uploaded dataset. '
                f'Target-dependent leakage checks were '
                f'not performed.'
            ),

            "risk_signal_count": len(
                independent_findings
            ),

            "high_risk_count": sum(
                1
                for finding in independent_findings
                if finding.get("severity") == "high"
            ),

            "medium_risk_count": sum(
                1
                for finding in independent_findings
                if finding.get("severity") == "medium"
            ),

            "findings": independent_findings,

            "target_findings": [],
        }

    # -----------------------------------------------------
    # No target
    # -----------------------------------------------------

    if not target_provided:

        return {
            "target_provided": False,
            "target_available": False,
            "target_exists": False,
            "target": None,

            "target_independent_checks": True,
            "target_dependent_checks": False,

            "status": "Target not provided",

            "summary": (
                "Target-independent leakage screening "
                "was performed. Target-dependent "
                "leakage checks were not performed "
                "because no target column was provided."
            ),

            "risk_signal_count": len(
                independent_findings
            ),

            "high_risk_count": sum(
                1
                for finding in independent_findings
                if finding.get("severity") == "high"
            ),

            "medium_risk_count": sum(
                1
                for finding in independent_findings
                if finding.get("severity") == "medium"
            ),

            "findings": independent_findings,

            "target_findings": [],
        }

    # -----------------------------------------------------
    # Valid target
    # -----------------------------------------------------

    target_findings = []

    target_findings.extend(
        _check_direct_target_copy(
            df,
            target,
        )
    )

    target_findings.extend(
        _check_target_name_signals(
            df,
            target,
        )
    )

    target_findings.extend(
        _check_categorical_association(
            df,
            target,
        )
    )

    target_findings.extend(
        _check_numeric_association(
            df,
            target,
        )
    )

    all_findings = (
        independent_findings
        + target_findings
    )

    high_risk_count = sum(
        1
        for finding in target_findings
        if finding.get("severity") == "high"
    )

    medium_risk_count = sum(
        1
        for finding in target_findings
        if finding.get("severity") == "medium"
    )

    # -----------------------------------------------------
    # Status
    # -----------------------------------------------------

    if high_risk_count > 0:

        status = "Needs investigation"

        summary = (
            f'Target-dependent leakage checks were '
            f'performed using "{target}" as the target. '
            f'The analysis identified '
            f'{high_risk_count} high-risk target-related '
            f'signal(s) that should be investigated '
            f'before model training.'
        )

    elif medium_risk_count > 0:

        status = "Potential risk"

        summary = (
            f'Target-dependent leakage checks were '
            f'performed using "{target}" as the target. '
            f'The analysis identified '
            f'{medium_risk_count} potential target-related '
            f'risk signal(s) that should be reviewed.'
        )

    else:

        status = "No strong target leakage signals"

        summary = (
            f'Target-dependent leakage checks were '
            f'performed using "{target}" as the target. '
            f'No strong target leakage signals were '
            f'detected by the current screening rules.'
        )

    return {
        "target_provided": True,
        "target_available": True,
        "target_exists": True,
        "target": target,

        "target_independent_checks": True,
        "target_dependent_checks": True,

        "status": status,
        "summary": summary,

        "risk_signal_count": len(all_findings),

        "high_risk_count": high_risk_count,

        "medium_risk_count": medium_risk_count,

        "findings": all_findings,

        "target_findings": target_findings,
    }