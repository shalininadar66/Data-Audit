import pandas as pd
import numpy as np


def calculate_quality_score(
    missing_percentage,
    duplicate_percentage,
    outlier_percentage,
    validity_issue_percentage,
    consistency_issue_percentage,
):
    """
    Calculate an overall dataset quality score.

    The score starts at 100 and applies penalties
    based on measurable data-quality problems.
    """

    score = 100.0

    # -----------------------------
    # Missing values
    # -----------------------------
    if missing_percentage > 30:
        score -= 25
    elif missing_percentage > 15:
        score -= 18
    elif missing_percentage > 5:
        score -= 10
    elif missing_percentage > 1:
        score -= 5

    # -----------------------------
    # Duplicate rows
    # -----------------------------
    if duplicate_percentage > 30:
        score -= 25
    elif duplicate_percentage > 15:
        score -= 18
    elif duplicate_percentage > 5:
        score -= 10
    elif duplicate_percentage > 1:
        score -= 5

    # -----------------------------
    # Outliers
    # -----------------------------
    if outlier_percentage > 20:
        score -= 15
    elif outlier_percentage > 10:
        score -= 10
    elif outlier_percentage > 5:
        score -= 5
    elif outlier_percentage > 1:
        score -= 2

    # -----------------------------
    # Validity problems
    # -----------------------------
    if validity_issue_percentage > 20:
        score -= 15
    elif validity_issue_percentage > 10:
        score -= 10
    elif validity_issue_percentage > 5:
        score -= 5

    # -----------------------------
    # Consistency problems
    # -----------------------------
    if consistency_issue_percentage > 20:
        score -= 10
    elif consistency_issue_percentage > 10:
        score -= 7
    elif consistency_issue_percentage > 5:
        score -= 4

    return max(0, min(100, round(score)))


def get_severity(percentage):
    """
    Convert an issue percentage into a severity level.
    """

    if percentage >= 20:
        return "critical"

    if percentage >= 10:
        return "high"

    if percentage >= 5:
        return "medium"

    if percentage > 0:
        return "low"

    return "none"


def analyze_missing_values(df):
    """
    Analyze missing values column by column.
    """

    counts = df.isna().sum()

    percentages = (
        (counts / len(df)) * 100
    ).round(2)

    affected_columns = []

    details = []

    for column in df.columns:

        missing_count = int(counts[column])
        missing_percentage = float(percentages[column])

        if missing_count > 0:

            affected_columns.append(column)

            details.append({
                "column": column,
                "missing_count": missing_count,
                "missing_percentage": missing_percentage,
                "severity": get_severity(missing_percentage),
            })

    return {
        "total_missing_cells": int(counts.sum()),

        "affected_columns": affected_columns,

        "counts": {
            column: int(value)
            for column, value in counts.items()
            if value > 0
        },

        "percentages": {
            column: float(value)
            for column, value in percentages.items()
            if counts[column] > 0
        },

        "details": details,
    }


def analyze_duplicates(df):
    """
    Analyze duplicate rows.
    """

    duplicate_mask = df.duplicated()

    duplicate_rows = int(
        duplicate_mask.sum()
    )

    total_rows = len(df)

    duplicate_percentage = (
        duplicate_rows / total_rows * 100
        if total_rows > 0
        else 0
    )

    return {
        "duplicate_rows": duplicate_rows,

        "duplicate_percentage": round(
            duplicate_percentage,
            2
        ),

        "severity": get_severity(
            duplicate_percentage
        ),
    }


def analyze_column_types(df):
    """
    Classify columns into useful categories.
    """

    numeric_columns = []

    categorical_columns = []

    date_like_columns = []

    boolean_columns = []

    for column in df.columns:

        series = df[column]

        if pd.api.types.is_bool_dtype(series):
            boolean_columns.append(column)

        elif pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(column)

        else:

            # Try detecting date-like columns
            converted = pd.to_datetime(
                series,
                errors="coerce"
            )

            non_null = series.notna().sum()

            if (
                non_null > 0
                and converted.notna().sum() / non_null >= 0.8
            ):
                date_like_columns.append(column)

            else:
                categorical_columns.append(column)

    return {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "date_like_columns": date_like_columns,
        "boolean_columns": boolean_columns,
    }


def analyze_constant_columns(df):
    """
    Detect columns containing only one unique value.
    """

    constant_columns = []

    for column in df.columns:

        unique_count = df[column].nunique(
            dropna=False
        )

        if unique_count <= 1:
            constant_columns.append(column)

    return constant_columns


def analyze_high_cardinality(df):
    """
    Detect categorical columns with unusually
    high numbers of unique values.
    """

    high_cardinality_columns = []

    details = []

    row_count = len(df)

    if row_count == 0:
        return {
            "columns": [],
            "details": []
        }

    for column in df.columns:

        series = df[column]

        if pd.api.types.is_numeric_dtype(series):
            continue

        unique_count = series.nunique(
            dropna=True
        )

        cardinality_percentage = (
            unique_count / row_count * 100
        )

        # High cardinality heuristic
        if (
            unique_count > 100
            and cardinality_percentage > 20
        ):
            high_cardinality_columns.append(
                column
            )

            details.append({
                "column": column,
                "unique_values": int(unique_count),
                "cardinality_percentage": round(
                    cardinality_percentage,
                    2
                ),
            })

    return {
        "columns": high_cardinality_columns,
        "details": details,
    }


def analyze_id_like_columns(df):
    """
    Detect columns that look like identifiers.
    """

    id_like_columns = []

    details = []

    row_count = len(df)

    if row_count == 0:
        return {
            "columns": [],
            "details": []
        }

    for column in df.columns:

        series = df[column]

        unique_count = series.nunique(
            dropna=True
        )

        uniqueness_percentage = (
            unique_count / row_count * 100
        )

        column_name = str(column).lower()

        name_signal = any(
            keyword in column_name
            for keyword in [
                "id",
                "uuid",
                "identifier",
                "code",
            ]
        )

        uniqueness_signal = (
            uniqueness_percentage >= 95
        )

        if name_signal and uniqueness_signal:

            id_like_columns.append(column)

            details.append({
                "column": column,
                "uniqueness_percentage": round(
                    uniqueness_percentage,
                    2
                ),
            })

    return {
        "columns": id_like_columns,
        "details": details,
    }


def analyze_outliers(df):
    """
    Detect numerical outliers using the IQR method.
    """

    details = []

    total_outlier_cells = 0
    total_numeric_cells = 0

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:

        series = df[column].dropna()

        if len(series) < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        if iqr == 0:
            continue

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_mask = (
            (series < lower_bound)
            | (series > upper_bound)
        )

        outlier_count = int(
            outlier_mask.sum()
        )

        valid_count = len(series)

        outlier_percentage = (
            outlier_count / valid_count * 100
        )

        total_outlier_cells += outlier_count
        total_numeric_cells += valid_count

        if outlier_count > 0:

            details.append({
                "column": column,
                "outlier_count": outlier_count,
                "outlier_percentage": round(
                    outlier_percentage,
                    2
                ),
                "lower_bound": round(
                    float(lower_bound),
                    4
                ),
                "upper_bound": round(
                    float(upper_bound),
                    4
                ),
                "severity": get_severity(
                    outlier_percentage
                ),
            })

    overall_percentage = (
        total_outlier_cells
        / total_numeric_cells
        * 100
        if total_numeric_cells > 0
        else 0
    )

    return {
        "total_outlier_cells": int(
            total_outlier_cells
        ),

        "overall_percentage": round(
            overall_percentage,
            2
        ),

        "details": details,
    }


def analyze_categorical_consistency(df):
    """
    Look for basic categorical consistency issues.

    Examples:
    'North Indian'
    ' north indian'
    'North indian '
    """

    details = []

    affected_cells = 0
    categorical_cells = 0

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for column in categorical_columns:

        series = df[column].dropna().astype(str)

        if len(series) == 0:
            continue

        categorical_cells += len(series)

        normalized = (
            series
            .str.strip()
            .str.lower()
        )

        inconsistent_count = int(
            (
                series != normalized
            ).sum()
        )

        if inconsistent_count > 0:

            affected_cells += (
                inconsistent_count
            )

            details.append({
                "column": column,
                "affected_values": inconsistent_count,
            })

    percentage = (
        affected_cells
        / categorical_cells
        * 100
        if categorical_cells > 0
        else 0
    )

    return {
        "affected_cells": int(
            affected_cells
        ),

        "percentage": round(
            percentage,
            2
        ),

        "details": details,
    }


def run_quality_audit(df):
    """
    Run the complete data-quality audit.
    """

    row_count = len(df)
    column_count = len(df.columns)

    # -----------------------------
    # Missing values
    # -----------------------------

    missing = analyze_missing_values(df)

    missing_percentage = (
        missing["total_missing_cells"]
        / (row_count * column_count)
        * 100
        if row_count > 0
        and column_count > 0
        else 0
    )

    # -----------------------------
    # Duplicates
    # -----------------------------

    duplicates = analyze_duplicates(df)

    # -----------------------------
    # Column types
    # -----------------------------

    types = analyze_column_types(df)

    # -----------------------------
    # Structural checks
    # -----------------------------

    constant_columns = (
        analyze_constant_columns(df)
    )

    high_cardinality = (
        analyze_high_cardinality(df)
    )

    id_like = (
        analyze_id_like_columns(df)
    )

    # -----------------------------
    # Outliers
    # -----------------------------

    outliers = analyze_outliers(df)

    # -----------------------------
    # Consistency
    # -----------------------------

    consistency = (
        analyze_categorical_consistency(df)
    )

    # -----------------------------
    # Validity
    # -----------------------------

    # Initial validity measure.
    # This will be expanded later with
    # column-specific validation rules.
    validity_issue_percentage = 0

    # -----------------------------
    # Quality score
    # -----------------------------

    quality_score = calculate_quality_score(
        missing_percentage=missing_percentage,
        duplicate_percentage=duplicates[
            "duplicate_percentage"
        ],
        outlier_percentage=outliers[
            "overall_percentage"
        ],
        validity_issue_percentage=
            validity_issue_percentage,
        consistency_issue_percentage=
            consistency["percentage"],
    )

    return {
        "score": quality_score,

        "missing": missing,

        "duplicates": duplicates,

        "types": types,

        "constant_columns": constant_columns,

        "high_cardinality": high_cardinality,

        "id_like": id_like,

        "outliers": outliers,

        "consistency": consistency,

        "validity": {
            "issue_percentage":
                validity_issue_percentage
        },

        "summary": {
            "rows": row_count,
            "columns": column_count,
        },
    }