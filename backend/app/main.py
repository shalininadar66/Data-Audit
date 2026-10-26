from io import BytesIO
from math import log2
import re
from typing import Any
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title='Data Audit API', description='Dataset quality, leakage, task and model readiness audit', version='1.1.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    # Allows your deployed Vercel website, including Vercel preview URLs.
    allow_origin_regex=r"https://.*[.]vercel[.]app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
MODEL_TASK = {'logistic regression': 'classification', 'decision tree': 'classification', 'random forest': 'classification', 'gradient boosting': 'classification', 'xgboost': 'classification', 'support vector machine (svm)': 'classification', 'k-nearest neighbors (knn)': 'classification', 'naive bayes': 'classification', 'adaboost': 'classification', 'extra trees': 'classification', 'linear regression': 'regression', 'ridge regression': 'regression', 'lasso regression': 'regression', 'elastic net': 'regression', 'decision tree regressor': 'regression', 'random forest regressor': 'regression', 'gradient boosting regressor': 'regression', 'xgboost regressor': 'regression', 'support vector regression (svr)': 'regression', 'k-nearest neighbors regressor': 'regression', 'k-means': 'clustering', 'hierarchical clustering': 'clustering', 'dbscan': 'clustering', 'gaussian mixture model': 'clustering', 'mean shift': 'clustering', 'spectral clustering': 'clustering', 'arima': 'time-series', 'sarima': 'time-series', 'exponential smoothing': 'time-series', 'prophet': 'time-series', 'random forest (forecasting)': 'time-series', 'xgboost (forecasting)': 'time-series', 'lstm': 'time-series', 'isolation forest': 'anomaly', 'local outlier factor': 'anomaly', 'one-class svm': 'anomaly', 'autoencoder': 'anomaly', 'pca': 'dimensionality-reduction', 'kernel pca': 'dimensionality-reduction', 'truncated svd': 'dimensionality-reduction', 't-sne': 'dimensionality-reduction', 'umap': 'dimensionality-reduction', 'lda': 'dimensionality-reduction', 'collaborative filtering': 'recommendation', 'content-based filtering': 'recommendation', 'matrix factorization': 'recommendation', 'k-nearest neighbors': 'recommendation', 'hybrid recommendation': 'recommendation', 'learning to rank': 'ranking', 'ranknet': 'ranking', 'lambdamart': 'ranking', 'xgboost ranking': 'ranking'}
TASK_DISPLAY = {'classification': 'classification', 'regression': 'regression', 'clustering': 'clustering', 'time-series': 'time-series forecasting', 'anomaly': 'anomaly detection', 'dimensionality-reduction': 'dimensionality reduction', 'recommendation': 'recommendation', 'ranking': 'ranking'}
SUPERVISED = {'classification', 'regression', 'ranking', 'recommendation', 'time-series'}

def clean(value: str | None) -> str:
    return str(value or '').strip()

def norm_model(model: str) -> str:
    return re.sub('\\s+', ' ', model.strip().lower())

def date_like_columns(df: pd.DataFrame) -> list[str]:
    result = []
    for col in df.columns:
        if df[col].dtype == 'object':
            sample = df[col].dropna().head(100)
            if len(sample) >= 5:
                parsed = pd.to_datetime(sample, errors='coerce')
                if float(parsed.notna().mean()) >= 0.8:
                    result.append(col)
    return result

def outlier_counts(df: pd.DataFrame, numeric_cols: list[str]) -> tuple[dict[str, int], dict[str, float], int, float]:
    counts: dict[str, int] = {}
    percentages: dict[str, float] = {}
    total_values = total_outliers = 0
    for col in numeric_cols:
        x = df[col].dropna()
        if len(x) < 10:
            counts[col] = 0
            percentages[col] = 0.0
            continue
        q1, q3 = (x.quantile(0.25), x.quantile(0.75))
        iqr = q3 - q1
        count = int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0
        counts[col] = count
        percentages[col] = round(count / len(x) * 100, 2)
        total_values += len(x)
        total_outliers += count
    rate = round(total_outliers / total_values * 100, 2) if total_values else 0.0
    return (counts, percentages, total_outliers, rate)

def structural_analysis(df: pd.DataFrame, date_cols: list[str]) -> dict[str, Any]:
    rows = len(df)
    numeric = df.select_dtypes(include='number').columns.tolist()
    categorical = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    missing_counts = df.isna().sum().astype(int).to_dict()
    missing_pct = (df.isna().mean() * 100).round(2).to_dict()
    duplicates = int(df.duplicated().sum())
    constants = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    id_like, high_card = ([], [])
    # Be conservative: uniqueness alone does not make a column an identifier.
    # Numeric measurements can legitimately be unique, so ID-like detection is
    # primarily based on explicit identifier tokens in the column name.
    identifier_name_pattern = re.compile(
        r'(?:^|[\W_])(id|uuid|guid|identifier|record_key|row_key|row_index|index)(?:$|[\W_])',
        flags=re.IGNORECASE,
    )
    for c in df.columns:
        ratio = df[c].nunique(dropna=True) / rows if rows else 0
        name = str(c).strip().lower().replace('-', '_').replace(' ', '_')
        name_looks_like_identifier = bool(identifier_name_pattern.search(name))
        if name_looks_like_identifier and ratio >= 0.5:
            id_like.append(c)
        if c in categorical and rows and (df[c].nunique(dropna=True) >= 20) and (ratio >= 0.5):
            high_card.append(c)
    empty_rows = int(df.isna().all(axis=1).sum())
    out_counts, out_pcts, total_outliers, outlier_rate = outlier_counts(df, numeric)
    ambiguous_types = []
    for c in categorical:
        sample = df[c].dropna().head(500)
        if len(sample) >= 20:
            parsed = pd.to_numeric(sample, errors='coerce')
            rate = float(parsed.notna().mean())
            if 0.2 <= rate < 0.9:
                ambiguous_types.append(c)
    return {'rows': rows, 'columns': df.shape[1], 'numeric_columns': numeric, 'categorical_columns': categorical, 'date_like_columns': date_cols, 'missing_counts': missing_counts, 'missing_percentages': missing_pct, 'missing_columns': [c for c in df.columns if missing_counts[c] > 0], 'duplicate_rows': duplicates, 'duplicate_percentage': round(duplicates / rows * 100, 2) if rows else 0, 'constant_columns': constants, 'id_like_columns': id_like, 'high_cardinality_columns': high_card, 'empty_rows': empty_rows, 'outlier_counts': out_counts, 'outlier_percentages': out_pcts, 'total_outliers': total_outliers, 'outlier_rate': outlier_rate, 'ambiguous_type_columns': ambiguous_types, 'unique_values': {c: int(df[c].nunique(dropna=True)) for c in df.columns}, 'data_types': {c: str(df[c].dtype) for c in df.columns}}

def quality_analysis(s: dict[str, Any], df: pd.DataFrame | None = None) -> dict[str, Any]:
    """Calculate a deterministic quality score with stronger validity checks.

    Quality checks use explicit, bounded rules; recognized range checks are disclosed in the report.
    leakage_analysis() is intentionally untouched.
    """
    weights = {
        'Completeness': 22,
        'Duplicate integrity': 18,
        'Outlier burden': 8,
        'Constant columns': 4,
        'High-cardinality columns': 4,
        'ID-like columns': 4,
        'Empty rows': 4,
        'Data type validity': 15,
        'Text consistency': 11,
        'Range validity': 10,
    }

    avg_missing = sum(s.get('missing_percentages', {}).values()) / max(int(s.get('columns', 0)), 1)
    rows = len(df) if df is not None else int(s.get('rows', 0))

    # Identify numeric-like fields, including text columns that contain mostly
    # numbers, so values such as "twenty" in Age are counted as invalid.
    numeric_series: dict[str, pd.Series] = {}
    invalid_by_column: dict[str, int] = {}
    numeric_like_columns: list[str] = []
    if df is not None:
        numeric_name_pattern = re.compile(
            r'(^|[^a-z])(age|years?|marks?|score|amount|price|cost|quantity|qty|count|salary|income|height|weight|percent(age)?|rate|duration|latitude|longitude)([^a-z]|$)',
            flags=re.IGNORECASE,
        )
        for col in df.columns:
            original = df[col]
            non_missing = original.notna()
            values = original[non_missing]
            if values.empty:
                continue
            if pd.api.types.is_numeric_dtype(original):
                parsed = pd.to_numeric(original, errors='coerce')
                is_numeric_like = True
            else:
                parsed_values = pd.to_numeric(values.astype(str).str.strip(), errors='coerce')
                parse_rate = float(parsed_values.notna().mean()) if len(parsed_values) else 0.0
                name_hint = bool(numeric_name_pattern.search(str(col)))
                is_numeric_like = parse_rate >= 0.80 or (name_hint and parse_rate >= 0.30)
                parsed = pd.to_numeric(original.astype('string').str.strip(), errors='coerce')
            if is_numeric_like:
                numeric_like_columns.append(str(col))
                invalid_by_column[str(col)] = int((non_missing & parsed.isna()).sum())
                numeric_series[str(col)] = parsed

    invalid_total = sum(invalid_by_column.values())
    numeric_non_missing_total = sum(int(df[c].notna().sum()) for c in numeric_like_columns) if df is not None else 0
    invalid_pct = invalid_total / numeric_non_missing_total * 100 if numeric_non_missing_total else 0.0

    # Count text cells with surrounding whitespace or tabs/newlines.
    whitespace_by_column: dict[str, int] = {}
    text_cell_total = 0
    if df is not None:
        for col in df.select_dtypes(include=['object', 'string']).columns:
            values = df[col].dropna().astype(str)
            text_cell_total += len(values)
            whitespace_by_column[str(col)] = int(values.ne(values.str.strip()).sum())
    whitespace_total = sum(whitespace_by_column.values())
    whitespace_pct = whitespace_total / text_cell_total * 100 if text_cell_total else 0.0

    # Apply conservative, documented range rules only to recognizable fields.
    # These are configurable screening rules, not universal truths for all datasets.
    range_rules = {
        'age': (0.0, 120.0),
        'years_old': (0.0, 120.0),
        'marks': (0.0, 100.0),
        'mark': (0.0, 100.0),
        'score': (0.0, 100.0),
        'percentage': (0.0, 100.0),
        'percent': (0.0, 100.0),
    }
    range_by_column: dict[str, dict[str, Any]] = {}
    range_checked_total = 0
    range_violation_total = 0
    if df is not None:
        for col in df.columns:
            normalized_name = re.sub(r'[^a-z0-9]+', '_', str(col).strip().lower()).strip('_')
            rule = range_rules.get(normalized_name)
            if rule is None:
                continue
            low, high = rule
            parsed = pd.to_numeric(df[col].astype('string').str.strip(), errors='coerce')
            non_missing_numeric = parsed.notna()
            violations = non_missing_numeric & ((parsed < low) | (parsed > high))
            checked = int(non_missing_numeric.sum())
            count = int(violations.sum())
            range_checked_total += checked
            range_violation_total += count
            range_by_column[str(col)] = {
                'minimum': low,
                'maximum': high,
                'checked_values': checked,
                'out_of_range_values': count,
                'percentage_of_parsed_values': round(count / checked * 100, 2) if checked else 0.0,
            }
    range_violation_pct = range_violation_total / range_checked_total * 100 if range_checked_total else 0.0

    # IQR flags are screening signals, not proof that a value is erroneous.
    quality_outlier_count = 0
    quality_numeric_value_count = 0
    quality_outlier_by_column: dict[str, int] = {}
    for col, values in numeric_series.items():
        x = values.dropna()
        if len(x) < 10:
            quality_outlier_by_column[col] = 0
            continue
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        count = int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0
        quality_outlier_by_column[col] = count
        quality_outlier_count += count
        quality_numeric_value_count += len(x)
    quality_outlier_rate = quality_outlier_count / quality_numeric_value_count * 100 if quality_numeric_value_count else 0.0

    # Each category's deduction scales to the observed rate and is capped at
    # its allocated weight. Rates intentionally penalize common data problems
    # more strongly than the previous formula did.
    penalties = {
        'Completeness': min(weights['Completeness'], avg_missing / 15 * weights['Completeness']),
        'Duplicate integrity': min(weights['Duplicate integrity'], s.get('duplicate_percentage', 0) / 5 * weights['Duplicate integrity']),
        'Outlier burden': min(weights['Outlier burden'], quality_outlier_rate / 10 * weights['Outlier burden']),
        'Constant columns': min(weights['Constant columns'], len(s.get('constant_columns', [])) * 2.5),
        'High-cardinality columns': min(weights['High-cardinality columns'], len(s.get('high_cardinality_columns', [])) * 2.5),
        'ID-like columns': min(weights['ID-like columns'], len(s.get('id_like_columns', [])) * 2.5),
        'Empty rows': min(weights['Empty rows'], (s.get('empty_rows', 0) / max(rows, 1)) * weights['Empty rows']),
        'Data type validity': min(weights['Data type validity'], invalid_pct / 1.8 * weights['Data type validity']),
        'Text consistency': min(weights['Text consistency'], whitespace_pct / 5 * weights['Text consistency']),
        'Range validity': min(weights['Range validity'], range_violation_pct / 10 * weights['Range validity']),
    }

    metrics = {
        'Completeness': f'{avg_missing:.2f}% average missingness',
        'Duplicate integrity': f"{s.get('duplicate_percentage', 0):.2f}% duplicate rows ({s.get('duplicate_rows', 0):,})",
        'Outlier burden': f'{quality_outlier_rate:.2f}% of parsed numeric observations ({quality_outlier_count:,}); investigate flagged values',
        'Constant columns': f"{len(s.get('constant_columns', []))} column(s)",
        'High-cardinality columns': f"{len(s.get('high_cardinality_columns', []))} column(s)",
        'ID-like columns': f"{len(s.get('id_like_columns', []))} column(s)",
        'Empty rows': f"{s.get('empty_rows', 0):,} row(s)",
        'Data type validity': f'{invalid_total:,} non-numeric value(s) in numeric-like columns; {invalid_pct:.2f}% of checked values',
        'Text consistency': f'{whitespace_total:,} whitespace-inconsistent cell(s); {whitespace_pct:.2f}% of non-missing text cells',
        'Range validity': f'{range_violation_total:,} out-of-range value(s) across checked columns; {range_violation_pct:.2f}% of parsed values in columns with configured range rules',
    }
    params = []
    for name, weight in weights.items():
        deduction = round(penalties[name], 2)
        params.append({
            'parameter': name,
            'weight': weight,
            'metric': metrics[name],
            'deduction': deduction,
            'points_awarded': round(weight - deduction, 2),
        })

    score = max(0, min(100, round(sum(x['points_awarded'] for x in params))))
    if score >= 90:
        classification = 'Excellent'
        classification_description = 'Few issues detected by the checks performed; this is not a guarantee that the data is error-free.'
    elif score >= 75:
        classification = 'Good, with issues to review'
        classification_description = 'Some measurable issues were found. Review the listed findings before analysis or training.'
    elif score >= 50:
        classification = 'Needs cleaning'
        classification_description = 'Several measurable issues were found. Clean and re-audit the dataset before relying on it.'
    else:
        classification = 'Poor'
        classification_description = 'Substantial issues were detected. Clean the dataset and run the audit again before relying on it.'

    details = {
        'numeric_like_columns': numeric_like_columns,
        'invalid_numeric_values': invalid_by_column,
        'whitespace_inconsistent_cells': whitespace_by_column,
        'quality_outlier_counts': quality_outlier_by_column,
        'range_violations': range_by_column,
        'classification_description': classification_description,
    }
    return {
        'score': score,
        'classification': classification,
        'classification_description': classification_description,
        'method': 'Deterministic weighted rule-based score based on measured missingness, duplicates, numeric validity, text consistency, outlier flags, structural issues and empty rows. Outlier flags are screening signals, not proof of errors. This is not a model accuracy score or external certification.',
        'formula': '100 points are distributed across completeness (22), duplicate integrity (18), outlier burden (8), constant columns (4), high-cardinality columns (4), ID-like columns (4), empty rows (4), data type validity (15), text consistency (11), and range validity (10). Each deduction is bounded by its category weight. Numeric validity uses a 1.8% full-deduction threshold; whitespace inconsistency uses a 5% threshold; range validity uses a 10% threshold in recognized columns.',
        'classification_bands': [
            {'minimum': 90, 'maximum': 100, 'label': 'Excellent'},
            {'minimum': 75, 'maximum': 89, 'label': 'Good, with issues to review'},
            {'minimum': 50, 'maximum': 74, 'label': 'Needs cleaning'},
            {'minimum': 0, 'maximum': 49, 'label': 'Poor'},
        ],
        'parameters': params,
        'checked_for': [x['parameter'] for x in params],
        'details': details,
    }

def target_analysis(df: pd.DataFrame, target: str, task: str) -> dict[str, Any]:
    if not target:
        return {'provided': False, 'exists': False, 'analysis': None}
    if target not in df.columns:
        return {'provided': True, 'exists': False, 'analysis': {'column': target, 'exists': False, 'message': f'The target column "{target}" was not found in the dataset.'}}
    s = df[target]
    numeric_parse_rate = None
    if s.dtype == 'object':
        parsed = pd.to_numeric(s, errors='coerce')
        numeric_parse_rate = round(float(parsed.notna().mean() * 100), 2)
    unique = int(s.nunique(dropna=True))
    missing = int(s.isna().sum())
    inferred_type = 'numeric' if pd.api.types.is_numeric_dtype(s) or (numeric_parse_rate is not None and numeric_parse_rate >= 90) else 'categorical'
    user_cols = [c for c in df.columns if any((k in norm_model(c) for k in ('user', 'customer', 'member', 'account')))]
    item_cols = [c for c in df.columns if any((k in norm_model(c) for k in ('item', 'product', 'post', 'content', 'document', 'movie', 'listing')))]
    ranking_signal = any((k in norm_model(target) for k in ('impression', 'engagement', 'click', 'relevance', 'rank', 'position', 'score', 'rating', 'view', 'like', 'conversion')))
    if inferred_type == 'categorical':
        data_suggested_task = 'classification'
    elif user_cols and item_cols and (ranking_signal or unique >= 5):
        data_suggested_task = 'ranking'
    elif unique <= 10:
        data_suggested_task = 'classification'
    else:
        data_suggested_task = 'regression'
    info = {'column': target, 'exists': True, 'type': str(s.dtype), 'unique_values': unique, 'missing_values': missing, 'missing_percentage': round(missing / len(df) * 100, 2) if len(df) else 0, 'numeric_parse_rate': numeric_parse_rate, 'inferred_task': data_suggested_task, 'inferred_type': inferred_type, 'suggested_task': data_suggested_task, 'selected_task': task or None}
    if data_suggested_task == 'classification':
        vc = s.value_counts(dropna=True)
        if len(vc):
            info.update({'class_count': unique, 'largest_class_percentage': round(vc.iloc[0] / vc.sum() * 100, 2), 'smallest_class_percentage': round(vc.iloc[-1] / vc.sum() * 100, 2), 'imbalance_ratio': round(float(vc.iloc[0] / vc.iloc[-1]), 2) if vc.iloc[-1] else None})
    return {'provided': True, 'exists': True, 'analysis': info}

def leakage_analysis(df: pd.DataFrame, target: str, s: dict[str, Any]) -> dict[str, Any]:
    """Calculate a transparent, bounded leakage-risk index from detected signals.

    The score is not a probability. Target-dependent findings are grouped by
    affected column so multiple rules describing the same suspicious feature do
    not repeatedly add the same high-severity points.
    """
    independent = []
    for c in s['id_like_columns']:
        independent.append({
            'column': c,
            'severity': 'low',
            'type': 'id_like',
            'metric': f"{int(df[c].nunique(dropna=True))} unique values",
            'message': f'The column "{c}" appears to be an identifier based on its name; review whether it should be used as a predictor.'
        })
    if s['duplicate_rows']:
        independent.append({
            'severity': 'medium',
            'type': 'duplicates',
            'metric': f"{s['duplicate_percentage']:.2f}% duplicate rows",
            'message': f"{s['duplicate_rows']:,} duplicate rows were detected ({s['duplicate_percentage']:.2f}% of the dataset)."
        })

    target_findings = []
    valid_target = bool(target and target in df.columns)
    if valid_target:
        target_series = df[target].astype('string').fillna('').str.strip().str.lower()
        for c in df.columns:
            if c == target:
                continue
            # Direct copy check: compare non-empty values only.
            if df[c].dtype == df[target].dtype:
                candidate = df[c].astype('string').fillna('').str.strip().str.lower()
                comparable = (target_series != '') & (candidate != '')
                if bool(comparable.any()):
                    match_rate = float((target_series[comparable] == candidate[comparable]).mean())
                    if match_rate >= 0.99:
                        target_findings.append({
                            'column': c,
                            'severity': 'high',
                            'type': 'direct_copy',
                            'metric': f'{match_rate * 100:.2f}% matching rows',
                            'message': f'The column "{c}" matches the target for {match_rate * 100:.2f}% of comparable rows.'
                        })
            # Correlation is a warning signal, not proof of leakage.
            if pd.api.types.is_numeric_dtype(df[target]) and pd.api.types.is_numeric_dtype(df[c]):
                pair = df[[c, target]].dropna()
                if len(pair) >= 20 and pair[c].nunique() > 1 and pair[target].nunique() > 1:
                    corr = pair[c].corr(pair[target], method='spearman')
                    if pd.notna(corr) and abs(float(corr)) >= 0.995:
                        target_findings.append({
                            'column': c,
                            'severity': 'high',
                            'type': 'strong_association',
                            'metric': f'Spearman {float(corr):.4f}',
                            'message': f'The numeric column "{c}" has a very strong Spearman relationship with "{target}" ({float(corr):.4f}). This is a review signal, not proof of leakage.'
                        })
            shared = set(re.findall(r'[a-z0-9]+', str(c).lower())) & set(re.findall(r'[a-z0-9]+', str(target).lower()))
            if shared and any(k in str(c).lower() for k in ('target', 'label', 'prediction', 'outcome', 'result', 'score', 'rate', 'rating')):
                target_findings.append({
                    'column': c,
                    'severity': 'medium',
                    'type': 'target_related_name',
                    'metric': 'shared target-related naming',
                    'message': f'The name of "{c}" is related to the target "{target}" and should be reviewed.'
                })

    id_points = min(15.0, len(s['id_like_columns']) * 2.5)
    duplicate_points = min(15.0, float(s['duplicate_percentage']) / 20.0 * 15.0)

    # Score target-dependent evidence once per affected column. A high-severity
    # signal gives the target component 70 points; medium-only findings give 25
    # points per distinct column, capped at 70. Multiple rules for one column
    # do not repeatedly inflate the score.
    severity_by_column: dict[str, str] = {}
    for finding in target_findings:
        column = str(finding.get('column', finding.get('type', 'unknown')))
        previous = severity_by_column.get(column)
        if finding.get('severity') == 'high' or previous is None:
            severity_by_column[column] = finding.get('severity', 'medium')
    if any(severity == 'high' for severity in severity_by_column.values()):
        target_points = 70.0
    else:
        medium_columns = sum(1 for severity in severity_by_column.values() if severity == 'medium')
        target_points = min(70.0, medium_columns * 25.0)

    score = int(round(min(100.0, id_points + duplicate_points + target_points)))
    checks = ['ID-like columns', 'duplicate rows']
    if valid_target:
        checks += ['direct target copies', 'very strong numeric target associations', 'target-related column names']
        status = 'Target-dependent screening completed'
        scope = 'Target-independent and target-dependent screening'
    elif target:
        status = 'Target not found'
        scope = 'Target-independent screening only; the supplied target name was not found'
    else:
        status = 'Target not provided'
        scope = 'Target-independent screening only'

    if target_findings:
        summary = f'{len(target_findings)} target-related signal(s) were detected. Strong correlation is a review signal and does not by itself prove leakage.'
    elif valid_target:
        summary = 'No target-dependent signal matched the current screening rules. This does not guarantee that the dataset is leakage-free.'
    else:
        summary = 'Target-dependent checks were not performed; only target-independent screening was available.'

    return {
        'score': score,
        'scale': '0 = no detected risk; 100 = high detected leakage risk. This is a rule-based index, not a probability.',
        'target_provided': bool(target),
        'target_exists': valid_target,
        'target_dependent': valid_target,
        'status': status,
        'scope': scope,
        'score_components': {
            'id_like_columns_points': round(id_points, 2),
            'duplicate_rows_points': round(duplicate_points, 2),
            'target_signal_points': round(target_points, 2),
            'maximum_target_independent_points': 30,
            'target_signal_scoring': 'High signal: 70 points once; medium-only signals: 25 points per distinct column, capped at 70.'
        },
        'summary': summary,
        'checks': checks,
        'findings': independent + target_findings,
        'target_findings': target_findings,
    }

def infer_task(df: pd.DataFrame, target_info: dict[str, Any], explicit_task: str, model: str) -> str | None:
    if explicit_task:
        return explicit_task
    if model and MODEL_TASK.get(norm_model(model)):
        return MODEL_TASK[norm_model(model)]
    if target_info.get('exists'):
        return target_info['analysis']['suggested_task']
    return 'clustering' if len(df.select_dtypes(include='number').columns) >= 2 else None

def _columns_by_role(df: pd.DataFrame, keywords: tuple[str, ...]) -> list[str]:
    return [c for c in df.columns if any((k in norm_model(c) for k in keywords))]

def _size_points(rows: int, weight: int) -> float:
    if rows >= 10000:
        return float(weight)
    if rows >= 5000:
        return round(weight * 0.9, 2)
    if rows >= 1000:
        return round(weight * 0.75, 2)
    if rows >= 500:
        return round(weight * 0.55, 2)
    if rows >= 100:
        return round(weight * 0.35, 2)
    return round(weight * 0.15, 2)

def _task_status(score: int) -> str:
    return 'Strong task-data complement' if score >= 85 else 'Good task-data complement' if score >= 70 else 'Moderate task-data complement' if score >= 50 else 'Weak task-data complement'

def _quality_ready_points(df: pd.DataFrame) -> tuple[float, str]:
    avg_missing = float(df.isna().mean().mean() * 100) if not df.empty else 100.0
    dup = float(df.duplicated().mean() * 100) if len(df) else 100.0
    score = 10.0 if avg_missing <= 2 and dup < 5 else 7.0 if avg_missing <= 10 and dup < 15 else 4.0 if avg_missing <= 20 and dup < 30 else 1.0
    return (score, f'{avg_missing:.2f}% average missingness; {dup:.2f}% duplicates')

SUPPORTED_TASKS = tuple(TASK_DISPLAY.keys())


def task_audit(df: pd.DataFrame, task: str, target_info: dict[str, Any], date_cols: list[str], compare_tasks: bool = False) -> dict[str, Any]:
    if not task:
        return {'provided': False, 'status': 'Task not provided', 'score': None, 'parameters': [], 'findings': [], 'recommendations': []}
    rows = len(df)
    target_exists = bool(target_info.get('exists'))
    target = target_info.get('analysis') if target_exists else None
    feature_df = df.drop(columns=[target['column']]) if target_exists and target['column'] in df.columns else df
    numeric = feature_df.select_dtypes(include='number').columns.tolist()
    categorical = feature_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    quality_points, quality_metric = _quality_ready_points(df)
    findings, recommendations, parameters = ([], [], [])

    def add(name: str, weight: int, points: float, metric: str, status: str='Assessed'):
        parameters.append({'parameter': name, 'weight': weight, 'points_awarded': round(max(0.0, min(float(weight), points)), 2), 'metric': metric, 'status': status})
    if task == 'classification':
        if target_exists:
            classes = int(target.get('class_count', target.get('unique_values', 0)))
            missing = float(target.get('missing_percentage', 0))
            target_points = 25.0 if target.get('inferred_type') == 'categorical' and 2 <= classes <= 20 else 17.0 if target.get('inferred_type') == 'categorical' and classes <= 50 else 8.0
            balance = 15.0 if target.get('largest_class_percentage', 100) <= 60 else 10.0 if target.get('largest_class_percentage', 100) <= 80 else 4.0
            complete = 10.0 if missing <= 1 else 7.0 if missing <= 5 else 3.0 if missing <= 15 else 0.0
            if target.get('inferred_type') != 'categorical':
                findings.append({'severity': 'high', 'type': 'classification_target_type', 'message': f'''Target "{target['column']}" is not clearly categorical for a standard classification workflow.'''})
        else:
            target_points, balance, complete = (12.0, 7.0, 5.0)
            findings.append({'severity': 'info', 'type': 'classification_no_target', 'message': 'Classification can be assessed structurally, but a target is needed for class-specific evaluation.'})
            recommendations.append('Provide a target to verify class type, class balance and target completeness.')
        add('Target compatibility', 25, target_points, f"{(target.get('inferred_type') if target_exists else 'Not supplied')}", 'Assessed' if target_exists else 'Partial')
        add('Class balance', 15, balance, f"Largest class: {target.get('largest_class_percentage', 0):.2f}%" if target_exists else 'Not available', 'Assessed' if target_exists else 'Not assessed')
        add('Target completeness', 10, complete, f"{target.get('missing_percentage', 0):.2f}% missing" if target_exists else 'Not available', 'Assessed' if target_exists else 'Not assessed')
        add('Feature compatibility', 20, 20 if numeric or categorical else 0, f'{len(numeric)} numeric; {len(categorical)} categorical')
        add('Dataset size', 10, _size_points(rows, 10), f'{rows:,} rows')
        add('General data quality', 20, 20 if quality_points >= 9 else 14 if quality_points >= 7 else 8, quality_metric)
    elif task == 'regression':
        if target_exists:
            unique = int(target.get('unique_values', 0))
            missing = float(target.get('missing_percentage', 0))
            target_points = 25.0 if target.get('inferred_type') == 'numeric' else 5.0
            variation = 15.0 if unique >= 100 else 11.0 if unique >= 20 else 6.0
            complete = 10.0 if missing <= 1 else 7.0 if missing <= 5 else 3.0 if missing <= 15 else 0.0
            if target.get('inferred_type') != 'numeric':
                findings.append({'severity': 'high', 'type': 'regression_target_type', 'message': f'''Target "{target['column']}" is not numeric for standard regression.'''})
        else:
            target_points, variation, complete = (12.0, 6.0, 5.0)
            findings.append({'severity': 'info', 'type': 'regression_no_target', 'message': 'Regression can be assessed structurally, but target-specific compatibility requires a target.'})
            recommendations.append('Provide a numeric target to complete regression-specific assessment.')
        add('Target compatibility', 25, target_points, f"{(target.get('inferred_type') if target_exists else 'Not supplied')}", 'Assessed' if target_exists else 'Partial')
        add('Target variation', 15, variation, f"{target.get('unique_values', 0):,} unique values" if target_exists else 'Not available', 'Assessed' if target_exists else 'Not assessed')
        add('Target completeness', 10, complete, f"{target.get('missing_percentage', 0):.2f}% missing" if target_exists else 'Not available', 'Assessed' if target_exists else 'Not assessed')
        add('Feature compatibility', 20, 20 if numeric or categorical else 0, f'{len(numeric)} numeric; {len(categorical)} categorical')
        add('Dataset size', 10, _size_points(rows, 10), f'{rows:,} rows')
        add('General data quality', 20, 20 if quality_points >= 9 else 14 if quality_points >= 7 else 8, quality_metric)
    elif task == 'ranking':
        relevance_fields = [c for c in df.columns if any((k in norm_model(c) for k in ('relevance', 'impression', 'click', 'engagement', 'rating', 'rank', 'position', 'score', 'conversion', 'view', 'like', 'purchase')))]
        user_cols = _columns_by_role(df, ('user', 'customer', 'member', 'account'))
        item_cols = _columns_by_role(df, ('item', 'product', 'post', 'content', 'document', 'movie', 'listing'))
        group_cols = _columns_by_role(df, ('query', 'group', 'session', 'user', 'customer'))
        relevance_points = 25.0 if target_exists and target.get('inferred_type') == 'numeric' and (int(target.get('unique_values', 0)) >= 5) else 18.0 if relevance_fields else 5.0
        group_points = 25.0 if user_cols and df[user_cols[0]].nunique(dropna=True) < rows * 0.8 else 18.0 if group_cols else 5.0
        item_points = 20.0 if item_cols else 5.0
        interaction_points = 15.0 if len(relevance_fields) >= 2 else 10.0 if relevance_fields else 4.0
        data_points = 15.0 if quality_points >= 9 else 10.0 if quality_points >= 7 else 5.0
        add('Relevance / target signal', 25, relevance_points, f"{(target['column'] if target_exists else ', '.join(relevance_fields[:3]) if relevance_fields else 'No obvious relevance field')}", 'Assessed' if target_exists or relevance_fields else 'Partial')
        add('Query / group structure', 25, group_points, f"{(', '.join(group_cols[:3]) if group_cols else 'No obvious group/query field')}")
        add('Item / entity structure', 20, item_points, f"{(', '.join(item_cols[:3]) if item_cols else 'No obvious item/entity field')}")
        add('Interaction / engagement evidence', 15, interaction_points, f"{(', '.join(relevance_fields[:4]) if relevance_fields else 'No obvious interaction field')}")
        add('Dataset quality and size', 15, data_points, quality_metric + f'; {rows:,} rows')
        if not target_exists:
            findings.append({'severity': 'info', 'type': 'ranking_no_target', 'message': 'Ranking can be assessed from structure without a target, but the relevance signal cannot be validated against a selected target.'})
            recommendations.append('Provide a relevance or engagement target when available to strengthen ranking assessment.')
        if not group_cols:
            findings.append({'severity': 'medium', 'type': 'ranking_group_structure', 'message': 'No obvious query, session, user or customer grouping field was detected.'})
            recommendations.append('Verify that multiple candidate items can be compared within the same group or query.')
        if not item_cols:
            findings.append({'severity': 'medium', 'type': 'ranking_item_structure', 'message': 'No obvious item, content, product or post identifier was detected.'})
            recommendations.append('Verify that rankable items can be identified consistently.')
    elif task == 'recommendation':
        user_cols = _columns_by_role(df, ('user', 'customer', 'member', 'account'))
        item_cols = _columns_by_role(df, ('item', 'product', 'post', 'content', 'movie', 'listing'))
        interaction_cols = [c for c in df.columns if any((k in norm_model(c) for k in ('click', 'view', 'rating', 'purchase', 'impression', 'engagement', 'like', 'watch', 'conversion')))]
        add('User / customer structure', 25, 25 if user_cols else 5, f"{(', '.join(user_cols[:3]) if user_cols else 'Not detected')}")
        add('Item / content structure', 25, 25 if item_cols else 5, f"{(', '.join(item_cols[:3]) if item_cols else 'Not detected')}")
        add('Interaction evidence', 30, 30 if interaction_cols else 5, f"{(', '.join(interaction_cols[:4]) if interaction_cols else 'Not detected')}")
        add('Feature compatibility', 10, 10 if numeric or categorical else 0, f'{len(numeric)} numeric; {len(categorical)} categorical')
        add('Dataset size', 10, _size_points(rows, 10), f'{rows:,} rows')
        if not user_cols:
            findings.append({'severity': 'high', 'message': 'No obvious user/customer identifier was detected for recommendation analysis.'})
        if not item_cols:
            findings.append({'severity': 'high', 'message': 'No obvious item/content identifier was detected for recommendation analysis.'})
        if not interaction_cols:
            findings.append({'severity': 'high', 'message': 'No obvious interaction or engagement signal was detected.'})
    elif task == 'clustering':
        usable = [c for c in numeric if df[c].nunique(dropna=True) > 1]
        varied = sum((df[c].nunique(dropna=True) > 5 for c in usable))
        add('Usable numeric features', 30, 30 if len(usable) >= 5 else 23 if len(usable) >= 2 else 5, f'{len(usable)} usable numeric feature(s)')
        add('Feature variation', 20, 20 if varied >= 3 else 12 if varied >= 2 else 5, f'{varied} varied numeric feature(s)')
        add('Missingness', 20, 20 if df.isna().mean().mean() * 100 <= 2 else 14 if df.isna().mean().mean() * 100 <= 10 else 5, f'{df.isna().mean().mean() * 100:.2f}% average missingness')
        add('Categorical burden', 15, 15 if len(categorical) <= 2 else 8, f'{len(categorical)} categorical feature(s)')
        add('Dataset size', 15, _size_points(rows, 15), f'{rows:,} rows')
        if len(usable) < 2:
            findings.append({'severity': 'high', 'message': 'Fewer than two usable numeric features were detected for clustering.'})
        recommendations.append('Consider scaling numeric features for distance-based clustering algorithms.')
    elif task == 'anomaly':
        usable = [c for c in numeric if df[c].nunique(dropna=True) > 1]
        varied = sum((df[c].nunique(dropna=True) > 5 for c in usable))
        outlier_columns = 0
        for c in usable:
            x = df[c].dropna()
            if len(x) >= 10:
                q1, q3 = (x.quantile(0.25), x.quantile(0.75))
                iqr = q3 - q1
                if iqr > 0 and int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum()) > 0:
                    outlier_columns += 1
        add('Usable feature space', 30, 30 if len(usable) >= 5 else 22 if len(usable) >= 2 else 6, f'{len(usable)} usable numeric feature(s)')
        add('Feature variation', 20, 20 if varied >= 3 else 12 if varied >= 2 else 5, f'{varied} varied numeric feature(s)')
        add('Outlier structure', 20, 20 if outlier_columns >= 2 else 14 if outlier_columns == 1 else 8, f'{outlier_columns} numeric feature(s) contain IQR outliers')
        add('Missingness', 15, 15 if df.isna().mean().mean() * 100 <= 5 else 8, f'{df.isna().mean().mean() * 100:.2f}% average missingness')
        add('Dataset size', 15, _size_points(rows, 15), f'{rows:,} rows')
        recommendations.append('Review scaling and missing-value handling before anomaly detection.')
    elif task == 'time-series':
        has_date = bool(date_cols)
        target_numeric = target_exists and target.get('inferred_type') == 'numeric'
        add('Temporal index', 35, 35 if has_date else 5, f"{(', '.join(date_cols[:3]) if date_cols else 'No date-like column')}")
        add('Forecast target compatibility', 25, 25 if target_numeric else 12 if target_exists else 7, f"{(target.get('inferred_type') if target_exists else 'Target not supplied')}", 'Assessed' if target_exists else 'Partial')
        add('Temporal coverage', 15, 15 if has_date and df[date_cols[0]].nunique(dropna=True) >= 20 else 8 if has_date else 0, f'{(int(df[date_cols[0]].nunique(dropna=True)) if has_date else 0)} distinct time values')
        add('Missingness', 15, 15 if df.isna().mean().mean() * 100 <= 5 else 8, f'{df.isna().mean().mean() * 100:.2f}% average')
        add('Dataset size', 10, _size_points(rows, 10), f'{rows:,} rows')
        if not has_date:
            findings.append({'severity': 'high', 'message': 'No date-like feature was detected for the selected time-series task.'})
            recommendations.append('Provide a timestamp or date column for time-series modelling.')
    elif task == 'dimensionality-reduction':
        usable = [c for c in numeric if df[c].nunique(dropna=True) > 1]
        add('Feature dimensionality', 30, 30 if len(usable) >= 10 else 24 if len(usable) >= 5 else 15 if len(usable) >= 2 else 4, f'{len(usable)} usable numeric dimensions')
        add('Feature variation', 20, 20 if len(usable) >= 5 else 10 if len(usable) >= 2 else 3, 'Non-constant numeric dimensions')
        add('Missingness', 20, 20 if df.isna().mean().mean() * 100 <= 2 else 14 if df.isna().mean().mean() * 100 <= 10 else 5, f'{df.isna().mean().mean() * 100:.2f}% average')
        add('Redundancy potential', 15, 15 if len(usable) >= 5 else 8, 'Multiple numeric dimensions available')
        add('Dataset size', 15, _size_points(rows, 15), f'{rows:,} rows')
        if len(usable) < 2:
            findings.append({'severity': 'high', 'message': 'Fewer than two usable numeric dimensions were detected.'})
    else:
        add('Task-specific evidence', 40, 12, 'No dedicated rubric for this task', 'Partial')
        add('Feature compatibility', 30, 20 if numeric or categorical else 0, f'{len(numeric)} numeric; {len(categorical)} categorical')
        add('Dataset size', 15, _size_points(rows, 15), f'{rows:,} rows')
        add('General data quality', 15, quality_points * 1.5, quality_metric)
        findings.append({'severity': 'medium', 'type': 'unsupported_task_rubric', 'message': f'No dedicated scoring rubric is implemented for "{task}"; the task score is intentionally conservative.'})
    score = max(0, min(100, int(round(sum((p['points_awarded'] for p in parameters))))))
    result = {
        'provided': True, 'task': task, 'status': _task_status(score), 'score': score,
        'summary': f'The CSV was scored for how well its observed structure and available information complement the selected {TASK_DISPLAY.get(task, task)} task.',
        'parameters': parameters, 'findings': findings, 'recommendations': list(dict.fromkeys(recommendations)),
        'target_required': task in SUPERVISED, 'target_available': target_exists,
    }
    if compare_tasks:
        comparisons = []
        for candidate_task in SUPPORTED_TASKS:
            if candidate_task == task:
                continue
            candidate = task_audit(df, candidate_task, target_info, date_cols, compare_tasks=False)
            comparisons.append({'task': candidate_task, 'display': TASK_DISPLAY.get(candidate_task, candidate_task), 'score': int(candidate.get('score') or 0), 'status': candidate.get('status', '')})
        comparisons.sort(key=lambda x: x['score'], reverse=True)
        result['task_comparisons'] = comparisons
        best = comparisons[0] if comparisons else None
        if best and best['score'] >= 60 and best['score'] >= score + 10:
            result['better_task'] = {
                'task': best['task'], 'display': best['display'], 'score': best['score'],
                'current_task': task, 'current_display': TASK_DISPLAY.get(task, task), 'current_score': score,
                'difference': best['score'] - score,
                'message': (f'Based on the CSV structure, {best["display"]} currently complements this dataset more strongly than the selected {TASK_DISPLAY.get(task, task)} task ({best["score"]}/100 vs {score}/100). This is a data-driven alternative suggestion and does not replace your selected task.')
            }
        else:
            result['better_task'] = None
    else:
        result['task_comparisons'] = []
        result['better_task'] = None
    return result

def model_audit(df: pd.DataFrame, model: str, task: str, target_info: dict[str, Any], date_cols: list[str], s: dict[str, Any]) -> dict[str, Any]:
    if not model:
        return {'provided': False, 'status': 'Model not provided', 'score': None, 'parameters': [], 'findings': [], 'recommendations': []}
    normalized = norm_model(model)
    implied = MODEL_TASK.get(normalized)
    effective = task or implied
    findings, recs, params = ([], [], [])
    earned = assessed = 0

    def add_param(name: str, weight: int, points: float, metric: str, status: str='Assessed'):
        nonlocal earned, assessed
        pts = max(0.0, min(float(weight), points))
        if status in {'Assessed', 'Partial'}:
            earned += pts
            assessed += weight
        params.append({'parameter': name, 'weight': weight, 'points_awarded': round(pts, 2), 'metric': metric, 'status': status})
    target_exists = bool(target_info.get('exists'))
    target_column = target_info.get('analysis', {}).get('column') if target_exists else None
    feature_df = df.drop(columns=[target_column]) if target_column in df.columns else df
    feature_categorical = [c for c in s['categorical_columns'] if c != target_column]
    feature_numeric = [c for c in s['numeric_columns'] if c != target_column]
    feature_high_card = [c for c in s['high_cardinality_columns'] if c != target_column]
    feature_id_like = [c for c in s['id_like_columns'] if c != target_column]
    categorical = len(feature_categorical)
    numeric = len(feature_numeric)
    rows = len(df)
    user_cols = _columns_by_role(feature_df, ('user', 'customer', 'member', 'account'))
    item_cols = _columns_by_role(feature_df, ('item', 'product', 'post', 'content', 'document', 'movie', 'listing'))
    group_cols = _columns_by_role(feature_df, ('query', 'group', 'session', 'user', 'customer'))
    interaction_cols = [c for c in feature_df.columns if any((k in norm_model(c) for k in ('click', 'view', 'rating', 'purchase', 'impression', 'engagement', 'like', 'watch', 'conversion', 'relevance')))]
    relevance_cols = [c for c in feature_df.columns if any((k in norm_model(c) for k in ('relevance', 'impression', 'click', 'engagement', 'rating', 'rank', 'position', 'score', 'conversion', 'view', 'like', 'purchase')))]
    if implied and task:
        ok = implied == task
        add_param('Task-model compatibility', 20, 20 if ok else 0, f'Model maps to {implied}; selected task is {task}')
        if not ok:
            findings.append({'severity': 'high', 'type': 'task_model_mismatch', 'message': f'''"{model}" is associated with {implied.replace('-', ' ')} while the selected task is {task.replace('-', ' ')}.'''})
    elif implied:
        add_param('Task-model compatibility', 20, 20, f'Task inferred as {implied}')
    else:
        add_param('Task-model compatibility', 20, 10, 'Model-task mapping not available', 'Partial')
        findings.append({'severity': 'medium', 'type': 'unknown_model_mapping', 'message': f'No exact task mapping was found for "{model}"; structural checks were still performed.'})
    if effective == 'recommendation':
        user_pts = 8 if user_cols else 0
        item_pts = 8 if item_cols else 0
        interaction_pts = 9 if interaction_cols or target_info.get('exists') else 0
        feature_points = user_pts + item_pts + interaction_pts
        feature_metric = f"User/customer: {(', '.join(user_cols[:2]) if user_cols else 'none')}; item/content: {(', '.join(item_cols[:2]) if item_cols else 'none')}; interaction/relevance: {(', '.join(interaction_cols[:2]) if interaction_cols else 'none')}"
        add_param('Recommendation structure', 25, feature_points, feature_metric, 'Assessed' if any((user_cols, item_cols, interaction_cols, target_info.get('exists'))) else 'Partial')
        if not user_cols:
            recs.append('A user/customer identifier would strengthen personalized recommendation modelling.')
        if not item_cols:
            recs.append('An item/content identifier is needed to define recommendation candidates.')
        if not interaction_cols and (not target_info.get('exists')):
            recs.append('An interaction or relevance signal would strengthen recommendation modelling.')
    elif effective == 'ranking':
        target_ok = bool(target_info.get('exists') and target_info['analysis'].get('inferred_type') == 'numeric' and (target_info['analysis'].get('unique_values', 0) >= 5))
        relevance_pts = 10 if target_ok else 7 if target_info.get('exists') or relevance_cols else 2
        group_pts = 8 if group_cols else 0
        item_pts = 7 if item_cols else 0
        feature_points = relevance_pts + group_pts + item_pts
        feature_metric = f"Relevance/target: {('available' if target_ok or target_info.get('exists') else 'none')}; group/query: {(', '.join(group_cols[:2]) if group_cols else 'none')}; item/entity: {(', '.join(item_cols[:2]) if item_cols else 'none')}"
        add_param('Ranking structure', 25, feature_points, feature_metric, 'Assessed' if any((target_info.get('exists'), relevance_cols, group_cols, item_cols)) else 'Partial')
        if not group_cols:
            recs.append('Verify that multiple candidate items can be compared within the same query, user, session or group.')
        if not item_cols:
            recs.append('Verify that rankable items can be identified consistently.')
        if not target_info.get('exists') and (not relevance_cols):
            recs.append('Provide a relevance or engagement target to strengthen ranking-specific model assessment.')
    else:
        feature_penalty = min(25, (6 if categorical else 0) + (5 if feature_high_card else 0) + (5 if feature_id_like else 0))
        add_param('Feature compatibility', 25, 25 - feature_penalty, f"{categorical} categorical, {len(feature_high_card)} high-cardinality, {len(feature_id_like)} ID-like predictor feature(s)")
        if categorical:
            recs.append('Encode categorical features or use an implementation that explicitly supports them.')
        if feature_high_card:
            recs.append('Review high-cardinality predictor features before training.')
    quality_points = 25 - min(25, s['duplicate_percentage'] / 50 * 15 + sum(s['missing_percentages'].values()) / max(s['columns'], 1) / 10 * 5 + s['outlier_rate'] / 10 * 5)
    add_param('Dataset readiness', 25, quality_points, f"{s['duplicate_percentage']:.2f}% duplicates; {sum(s['missing_percentages'].values()) / max(s['columns'], 1):.2f}% average missingness; {s['outlier_rate']:.2f}% numeric outliers")
    if s['duplicate_percentage'] >= 20:
        findings.append({'severity': 'medium', 'type': 'duplicates', 'message': 'The duplicate rate may affect training and validation quality.'})
    if s['missing_columns']:
        recs.append('Handle missing values before training.')
    scaling_models = {'k-nearest neighbors (knn)', 'support vector machine (svm)', 'logistic regression', 'linear regression', 'ridge regression', 'lasso regression', 'elastic net', 'k-means', 'dbscan'}
    scaling_need = normalized in scaling_models
    add_param('Preprocessing fit', 15, 12 if scaling_need else 15, 'Scaling/encoding considerations are reviewed')
    if scaling_need:
        recs.append('Consider feature scaling because this model or algorithm is sensitive to feature magnitude or distances.')
    if effective == 'time-series' and (not date_cols):
        findings.append({'severity': 'high', 'type': 'no_date', 'message': f'"{model}" is being assessed for a time-series workflow, but no date-like column was found.'})
    if effective == 'regression' and target_info.get('exists'):
        ok = target_info['analysis']['inferred_type'] == 'numeric'
        add_param('Target compatibility', 10, 10 if ok else 0, f"Target inferred as {target_info['analysis']['inferred_type']}")
        if not ok:
            findings.append({'severity': 'high', 'type': 'bad_target_type', 'message': f'''The target "{target_info['analysis']['column']}" is not suitable for direct regression use.'''})
    elif effective == 'classification' and target_info.get('exists'):
        a = target_info['analysis']
        classes = int(a.get('class_count', a.get('unique_values', 0)))
        ok = a.get('inferred_type') == 'categorical' and 2 <= classes <= 50
        add_param('Target compatibility', 10, 10 if ok else 2, f'{classes} usable classes')
        if not ok:
            findings.append({'severity': 'high', 'type': 'bad_target', 'message': f'The target contains {classes} usable classes or is not categorical; review classification suitability.'})
    elif effective not in {'recommendation', 'ranking'}:
        add_param('Target compatibility', 10, 10, 'Not required or target not supplied', 'Not applicable')
    else:
        pass
    score = int(round(earned / assessed * 100)) if assessed else 0
    status = 'Strong model-data alignment' if score >= 80 else 'Review model-data alignment' if score >= 60 else 'Significant model-data concerns'
    return {'provided': True, 'model': model, 'inferred_task': implied, 'effective_task': effective, 'score': max(0, min(100, score)), 'status': status, 'parameters': params, 'findings': findings, 'recommendations': list(dict.fromkeys(recs)), 'summary': f'Model suitability for "{model}" was assessed against the selected task, target information when available, and the measured dataset structure.'}
MODEL_CANDIDATES = {'classification': ['Random Forest', 'Gradient Boosting', 'XGBoost', 'Logistic Regression', 'Decision Tree', 'K-Nearest Neighbors (KNN)', 'Naive Bayes', 'Extra Trees'], 'regression': ['Random Forest Regressor', 'Gradient Boosting Regressor', 'XGBoost Regressor', 'Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Support Vector Regression (SVR)', 'K-Nearest Neighbors Regressor'], 'clustering': ['K-Means', 'DBSCAN', 'Hierarchical Clustering', 'Gaussian Mixture Model', 'Mean Shift', 'Spectral Clustering'], 'anomaly': ['Isolation Forest', 'Local Outlier Factor', 'One-Class SVM', 'Autoencoder'], 'dimensionality-reduction': ['PCA', 'Kernel PCA', 'Truncated SVD', 'UMAP', 't-SNE', 'LDA'], 'time-series': ['ARIMA', 'SARIMA', 'Exponential Smoothing', 'Prophet', 'Random Forest (Forecasting)', 'XGBoost (Forecasting)', 'LSTM'], 'recommendation': ['Content-Based Filtering', 'Collaborative Filtering', 'Matrix Factorization', 'K-Nearest Neighbors', 'Hybrid Recommendation'], 'ranking': ['Learning to Rank', 'RankNet', 'LambdaMART', 'XGBoost Ranking']}

def suggestions(df: pd.DataFrame, target_info: dict[str, Any], task: str | None) -> list[dict[str, Any]]:
    """
    Score model candidates across ALL supported ML tasks.

    The selected task is not used as a filter. This allows Data Audit to
    discover strong alternatives from other ML task families.

    Each candidate receives:
      - task_score: how well the CSV complements the task
      - model_score: how well the specific model complements the CSV + task
      - score: combined task/model fit score
    """
    date_cols = date_like_columns(df)
    structure = structural_analysis(df, date_cols)
    target_column = target_info.get('analysis', {}).get('column', '') if target_info.get('exists') else ''
    leakage = leakage_analysis(df, target_column, structure)
    results: list[dict[str, Any]] = []
    for candidate_task, candidate_models in MODEL_CANDIDATES.items():
        task_info = task_audit(df, candidate_task, target_info, date_cols, compare_tasks=False)
        task_score = int(task_info.get('score') or 0)
        for candidate_model in candidate_models:
            model_info = model_audit(df, candidate_model, candidate_task, target_info, date_cols, structure)
            model_score = int(model_info.get('score') or 0)
            combined_score = int(round(task_score * 0.55 + model_score * 0.45))
            selected_note = ''
            if task and candidate_task == task:
                selected_note = ' Matches the task you selected.'
            elif task and candidate_task != task:
                selected_note = ' Alternative task/model combination for this dataset.'
            if candidate_task == 'classification':
                reason = 'Classification is supported when the dataset has a valid categorical target. The model score also considers feature types, missingness and dataset readiness.'
            elif candidate_task == 'regression':
                reason = 'Regression is supported when the target is numeric and sufficiently varied. The model score also considers feature types, missingness and dataset readiness.'
            elif candidate_task == 'ranking':
                reason = 'Ranking is assessed using relevance/engagement signals together with query/group and item/entity structure.'
            elif candidate_task == 'recommendation':
                reason = 'Recommendation is assessed using user/customer, item/content and interaction structure in the CSV.'
            elif candidate_task == 'clustering':
                reason = 'Clustering is assessed from usable numeric feature space, variation, missingness and dataset size.'
            elif candidate_task == 'anomaly':
                reason = 'Anomaly detection is assessed from usable feature space, variation and observable outlier structure.'
            elif candidate_task == 'time-series':
                reason = 'Time-series suitability is assessed from temporal fields, temporal coverage and forecast-target compatibility.'
            else:
                reason = 'Dimensionality reduction is assessed from usable feature dimensionality, variation, missingness and dataset size.'
            results.append({'model': candidate_model, 'task': candidate_task, 'task_score': task_score, 'model_score': model_score, 'score': combined_score, 'reason': f'Task fit: {task_score}/100. Model fit: {model_score}/100. {reason}{selected_note}'})
    results.sort(key=lambda x: (x['score'], x['task_score'], x['model_score']), reverse=True)
    selected: list[dict[str, Any]] = []
    task_counts: dict[str, int] = {}
    for item in results:
        count = task_counts.get(item['task'], 0)
        if count >= 2:
            continue
        selected.append(item)
        task_counts[item['task']] = count + 1
        if len(selected) >= 10:
            break
    if selected:
        selected[0]['is_best'] = True
        selected[0]['best_choice_label'] = 'Best-supported task + model combination for this dataset'
    for item in selected[1:]:
        item['is_best'] = False
        item['best_choice_label'] = None
    return selected

def context_audit(intended_use: str, description: str, target: str, task: str, model: str, date_cols: list[str]) -> dict[str, Any]:
    if not any((intended_use, description)):
        return {'provided': False, 'score': None, 'summary': 'No additional context was provided.', 'findings': [], 'recommendations': []}
    score, findings, recommendations = (100, [], [])
    text = f'{intended_use} {description}'.lower()
    if intended_use:
        temporal_words = ('forecast', 'next month', 'future', 'tomorrow', 'over time', 'next week')
        if any((x in intended_use.lower() for x in temporal_words)) and (not date_cols):
            score -= 25
            findings.append({'severity': 'medium', 'type': 'context_time_mismatch', 'message': 'The intended use suggests a temporal prediction scenario, but no date-like field was detected.'})
        findings.append({'severity': 'info', 'type': 'intended_use', 'message': f'Intended use used in the audit: "{intended_use}".'})
        recommendations.append('Confirm that every feature used for the intended decision would be available at prediction time.')
    if description:
        findings.append({'severity': 'info', 'type': 'description', 'message': 'The dataset description was used as context when interpreting the selected task, target and intended use.'})
    if target and target.lower() in text:
        findings.append({'severity': 'info', 'type': 'target_context_match', 'message': f'The supplied context explicitly mentions the target "{target}".'})
    if task:
        keywords = {'classification': ('classif', 'category', 'label', 'class'), 'regression': ('regress', 'predict', 'value', 'price', 'rate', 'amount'), 'clustering': ('cluster', 'segment', 'group'), 'anomaly': ('anomaly', 'outlier', 'unusual'), 'time-series': ('forecast', 'future', 'next month', 'over time')}.get(task, ())
        if keywords and any((k in text for k in keywords)):
            findings.append({'severity': 'info', 'type': 'task_context_alignment', 'message': f'The description/intended use contains language consistent with the selected {TASK_DISPLAY.get(task, task)} task.'})
        elif keywords:
            score -= 10
            findings.append({'severity': 'medium', 'type': 'task_context_unclear', 'message': 'The provided description/intended use does not clearly mention language associated with the selected ML task; this is a context warning, not proof of a mismatch.'})
    score = max(0, score)
    return {'provided': True, 'score': score, 'context': {'intended_use': intended_use or None, 'description': description or None}, 'findings': findings, 'recommendations': recommendations, 'summary': 'The supplied intended use and dataset description were actively incorporated into the audit context.'}

def column_details(df: pd.DataFrame, s: dict[str, Any]) -> list[dict[str, Any]]:
    details = []
    for c in df.columns:
        values = df[c].dropna()
        unique = int(df[c].nunique(dropna=True))
        repeated = max(0, len(values) - unique)
        samples = []
        for v in values.head(5).tolist():
            text = str(v)
            samples.append(text if len(text) <= 36 else text[:33] + '...')
        details.append({'column': c, 'dtype': s['data_types'][c], 'missing_count': int(df[c].isna().sum()), 'missing_percentage': float(s['missing_percentages'][c]), 'unique_count': unique, 'unique_percentage': round(unique / len(df) * 100, 2) if len(df) else 0, 'repeated_value_count': repeated, 'repeated_value_percentage': round(repeated / len(values) * 100, 2) if len(values) else 0, 'outlier_count': int(s['outlier_counts'].get(c, 0)), 'outlier_percentage': float(s['outlier_percentages'].get(c, 0)), 'is_constant': c in s['constant_columns'], 'is_id_like': c in s['id_like_columns'], 'is_high_cardinality': c in s['high_cardinality_columns'], 'is_date_like': c in s['date_like_columns'], 'sample_values': samples})
    return details

def final_assessment(q: dict[str, Any], target_info: dict[str, Any], leakage: dict[str, Any], task_info: dict[str, Any], model_info: dict[str, Any], context_info: dict[str, Any], s: dict[str, Any], suggested: list[dict[str, Any]]) -> dict[str, Any]:
    findings, recs = ([], [])
    if q['score'] < 75:
        findings.append({'severity': 'high' if q['score'] < 50 else 'medium', 'type': 'quality_score', 'message': f"Dataset quality is classified as '{q.get('classification', 'Needs cleaning')}' ({q['score']}/100). Review the quality findings before relying on this dataset."})
    elif q['score'] < 90:
        findings.append({'severity': 'low', 'type': 'quality_score', 'message': f"Dataset quality is classified as '{q.get('classification', 'Good, with issues to review')}' ({q['score']}/100). Review the detected issues before modelling."})
    if s['duplicate_rows']:
        recs.append(f"Investigate the {s['duplicate_rows']:,} duplicate rows before training.")
    if s['missing_columns']:
        recs.append('Review column-level missingness and choose an appropriate preprocessing strategy.')
    if target_info.get('exists') and target_info['analysis']['missing_values']:
        recs.append(f'''Handle missing values in target "{target_info['analysis']['column']}" before supervised training.''')
    if leakage.get('target_findings'):
        recs.append('Investigate the leakage signals listed in the report and verify feature availability at prediction time.')
    if task_info.get('provided'):
        recs.extend(task_info.get('recommendations', []))
    if model_info.get('provided'):
        recs.extend(model_info.get('recommendations', []))
    recs.extend(context_info.get('recommendations', []))
    recs = list(dict.fromkeys(recs))
    high = sum((1 for x in leakage.get('findings', []) if x.get('severity') == 'high')) + sum((1 for x in task_info.get('findings', []) if x.get('severity') == 'high')) + sum((1 for x in model_info.get('findings', []) if x.get('severity') == 'high'))
    status = 'Needs attention' if high or q['score'] < 50 else 'Review recommended' if q['score'] < 90 or leakage['score'] >= 30 else 'No major issues detected by the audit'
    parts = [f"The dataset contains {s['rows']:,} rows and {s['columns']} columns and received a deterministic quality score of {q['score']}/100."]
    if target_info.get('exists'):
        parts.append(f'''The supplied target "{target_info['analysis']['column']}" was found, so target-specific analysis was performed.''')
    elif target_info.get('provided'):
        parts.append('A target was supplied but was not found, so target-specific checks were not performed.')
    else:
        parts.append('No target was supplied, so target-dependent checks were not performed.')
    if task_info.get('provided'):
        parts.append(f"The selected task was {TASK_DISPLAY.get(task_info['task'], task_info['task'])} and received a {task_info['score']}/100 structural fit score.")
        if task_info.get('better_task'):
            bt = task_info['better_task']
            parts.append(f'{bt["display"]} has stronger dataset-structure evidence at {bt["score"]}/100, which is {bt["difference"]} points above the selected task.')
    if model_info.get('provided'):
        parts.append(f'''The selected model "{model_info['model']}" received a {model_info['score']}/100 fit score based on the evidence available.''')
    if context_info.get('provided'):
        parts.append('The intended use and dataset description were incorporated into the contextual assessment.')
    if suggested:
        best = suggested[0]
        parts.append(f'''The highest-scoring task/model combination in this dataset-level screening was "{best['task']} + {best['model']}" with a combined fit score of {best['score']}/100.''')
    return {'status': status, 'quality_classification': q.get('classification'), 'quality_classification_description': q.get('classification_description'), 'summary': ' '.join(parts), 'findings': findings, 'recommendations': recs, 'suggested_models': suggested, 'best_recommendation': suggested[0] if suggested else None}

@app.get('/')
def root():
    return {'name': 'Data Audit', 'status': 'running'}

@app.get('/health')
def health():
    return {'status': 'healthy'}

@app.post('/api/audit')
async def audit_dataset(file: UploadFile=File(...), target: str=Form(''), task: str=Form(''), model: str=Form(''), intended_use: str=Form(''), description: str=Form('')):
    if not file.filename or not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail='Please upload a CSV file.')
    try:
        data = await file.read()
        df = pd.read_csv(BytesIO(data))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f'Unable to read this CSV. Check its encoding, delimiter, and file contents. Details: {exc}')
    if df.empty or df.shape[1] == 0:
        raise HTTPException(status_code=400, detail='The CSV contains no usable data.')
    target, task, model = (clean(target), clean(task).lower(), clean(model))
    intended_use, description = (clean(intended_use), clean(description))
    date_cols = date_like_columns(df)
    s = structural_analysis(df, date_cols)
    q = quality_analysis(s, df)
    tinfo = target_analysis(df, target, task)
    effective_task = infer_task(df, tinfo, task, model)
    leakage = leakage_analysis(df, target, s)
    task_info = task_audit(df, task, tinfo, date_cols, compare_tasks=bool(task))
    model_info = model_audit(df, model, effective_task or '', tinfo, date_cols, s)
    context_info = context_audit(intended_use, description, target, effective_task or '', model, date_cols)
    suggested = suggestions(df, tinfo, effective_task)
    final = final_assessment(q, tinfo, leakage, task_info, model_info, context_info, s, suggested)
    scope = ['Dataset quality', 'Structural analysis', 'Basic leakage screening', 'Column-level statistics']
    if target:
        scope.append('Target analysis')
    if target in df.columns and target:
        scope.append('Target-dependent leakage screening')
    if task:
        scope.append('ML task suitability')
        if task_info.get('better_task'):
            scope.append('Alternative task comparison')
    elif model and effective_task:
        scope.append('Model-associated task inference')
    if model:
        scope.append('Model suitability')
    if intended_use or description:
        scope.append('Intended-use and description context')
    if suggested:
        scope.append('Model suggestions')
    return {'status': 'success', 'audit_scope': scope, 'dataset': {'filename': file.filename, 'rows': s['rows'], 'columns': s['columns'], 'column_names': df.columns.tolist(), 'data_types': s['data_types']}, 'quality': {'score': q['score'], 'classification': q.get('classification'), 'classification_description': q.get('classification_description'), 'classification_bands': q.get('classification_bands'), 'method': q['method'], 'formula': q['formula'], 'parameters': q['parameters'], 'checked_for': q['checked_for'], 'duplicate_rows': s['duplicate_rows'], 'duplicate_percentage': s['duplicate_percentage'], 'columns_with_missing_values': s['missing_columns'], 'constant_columns': s['constant_columns'], 'numeric_columns': s['numeric_columns'], 'categorical_columns': s['categorical_columns'], 'date_like_columns': s['date_like_columns'], 'id_like_columns': s['id_like_columns'], 'high_cardinality_columns': s['high_cardinality_columns'], 'outliers': s['outlier_counts'], 'quality_details': q.get('details', {})}, 'missing_values': {'counts': s['missing_counts'], 'percentages': s['missing_percentages']}, 'column_details': column_details(df, s), 'target': tinfo, 'task': {'provided': bool(task), 'value': task or None}, 'task_analysis': task_info, 'model': {'provided': bool(model), 'value': model or None}, 'model_analysis': model_info, 'intended_use': intended_use or None, 'description': description or None, 'context_analysis': context_info, 'leakage': leakage, 'suggested_models': suggested, 'final_assessment': final}
