from typing import Any

import pandas as pd


# =========================================================
# CONTEXT ENGINE
# =========================================================

def run_context_audit(
    df: pd.DataFrame,
    target: str = "",
    task: str = "",
    model: str = "",
    intended_use: str = "",
    description: str = "",
) -> dict[str, Any]:

    target = target.strip()
    task = task.strip()
    model = model.strip()
    intended_use = intended_use.strip()
    description = description.strip()

    findings = []
    recommendations = []

    # -----------------------------------------------------
    # No context supplied
    # -----------------------------------------------------

    if not any([
        target,
        task,
        model,
        intended_use,
        description,
    ]):

        return {
            "provided": False,
            "summary": (
                "No additional user context was provided. "
                "The audit is based on the uploaded dataset."
            ),
            "findings": [],
            "recommendations": [],
        }

    # -----------------------------------------------------
    # Intended use
    # -----------------------------------------------------

    if intended_use:

        findings.append({
            "type": "intended_use_recorded",
            "severity": "info",
            "message": (
                f'The intended use provided by the user is: '
                f'"{intended_use}".'
            ),
        })

        recommendations.append(
            "Confirm that every feature used for modelling "
            "would be available at the time the intended "
            "prediction or decision is made."
        )

    # -----------------------------------------------------
    # Dataset description
    # -----------------------------------------------------

    if description:

        findings.append({
            "type": "dataset_description_recorded",
            "severity": "info",
            "message": (
                "A dataset description was provided and "
                "has been included as audit context."
            ),
        })

    # -----------------------------------------------------
    # Target context
    # -----------------------------------------------------

    if target:

        if target in df.columns:

            findings.append({
                "type": "target_context_valid",
                "severity": "info",
                "message": (
                    f'The stated target "{target}" '
                    "exists in the uploaded dataset."
                ),
            })

        else:

            findings.append({
                "type": "target_context_invalid",
                "severity": "high",
                "message": (
                    f'The stated target "{target}" '
                    "does not exist in the uploaded dataset."
                ),
            })

    # -----------------------------------------------------
    # Task context
    # -----------------------------------------------------

    if task:

        findings.append({
            "type": "task_context_recorded",
            "severity": "info",
            "message": (
                f'The selected machine-learning task '
                f'is "{task}".'
            ),
        })

    # -----------------------------------------------------
    # Model context
    # -----------------------------------------------------

    if model:

        findings.append({
            "type": "model_context_recorded",
            "severity": "info",
            "message": (
                f'The selected model or algorithm '
                f'is "{model}".'
            ),
        })

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    return {
        "provided": True,

        "summary": (
            "The additional information supplied by the "
            "user was incorporated into the audit context."
        ),

        "context": {
            "target": target or None,
            "task": task or None,
            "model": model or None,
            "intended_use": intended_use or None,
            "description": description or None,
        },

        "findings": findings,

        "recommendations": recommendations,
    }