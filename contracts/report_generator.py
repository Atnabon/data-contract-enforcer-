"""
ReportGenerator — produces the auto-generated Enforcer Report.

Reads from violation_log/ and validation_reports/ to construct a
data-driven report with:
  1. Data Health Score (0-100)
  2. Violations this week (by severity, plain language)
  3. Schema changes detected
  4. AI system risk assessment
  5. Recommended actions (specific file + clause references)

Usage:
    python contracts/report_generator.py \
        --output enforcer_report/report_data.json
"""
import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

SEVERITY_DEDUCTIONS = {"CRITICAL": 20, "HIGH": 10, "MEDIUM": 5, "LOW": 1, "WARNING": 2}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_validation_reports(reports_dir: str = "validation_reports/") -> List[Dict]:
    """Load all JSON validation reports."""
    reports = []
    rdir = Path(reports_dir)
    if not rdir.exists():
        return reports
    for p in sorted(rdir.glob("*.json")):
        try:
            with open(p) as f:
                data = json.load(f)
            # Only load files that look like validation reports
            if "results" in data and "total_checks" in data:
                data["_source_file"] = str(p)
                reports.append(data)
        except (json.JSONDecodeError, KeyError):
            pass
    return reports


def load_violation_log(log_path: str = "violation_log/violations.jsonl") -> List[Dict]:
    """Load violation log entries."""
    violations = []
    p = Path(log_path)
    if not p.exists():
        return violations
    with open(p) as f:
        for line in f:
            if line.strip():
                try:
                    violations.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return violations


def load_ai_extensions(ai_path: str = "validation_reports/ai_extensions.json") -> Dict:
    """Load AI extension results."""
    p = Path(ai_path)
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def load_schema_evolution(evo_path: str = "validation_reports/schema_evolution.json") -> Dict:
    """Load schema evolution report."""
    p = Path(evo_path)
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(p)


def load_registry(registry_path: str = "contract_registry/subscriptions.yaml") -> Dict:
    """Load contract registry."""
    p = Path(registry_path)
    if not p.exists():
        return {"subscriptions": []}
    with open(p) as f:
        return yaml.safe_load(f) or {"subscriptions": []}


# ---------------------------------------------------------------------------
# Health score computation
# ---------------------------------------------------------------------------

def compute_health_score(reports: List[Dict]) -> tuple:
    """
    Health score = (checks_passed / total_checks) * 100
    adjusted down by 20 points per CRITICAL violation.
    """
    total_checks = 0
    passed_checks = 0
    all_fails = []

    for report in reports:
        total_checks += report.get("total_checks", 0)
        passed_checks += report.get("passed", 0)
        for result in report.get("results", []):
            if result.get("status") in ("FAIL", "ERROR"):
                all_fails.append(result)

    if total_checks == 0:
        return 100, all_fails

    base_score = (passed_checks / total_checks) * 100

    # Deduct 20 per CRITICAL violation
    critical_count = sum(1 for f in all_fails if f.get("severity") == "CRITICAL")
    high_count = sum(1 for f in all_fails if f.get("severity") == "HIGH")

    adjusted = base_score - (critical_count * 20) - (high_count * 5)
    return max(0, min(100, round(adjusted, 1))), all_fails


# ---------------------------------------------------------------------------
# Plain language violation descriptions
# ---------------------------------------------------------------------------

def describe_violation(result: Dict, registry: Dict) -> str:
    """
    Produce a plain-language description of a violation.
    References the failing system, field, and downstream impact.
    """
    col = result.get("column_name", "unknown field")
    check_type = result.get("check_type", "unknown check")
    actual = result.get("actual_value", "unknown")
    expected = result.get("expected", "unknown")
    severity = result.get("severity", "UNKNOWN")
    message = result.get("message", "")

    # Find affected subscribers
    check_id = result.get("check_id", "")
    contract_keyword = check_id.split("-")[0] if "-" in check_id else ""
    subscribers = []
    for sub in registry.get("subscriptions", []):
        if contract_keyword and contract_keyword in sub.get("contract_id", ""):
            subscribers.append(sub["subscriber_id"])

    sub_str = ", ".join(set(subscribers)) if subscribers else "no registered subscribers"

    return (
        f"[{severity}] The '{col}' field failed its {check_type} check. "
        f"Expected: {expected}. Found: {actual}. "
        f"{message} "
        f"Downstream subscribers affected: {sub_str}. "
        f"Records failing: {result.get('records_failing', 'unknown')}."
    )


# ---------------------------------------------------------------------------
# Report construction
# ---------------------------------------------------------------------------

def generate_report(
    reports_dir: str = "validation_reports/",
    violation_log_path: str = "violation_log/violations.jsonl",
    ai_extensions_path: str = "validation_reports/ai_extensions.json",
    schema_evo_path: str = "validation_reports/schema_evolution.json",
    registry_path: str = "contract_registry/subscriptions.yaml",
) -> Dict[str, Any]:
    """Build the full Enforcer Report from live data."""
    # Load all data sources
    reports = load_validation_reports(reports_dir)
    violations = load_violation_log(violation_log_path)
    ai = load_ai_extensions(ai_extensions_path)
    registry = load_registry(registry_path)

    # Load schema evolution
    schema_evo = {}
    evo_path = Path(schema_evo_path)
    if evo_path.exists():
        try:
            with open(evo_path) as f:
                schema_evo = json.load(f)
        except (json.JSONDecodeError, KeyError):
            pass

    # Section 1: Data Health Score
    score, all_fails = compute_health_score(reports)
    critical_count = sum(1 for f in all_fails if f.get("severity") == "CRITICAL")

    health_narrative = f"Score {score}/100. "
    if score >= 90:
        health_narrative += "All systems operating within contract parameters."
    elif score >= 70:
        health_narrative += (
            f"{critical_count} critical issue(s) detected. "
            f"Most contract checks pass but immediate attention needed on failures."
        )
    else:
        health_narrative += (
            f"{critical_count} critical issue(s) require immediate action. "
            f"Data pipeline should not proceed to production without resolution."
        )

    # Section 2: Violations this week
    severity_counts = {
        "CRITICAL": sum(1 for f in all_fails if f.get("severity") == "CRITICAL"),
        "HIGH": sum(1 for f in all_fails if f.get("severity") == "HIGH"),
        "MEDIUM": sum(1 for f in all_fails if f.get("severity") == "MEDIUM"),
        "LOW": sum(1 for f in all_fails if f.get("severity") == "LOW"),
        "WARNING": sum(1 for f in all_fails if f.get("severity") == "WARNING"),
    }

    # Top 3 violations by severity
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "WARNING": 4}
    sorted_fails = sorted(
        all_fails,
        key=lambda x: severity_order.get(x.get("severity", "LOW"), 99)
    )
    top_violations = [
        describe_violation(f, registry) for f in sorted_fails[:3]
    ]

    # Section 3: Schema changes detected
    schema_changes_summary = "No schema evolution analysis available."
    if schema_evo:
        verdict = schema_evo.get("overall_verdict", "UNKNOWN")
        n_changes = schema_evo.get("total_changes", 0)
        n_breaking = schema_evo.get("breaking_changes", 0)
        changes_detail = []
        for c in schema_evo.get("changes", []):
            changes_detail.append(
                f"  - {c.get('field', 'unknown')}: {c.get('change_type', 'unknown')} "
                f"({c.get('verdict', 'UNKNOWN')})"
            )
        schema_changes_summary = (
            f"Overall verdict: {verdict}. "
            f"{n_changes} change(s) detected, {n_breaking} breaking. "
            + ("\n".join(changes_detail) if changes_detail else "")
        )

    # Section 4: AI system risk assessment
    ai_risk = {
        "embedding_drift": {
            "status": ai.get("embedding_drift", {}).get("status", "NOT_RUN"),
            "drift_score": ai.get("embedding_drift", {}).get("drift_score", "N/A"),
            "threshold": ai.get("embedding_drift", {}).get("threshold", 0.15),
            "interpretation": ai.get("embedding_drift", {}).get("interpretation", "Not assessed"),
        },
        "prompt_input_validation": {
            "status": ai.get("prompt_input_validation", {}).get("status", "NOT_RUN"),
            "valid": ai.get("prompt_input_validation", {}).get("valid", "N/A"),
            "quarantined": ai.get("prompt_input_validation", {}).get("quarantined", "N/A"),
        },
        "output_violation_rate": {
            "status": ai.get("output_violation_rate", {}).get("status", "NOT_RUN"),
            "violation_rate": ai.get("output_violation_rate", {}).get("violation_rate", "N/A"),
            "trend": ai.get("output_violation_rate", {}).get("trend", "unknown"),
        },
        "overall_assessment": _ai_overall_assessment(ai),
    }

    # Section 5: Recommended actions (specific file + clause)
    recommendations = _build_recommendations(all_fails, violations, schema_evo, ai, registry)

    # Assemble report
    now = datetime.now(timezone.utc)
    report = {
        "report_id": f"enforcer-{now.strftime('%Y%m%d-%H%M%S')}",
        "generated_at": now.isoformat(),
        "period": f"{(now - timedelta(days=7)).date()} to {now.date()}",
        "data_health_score": score,
        "health_score_formula": "(checks_passed / total_checks) * 100, minus 20 per CRITICAL violation",
        "health_narrative": health_narrative,
        "total_checks_run": sum(r.get("total_checks", 0) for r in reports),
        "total_checks_passed": sum(r.get("passed", 0) for r in reports),
        "total_checks_failed": sum(r.get("failed", 0) for r in reports),
        "violations_by_severity": severity_counts,
        "top_violations": top_violations,
        "schema_changes": schema_changes_summary,
        "ai_risk_assessment": ai_risk,
        "recommended_actions": recommendations,
        "validation_reports_analyzed": len(reports),
        "violation_log_entries": len(violations),
    }

    return report


def _ai_overall_assessment(ai: Dict) -> str:
    """Produce an overall AI system risk assessment narrative."""
    if not ai:
        return "AI contract extensions have not been run. Cannot assess AI system reliability."

    drift_status = ai.get("embedding_drift", {}).get("status", "NOT_RUN")
    rate_status = ai.get("output_violation_rate", {}).get("status", "NOT_RUN")
    prompt_status = ai.get("prompt_input_validation", {}).get("status", "NOT_RUN")

    if all(s == "PASS" for s in [drift_status, rate_status, prompt_status]):
        return (
            "All AI systems are currently consuming reliable data. "
            "Embedding drift is within acceptable bounds. "
            "LLM output schema violation rate is stable. "
            "Prompt inputs pass schema validation."
        )
    issues = []
    if drift_status == "FAIL":
        issues.append("Embedding drift exceeds threshold — content may have shifted")
    if rate_status == "WARN":
        rate = ai.get("output_violation_rate", {}).get("violation_rate", 0)
        issues.append(f"LLM output violation rate ({rate:.2%}) requires monitoring")
    if prompt_status == "WARN":
        q = ai.get("prompt_input_validation", {}).get("quarantined", 0)
        issues.append(f"{q} prompt inputs failed validation and were quarantined")
    if issues:
        return "ATTENTION REQUIRED: " + "; ".join(issues) + "."
    return "AI systems are operating normally with minor observations."


def _build_recommendations(
    all_fails: List[Dict],
    violations: List[Dict],
    schema_evo: Dict,
    ai: Dict,
    registry: Dict,
) -> List[Dict]:
    """Build specific, actionable recommendations with file + clause references."""
    recs = []
    priority = 1

    # Recommendation from CRITICAL failures
    for fail in all_fails:
        if fail.get("severity") != "CRITICAL":
            continue
        col = fail.get("column_name", "unknown")
        check = fail.get("check_type", "unknown")

        if "confidence" in col:
            recs.append({
                "priority": priority,
                "action": (
                    f"Update the upstream producer to output '{col}' as float 0.0-1.0 "
                    f"per contract clause {col}.range. "
                    f"File: contracts/generator.py — clause 'evidence_confidence' with "
                    f"minimum: 0.0, maximum: 1.0. "
                    f"The current data shows values in 0-100 scale, indicating a scale change."
                ),
                "severity": "CRITICAL",
                "field": col,
                "file_reference": "contracts/generator.py",
                "clause_reference": f"{col}.range (minimum: 0.0, maximum: 1.0)",
            })
        else:
            recs.append({
                "priority": priority,
                "action": (
                    f"Fix {check} violation on field '{col}'. "
                    f"Expected: {fail.get('expected', 'N/A')}. "
                    f"Found: {fail.get('actual_value', 'N/A')}."
                ),
                "severity": fail.get("severity", "UNKNOWN"),
                "field": col,
            })
        priority += 1
        if priority > 3:
            break

    # Add CI/CD recommendation if not already 3
    if len(recs) < 3:
        recs.append({
            "priority": priority,
            "action": (
                "Add contracts/runner.py as a required CI step before any deployment. "
                "Run in AUDIT mode for the first 2 weeks, then switch to ENFORCE mode. "
                "File: .github/workflows/contract-validation.yml"
            ),
            "severity": "HIGH",
            "field": "__pipeline__",
            "file_reference": "contracts/runner.py",
        })
        priority += 1

    if len(recs) < 3:
        recs.append({
            "priority": priority,
            "action": (
                "Schedule monthly baseline refresh for statistical drift thresholds "
                "in schema_snapshots/baselines.json. Current baselines may become stale "
                "as data distributions evolve. "
                "File: schema_snapshots/baselines.json"
            ),
            "severity": "MEDIUM",
            "field": "__baselines__",
            "file_reference": "schema_snapshots/baselines.json",
        })

    return recs[:3]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ReportGenerator")
    parser.add_argument("--output", default="enforcer_report/report_data.json",
                        help="Output path for report JSON")
    parser.add_argument("--reports-dir", default="validation_reports/",
                        help="Directory containing validation reports")
    parser.add_argument("--violation-log", default="violation_log/violations.jsonl",
                        help="Path to violation log")
    parser.add_argument("--registry", default="contract_registry/subscriptions.yaml",
                        help="Path to contract registry")
    args = parser.parse_args()

    print("[1/3] Loading validation data...")
    report = generate_report(
        reports_dir=args.reports_dir,
        violation_log_path=args.violation_log,
        registry_path=args.registry,
    )

    print(f"[2/3] Report generated.")
    print(f"  Data Health Score: {report['data_health_score']}/100")
    print(f"  {report['health_narrative']}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[3/3] Written: {out_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Enforcer Report Summary")
    print(f"{'='*60}")
    print(f"  Health Score:      {report['data_health_score']}/100")
    print(f"  Checks Run:        {report['total_checks_run']}")
    print(f"  Checks Passed:     {report['total_checks_passed']}")
    print(f"  Checks Failed:     {report['total_checks_failed']}")
    print(f"  Violations Logged: {report['violation_log_entries']}")

    print(f"\n  Top violations:")
    for i, v in enumerate(report["top_violations"], 1):
        print(f"    {i}. {v[:120]}...")

    print(f"\n  Recommended actions:")
    for rec in report["recommended_actions"]:
        print(f"    P{rec['priority']}: {rec['action'][:120]}...")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
