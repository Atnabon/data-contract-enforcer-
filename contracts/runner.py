"""
ValidationRunner — evaluates contract rules against JSONL data.

Loads a Bitol YAML contract and runs every clause as a machine-checkable
rule against each record in a JSONL source file. Produces a structured
ValidationReport with per-check pass/fail/warn status and writes it to
validation_reports/.

Usage:
    python contracts/runner.py \
        --source outputs/week3/verdicts.jsonl \
        --contract generated_contracts/week3_verdicts.yaml \
        --output validation_reports/

    python contracts/runner.py \
        --source outputs/week5/events.jsonl \
        --contract generated_contracts/week5_events.yaml \
        --output validation_reports/
"""
import argparse
import json
import re
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from contracts.models import ValidationReport, ValidationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _is_valid_uuid(value: str) -> bool:
    return bool(UUID_RE.match(str(value)))


def _is_valid_datetime(value: str) -> bool:
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S+00:00",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            datetime.strptime(str(value), fmt)
            return True
        except ValueError:
            pass
    return False


def _coerce_numeric(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Column-level checks
# ---------------------------------------------------------------------------

def _check_required(col_name: str, records: List[Dict],
                    check_id: str) -> ValidationResult:
    """All records must have a non-null value for this column."""
    failing = []
    for i, rec in enumerate(records):
        val = rec.get(col_name)
        if val is None or val == "":
            failing.append(str(i))

    if failing:
        return ValidationResult(
            check_id=check_id,
            column_name=col_name,
            check_type="required",
            status="FAIL",
            actual_value=f"{len(failing)} null/missing",
            expected="0 nulls",
            severity="CRITICAL",
            records_failing=len(failing),
            sample_failing=failing[:5],
            message=f"Column '{col_name}' has {len(failing)} null or missing values.",
        )
    return ValidationResult(
        check_id=check_id,
        column_name=col_name,
        check_type="required",
        status="PASS",
        actual_value="0 nulls",
        expected="0 nulls",
        severity="CRITICAL",
        message=f"Column '{col_name}' has no null values across all records.",
    )


def _check_format_uuid(col_name: str, records: List[Dict],
                       check_id: str) -> ValidationResult:
    """All non-null values must match UUID v4 format."""
    failing = []
    for i, rec in enumerate(records):
        val = rec.get(col_name)
        if val is not None and not _is_valid_uuid(val):
            failing.append(str(val))

    if failing:
        return ValidationResult(
            check_id=check_id,
            column_name=col_name,
            check_type="format_uuid",
            status="FAIL",
            actual_value=f"{len(failing)} invalid UUIDs",
            expected="all values match ^[0-9a-f]{{8}}-...$",
            severity="HIGH",
            records_failing=len(failing),
            sample_failing=failing[:3],
            message=f"Column '{col_name}' contains {len(failing)} non-UUID values.",
        )
    return ValidationResult(
        check_id=check_id,
        column_name=col_name,
        check_type="format_uuid",
        status="PASS",
        actual_value="all valid UUIDs",
        expected="all values match UUID pattern",
        severity="HIGH",
        message=f"Column '{col_name}' — all values are valid UUIDs.",
    )


def _check_format_datetime(col_name: str, records: List[Dict],
                           check_id: str) -> ValidationResult:
    """All non-null values must be ISO 8601 date-time strings."""
    failing = []
    for i, rec in enumerate(records):
        val = rec.get(col_name)
        if val is not None and not _is_valid_datetime(val):
            failing.append(str(val))

    if failing:
        return ValidationResult(
            check_id=check_id,
            column_name=col_name,
            check_type="format_datetime",
            status="FAIL",
            actual_value=f"{len(failing)} invalid datetimes",
            expected="ISO 8601 date-time strings",
            severity="HIGH",
            records_failing=len(failing),
            sample_failing=failing[:3],
            message=f"Column '{col_name}' contains {len(failing)} non-datetime values.",
        )
    return ValidationResult(
        check_id=check_id,
        column_name=col_name,
        check_type="format_datetime",
        status="PASS",
        actual_value="all valid ISO 8601",
        expected="ISO 8601 date-time strings",
        severity="HIGH",
        message=f"Column '{col_name}' — all values are valid ISO 8601 datetimes.",
    )


def _check_enum(col_name: str, allowed: List[str],
                records: List[Dict], check_id: str) -> ValidationResult:
    """All non-null values must be in the allowed enum set."""
    allowed_set = set(str(v) for v in allowed)
    failing = []
    for rec in records:
        val = rec.get(col_name)
        if val is not None and str(val) not in allowed_set:
            failing.append(str(val))

    if failing:
        return ValidationResult(
            check_id=check_id,
            column_name=col_name,
            check_type="enum",
            status="FAIL",
            actual_value=f"{len(failing)} values outside enum",
            expected=f"one of {sorted(allowed_set)[:5]}...",
            severity="HIGH",
            records_failing=len(failing),
            sample_failing=list(set(failing))[:5],
            message=f"Column '{col_name}' contains values not in the defined enum.",
        )
    return ValidationResult(
        check_id=check_id,
        column_name=col_name,
        check_type="enum",
        status="PASS",
        actual_value="all in enum",
        expected=f"one of {sorted(allowed_set)[:5]}...",
        severity="HIGH",
        message=f"Column '{col_name}' — all values are within the allowed enum.",
    )


def _check_minimum(col_name: str, minimum: float,
                   records: List[Dict], check_id: str) -> ValidationResult:
    """All numeric values must be >= minimum."""
    failing = []
    for rec in records:
        val = _coerce_numeric(rec.get(col_name))
        if val is not None and val < minimum:
            failing.append(str(val))

    if failing:
        return ValidationResult(
            check_id=check_id,
            column_name=col_name,
            check_type="minimum",
            status="FAIL",
            actual_value=f"min observed = {min(float(x) for x in failing):.4f}",
            expected=f">= {minimum}",
            severity="HIGH",
            records_failing=len(failing),
            sample_failing=failing[:3],
            message=f"Column '{col_name}' has {len(failing)} values below minimum {minimum}.",
        )
    return ValidationResult(
        check_id=check_id,
        column_name=col_name,
        check_type="minimum",
        status="PASS",
        actual_value=f"all >= {minimum}",
        expected=f">= {minimum}",
        severity="HIGH",
        message=f"Column '{col_name}' — all values meet minimum {minimum}.",
    )


def _check_maximum(col_name: str, maximum: float,
                   records: List[Dict], check_id: str) -> ValidationResult:
    """All numeric values must be <= maximum."""
    failing = []
    for rec in records:
        val = _coerce_numeric(rec.get(col_name))
        if val is not None and val > maximum:
            failing.append(str(val))

    # Confidence/score range violations are CRITICAL
    severity = "CRITICAL" if "confidence" in col_name else "HIGH"

    if failing:
        return ValidationResult(
            check_id=check_id,
            column_name=col_name,
            check_type="maximum",
            status="FAIL",
            actual_value=f"max observed = {max(float(x) for x in failing):.4f}",
            expected=f"<= {maximum}",
            severity=severity,
            records_failing=len(failing),
            sample_failing=failing[:3],
            message=f"Column '{col_name}' has {len(failing)} values above maximum {maximum}.",
        )
    return ValidationResult(
        check_id=check_id,
        column_name=col_name,
        check_type="maximum",
        status="PASS",
        actual_value=f"all <= {maximum}",
        expected=f"<= {maximum}",
        severity=severity,
        message=f"Column '{col_name}' — all values meet maximum {maximum}.",
    )


def _check_type(col_name: str, expected_type: str,
                records: List[Dict], check_id: str) -> ValidationResult:
    """Type compliance check: string/number/integer/boolean."""
    type_map = {
        "string":  str,
        "number":  (int, float),
        "integer": int,
        "boolean": bool,
    }
    py_type = type_map.get(expected_type)
    if py_type is None:
        return ValidationResult(
            check_id=check_id,
            column_name=col_name,
            check_type="type",
            status="WARN",
            actual_value="unknown",
            expected=expected_type,
            severity="LOW",
            message=f"Type '{expected_type}' not checked (unsupported in runner).",
        )

    failing = []
    for rec in records:
        val = rec.get(col_name)
        if val is not None and not isinstance(val, py_type):
            failing.append(f"{type(val).__name__}:{val!r}")

    if failing:
        return ValidationResult(
            check_id=check_id,
            column_name=col_name,
            check_type="type",
            status="FAIL",
            actual_value=f"{len(failing)} wrong-type values",
            expected=f"all {expected_type}",
            severity="HIGH",
            records_failing=len(failing),
            sample_failing=failing[:3],
            message=f"Column '{col_name}' has {len(failing)} values with wrong type (expected {expected_type}).",
        )
    return ValidationResult(
        check_id=check_id,
        column_name=col_name,
        check_type="type",
        status="PASS",
        actual_value=f"all {expected_type}",
        expected=f"all {expected_type}",
        severity="LOW",
        message=f"Column '{col_name}' — all values match expected type {expected_type}.",
    )


# ---------------------------------------------------------------------------
# Cross-field / dataset-level checks
# ---------------------------------------------------------------------------

def _check_row_count(records: List[Dict], check_id: str) -> ValidationResult:
    """Dataset must have at least one row."""
    count = len(records)
    if count < 1:
        return ValidationResult(
            check_id=check_id,
            column_name="__dataset__",
            check_type="row_count",
            status="FAIL",
            actual_value="0",
            expected=">= 1",
            severity="CRITICAL",
            message="Dataset is empty — no records found.",
        )
    return ValidationResult(
        check_id=check_id,
        column_name="__dataset__",
        check_type="row_count",
        status="PASS",
        actual_value=str(count),
        expected=">= 1",
        severity="CRITICAL",
        message=f"Dataset has {count} records.",
    )


def _check_uniqueness(col_name: str, records: List[Dict],
                      check_id: str) -> ValidationResult:
    """Values in a column (typically _id fields) must be unique."""
    values = [str(r.get(col_name)) for r in records if r.get(col_name) is not None]
    duplicates = len(values) - len(set(values))

    if duplicates > 0:
        from collections import Counter
        counts = Counter(values)
        dupes = [v for v, c in counts.items() if c > 1]
        return ValidationResult(
            check_id=check_id,
            column_name=col_name,
            check_type="uniqueness",
            status="FAIL",
            actual_value=f"{duplicates} duplicates",
            expected="0 duplicates",
            severity="CRITICAL",
            records_failing=duplicates,
            sample_failing=dupes[:3],
            message=f"Column '{col_name}' has {duplicates} duplicate values — ID uniqueness violated.",
        )
    return ValidationResult(
        check_id=check_id,
        column_name=col_name,
        check_type="uniqueness",
        status="PASS",
        actual_value="0 duplicates",
        expected="0 duplicates",
        severity="CRITICAL",
        message=f"Column '{col_name}' — all values are unique.",
    )


def _check_monotonic(col_name: str, records: List[Dict],
                     check_id: str) -> ValidationResult:
    """
    Numeric column must be monotonically non-decreasing.

    - global_position: checked across all records (truly global)
    - stream_position: checked within each stream_id group (per-aggregate)
      because stream_position restarts at 1 for each new stream.
    """
    violations = []

    if col_name == "stream_position":
        # Group by stream_id, check monotonic within each group
        from collections import defaultdict
        streams: dict = defaultdict(list)
        for i, rec in enumerate(records):
            sid = rec.get("stream_id", "__unknown__")
            val = _coerce_numeric(rec.get(col_name))
            if val is not None:
                streams[sid].append((i, val))

        for sid, vals in streams.items():
            for idx in range(1, len(vals)):
                if vals[idx][1] < vals[idx - 1][1]:
                    violations.append(
                        f"stream={sid!r} record {vals[idx][0]}: "
                        f"{vals[idx][1]} < {vals[idx-1][1]}"
                    )
    else:
        # Global monotonic (global_position)
        values = []
        for i, rec in enumerate(records):
            val = _coerce_numeric(rec.get(col_name))
            if val is not None:
                values.append((i, val))
        for idx in range(1, len(values)):
            if values[idx][1] < values[idx - 1][1]:
                violations.append(
                    f"record {values[idx][0]}: {values[idx][1]} < {values[idx-1][1]}"
                )

    if violations:
        return ValidationResult(
            check_id=check_id,
            column_name=col_name,
            check_type="monotonic",
            status="FAIL",
            actual_value=f"{len(violations)} ordering violations",
            expected="monotonically non-decreasing",
            severity="HIGH",
            records_failing=len(violations),
            sample_failing=violations[:3],
            message=(
                f"Column '{col_name}' is not monotonically non-decreasing. "
                f"Optimistic concurrency control or replay logic may break."
            ),
        )
    return ValidationResult(
        check_id=check_id,
        column_name=col_name,
        check_type="monotonic",
        status="PASS",
        actual_value="monotonically non-decreasing",
        expected="monotonically non-decreasing",
        severity="HIGH",
        message=(
            f"Column '{col_name}' — ordering is monotonically non-decreasing "
            f"{'within each stream_id' if col_name == 'stream_position' else '(global)'}."
        ),
    )


def _check_score_consistency(records: List[Dict],
                              check_id: str) -> ValidationResult:
    """
    Cross-field check for automaton-auditor verdict records:
    final_score must lie within the observed range of
    [min(prosecutor, defense, tech_lead), max(prosecutor, defense, tech_lead)].
    Also flags missing dissent_summary when score variance > 2.
    """
    anomalies = []
    missing_dissent = []

    for i, rec in enumerate(records):
        final  = _coerce_numeric(rec.get("final_score"))
        pros   = _coerce_numeric(rec.get("prosecutor_score"))
        defn   = _coerce_numeric(rec.get("defense_score"))
        tech   = _coerce_numeric(rec.get("tech_lead_score"))

        if None in (final, pros, defn, tech):
            continue

        lo = min(pros, defn, tech)
        hi = max(pros, defn, tech)
        if not (lo - 1 <= final <= hi + 1):
            anomalies.append(
                f"record {i}: final={final} outside [{lo},{hi}]"
            )

        if (hi - lo) > 2 and not rec.get("dissent_summary"):
            missing_dissent.append(
                f"record {i}: variance={hi-lo:.0f}, dissent_summary=null"
            )

    issues = anomalies + missing_dissent
    if issues:
        return ValidationResult(
            check_id=check_id,
            column_name="final_score",
            check_type="score_consistency",
            status="WARN",
            actual_value=f"{len(issues)} score anomalies",
            expected="final_score within judge range; dissent when variance>2",
            severity="MEDIUM",
            records_failing=len(issues),
            sample_failing=issues[:3],
            message=(
                f"Score consistency: {len(anomalies)} final_score outliers, "
                f"{len(missing_dissent)} missing dissent summaries."
            ),
        )
    return ValidationResult(
        check_id=check_id,
        column_name="final_score",
        check_type="score_consistency",
        status="PASS",
        actual_value="all scores internally consistent",
        expected="final_score within judge range; dissent when variance>2",
        severity="MEDIUM",
        message="Cross-field score consistency check passed for all records.",
    )


def _check_temporal_ordering(
    ts_col: str, records: List[Dict], check_id: str
) -> ValidationResult:
    """
    For event records: occurred_at (if present) should be <= recorded_at.
    For verdict records: audited_at should be a recent timestamp.
    """
    if not records or ts_col not in records[0]:
        return ValidationResult(
            check_id=check_id,
            column_name=ts_col,
            check_type="temporal_ordering",
            status="WARN",
            actual_value="column not found",
            expected="timestamps present",
            severity="LOW",
            message=f"Column '{ts_col}' not found in records — skipping temporal check.",
        )

    violations = []
    for i, rec in enumerate(records):
        occurred = rec.get("occurred_at")
        recorded = rec.get("recorded_at")
        if occurred and recorded:
            try:
                occ_dt = datetime.fromisoformat(str(occurred).replace("Z", "+00:00"))
                rec_dt = datetime.fromisoformat(str(recorded).replace("Z", "+00:00"))
                if occ_dt > rec_dt:
                    violations.append(
                        f"record {i}: occurred_at={occurred} > recorded_at={recorded}"
                    )
            except ValueError:
                pass

    if violations:
        return ValidationResult(
            check_id=check_id,
            column_name=ts_col,
            check_type="temporal_ordering",
            status="FAIL",
            actual_value=f"{len(violations)} temporal violations",
            expected="occurred_at <= recorded_at",
            severity="HIGH",
            records_failing=len(violations),
            sample_failing=violations[:3],
            message=(
                f"Temporal ordering violated: {len(violations)} records have "
                f"occurred_at after recorded_at."
            ),
        )
    return ValidationResult(
        check_id=check_id,
        column_name=ts_col,
        check_type="temporal_ordering",
        status="PASS",
        actual_value="occurred_at <= recorded_at for all records",
        expected="occurred_at <= recorded_at",
        severity="HIGH",
        message="Temporal ordering check passed — occurred_at never exceeds recorded_at.",
    )


# ---------------------------------------------------------------------------
# Runner core
# ---------------------------------------------------------------------------

def _flatten_record(record: Dict) -> Dict:
    """
    Flatten one level of nested dicts (payload, metadata) into the record
    using prefixed keys, matching what the generator produces.
    """
    flat = {}
    for k, v in record.items():
        if isinstance(v, dict):
            for dk, dv in v.items():
                if not isinstance(dv, (list, dict)):
                    flat[f"{k}_{dk}"] = dv
        elif not isinstance(v, list):
            flat[k] = v
    return flat


def _check_statistical_drift(
    col_name: str,
    records: List[Dict],
    baselines_path: str,
    check_id: str,
) -> Optional[ValidationResult]:
    """
    Statistical drift detection: compare current mean against stored baseline.
    Emit WARNING at 2 stddev deviation and FAIL at 3 stddev.
    """
    bp = Path(baselines_path)
    if not bp.exists():
        return None

    with open(bp) as f:
        baselines = json.load(f).get("columns", {})
    if col_name not in baselines:
        return None

    values = [_coerce_numeric(r.get(col_name)) for r in records]
    values = [v for v in values if v is not None]
    if not values:
        return None

    import statistics
    current_mean = statistics.mean(values)
    baseline = baselines[col_name]
    b_mean = baseline["mean"]
    b_stddev = max(baseline.get("stddev", 0.001), 1e-9)

    z_score = abs(current_mean - b_mean) / b_stddev

    if z_score > 3:
        return ValidationResult(
            check_id=check_id,
            column_name=col_name,
            check_type="statistical_drift",
            status="FAIL",
            actual_value=f"mean={current_mean:.4f}, z={z_score:.1f}",
            expected=f"mean within 3 stddev of baseline {b_mean:.4f}",
            severity="HIGH",
            records_failing=len(values),
            message=(
                f"Statistical drift FAIL: '{col_name}' mean drifted {z_score:.1f} stddev "
                f"from baseline (current={current_mean:.4f}, baseline={b_mean:.4f}). "
                f"Possible scale change detected."
            ),
        )
    elif z_score > 2:
        return ValidationResult(
            check_id=check_id,
            column_name=col_name,
            check_type="statistical_drift",
            status="WARN",
            actual_value=f"mean={current_mean:.4f}, z={z_score:.1f}",
            expected=f"mean within 2 stddev of baseline {b_mean:.4f}",
            severity="MEDIUM",
            records_failing=0,
            message=(
                f"Statistical drift WARNING: '{col_name}' approaching drift threshold "
                f"({z_score:.1f} stddev from baseline)."
            ),
        )
    return ValidationResult(
        check_id=check_id,
        column_name=col_name,
        check_type="statistical_drift",
        status="PASS",
        actual_value=f"mean={current_mean:.4f}, z={z_score:.1f}",
        expected=f"mean within 2 stddev of baseline {b_mean:.4f}",
        severity="LOW",
        message=f"Statistical drift check passed for '{col_name}' (z={z_score:.1f}).",
    )


def run_validation(
    source_path: str,
    contract_path: str,
    mode: str = "AUDIT",
    baselines_path: str = "schema_snapshots/baselines.json",
) -> ValidationReport:
    """
    Run all contract clauses against the source JSONL.
    Returns a structured ValidationReport.
    """
    # Load contract
    with open(contract_path) as f:
        contract = yaml.safe_load(f)

    contract_id = contract.get("id", "unknown")
    schema = contract.get("schema", {})

    # Load + flatten records
    raw_records = []
    with open(source_path) as f:
        for line in f:
            line = line.strip()
            if line:
                raw_records.append(json.loads(line))

    flat_records = [_flatten_record(r) for r in raw_records]

    results: List[ValidationResult] = []
    check_counter = 0

    def next_id(prefix: str) -> str:
        nonlocal check_counter
        check_counter += 1
        return f"{prefix}-{check_counter:03d}"

    # --- Dataset-level check ---
    results.append(_check_row_count(flat_records, next_id("ROW")))

    # --- Per-column checks ---
    for col_name, clause in schema.items():
        col_type = clause.get("type", "string")

        # Type check
        results.append(_check_type(col_name, col_type, flat_records, next_id("TYP")))

        # Required check
        if clause.get("required"):
            results.append(_check_required(col_name, flat_records, next_id("REQ")))

        # UUID format
        if clause.get("format") == "uuid":
            results.append(_check_format_uuid(col_name, flat_records, next_id("UUID")))
            # Also check uniqueness for id fields
            if col_name.endswith("_id") or col_name in ("audit_id", "event_id"):
                results.append(_check_uniqueness(col_name, flat_records, next_id("UNIQ")))

        # Date-time format
        if clause.get("format") == "date-time":
            results.append(_check_format_datetime(col_name, flat_records, next_id("DT")))

        # Enum check
        if clause.get("enum"):
            results.append(_check_enum(col_name, clause["enum"], flat_records, next_id("ENUM")))

        # Numeric bounds
        if clause.get("minimum") is not None:
            results.append(_check_minimum(col_name, float(clause["minimum"]),
                                          flat_records, next_id("MIN")))
        if clause.get("maximum") is not None:
            results.append(_check_maximum(col_name, float(clause["maximum"]),
                                          flat_records, next_id("MAX")))

    # --- Cross-field checks ---

    # Score consistency for week3 audit verdicts
    if "final_score" in schema and "prosecutor_score" in schema:
        results.append(_check_score_consistency(flat_records, next_id("XFD")))

    # Monotonic ordering for event streams
    if "global_position" in schema:
        results.append(_check_monotonic("global_position", flat_records, next_id("MON")))
    if "stream_position" in schema:
        results.append(_check_monotonic("stream_position", flat_records, next_id("MON")))

    # Temporal ordering for event records
    if "recorded_at" in schema:
        results.append(_check_temporal_ordering("recorded_at", flat_records, next_id("TEMP")))

    # Statistical drift checks for numeric columns
    for col_name, clause in schema.items():
        if clause.get("type") in ("number", "integer"):
            drift_result = _check_statistical_drift(
                col_name, flat_records, baselines_path, next_id("DRIFT")
            )
            if drift_result is not None:
                results.append(drift_result)

    # Tally
    passed  = sum(1 for r in results if r.status == "PASS")
    failed  = sum(1 for r in results if r.status == "FAIL")
    warned  = sum(1 for r in results if r.status == "WARN")
    errored = sum(1 for r in results if r.status == "ERROR")

    report = ValidationReport(
        report_id     = str(uuid.uuid4()),
        contract_id   = contract_id,
        snapshot_id   = contract.get("info", {}).get("version", "1.0.0"),
        run_timestamp = datetime.now(timezone.utc).isoformat(),
        total_checks  = len(results),
        passed        = passed,
        failed        = failed,
        warned        = warned,
        errored       = errored,
        results       = results,
    )
    return report


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _report_to_dict(report: ValidationReport) -> Dict:
    """Serialize ValidationReport to a plain dict for JSON output."""
    d = asdict(report)
    return d


def print_summary(report: ValidationReport) -> None:
    """Print a human-readable summary to stdout."""
    print(f"\n{'='*60}")
    print(f"Validation Report: {report.contract_id}")
    print(f"Run: {report.run_timestamp}")
    print(f"{'='*60}")
    print(f"  Total checks : {report.total_checks}")
    print(f"  PASS         : {report.passed}")
    print(f"  FAIL         : {report.failed}")
    print(f"  WARN         : {report.warned}")
    print(f"  ERROR        : {report.errored}")

    if report.failed > 0 or report.warned > 0:
        print(f"\nIssues:")
        for r in report.results:
            if r.status in ("FAIL", "WARN", "ERROR"):
                print(f"  [{r.status}] {r.check_type} on '{r.column_name}': {r.message}")
                if r.sample_failing:
                    print(f"         Samples: {r.sample_failing[:3]}")
    else:
        print("\nAll checks passed.")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ValidationRunner")
    parser.add_argument("--source",   required=True,
                        help="Path to the JSONL data file to validate")
    parser.add_argument("--contract", required=True,
                        help="Path to the Bitol contract YAML file")
    parser.add_argument("--output",   default="validation_reports",
                        help="Directory to write the JSON validation report")
    parser.add_argument("--mode", default="AUDIT",
                        choices=["AUDIT", "WARN", "ENFORCE"],
                        help="Enforcement mode: AUDIT (log only), WARN (block on CRITICAL), ENFORCE (block on CRITICAL+HIGH)")
    args = parser.parse_args()

    print(f"[1/3] Loading contract: {args.contract}")
    print(f"[2/3] Validating:       {args.source}")
    print(f"       Mode:            {args.mode}")

    report = run_validation(args.source, args.contract, mode=args.mode)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    contract_slug = Path(args.contract).stem
    out_path = output_dir / f"{contract_slug}_{ts}.json"

    with open(out_path, "w") as f:
        json.dump(_report_to_dict(report), f, indent=2)

    print(f"[3/3] Report written: {out_path}")
    print_summary(report)

    # Enforcement mode determines exit behavior
    # AUDIT: log only, never block
    # WARN: block on CRITICAL violations only
    # ENFORCE: block on CRITICAL and HIGH violations
    if args.mode == "AUDIT":
        print(f"  Mode AUDIT: logging only, pipeline continues.")
    elif args.mode == "WARN":
        critical = sum(1 for r in report.results if r.status == "FAIL" and r.severity == "CRITICAL")
        if critical > 0:
            print(f"  Mode WARN: {critical} CRITICAL violation(s) — pipeline BLOCKED.")
            raise SystemExit(1)
        print(f"  Mode WARN: no CRITICAL violations, pipeline continues.")
    elif args.mode == "ENFORCE":
        blocking = sum(
            1 for r in report.results
            if r.status == "FAIL" and r.severity in ("CRITICAL", "HIGH")
        )
        if blocking > 0:
            print(f"  Mode ENFORCE: {blocking} CRITICAL/HIGH violation(s) — pipeline BLOCKED.")
            raise SystemExit(1)
        print(f"  Mode ENFORCE: no CRITICAL/HIGH violations, pipeline continues.")


if __name__ == "__main__":
    main()
