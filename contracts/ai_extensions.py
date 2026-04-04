"""
AI Contract Extensions — three AI-specific data contract checks.

  1. Embedding drift detection (cosine distance from stored baseline centroid)
  2. Prompt input schema validation (JSON Schema enforcement + quarantine)
  3. LLM output schema violation rate tracking (trend analysis)

Usage:
    python contracts/ai_extensions.py \
        --extractions outputs/week3/verdicts.jsonl \
        --verdicts outputs/week3/verdicts.jsonl \
        --output validation_reports/ai_extensions.json
"""
import argparse
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from jsonschema import validate, ValidationError
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


# ---------------------------------------------------------------------------
# Extension 1: Embedding Drift Detection
# ---------------------------------------------------------------------------

def _compute_centroid(vectors: List[List[float]]) -> List[float]:
    """Compute mean centroid of embedding vectors (numpy-free fallback)."""
    if HAS_NUMPY:
        arr = np.array(vectors)
        return arr.mean(axis=0).tolist()
    n = len(vectors)
    dim = len(vectors[0])
    centroid = [0.0] * dim
    for v in vectors:
        for i in range(dim):
            centroid[i] += v[i]
    return [c / n for c in centroid]


def _cosine_distance(a: List[float], b: List[float]) -> float:
    """Compute cosine distance = 1 - cosine_similarity."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 1.0
    similarity = dot / (norm_a * norm_b)
    return max(0.0, 1.0 - similarity)


def _simple_text_embedding(texts: List[str], dim: int = 64) -> List[List[float]]:
    """
    Simple deterministic text embedding for drift detection.
    Uses character-frequency features — no API calls needed.
    This provides meaningful drift detection for text content changes.
    """
    vectors = []
    for text in texts:
        text_lower = text.lower()
        # Character frequency features
        vec = [0.0] * dim
        for i, ch in enumerate(text_lower):
            idx = ord(ch) % dim
            vec[idx] += 1.0
        # Normalize
        magnitude = math.sqrt(sum(x * x for x in vec))
        if magnitude > 0:
            vec = [x / magnitude for x in vec]
        # Add text length and word count features
        vec[0] = len(text) / 1000.0
        vec[1] = len(text.split()) / 100.0
        vectors.append(vec)
    return vectors


def check_embedding_drift(
    texts: List[str],
    baseline_path: str = "schema_snapshots/embedding_baselines.json",
    threshold: float = 0.15,
    sample_size: int = 200,
) -> Dict[str, Any]:
    """
    Check for embedding drift by comparing current text centroid
    against stored baseline centroid using cosine distance.

    Reads from extraction text values (e.g., outputs/week3/extractions.jsonl
    extracted_facts[*].text or reasoning fields).
    """
    # Sample texts
    sample = texts[:sample_size]
    if not sample:
        return {
            "status": "ERROR",
            "drift_score": None,
            "threshold": threshold,
            "message": "No text samples provided for embedding drift check.",
        }

    # Compute embeddings and centroid
    vectors = _simple_text_embedding(sample)
    current_centroid = _compute_centroid(vectors)

    baseline_file = Path(baseline_path)
    if not baseline_file.exists():
        # Write baseline
        baseline_file.parent.mkdir(parents=True, exist_ok=True)
        with open(baseline_file, "w") as f:
            json.dump({
                "centroid": current_centroid,
                "sample_size": len(sample),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }, f)
        return {
            "status": "BASELINE_SET",
            "drift_score": 0.0,
            "threshold": threshold,
            "sample_size": len(sample),
            "message": "Baseline established. Run again to detect drift.",
        }

    # Load baseline and compute distance
    with open(baseline_file) as f:
        baseline_data = json.load(f)
    baseline_centroid = baseline_data["centroid"]

    drift = _cosine_distance(current_centroid, baseline_centroid)
    drift_rounded = round(drift, 4)

    status = "FAIL" if drift > threshold else "PASS"
    interpretation = (
        "Semantic content has shifted significantly from baseline. "
        "Investigate whether extraction quality or source documents have changed."
        if status == "FAIL"
        else "Embedding drift within acceptable bounds — content is stable."
    )

    return {
        "status": status,
        "drift_score": drift_rounded,
        "threshold": threshold,
        "sample_size": len(sample),
        "baseline_sample_size": baseline_data.get("sample_size", "unknown"),
        "baseline_created_at": baseline_data.get("created_at", "unknown"),
        "method": "cosine_distance",
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# Extension 2: Prompt Input Schema Validation
# ---------------------------------------------------------------------------

PROMPT_INPUT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["doc_id", "source_path"],
    "properties": {
        "doc_id": {"type": "string", "minLength": 36, "maxLength": 36},
        "source_path": {"type": "string", "minLength": 1},
    },
}

# Week 3 verdict prompt schema — validates document metadata
WEEK3_PROMPT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["audit_id", "target_repo", "dimension_id"],
    "properties": {
        "audit_id": {"type": "string", "minLength": 36, "maxLength": 36},
        "target_repo": {"type": "string", "minLength": 1},
        "dimension_id": {"type": "string", "minLength": 1},
    },
}


def validate_prompt_inputs(
    records: List[Dict],
    schema: Optional[Dict] = None,
    quarantine_path: str = "outputs/quarantine/",
) -> Dict[str, Any]:
    """
    Validate prompt input records against a JSON Schema.
    Non-conforming records are routed to quarantine — never silently dropped.
    """
    if schema is None:
        schema = WEEK3_PROMPT_SCHEMA

    valid_count = 0
    quarantined = []

    if HAS_JSONSCHEMA:
        for i, r in enumerate(records):
            try:
                validate(instance=r, schema=schema)
                valid_count += 1
            except ValidationError as e:
                quarantined.append({
                    "record_index": i,
                    "record_id": r.get("audit_id", r.get("doc_id", f"record_{i}")),
                    "error": str(e.message),
                    "path": list(e.path),
                })
    else:
        # Fallback: manual required field check
        required = schema.get("required", [])
        for i, r in enumerate(records):
            missing = [f for f in required if f not in r or r[f] is None]
            if missing:
                quarantined.append({
                    "record_index": i,
                    "record_id": r.get("audit_id", r.get("doc_id", f"record_{i}")),
                    "error": f"Missing required fields: {missing}",
                    "path": [],
                })
            else:
                valid_count += 1

    # Write quarantined records
    if quarantined:
        qpath = Path(quarantine_path)
        qpath.mkdir(parents=True, exist_ok=True)
        with open(qpath / "quarantine.jsonl", "a") as f:
            for q in quarantined:
                f.write(json.dumps(q) + "\n")

    return {
        "status": "PASS" if not quarantined else "WARN",
        "valid": valid_count,
        "quarantined": len(quarantined),
        "total": len(records),
        "quarantine_path": quarantine_path if quarantined else None,
        "message": (
            f"{valid_count} records passed schema validation, "
            f"{len(quarantined)} quarantined."
        ),
    }


# ---------------------------------------------------------------------------
# Extension 3: LLM Output Schema Violation Rate Tracking
# ---------------------------------------------------------------------------

def check_output_violation_rate(
    verdict_records: List[Dict],
    expected_enum_field: str = "dimension_id",
    expected_values: Optional[List[str]] = None,
    baseline_rate: Optional[float] = None,
    warn_threshold: float = 0.02,
    violation_log_path: str = "violation_log/violations.jsonl",
) -> Dict[str, Any]:
    """
    Track LLM output schema violation rate.
    Reads from Week 2 verdict records (or Week 3 as available).
    Computes violation_rate and trend, writes WARN to violation log on threshold breach.
    """
    if expected_values is None:
        expected_values = [
            "git_forensic_analysis", "state_management_rigor",
            "graph_orchestration", "safe_tool_engineering",
            "structured_output_enforcement", "judicial_nuance",
            "chief_justice_synthesis", "theoretical_depth",
            "report_accuracy", "swarm_visual",
        ]

    total = len(verdict_records)
    if total == 0:
        return {
            "status": "ERROR",
            "total_outputs": 0,
            "schema_violations": 0,
            "violation_rate": 0.0,
            "trend": "unknown",
            "message": "No verdict records provided.",
        }

    # Check each verdict for schema conformance
    violations = 0
    violation_details = []
    for i, v in enumerate(verdict_records):
        issues = []
        # Check enum field
        val = v.get(expected_enum_field)
        if val not in expected_values:
            issues.append(f"{expected_enum_field}={val!r} not in expected enum")

        # Check score ranges
        for score_field in ("final_score", "prosecutor_score", "defense_score", "tech_lead_score"):
            score = v.get(score_field)
            if score is not None and (not isinstance(score, int) or score < 1 or score > 5):
                issues.append(f"{score_field}={score!r} not in 1-5 range")

        # Check verdict enum if present
        verdict_val = v.get("overall_verdict")
        if verdict_val is not None and verdict_val not in ("PASS", "FAIL", "WARN"):
            issues.append(f"overall_verdict={verdict_val!r} not in (PASS, FAIL, WARN)")

        # Check confidence range
        conf = v.get("evidence_confidence", v.get("confidence"))
        if conf is not None:
            try:
                conf_float = float(conf)
                if conf_float < 0.0 or conf_float > 1.0:
                    issues.append(f"confidence={conf_float} outside 0.0-1.0 range")
            except (TypeError, ValueError):
                issues.append(f"confidence={conf!r} is not numeric")

        if issues:
            violations += 1
            violation_details.append({
                "record_index": i,
                "issues": issues,
            })

    rate = round(violations / total, 4)

    # Determine trend
    trend = "unknown"
    if baseline_rate is not None:
        if rate > baseline_rate * 1.5:
            trend = "rising"
        elif rate < baseline_rate * 0.5:
            trend = "falling"
        else:
            trend = "stable"

    status = "PASS"
    if rate > warn_threshold or trend == "rising":
        status = "WARN"

    # Write WARN to violation log if threshold breached
    if status == "WARN":
        vlog_path = Path(violation_log_path)
        vlog_path.parent.mkdir(parents=True, exist_ok=True)
        warn_entry = {
            "violation_id": str(uuid.uuid4()),
            "check_id": "ai_ext.output_schema_violation_rate",
            "contract_id": "week3-automaton-auditor-verdicts",
            "column_name": "llm_output_schema",
            "check_type": "output_violation_rate",
            "severity": "WARNING",
            "detected_at": datetime.now(timezone.utc).isoformat(),
            "actual_value": f"violation_rate={rate}, trend={trend}",
            "expected": f"rate <= {warn_threshold}",
            "message": (
                f"LLM output schema violation rate is {rate:.2%} ({violations}/{total}). "
                f"Trend: {trend}. Investigate prompt degradation or model behavior change."
            ),
            "records_failing": violations,
        }
        with open(vlog_path, "a") as f:
            f.write(json.dumps(warn_entry) + "\n")

    return {
        "status": status,
        "total_outputs": total,
        "schema_violations": violations,
        "violation_rate": rate,
        "trend": trend,
        "baseline_rate": baseline_rate,
        "warn_threshold": warn_threshold,
        "violation_details": violation_details[:5],
        "message": (
            f"LLM output schema violation rate: {rate:.2%} ({violations}/{total}). "
            f"Trend: {trend}."
        ),
    }


# ---------------------------------------------------------------------------
# Single entry point
# ---------------------------------------------------------------------------

def run_all_extensions(
    extraction_records: List[Dict],
    verdict_records: List[Dict],
    embedding_baseline_path: str = "schema_snapshots/embedding_baselines.json",
    quarantine_path: str = "outputs/quarantine/",
    violation_log_path: str = "violation_log/violations.jsonl",
) -> Dict[str, Any]:
    """Run all three AI contract extensions and return combined results."""
    results = {}

    # Extension 1: Embedding drift
    print("  [1/3] Checking embedding drift...")
    texts = []
    for r in extraction_records:
        # Extract text from reasoning field (Week 3 verdicts)
        if "reasoning" in r:
            texts.append(r["reasoning"])
        # Extract from extracted_facts if present
        for fact in r.get("extracted_facts", []):
            if "text" in fact:
                texts.append(fact["text"])

    results["embedding_drift"] = check_embedding_drift(
        texts, baseline_path=embedding_baseline_path
    )
    print(f"    Status: {results['embedding_drift']['status']}, "
          f"Drift: {results['embedding_drift'].get('drift_score', 'N/A')}")

    # Extension 2: Prompt input validation
    print("  [2/3] Validating prompt inputs...")
    results["prompt_input_validation"] = validate_prompt_inputs(
        extraction_records, quarantine_path=quarantine_path
    )
    print(f"    Status: {results['prompt_input_validation']['status']}, "
          f"Valid: {results['prompt_input_validation']['valid']}, "
          f"Quarantined: {results['prompt_input_validation']['quarantined']}")

    # Extension 3: LLM output schema violation rate
    print("  [3/3] Computing LLM output schema violation rate...")
    results["output_violation_rate"] = check_output_violation_rate(
        verdict_records, violation_log_path=violation_log_path
    )
    print(f"    Status: {results['output_violation_rate']['status']}, "
          f"Rate: {results['output_violation_rate']['violation_rate']:.2%}, "
          f"Trend: {results['output_violation_rate']['trend']}")

    results["run_timestamp"] = datetime.now(timezone.utc).isoformat()
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description="AI Contract Extensions")
    parser.add_argument("--extractions", required=True,
                        help="Path to extraction/verdict JSONL (Week 3)")
    parser.add_argument("--verdicts", required=True,
                        help="Path to verdict JSONL for output schema check")
    parser.add_argument("--output", default="validation_reports/ai_extensions.json",
                        help="Output path for combined results")
    args = parser.parse_args()

    print(f"[1/2] Loading data...")
    print(f"  Extractions: {args.extractions}")
    print(f"  Verdicts:    {args.verdicts}")

    extraction_records = _load_jsonl(args.extractions)
    verdict_records = _load_jsonl(args.verdicts)

    print(f"  Loaded {len(extraction_records)} extraction records")
    print(f"  Loaded {len(verdict_records)} verdict records")

    print(f"\n[2/2] Running AI contract extensions...")
    results = run_all_extensions(extraction_records, verdict_records)

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written: {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"AI Contract Extension Results")
    print(f"{'='*60}")
    for ext_name, ext_result in results.items():
        if isinstance(ext_result, dict) and "status" in ext_result:
            print(f"  {ext_name}: {ext_result['status']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
