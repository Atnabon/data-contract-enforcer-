"""
ContractGenerator — reads JSONL outputs and produces Bitol YAML contracts.

Maps the actual weekly project outputs to Bitol v3.0.0 data contracts:
  - Week 3 (automaton-auditor): audit verdict records
  - Week 5 (The-Ledger): stored event records

Usage:
    python contracts/generator.py \
        --source outputs/week3/verdicts.jsonl \
        --contract-id week3-automaton-auditor-verdicts \
        --lineage outputs/week4/lineage_snapshots.jsonl \
        --output generated_contracts/

    python contracts/generator.py \
        --source outputs/week5/events.jsonl \
        --contract-id week5-ledger-events \
        --lineage outputs/week4/lineage_snapshots.jsonl \
        --output generated_contracts/
"""
import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from dotenv import load_dotenv

load_dotenv()

from contracts.models import ColumnProfile, ContractClause

# ---------------------------------------------------------------------------
# STAGE 1 — Load and flatten
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict]:
    """Read a JSONL file into a list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def flatten_for_profile(records: List[Dict]) -> pd.DataFrame:
    """
    Flatten nested JSONL to a flat DataFrame for profiling.

    Handles two schemas:
    - Week 3 verdict records: flat structure with nested code_refs[] array
    - Week 5 event records: flat with nested payload{} and metadata{}

    For nested dicts (payload, metadata): extracts top-level scalar keys
    with prefixed column names.
    For nested lists (code_refs): drops (too variable to profile per column).
    """
    rows = []
    for r in records:
        row = {}
        for k, v in r.items():
            if isinstance(v, dict):
                # Flatten one level of nested dicts
                for dk, dv in v.items():
                    if not isinstance(dv, (list, dict)):
                        row[f"{k}_{dk}"] = dv
            elif isinstance(v, list):
                # Skip list columns — too heterogeneous to profile as columns
                pass
            else:
                row[k] = v
        rows.append(row)

    df = pd.DataFrame(rows)

    # Warn if any confidence-like field is not float
    for col in df.columns:
        if "confidence" in col or "score" in col:
            if str(df[col].dtype) not in ("float64", "int64"):
                print(
                    f"WARNING: {col} dtype is '{df[col].dtype}', expected numeric. "
                    f"Check for mixed types — may indicate a contract violation."
                )

    return df


# ---------------------------------------------------------------------------
# STAGE 2 — Per-column profiling
# ---------------------------------------------------------------------------

def profile_column(series: pd.Series, col_name: str) -> ColumnProfile:
    """Build a ColumnProfile for one column."""
    normalized = series.dropna().map(
        lambda v: json.dumps(v, sort_keys=True)
        if isinstance(v, (list, dict))
        else v
    )

    profile = ColumnProfile(
        name=col_name,
        dtype=str(series.dtype),
        null_fraction=float(series.isna().mean()),
        cardinality_estimate=int(normalized.nunique()),
        sample_values=[str(v) for v in normalized.unique()[:5]],
    )
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        numeric = series.dropna().astype(float)
        profile.stats = {
            "min":    float(numeric.min()),
            "max":    float(numeric.max()),
            "mean":   float(numeric.mean()),
            "p25":    float(numeric.quantile(0.25)),
            "p50":    float(numeric.quantile(0.50)),
            "p75":    float(numeric.quantile(0.75)),
            "p95":    float(numeric.quantile(0.95)),
            "p99":    float(numeric.quantile(0.99)),
            "stddev": float(numeric.std()) if len(numeric) > 1 else 0.0,
        }
    return profile


# ---------------------------------------------------------------------------
# STAGE 3 — Bitol clause generation
# ---------------------------------------------------------------------------

def infer_type(dtype_str: str) -> str:
    mapping = {
        "float64": "number",
        "float32": "number",
        "int64":   "integer",
        "int32":   "integer",
        "bool":    "boolean",
        "object":  "string",
    }
    return mapping.get(dtype_str, "string")


def column_to_clause(profile: ColumnProfile) -> Dict[str, Any]:
    """
    Map a ColumnProfile to a Bitol contract clause dict.
    Domain rules applied in priority order.
    """
    clause: Dict[str, Any] = {
        "type":     infer_type(profile.dtype),
        "required": profile.null_fraction == 0.0,
    }

    # Confidence/score fields: 0.0-1.0 float range — BREAKING if scale changed
    if "confidence" in profile.name and clause["type"] == "number":
        clause["minimum"]     = 0.0
        clause["maximum"]     = 1.0
        clause["description"] = (
            "Confidence score from automaton-auditor detective evidence. "
            "Must remain 0.0-1.0 float. BREAKING if changed to 0-100 integer scale."
        )

    # final_score, prosecutor_score, defense_score, tech_lead_score: 1-5 integer
    if profile.name in ("final_score", "prosecutor_score", "defense_score", "tech_lead_score"):
        clause["minimum"] = 1
        clause["maximum"] = 5
        clause["description"] = (
            f"Judicial score (1=Vibe Coder, 5=Master Thinker). "
            f"BREAKING if scale changes from 1-5."
        )

    # stream_position and global_position: must be >= 1
    if profile.name in ("stream_position", "global_position"):
        clause["minimum"] = 1
        clause["description"] = (
            f"{profile.name}: append-order counter. "
            f"Must be >= 1. BREAKING if reset to 0 (OCC would break)."
        )
        if profile.stats:
            clause["maximum"] = profile.stats["max"]

    # event_version: must be >= 1, bounded by schema evolution history
    if profile.name == "event_version":
        clause["minimum"] = 1
        clause["maximum"] = 3
        clause["description"] = (
            "Schema version for upcasting. Max=3 reflects current upcaster registry. "
            "Increment only with a registered upcaster."
        )

    # Low-cardinality string columns → enum constraint
    if (profile.cardinality_estimate <= 12
            and profile.dtype == "object"
            and len(profile.sample_values) >= profile.cardinality_estimate
            and profile.name not in ("audit_id", "target_repo", "stream_id",
                                     "reasoning", "dissent_summary")):
        clause["enum"] = sorted(set(profile.sample_values))

    # _id suffix → UUID format
    if profile.name.endswith("_id") or profile.name in ("audit_id", "event_id"):
        clause["format"]  = "uuid"
        clause["pattern"] = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"

    # _at suffix → ISO 8601 date-time
    if profile.name.endswith("_at"):
        clause["format"] = "date-time"

    # Numeric range from profiled stats (skip fields with domain-specific bounds)
    skip_stats = {
        "confidence", "final_score", "prosecutor_score",
        "defense_score", "tech_lead_score", "event_version",
        "stream_position", "global_position",
    }
    if profile.stats and not any(s in profile.name for s in skip_stats):
        clause["minimum"] = profile.stats["min"]
        clause["maximum"] = profile.stats["max"]

    return clause


# ---------------------------------------------------------------------------
# STAGE 4A — Lineage injection
# ---------------------------------------------------------------------------

def inject_lineage(contract: Dict, lineage_path: str,
                   contract_id: str) -> Dict:
    """
    Read the latest lineage snapshot and find downstream consumers.
    Inject into contract['lineage']['downstream'].
    """
    with open(lineage_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    snapshot = json.loads(lines[-1])

    source_keyword = contract_id.split("-")[0]  # e.g. 'week3' or 'week5'

    downstream = []
    for edge in snapshot.get("edges", []):
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        rel = edge.get("relationship", "UNKNOWN")
        if source_keyword in src or source_keyword in tgt:
            downstream.append({
                "id":          tgt,
                "description": f"Downstream consumer via {rel} edge",
                "fields_consumed": _infer_consumed_fields(contract_id),
                "breaking_if_changed": _infer_breaking_fields(contract_id),
            })

    contract["lineage"] = {
        "upstream":   _infer_upstream(contract_id),
        "downstream": downstream,
    }
    return contract


def _infer_consumed_fields(contract_id: str) -> List[str]:
    """Return the key fields consumed by downstream systems."""
    if "week3" in contract_id:
        return ["audit_id", "dimension_id", "final_score", "evidence_confidence"]
    if "week5" in contract_id:
        return ["event_id", "stream_id", "event_type", "stream_position", "payload"]
    return ["id"]


def _infer_breaking_fields(contract_id: str) -> List[str]:
    """Return fields whose type/scale change would be breaking."""
    if "week3" in contract_id:
        return [
            "final_score",       # scale 1-5 must not change
            "evidence_confidence",  # 0.0-1.0 float must not become 0-100
            "dimension_id",      # enum values must remain stable
        ]
    if "week5" in contract_id:
        return [
            "event_type",        # enum must remain stable
            "stream_position",   # OCC depends on monotonic increment
            "event_version",     # upcasters keyed on this
        ]
    return []


def _infer_upstream(contract_id: str) -> List[Dict]:
    """Return upstream producers for known contracts."""
    if "week3" in contract_id:
        return [{
            "id": "file::automaton-auditor/src/nodes/justice.py",
            "description": "Chief Justice synthesis node that writes CriterionVerdict records",
        }]
    if "week5" in contract_id:
        return [{
            "id": "file::The-Ledger/src/event_store.py",
            "description": "EventStore.append() — writes StoredEvent records to PostgreSQL then exports to JSONL",
        }]
    return []


# ---------------------------------------------------------------------------
# STAGE 4B — LLM annotation for ambiguous columns
# ---------------------------------------------------------------------------

AMBIGUOUS_PATTERNS = [
    "hash", "ref", "version", "summary", "reasoning",
    "basis", "metadata", "payload",
]


def is_ambiguous(col_name: str) -> bool:
    return any(p in col_name.lower() for p in AMBIGUOUS_PATTERNS)


def annotate_with_llm(col_name: str, table_name: str,
                      sample_values: List[str],
                      adjacent_cols: List[str]) -> Dict:
    """
    Call Claude to annotate ambiguous columns.
    Falls back to a structured placeholder if no API key is set.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key.startswith("sk-ant-PLACEHOLDER"):
        return _placeholder_annotation(col_name, table_name)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        prompt = (
            f"You are a data contract specialist for an AI training pipeline.\n"
            f"Table: {table_name}\n"
            f"Column: {col_name}\n"
            f"Sample values: {sample_values}\n"
            f"Adjacent columns: {adjacent_cols}\n\n"
            f"Provide exactly three things as JSON with keys "
            f"'description', 'business_rule', 'relationships':\n"
            f"(a) description: plain-English meaning of this column\n"
            f"(b) business_rule: a machine-checkable validation expression\n"
            f"(c) relationships: any cross-column dependency\n"
            f"Respond with valid JSON only."
        )
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        text = message.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except Exception as exc:
        return {
            "description":   f"Column {col_name} in {table_name}.",
            "business_rule": f"Annotation failed: {exc}",
            "relationships": "None identified.",
        }


def _placeholder_annotation(col_name: str, table_name: str) -> Dict:
    """Domain-aware fallback annotations for known ambiguous columns."""
    known = {
        "reasoning": {
            "description": "Chief Justice synthesis chain explaining how each deterministic rule was applied.",
            "business_rule": "len(reasoning) > 0; must not be empty string.",
            "relationships": "Derived from prosecutor_score, defense_score, tech_lead_score via synthesis rules.",
        },
        "dissent_summary": {
            "description": "Present only when judicial score variance exceeds 2. Null otherwise.",
            "business_rule": "IF max(scores) - min(scores) > 2 THEN dissent_summary IS NOT NULL.",
            "relationships": "Cross-field constraint with final_score, prosecutor_score, defense_score, tech_lead_score.",
        },
        "metadata_correlation_id": {
            "description": "Trace correlation ID linking causally related events across aggregate streams.",
            "business_rule": "If present, must be valid UUID or null.",
            "relationships": "Links event records that belong to the same logical business transaction.",
        },
        "metadata_causation_id": {
            "description": "ID of the command or event that caused this event to be raised.",
            "business_rule": "If present, must be valid UUID or null.",
            "relationships": "Forms a causal chain; downstream replay depends on this for gas-town recovery.",
        },
        "payload_summary": {
            "description": "JSON object containing the domain event payload. Schema varies per event_type.",
            "business_rule": "Must be parseable JSON; required keys vary by event_type.",
            "relationships": "payload schema is determined by event_type + event_version combination.",
        },
    }
    for key, annotation in known.items():
        if key in col_name.lower():
            return annotation
    return {
        "description":   f"Column {col_name} in {table_name}.",
        "business_rule": "No rule inferred (LLM annotation skipped).",
        "relationships": "None identified.",
    }


# ---------------------------------------------------------------------------
# STAGE 4C — dbt schema.yml output
# ---------------------------------------------------------------------------

def build_dbt_schema(contract_id: str, schema: Dict[str, Any]) -> Dict:
    """Produce a dbt-compatible schema.yml dict."""
    columns = []
    for col_name, clause in schema.items():
        col: Dict[str, Any] = {"name": col_name, "tests": []}
        if clause.get("required"):
            col["tests"].append("not_null")
        if clause.get("format") == "uuid":
            # avoid duplicate not_null
            if "not_null" not in col["tests"]:
                col["tests"].append("not_null")
        if clause.get("enum"):
            col["tests"].append({
                "accepted_values": {"values": clause["enum"]}
            })
        # dbt range tests via dbt-utils expression_is_true (represented as comment)
        if clause.get("minimum") is not None and clause.get("type") in ("integer", "number"):
            col["tests"].append({
                "dbt_utils.expression_is_true": {
                    "expression": f">= {clause['minimum']}"
                }
            })
        if clause.get("maximum") is not None and clause.get("type") in ("integer", "number"):
            col["tests"].append({
                "dbt_utils.expression_is_true": {
                    "expression": f"<= {clause['maximum']}"
                }
            })
        columns.append(col)

    return {
        "version": 2,
        "models": [{
            "name":        contract_id.replace("-", "_"),
            "description": f"dbt schema tests for {contract_id}",
            "columns":     columns,
        }],
    }


# ---------------------------------------------------------------------------
# MAIN — wire all stages together
# ---------------------------------------------------------------------------

def build_contract(contract_id: str, source_path: str,
                   df: pd.DataFrame,
                   profiles: Dict[str, ColumnProfile]) -> Dict:
    """Assemble the full Bitol v3.0.0 contract dict."""
    schema_clauses: Dict[str, Any] = {}
    llm_annotations: Dict[str, Any] = {}
    col_names = list(profiles.keys())

    for col_name, profile in profiles.items():
        clause = column_to_clause(profile)
        schema_clauses[col_name] = clause

        if is_ambiguous(col_name):
            adjacent = [c for c in col_names if c != col_name][:5]
            annotation = annotate_with_llm(
                col_name, contract_id, profile.sample_values, adjacent
            )
            llm_annotations[col_name] = annotation
            if "description" not in clause and annotation.get("description"):
                clause["description"] = annotation["description"]

    # Build quality checks appropriate to this contract
    quality_checks = ["row_count >= 1"]
    if "audit_id" in schema_clauses:
        quality_checks.append("missing_count(audit_id) = 0")
        quality_checks.append("duplicate_count(audit_id) = 0")
    if "event_id" in schema_clauses:
        quality_checks.append("missing_count(event_id) = 0")
        quality_checks.append("duplicate_count(event_id) = 0")
    if "final_score" in schema_clauses:
        quality_checks.append("min(final_score) >= 1")
        quality_checks.append("max(final_score) <= 5")
    if "stream_position" in schema_clauses:
        quality_checks.append("min(stream_position) >= 1")

    contract = {
        "kind":       "DataContract",
        "apiVersion": "v3.0.0",
        "id":         contract_id,
        "info": {
            "title":       f"Contract for {contract_id}",
            "version":     "1.0.0",
            "owner":       "week7-team",
            "description": (
                f"Auto-generated contract from {Path(source_path).name}. "
                f"Generated at {datetime.now(timezone.utc).isoformat()}. "
                f"Covers {len(schema_clauses)} schema fields with "
                f"{len([c for c in schema_clauses.values() if c.get('required')])} required."
            ),
        },
        "servers": {
            "local": {
                "type":   "local",
                "path":   source_path,
                "format": "jsonl",
            }
        },
        "terms": {
            "usage":       "Internal inter-system data contract. Do not publish externally.",
            "limitations": (
                "Score fields must remain on 1-5 integer scale. "
                "Confidence fields must remain on 0.0-1.0 float scale. "
                "Breaking changes require a new contract version and migration plan."
            ),
        },
        "schema":          schema_clauses,
        "llm_annotations": llm_annotations,
        "quality": {
            "type": "SodaChecks",
            "specification": {
                "checks": quality_checks,
            },
        },
    }
    return contract


def main():
    parser = argparse.ArgumentParser(description="ContractGenerator")
    parser.add_argument("--source",      required=True,
                        help="Path to input JSONL file")
    parser.add_argument("--contract-id", required=True,
                        help="Unique contract identifier")
    parser.add_argument("--lineage",     required=True,
                        help="Path to lineage_snapshots.jsonl")
    parser.add_argument("--output",      required=True,
                        help="Output directory for generated contracts")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading {args.source} ...")
    records = load_jsonl(args.source)
    print(f"      Loaded {len(records)} records.")

    print("[2/4] Flattening and profiling ...")
    df = flatten_for_profile(records)
    print(f"      DataFrame shape: {df.shape}")
    print(f"      Columns: {list(df.columns)}")

    profiles: Dict[str, ColumnProfile] = {}
    for col in df.columns:
        profiles[col] = profile_column(df[col], col)

    print("[3/4] Building Bitol contract ...")
    contract = build_contract(args.contract_id, args.source, df, profiles)

    print("[3/4] Injecting lineage ...")
    contract = inject_lineage(contract, args.lineage, args.contract_id)

    # Derive output filename from contract-id
    parts = args.contract_id.split("-")
    short_name = f"{parts[0]}_{parts[-1]}"
    yaml_path = output_dir / f"{short_name}.yaml"

    with open(yaml_path, "w") as f:
        yaml.dump(contract, f, default_flow_style=False,
                  sort_keys=False, allow_unicode=True)
    print(f"      Written: {yaml_path}")

    dbt_schema = build_dbt_schema(args.contract_id, contract["schema"])
    dbt_path = output_dir / f"{short_name}_dbt.yml"
    with open(dbt_path, "w") as f:
        yaml.dump(dbt_schema, f, default_flow_style=False,
                  sort_keys=False, allow_unicode=True)
    print(f"      Written: {dbt_path}")

    snapshot_dir = Path("schema_snapshots") / args.contract_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    snapshot_path = snapshot_dir / f"{ts}.yaml"
    shutil.copy(yaml_path, snapshot_path)
    print(f"      Snapshot: {snapshot_path}")

    # Write statistical baseline (mean, stddev per numeric column)
    baselines_path = Path("schema_snapshots") / "baselines.json"
    baselines = {
        "written_at": datetime.now(timezone.utc).isoformat(),
        "contract_id": args.contract_id,
        "columns": {},
    }
    for col in df.select_dtypes(include="number").columns:
        mean_val = float(df[col].mean())
        std_val = float(df[col].std()) if len(df[col]) > 1 else 0.0
        baselines["columns"][col] = {
            "mean": mean_val,
            "stddev": std_val,
            "min": float(df[col].min()),
            "max": float(df[col].max()),
        }
        # Flag suspicious distributions for confidence-like fields only
        if ("confidence" in col.lower()) and (mean_val > 0.99 or mean_val < 0.01):
            print(f"      WARNING: {col} mean={mean_val:.4f} — suspicious distribution "
                  f"(possibly clamped or broken)")
    with open(baselines_path, "w") as f:
        json.dump(baselines, f, indent=2)
    print(f"      Baselines: {baselines_path}")

    print("[4/4] Done.")
    print(f"\nSummary:")
    print(f"  Schema clauses    : {len(contract['schema'])}")
    print(f"  Required fields   : {sum(1 for c in contract['schema'].values() if c.get('required'))}")
    print(f"  LLM annotations   : {len(contract['llm_annotations'])}")
    print(f"  Downstream nodes  : {len(contract['lineage']['downstream'])}")
    print(f"  Quality checks    : {len(contract['quality']['specification']['checks'])}")
    print(f"  Contract YAML     : {yaml_path}")
    print(f"  dbt YAML          : {dbt_path}")
    print(f"  Snapshot          : {snapshot_path}")


if __name__ == "__main__":
    main()
