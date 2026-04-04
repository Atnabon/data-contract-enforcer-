"""
SchemaEvolutionAnalyzer — diffs consecutive schema snapshots and classifies changes.

Loads two timestamped snapshots from schema_snapshots/{contract_id}/ and:
  1. Diffs them field-by-field
  2. Classifies each change using the breaking-change taxonomy
  3. Generates a migration impact report with blast radius and rollback plan

Usage:
    python contracts/schema_analyzer.py \
        --contract-id week3-automaton-auditor-verdicts \
        --since "7 days ago" \
        --output validation_reports/schema_evolution.json
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# ---------------------------------------------------------------------------
# Taxonomy classification
# ---------------------------------------------------------------------------

# Breaking changes (backward-incompatible)
BREAKING_TYPES = {
    "add_required_field": "New required field added — consumers missing it will fail",
    "remove_field": "Field removed — consumers depending on it will fail",
    "rename_field": "Field renamed — consumers referencing old name will fail",
    "narrow_type": "Type narrowed (e.g. float -> int, wider range -> narrower range) — data loss",
    "remove_enum_value": "Enum value removed — consumers expecting it will fail",
    "change_type": "Type changed — consumers will encounter type errors",
    "narrow_range": "Numeric range narrowed — values previously valid now rejected",
}

# Compatible changes (backward-compatible)
COMPATIBLE_TYPES = {
    "add_optional_field": "New optional field added — consumers can ignore it",
    "widen_type": "Type widened (e.g. int -> float) — existing values still valid",
    "add_enum_value": "Enum value added — existing consumers unaffected",
    "widen_range": "Numeric range widened — more values accepted",
    "description_change": "Description updated — no structural impact",
}


def classify_change(
    field: str,
    old_clause: Optional[Dict],
    new_clause: Optional[Dict],
) -> Tuple[str, str, str]:
    """
    Classify a schema change between two contract versions.
    Returns: (verdict, change_type, description)
    """
    # Field added
    if old_clause is None and new_clause is not None:
        if new_clause.get("required", False):
            return ("BREAKING", "add_required_field",
                    f"New required field '{field}' added. "
                    f"All consumers must add support before migration.")
        return ("COMPATIBLE", "add_optional_field",
                f"New optional field '{field}' added. "
                f"Consumers can safely ignore this field.")

    # Field removed
    if old_clause is not None and new_clause is None:
        return ("BREAKING", "remove_field",
                f"Field '{field}' removed. "
                f"Deprecation period mandatory before removal.")

    if old_clause is None or new_clause is None:
        return ("COMPATIBLE", "no_change", f"No material change to '{field}'.")

    # Type change
    old_type = old_clause.get("type", "string")
    new_type = new_clause.get("type", "string")
    if old_type != new_type:
        # Detect narrow type: float -> int (e.g. confidence 0.0-1.0 -> 0-100)
        if old_type == "number" and new_type == "integer":
            return ("BREAKING", "narrow_type",
                    f"CRITICAL: Field '{field}' narrowed from {old_type} to {new_type}. "
                    f"Example: float 0.0-1.0 -> int 0-100 causes data loss and "
                    f"breaks all downstream consumers expecting float precision. "
                    f"This is the most dangerous schema change class.")
        if old_type == "integer" and new_type == "number":
            return ("COMPATIBLE", "widen_type",
                    f"Field '{field}' widened from {old_type} to {new_type}. "
                    f"Existing integer values are valid numbers.")
        return ("BREAKING", "change_type",
                f"Field '{field}' type changed from {old_type} to {new_type}.")

    # Range change — specifically catch the 0.0-1.0 -> 0-100 case
    old_min = old_clause.get("minimum")
    old_max = old_clause.get("maximum")
    new_min = new_clause.get("minimum")
    new_max = new_clause.get("maximum")

    if old_max is not None and new_max is not None:
        if new_max > old_max:
            # Check for the canonical scale change: max went from 1.0 to 100
            if old_max <= 1.0 and new_max >= 100:
                return ("BREAKING", "narrow_type",
                        f"CRITICAL: Field '{field}' range changed from "
                        f"[{old_min}, {old_max}] to [{new_min}, {new_max}]. "
                        f"This indicates a scale change (float 0.0-1.0 -> int 0-100). "
                        f"All downstream consumers interpreting values as fractions "
                        f"will produce incorrect results.")
            return ("COMPATIBLE", "widen_range",
                    f"Field '{field}' maximum widened from {old_max} to {new_max}.")
        elif new_max < old_max:
            return ("BREAKING", "narrow_range",
                    f"Field '{field}' maximum narrowed from {old_max} to {new_max}. "
                    f"Values previously valid may now be rejected.")

    if old_min is not None and new_min is not None:
        if new_min > old_min:
            return ("BREAKING", "narrow_range",
                    f"Field '{field}' minimum raised from {old_min} to {new_min}.")
        elif new_min < old_min:
            return ("COMPATIBLE", "widen_range",
                    f"Field '{field}' minimum lowered from {old_min} to {new_min}.")

    # Enum changes
    old_enum = set(old_clause.get("enum", []))
    new_enum = set(new_clause.get("enum", []))
    if old_enum and new_enum:
        removed = old_enum - new_enum
        added = new_enum - old_enum
        if removed:
            return ("BREAKING", "remove_enum_value",
                    f"Enum values removed from '{field}': {sorted(removed)}. "
                    f"Consumers expecting these values will fail.")
        if added:
            return ("COMPATIBLE", "add_enum_value",
                    f"Enum values added to '{field}': {sorted(added)}.")

    # Required change
    old_req = old_clause.get("required", False)
    new_req = new_clause.get("required", False)
    if not old_req and new_req:
        return ("BREAKING", "add_required_field",
                f"Field '{field}' changed from optional to required.")
    if old_req and not new_req:
        return ("COMPATIBLE", "add_optional_field",
                f"Field '{field}' changed from required to optional.")

    # Description-only change
    if old_clause.get("description") != new_clause.get("description"):
        return ("COMPATIBLE", "description_change",
                f"Description updated for '{field}'.")

    return ("COMPATIBLE", "no_change", f"No material change to '{field}'.")


# ---------------------------------------------------------------------------
# Snapshot loading and diffing
# ---------------------------------------------------------------------------

def load_snapshots(
    contract_id: str,
    snapshots_dir: str = "schema_snapshots",
    since: Optional[str] = None,
) -> List[Tuple[str, Dict]]:
    """Load timestamped snapshots sorted by filename (timestamp)."""
    snap_dir = Path(snapshots_dir) / contract_id
    if not snap_dir.exists():
        print(f"WARNING: Snapshot directory not found: {snap_dir}")
        return []

    yamls = sorted(snap_dir.glob("*.yaml"))
    if not yamls:
        print(f"WARNING: No snapshots found in {snap_dir}")
        return []

    snapshots = []
    for ypath in yamls:
        with open(ypath) as f:
            data = yaml.safe_load(f)
        snapshots.append((ypath.stem, data))

    return snapshots


def diff_schemas(
    old_schema: Dict[str, Dict],
    new_schema: Dict[str, Dict],
) -> List[Dict]:
    """Diff two schema dicts and classify each change."""
    changes = []
    all_fields = set(list(old_schema.keys()) + list(new_schema.keys()))

    for field in sorted(all_fields):
        old_clause = old_schema.get(field)
        new_clause = new_schema.get(field)

        if old_clause == new_clause:
            continue

        verdict, change_type, description = classify_change(
            field, old_clause, new_clause
        )
        changes.append({
            "field": field,
            "verdict": verdict,
            "change_type": change_type,
            "description": description,
            "old_clause": old_clause,
            "new_clause": new_clause,
        })

    return changes


# ---------------------------------------------------------------------------
# Migration report construction
# ---------------------------------------------------------------------------

def build_migration_report(
    contract_id: str,
    old_timestamp: str,
    new_timestamp: str,
    changes: List[Dict],
    registry_path: str = "contract_registry/subscriptions.yaml",
) -> Dict:
    """Build a full migration impact report."""
    breaking = [c for c in changes if c["verdict"] == "BREAKING"]
    compatible = [c for c in changes if c["verdict"] == "COMPATIBLE"]

    # Determine overall compatibility
    if not changes:
        overall = "NO_CHANGES"
    elif breaking:
        overall = "BREAKING"
    else:
        overall = "BACKWARD_COMPATIBLE"

    # Load registry for blast radius
    blast_radius = []
    try:
        with open(registry_path) as f:
            registry = yaml.safe_load(f)
        for sub in registry.get("subscriptions", []):
            if sub["contract_id"] == contract_id:
                for bf in sub.get("breaking_fields", []):
                    field_name = bf["field"] if isinstance(bf, dict) else bf
                    affected_changes = [c for c in breaking if field_name in c["field"]]
                    if affected_changes:
                        blast_radius.append({
                            "subscriber_id": sub["subscriber_id"],
                            "contact": sub.get("contact", "unknown"),
                            "affected_fields": [c["field"] for c in affected_changes],
                            "validation_mode": sub.get("validation_mode", "AUDIT"),
                        })
    except FileNotFoundError:
        pass

    # Build migration checklist
    checklist = []
    for i, bc in enumerate(breaking, 1):
        checklist.append({
            "step": i,
            "action": f"Update consumer code to handle new schema for field '{bc['field']}'",
            "field": bc["field"],
            "change_type": bc["change_type"],
            "description": bc["description"],
        })
    checklist.append({
        "step": len(breaking) + 1,
        "action": "Run ValidationRunner in AUDIT mode against new schema to verify no regressions",
        "field": "__all__",
        "change_type": "verification",
        "description": "End-to-end validation after migration",
    })
    checklist.append({
        "step": len(breaking) + 2,
        "action": "Update statistical baselines in schema_snapshots/baselines.json after schema migration",
        "field": "__baselines__",
        "change_type": "baseline_refresh",
        "description": "Baselines must be re-established post-migration to avoid false drift alerts",
    })

    # Build rollback plan
    rollback_plan = {
        "procedure": [
            f"Revert to snapshot {old_timestamp} by restoring schema_snapshots/{contract_id}/{old_timestamp}.yaml",
            "Re-run ContractGenerator with original data to regenerate contract",
            "Notify all affected subscribers listed in blast_radius",
            "Re-establish statistical baselines from pre-migration data",
        ],
        "baselines_to_restore": [
            "schema_snapshots/baselines.json — numeric column means and stddevs",
        ],
        "estimated_impact": f"{len(blast_radius)} subscribers require notification",
    }

    # Per-consumer failure mode analysis
    consumer_failure_modes = []
    for sub in blast_radius:
        consumer_failure_modes.append({
            "subscriber_id": sub["subscriber_id"],
            "contact": sub["contact"],
            "failure_mode": (
                f"Subscriber '{sub['subscriber_id']}' consumes {sub['affected_fields']}. "
                f"Without migration, validation in {sub['validation_mode']} mode will "
                f"{'block pipeline' if sub['validation_mode'] == 'ENFORCE' else 'emit warnings'}."
            ),
            "affected_fields": sub["affected_fields"],
        })

    return {
        "report_id": str(uuid.uuid4()) if 'uuid' in dir() else f"evo-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        "contract_id": contract_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_old": old_timestamp,
        "snapshot_new": new_timestamp,
        "overall_verdict": overall,
        "total_changes": len(changes),
        "breaking_changes": len(breaking),
        "compatible_changes": len(compatible),
        "changes": changes,
        "blast_radius": blast_radius,
        "migration_checklist": checklist,
        "rollback_plan": rollback_plan,
        "consumer_failure_modes": consumer_failure_modes,
        "taxonomy_reference": {
            "breaking": list(BREAKING_TYPES.keys()),
            "compatible": list(COMPATIBLE_TYPES.keys()),
        },
    }


# Need uuid import for report_id
import uuid as _uuid


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SchemaEvolutionAnalyzer")
    parser.add_argument("--contract-id", required=True,
                        help="Contract ID to analyze")
    parser.add_argument("--since", default=None,
                        help="Only consider snapshots since this date (unused, for CLI compat)")
    parser.add_argument("--output", default="validation_reports/schema_evolution.json",
                        help="Output path for migration report")
    parser.add_argument("--registry",
                        default="contract_registry/subscriptions.yaml",
                        help="Path to contract registry")
    args = parser.parse_args()

    print(f"[1/3] Loading snapshots for: {args.contract_id}")
    snapshots = load_snapshots(args.contract_id)

    if len(snapshots) < 2:
        print(f"ERROR: Need at least 2 snapshots to diff, found {len(snapshots)}.")
        print("Run the ContractGenerator twice (on clean + violated data) to create snapshots.")
        raise SystemExit(1)

    # Use the two most recent snapshots
    old_ts, old_contract = snapshots[-2]
    new_ts, new_contract = snapshots[-1]

    old_schema = old_contract.get("schema", {})
    new_schema = new_contract.get("schema", {})

    print(f"  Old snapshot: {old_ts} ({len(old_schema)} fields)")
    print(f"  New snapshot: {new_ts} ({len(new_schema)} fields)")

    print(f"\n[2/3] Diffing schemas...")
    changes = diff_schemas(old_schema, new_schema)

    breaking = [c for c in changes if c["verdict"] == "BREAKING"]
    compatible = [c for c in changes if c["verdict"] == "COMPATIBLE"]
    print(f"  Changes detected: {len(changes)}")
    print(f"  Breaking:         {len(breaking)}")
    print(f"  Compatible:       {len(compatible)}")

    print(f"\n[3/3] Building migration report...")
    report = build_migration_report(
        args.contract_id, old_ts, new_ts, changes, args.registry
    )

    # Write report
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report written: {out_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Schema Evolution Report: {args.contract_id}")
    print(f"{'='*60}")
    print(f"  Overall verdict: {report['overall_verdict']}")

    for c in changes:
        icon = "BREAKING" if c["verdict"] == "BREAKING" else "OK"
        print(f"  [{icon}] {c['field']}: {c['change_type']}")
        print(f"         {c['description']}")

    if report["blast_radius"]:
        print(f"\n  Blast radius:")
        for br in report["blast_radius"]:
            print(f"    - {br['subscriber_id']} ({br['contact']}): {br['affected_fields']}")

    if report["rollback_plan"]:
        print(f"\n  Rollback plan:")
        for step in report["rollback_plan"]["procedure"]:
            print(f"    - {step}")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
