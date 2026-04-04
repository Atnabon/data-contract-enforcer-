"""
ViolationAttributor — traces contract violations to their origin.

Pipeline:
  1. Registry blast radius query (primary source)
  2. Lineage traversal for transitive depth (enrichment)
  3. Git blame for cause attribution
  4. Write violation log entry

Usage:
    python contracts/attributor.py \
        --violation validation_reports/violated_run.json \
        --lineage outputs/week4/lineage_snapshots.jsonl \
        --registry contract_registry/subscriptions.yaml \
        --output violation_log/violations.jsonl
"""
import argparse
import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Step 1 — Registry blast radius query (primary source)
# ---------------------------------------------------------------------------

def registry_blast_radius(
    contract_id: str,
    failing_field: str,
    registry_path: str,
) -> List[Dict]:
    """
    Query the contract registry for subscribers affected by a breaking field change.
    This is the PRIMARY source for blast radius — not the lineage graph.
    """
    with open(registry_path) as f:
        registry = yaml.safe_load(f)

    affected = []
    for sub in registry.get("subscriptions", []):
        if sub["contract_id"] != contract_id:
            continue
        for bf in sub.get("breaking_fields", []):
            field_name = bf["field"] if isinstance(bf, dict) else bf
            reason = bf.get("reason", "No reason documented") if isinstance(bf, dict) else ""
            if field_name == failing_field or failing_field.startswith(field_name):
                affected.append({
                    "subscriber_id": sub["subscriber_id"],
                    "subscriber_team": sub.get("subscriber_team", "unknown"),
                    "contact": sub.get("contact", "unknown"),
                    "validation_mode": sub.get("validation_mode", "AUDIT"),
                    "reason": reason.strip(),
                    "fields_consumed": sub.get("fields_consumed", []),
                })
                break
    return affected


# ---------------------------------------------------------------------------
# Step 2 — Lineage transitive depth (enrichment)
# ---------------------------------------------------------------------------

def compute_transitive_depth(
    producer_node_id: str,
    lineage_path: str,
    max_depth: int = 3,
) -> Dict[str, Any]:
    """
    BFS traversal from the producer node through PRODUCES/CONSUMES edges.
    Returns direct and transitive downstream nodes with contamination depth.
    """
    with open(lineage_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    snapshot = json.loads(lines[-1])

    edges = snapshot.get("edges", [])
    nodes = {n["node_id"]: n for n in snapshot.get("nodes", [])}

    visited = set()
    frontier = {producer_node_id}
    depth_map: Dict[str, int] = {}

    for depth in range(1, max_depth + 1):
        next_frontier = set()
        for node in frontier:
            for edge in edges:
                if (edge["source"] == node
                        and edge["relationship"] in ("PRODUCES", "WRITES", "CONSUMES")):
                    target = edge["target"]
                    if target not in visited and target != producer_node_id:
                        depth_map[target] = depth
                        next_frontier.add(target)
                        visited.add(target)
        frontier = next_frontier
        if not frontier:
            break

    return {
        "direct": [n for n, d in depth_map.items() if d == 1],
        "transitive": [n for n, d in depth_map.items() if d > 1],
        "all_affected": list(depth_map.keys()),
        "contamination_depth": max(depth_map.values()) if depth_map else 0,
        "depth_map": {n: d for n, d in depth_map.items()},
    }


# ---------------------------------------------------------------------------
# Step 3 — Git blame for cause attribution
# ---------------------------------------------------------------------------

def get_recent_commits(
    file_path: str,
    repo_root: str = ".",
    days: int = 30,
) -> List[Dict]:
    """Run git log on the identified upstream file."""
    cmd = [
        "git", "log", "--follow",
        f"--since={days} days ago",
        "--format=%H|%ae|%ai|%s",
        "--", file_path,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=repo_root, timeout=10
        )
        commits = []
        for line in result.stdout.strip().split("\n"):
            if "|" not in line:
                continue
            parts = line.split("|", 3)
            if len(parts) >= 4:
                commits.append({
                    "commit_hash": parts[0],
                    "author": parts[1],
                    "commit_timestamp": parts[2].strip(),
                    "commit_message": parts[3],
                })
        return commits
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


def score_candidates(
    commits: List[Dict],
    violation_ts: str,
    lineage_distance: int = 1,
) -> List[Dict]:
    """
    Rank blame candidates by temporal proximity and lineage distance.
    Formula: base = 1.0 - (days_since_commit * 0.1), reduced by 0.2 per lineage hop.
    Returns up to 5 candidates.
    """
    try:
        vt = datetime.fromisoformat(violation_ts.replace("Z", "+00:00"))
    except ValueError:
        vt = datetime.now(timezone.utc)

    scored = []
    for c in commits[:5]:
        try:
            # Parse git timestamp format: "2026-04-01 19:44:05 +0300"
            ts_str = c["commit_timestamp"]
            ct = datetime.fromisoformat(ts_str.replace(" ", "T", 1).replace(" ", ""))
        except (ValueError, AttributeError):
            ct = datetime.now(timezone.utc)

        days = abs((vt - ct).days)
        base_score = max(0.0, 1.0 - (days * 0.1))
        hop_penalty = lineage_distance * 0.2
        confidence = max(0.0, round(base_score - hop_penalty, 3))

        scored.append({
            **c,
            "confidence_score": confidence,
            "days_since_commit": days,
            "lineage_hops": lineage_distance,
        })

    return sorted(scored, key=lambda x: x["confidence_score"], reverse=True)


# ---------------------------------------------------------------------------
# Step 4 — Build and write violation log entry
# ---------------------------------------------------------------------------

def find_upstream_file(contract_id: str, lineage_path: str) -> Optional[str]:
    """Find the upstream producer file from the lineage graph."""
    with open(lineage_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    snapshot = json.loads(lines[-1])

    # Find the pipeline node matching this contract
    pipeline_node = f"pipeline::{contract_id}"

    # Find edges where something PRODUCES into this pipeline
    for edge in snapshot.get("edges", []):
        tgt = edge.get("target", "")
        src = edge.get("source", "")
        rel = edge.get("relationship", "")
        if tgt == pipeline_node and rel == "PRODUCES":
            if "::" in src:
                return src.split("::", 1)[1]
            return src

    # Fallback: find file nodes with contract keyword
    keyword = contract_id.split("-")[0]
    for node in snapshot.get("nodes", []):
        nid = node.get("node_id", "")
        if keyword in nid and nid.startswith("file::"):
            return nid.split("::", 1)[1]
    return None


def build_violation_entry(
    check_result: Dict,
    contract_id: str,
    registry_blast: List[Dict],
    lineage_enrichment: Dict,
    blame_chain: List[Dict],
) -> Dict:
    """Construct a single violation log entry."""
    return {
        "violation_id": str(uuid.uuid4()),
        "check_id": check_result.get("check_id", "unknown"),
        "contract_id": contract_id,
        "column_name": check_result.get("column_name", "unknown"),
        "check_type": check_result.get("check_type", "unknown"),
        "severity": check_result.get("severity", "UNKNOWN"),
        "detected_at": datetime.now(timezone.utc).isoformat(),
        "actual_value": check_result.get("actual_value", ""),
        "expected": check_result.get("expected", ""),
        "message": check_result.get("message", ""),
        "records_failing": check_result.get("records_failing", 0),
        "blame_chain": blame_chain,
        "blast_radius": {
            "source": "registry",
            "direct_subscribers": registry_blast,
            "transitive_nodes": lineage_enrichment.get("transitive", []),
            "all_affected_nodes": lineage_enrichment.get("all_affected", []),
            "contamination_depth": lineage_enrichment.get("contamination_depth", 0),
            "note": "direct_subscribers from registry; transitive_nodes from lineage graph enrichment",
        },
    }


def write_violation(entry: Dict, out_path: str) -> None:
    """Append a violation entry to the JSONL log."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Main attribution pipeline
# ---------------------------------------------------------------------------

def attribute_violations(
    report_path: str,
    lineage_path: str,
    registry_path: str,
    output_path: str,
) -> List[Dict]:
    """
    Full attribution pipeline: for each FAIL in the validation report,
    trace it to its origin and write to the violation log.
    """
    with open(report_path) as f:
        report = json.load(f)

    contract_id = report.get("contract_id", "unknown")
    violations = []

    # Find failing checks
    failing_checks = [
        r for r in report.get("results", [])
        if r.get("status") in ("FAIL", "ERROR")
    ]

    if not failing_checks:
        print("No failing checks found in the report.")
        return []

    # Find upstream file from lineage
    upstream_file = find_upstream_file(contract_id, lineage_path)
    print(f"  Upstream file identified: {upstream_file}")

    # Get git commits for upstream file
    commits = []
    if upstream_file:
        commits = get_recent_commits(upstream_file, repo_root=".", days=30)
        if not commits:
            # Fallback: get recent commits from the whole repo
            commits = get_recent_commits(".", repo_root=".", days=30)
    print(f"  Found {len(commits)} recent commits")

    for check in failing_checks:
        col_name = check.get("column_name", "")
        print(f"\n  Attributing: {check.get('check_id')} on '{col_name}'")

        # Step 1: Registry blast radius
        blast = registry_blast_radius(contract_id, col_name, registry_path)
        print(f"    Registry subscribers affected: {len(blast)}")

        # Step 2: Lineage transitive depth (use pipeline node as starting point)
        producer_node = f"pipeline::{contract_id}"
        enrichment = compute_transitive_depth(producer_node, lineage_path)
        # Also add upstream file path to enrichment for context
        if upstream_file:
            enrichment["upstream_file"] = upstream_file
        print(f"    Contamination depth: {enrichment['contamination_depth']}")

        # Step 3: Blame chain
        blame = score_candidates(
            commits,
            violation_ts=report.get("run_timestamp", datetime.now(timezone.utc).isoformat()),
            lineage_distance=1,
        )
        if blame:
            print(f"    Top blame candidate: {blame[0].get('commit_hash', 'N/A')[:8]}... "
                  f"(confidence: {blame[0].get('confidence_score', 0):.3f})")

        # Step 4: Build and write
        entry = build_violation_entry(
            check, contract_id, blast, enrichment, blame
        )
        write_violation(entry, output_path)
        violations.append(entry)
        print(f"    Written violation: {entry['violation_id'][:8]}...")

    return violations


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ViolationAttributor")
    parser.add_argument("--violation", required=True,
                        help="Path to validation report JSON with failures")
    parser.add_argument("--lineage", required=True,
                        help="Path to lineage_snapshots.jsonl")
    parser.add_argument("--registry", required=True,
                        help="Path to contract_registry/subscriptions.yaml")
    parser.add_argument("--output", default="violation_log/violations.jsonl",
                        help="Output path for violation log JSONL")
    args = parser.parse_args()

    print(f"[1/4] Loading violation report: {args.violation}")
    print(f"[2/4] Loading lineage graph:    {args.lineage}")
    print(f"[3/4] Loading registry:         {args.registry}")
    print(f"[4/4] Attributing violations...")

    violations = attribute_violations(
        args.violation, args.lineage, args.registry, args.output
    )

    print(f"\n{'='*60}")
    print(f"Attribution complete: {len(violations)} violations attributed")
    print(f"Violation log: {args.output}")
    print(f"{'='*60}")

    for v in violations:
        print(f"\n  Violation: {v['violation_id'][:8]}...")
        print(f"  Check:     {v['check_id']}")
        print(f"  Field:     {v['column_name']}")
        print(f"  Severity:  {v['severity']}")
        print(f"  Blast:     {len(v['blast_radius']['direct_subscribers'])} direct subscribers")
        if v["blame_chain"]:
            top = v["blame_chain"][0]
            print(f"  Blame:     {top.get('commit_hash', 'N/A')[:12]}... "
                  f"by {top.get('author', 'unknown')} "
                  f"(confidence: {top.get('confidence_score', 0):.3f})")


if __name__ == "__main__":
    main()
