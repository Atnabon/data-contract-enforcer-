
# Data Contract Enforcer

A system for generating, validating, and enforcing machine-checkable data contracts across inter-system boundaries in the TRP1 program. Traces violations to their origin via blame chain attribution, detects schema evolution breaking changes, monitors AI-specific metrics, and produces auto-generated enforcer reports.

## Quick Start

```bash
# Install
pip install -e .

# Step 1: Generate contracts
python -m contracts.generator \
  --source outputs/week3/verdicts.jsonl \
  --contract-id week3-automaton-auditor-verdicts \
  --lineage outputs/week4/lineage_snapshots.jsonl \
  --output generated_contracts/
# Expected: generated_contracts/week3_verdicts.yaml (12 clauses), baselines written

# Step 2: Validate clean data (baseline)
python -m contracts.runner \
  --source outputs/week3/verdicts.jsonl \
  --contract generated_contracts/week3_verdicts.yaml \
  --mode AUDIT
# Expected: ALL PASS

# Step 3: Validate violated data (detects failures)
python -m contracts.runner \
  --source outputs/week3/verdicts_violated.jsonl \
  --contract generated_contracts/week3_verdicts.yaml \
  --mode ENFORCE
# Expected: 3 FAIL (confidence range, statistical drift, dimension_id format)

# Step 4: Attribute violations (blame chain + blast radius)
python -m contracts.attributor \
  --violation validation_reports/<latest_violated_report>.json \
  --lineage outputs/week4/lineage_snapshots.jsonl \
  --registry contract_registry/subscriptions.yaml
# Expected: 3 violations with blame chain, commit hash, 3 affected subscribers

# Step 5: Analyze schema evolution
python -m contracts.schema_analyzer \
  --contract-id week3-automaton-auditor-verdicts
# Expected: 1 BREAKING change (narrow_type: number -> integer)

# Step 6: Run AI extensions
python -m contracts.ai_extensions \
  --extractions outputs/week3/verdicts.jsonl \
  --verdicts outputs/week3/verdicts.jsonl
# Expected: Embedding drift PASS, Prompt validation PASS, Output violation rate PASS

# Step 7: Generate enforcer report
python -m contracts.report_generator
# Expected: enforcer_report/report_data.json with health score 0-100
```

## Architecture

```
outputs/week{3,5}/*.jsonl           # Raw JSONL from TRP1 systems
         │
         ▼
contracts/generator.py              # 4-stage contract generation
    Stage 1: Load & Flatten         # Normalize nested JSONL payloads
    Stage 2: Column Profiling       # Types, nullability, stats, cardinality
    Stage 3: Bitol Clause Gen       # Domain rules → Bitol clauses
    Stage 4: Lineage + LLM + dbt   # Inject lineage, annotate, dbt schema
         │
         ▼
generated_contracts/*.yaml          # Bitol v3.0.0 contracts
schema_snapshots/baselines.json     # Statistical baselines (mean, stddev)
         │
         ▼
contracts/runner.py                 # ValidationRunner (--mode AUDIT|WARN|ENFORCE)
    12 check types                  # type, required, uuid, datetime, enum,
    + statistical drift             # min, max, uniqueness, monotonic, temporal,
    + 3 enforcement modes           # score_consistency, statistical_drift
         │
         ▼
validation_reports/*.json           # Structured validation reports
         │
         ▼
contracts/attributor.py             # ViolationAttributor
    1. Registry blast radius        # Primary: contract_registry/subscriptions.yaml
    2. Lineage transitive depth     # Enrichment: outputs/week4/lineage_snapshots.jsonl
    3. Git blame + scoring          # Confidence = 1.0 - (days*0.1) - (hops*0.2)
    4. Violation log                # violation_log/violations.jsonl
         │
         ▼
contracts/schema_analyzer.py        # SchemaEvolutionAnalyzer
    Snapshot diffing                # Compare consecutive timestamped snapshots
    Taxonomy classification         # Breaking vs compatible (7 breaking, 5 compatible types)
    Migration report                # Rollback plan + per-consumer failure modes
         │
         ▼
contracts/ai_extensions.py          # AI Contract Extensions
    1. Embedding drift              # Cosine distance from baseline centroid
    2. Prompt input validation      # JSON Schema enforcement + quarantine
    3. Output violation rate        # Trend tracking + violation log writes
         │
         ▼
contracts/report_generator.py       # Enforcer Report
    1. Data Health Score            # (passed/total)*100 - 20*CRITICAL
    2. Violations this week         # By severity, plain language
    3. Schema changes               # From evolution analyzer
    4. AI system risk               # From AI extensions
    5. Recommended actions          # Specific file + clause references
```

## Project Structure

```
data-contract-enforcer/
├── contracts/
│   ├── generator.py          # ContractGenerator (4-stage pipeline)
│   ├── runner.py             # ValidationRunner (12 checks + drift + modes)
│   ├── attributor.py         # ViolationAttributor (registry + lineage + git)
│   ├── schema_analyzer.py    # SchemaEvolutionAnalyzer (diff + taxonomy)
│   ├── ai_extensions.py      # AI Extensions (drift, validation, rate)
│   ├── report_generator.py   # ReportGenerator (5-section report)
│   └── models.py             # Shared dataclasses
├── contract_registry/
│   └── subscriptions.yaml    # 6 inter-system dependency subscriptions
├── generated_contracts/      # Output Bitol YAML + dbt schema.yml
├── schema_snapshots/         # Timestamped snapshots + baselines
├── validation_reports/       # JSON validation reports
├── violation_log/            # JSONL violation records
├── enforcer_report/          # Auto-generated report
├── outputs/                  # Input JSONL from TRP1 weeks 3, 4, 5
└── interim_report.md         # Final submission report
```

## Contract Registry

The `contract_registry/subscriptions.yaml` file records 6 subscriptions covering all inter-system dependencies. The registry is the **primary source for blast radius** — not the lineage graph. This is the architecture that scales to Tier 2 (multi-team) and Tier 3 (cross-company).

Required minimum subscriptions:
- Week 3 → Week 4 (verdicts → cartographer)
- Week 4 → Week 7 (lineage → enforcer)
- Week 5 → Week 7 (events → enforcer)
- LangSmith → Week 7 (traces → AI extensions)

## Dependencies

```
pandas, numpy, pyyaml, anthropic, python-dotenv, jsonschema
```
