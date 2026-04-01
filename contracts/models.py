"""
Shared dataclasses used across all contracts components.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ColumnProfile:
    """Profile of a single column from a JSONL dataset."""
    name: str
    dtype: str
    null_fraction: float
    cardinality_estimate: int
    sample_values: List[str]
    stats: Optional[Dict[str, float]] = None  # only for numeric columns
    # stats keys: min, max, mean, p25, p50, p75, p95, p99, stddev


@dataclass
class ContractClause:
    """A single clause in a data contract."""
    type: str
    required: bool = False
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    format: Optional[str] = None
    pattern: Optional[str] = None
    enum: Optional[List[str]] = None
    description: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of a single contract check."""
    check_id: str
    column_name: str
    check_type: str
    status: str          # PASS | FAIL | WARN | ERROR
    actual_value: str
    expected: str
    severity: str        # CRITICAL | HIGH | MEDIUM | LOW
    records_failing: int = 0
    sample_failing: List[str] = field(default_factory=list)
    message: str = ""


@dataclass
class ValidationReport:
    """Full report from one ValidationRunner run."""
    report_id: str
    contract_id: str
    snapshot_id: str
    run_timestamp: str
    total_checks: int
    passed: int
    failed: int
    warned: int
    errored: int
    results: List[ValidationResult] = field(default_factory=list)
