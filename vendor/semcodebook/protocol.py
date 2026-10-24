from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class WatermarkSpec:
    wm_id: int
    ecc_scheme: str = "soft_secded84_adaptive_v1"
    detector_threshold: float = 0.5
    payload_bits: int = 4
    carrier_key: str = "semcodebook-demo-key"
    carrier_order: tuple[str, ...] = ()
    carrier_schedule: tuple["CarrierScheduleEntry", ...] = ()
    schedule_strategy: str = "typed_ast_cfg_ssa_v1"
    implementation_stage: str = "generation_stage_unspecified"


@dataclass(frozen=True, slots=True)
class CarrierScheduleEntry:
    family: str
    slot_index: int
    role: str = "data"
    bit_index: int | None = None
    target_bit: int | None = None
    applicable: bool = False
    applicability_score: float = 0.0
    schedule_priority: float = 0.0
    structural_level: str = ""
    structural_signal: str = ""
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class VariantExample:
    task_id: str
    language: str
    prompt: str
    reference_code: str
    family: str
    bit_value: int
    transformed_code: str
    applicable: bool = False
    validation_passed: bool = False
    applicability_score: float = 0.0
    schedule_priority: float = 0.0
    structural_level: str = ""
    structural_signal: str = ""
    validation_notes: tuple[str, ...] = ()
    split: str = ""
    validation_mode: str = ""
    training_eligible: bool = False
    transformed_code_hash: str = ""
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class VariantPool:
    records: tuple[VariantExample, ...]
    created_count: int
    language_distribution: dict[str, int]
    family_distribution: dict[str, int]
    attempted_count: int = 0
    applicable_count: int = 0
    validated_record_count: int = 0
    training_eligible_count: int = 0
    split_distribution: dict[str, int] = field(default_factory=dict)
    validation_distribution: dict[str, int] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CarrierEvidence:
    family: str
    option: str
    confidence: float
    evidence_source: str = ""
    structural_level: str = ""
    structural_signal: str = ""
    prob_zero: float = 0.5
    prob_one: float = 0.5
    applicable: bool = False
    applicability_score: float = 0.0
    schedule_priority: float = 0.0
    matches: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DetectionOutput:
    is_watermarked: bool
    wm_id_hat: int | None
    bit_error_rate: float | None
    confidence: float
    corrected_bits: int
    decoded_wm_id_candidate: int | None = None
    decoder_status: str = ""
    erasure_count: int = 0
    raw_bit_error_count: int = 0
    support_count: int = 0
    support_ratio: float = 0.0
    negative_control_score: float = 0.0
    ber_numerator: int | None = None
    ber_denominator: int = 0
    carrier_evidence: tuple[CarrierEvidence, ...] = ()
    carrier_trace: tuple[CarrierEvidence, ...] = ()
    carrier_schedule: tuple[CarrierScheduleEntry, ...] = ()
    implementation_stage: str = "generation_stage_unspecified"
    notes: tuple[str, ...] = ()
    decision_status: str = "abstain"
    abstain_reason: str | None = None
    positive_support_score: float = 0.0
    positive_support_family_count: int = 0
    positive_support_level_count: int = 0


@dataclass(frozen=True, slots=True)
class GenerationInterface:
    backbone_name: str = ""
    model_name: str = ""
    backend_name: str = "semcodebook_generation_interface"
    execution_mode: str = "materialized_interface_preferred"
    selection_strategy: str = "adaptive_keyed_carrier_schedule"
    checkpoint_path: str = ""
    adapter_path: str = ""
    embedding_table_path: str = ""
    variant_pool_path: str = ""
    dataset_manifest_path: str = ""
    contract_path: str = ""
    interface_path: str = ""
    base_code_source: str = "caller_supplied"
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    prompt: str
    language: str
    wm_id: int
    model_name: str
    task_id: str = ""
    carrier_key: str = "semcodebook-demo-key"
    max_new_tokens: int = 384
    temperature: float = 0.2
    backbone_name: str = ""
    backend_name: str = "semcodebook_generation_interface"
    execution_mode: str = "materialized_interface_preferred"
    selection_strategy: str = "adaptive_keyed_carrier_schedule"
    checkpoint_path: str = ""
    adapter_path: str = ""
    embedding_table_path: str = ""
    variant_pool_path: str = ""
    dataset_manifest_path: str = ""
    contract_path: str = ""
    interface_path: str = ""
    base_code_source: str = "caller_supplied"
    candidate_budget: int = 12
    schedule_strategy: str = "typed_ast_cfg_ssa_v1"
    validation_tests: tuple[str, ...] = ()
    validation_metadata: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class GenerationResult:
    watermarked_code: str
    language: str
    wm_id: int
    carrier_bits: tuple[int, ...]
    chosen_families: tuple[str, ...]
    applied_families: tuple[str, ...] = ()
    scheduled_families: tuple[str, ...] = ()
    coverage_ratio: float = 0.0
    model_name: str = ""
    backbone_name: str = ""
    backend_name: str = ""
    execution_mode: str = ""
    selection_strategy: str = ""
    checkpoint_path: str = ""
    adapter_path: str = ""
    embedding_table_path: str = ""
    variant_pool_path: str = ""
    dataset_manifest_path: str = ""
    contract_path: str = ""
    interface_path: str = ""
    base_code_source: str = ""
    candidate_budget: int = 0
    carrier_schedule: tuple[CarrierScheduleEntry, ...] = ()
    implementation_stage: str = "generation_stage_unspecified"
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AttackRecord:
    attack_name: str
    attack_category: str
    language: str
    original_code: str
    attacked_code: str
    changed: bool
    applicable: bool
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BenchmarkTask:
    benchmark: str
    task_id: str
    language: str
    prompt: str
    reference_code: str = ""
    tests: tuple[str, ...] = ()
    split: str = "smoke"
    source: str = "project_local"
    metadata: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    project: str
    method_name: str
    model_name: str
    benchmark: str
    task_id: str
    split: str
    source: str
    language: str
    attack_name: str | None
    attack_category: str | None
    attack_applicable: bool
    compile_supported: bool
    compile_ok: bool | None
    pass_supported: bool
    pass_ok: bool | None
    semantic_ok: bool | None
    compile_pass_preserved: bool | None
    negative_control: bool
    detected: bool
    wm_id_expected: int
    wm_id_hat: int | None
    payload_bits: int
    ecc_scheme: str
    exact_recovery: bool
    bit_error_count: int | None
    bit_error_rate: float | None
    corrected_bits: int
    confidence: float
    code_changed: bool
    carrier_coverage: float
    decoder_status: str = ""
    erasure_count: int = 0
    raw_bit_error_count: int = 0
    support_count: int = 0
    support_ratio: float = 0.0
    carrier_signal_coverage: float = 0.0
    negative_control_score: float = 0.0
    ber_numerator: int | None = None
    ber_denominator: int = 0
    executed_tests: tuple[str, ...] = ()
    failure_reason: str | None = None
    carrier_evidence: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    decision_status: str = "abstain"
    abstain_reason: str | None = None
    positive_support_score: float = 0.0
    positive_support_family_count: int = 0
    positive_support_level_count: int = 0


@dataclass(frozen=True, slots=True)
class RunManifest:
    run_name: str
    project: str
    model_name: str
    benchmark_names: tuple[str, ...]
    attack_names: tuple[str, ...]
    baseline_names: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BackboneConfig:
    name: str
    model_name: str
    family: str
    dtype: str
    context_length: int
    wm_embedding_dim: int
    lora_rank: int
    train_micro_batch_size: int


@dataclass(frozen=True, slots=True)
class ProjectAcceptance:
    stronger_than_runtime_frontier: bool
    stable_clean_recovery: bool
    stable_refactor_recovery: bool
    compile_collapse_observed: bool
    notes: tuple[str, ...] = field(default_factory=tuple)
