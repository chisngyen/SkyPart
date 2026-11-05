"""Versioned SkyPart checkpoint construction and compatibility validation."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any
import warnings

import torch

from . import __version__


CHECKPOINT_SCHEMA_VERSION = 1
FINGERPRINT_FIELDS = (
    "N_PARTS",
    "PART_DIM",
    "EMBED_DIM",
    "TEACHER_DIM",
    "NUM_ALTITUDES",
    "GAT_HEADS",
    "GAT_HIDDEN",
    "GRAPH_EMBED_DIM",
    "CLUSTER_TEMP",
    "MASK_RATIO",
    "UNFREEZE_BLOCKS",
)
DINO_IDENTITY_FIELDS = (
    "DINOV2_REPO",
    "DINOV2_STUDENT_MODEL",
    "DINOV2_TEACHER_MODEL",
)


class CheckpointCompatibilityError(ValueError):
    """Raised when checkpoint metadata cannot prove model compatibility."""


@dataclass(frozen=True)
class LoadedCheckpoint:
    state_dict: Mapping[str, Any]
    num_classes: int
    is_legacy: bool
    metadata: Mapping[str, Any]


def _config_dict(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, Mapping):
        return deepcopy(dict(config))
    return deepcopy(vars(config))


def _config_value(config: Any, field: str) -> Any:
    if isinstance(config, Mapping):
        return config[field]
    return getattr(config, field)


def _positive_int(value: Any) -> bool:
    return type(value) is int and value > 0


def _same_typed_value(actual: Any, expected: Any) -> bool:
    return type(actual) is type(expected) and actual == expected


def build_checkpoint(
    *,
    model_state_dict: Mapping[str, Any],
    epoch: int,
    metrics: Mapping[str, Any],
    config: Any,
) -> dict[str, Any]:
    """Build a complete schema-v1 checkpoint payload."""
    config_data = _config_dict(config)
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "package_version": __version__,
        "model_state_dict": model_state_dict,
        "epoch": epoch,
        "metrics": deepcopy(dict(metrics)),
        "config": config_data,
        "model_fingerprint": {
            field: _config_value(config, field) for field in FINGERPRINT_FIELDS
        },
        **{
            field: _config_value(config, field) for field in DINO_IDENTITY_FIELDS
        },
        "NUM_CLASSES": _config_value(config, "NUM_CLASSES"),
    }


def _legacy_checkpoint(
    checkpoint: Mapping[str, Any],
    config: Any,
) -> LoadedCheckpoint:
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    num_classes = _config_value(config, "NUM_CLASSES")
    errors = []
    if not isinstance(state_dict, Mapping):
        errors.append("model_state_dict: missing or not a mapping")
    if not _positive_int(num_classes):
        errors.append("NUM_CLASSES: legacy YAML value must be a positive integer")
    if errors:
        raise CheckpointCompatibilityError(
            "Legacy checkpoint is incompatible:\n- " + "\n- ".join(errors)
        )
    warnings.warn(
        "Loading a metadata-free legacy checkpoint: architecture/protocol "
        "provenance could not be verified; the selected YAML remains "
        "authoritative and strict state-dict loading is enforced.",
        UserWarning,
        stacklevel=3,
    )
    return LoadedCheckpoint(state_dict, num_classes, True, checkpoint)


def validate_checkpoint(
    checkpoint: Mapping[str, Any],
    config: Any,
    *,
    allow_legacy_checkpoint: bool = False,
) -> LoadedCheckpoint:
    """Validate checkpoint metadata without changing the evaluation config."""
    if not isinstance(checkpoint, Mapping):
        raise CheckpointCompatibilityError("checkpoint: expected a mapping")

    if "checkpoint_schema_version" not in checkpoint:
        if not allow_legacy_checkpoint:
            raise CheckpointCompatibilityError(
                "Metadata-free legacy checkpoint rejected; pass "
                "--allow-legacy-checkpoint to accept unverifiable provenance."
            )
        return _legacy_checkpoint(checkpoint, config)

    schema = checkpoint["checkpoint_schema_version"]
    if type(schema) is not int or schema != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointCompatibilityError(
            "checkpoint_schema_version: expected "
            f"int {CHECKPOINT_SCHEMA_VERSION!r}, found "
            f"{type(schema).__name__} {schema!r}"
        )

    errors: list[str] = []
    for field in ("package_version", "model_state_dict", "epoch", "metrics", "config"):
        if field not in checkpoint:
            errors.append(f"{field}: missing")
    if "package_version" in checkpoint and not isinstance(
        checkpoint["package_version"], str
    ):
        errors.append("package_version: expected a string")
    if "model_state_dict" in checkpoint and not isinstance(
        checkpoint["model_state_dict"], Mapping
    ):
        errors.append("model_state_dict: expected a mapping")
    if "config" in checkpoint and not isinstance(checkpoint["config"], Mapping):
        errors.append("config: expected a mapping")

    fingerprint = checkpoint.get("model_fingerprint")
    if not isinstance(fingerprint, Mapping):
        errors.append("model_fingerprint: missing or not a mapping")
        fingerprint = {}
    for field in sorted(set(fingerprint) - set(FINGERPRINT_FIELDS)):
        errors.append(f"{field}: unknown model_fingerprint field")
    for field in FINGERPRINT_FIELDS:
        if field not in fingerprint:
            errors.append(f"{field}: missing from model_fingerprint")
        else:
            expected = _config_value(config, field)
            if not _same_typed_value(fingerprint[field], expected):
                errors.append(
                    f"{field}: checkpoint={type(fingerprint[field]).__name__} "
                    f"{fingerprint[field]!r}, YAML={type(expected).__name__} "
                    f"{expected!r}"
                )

    for field in DINO_IDENTITY_FIELDS:
        if field not in checkpoint:
            errors.append(f"{field}: missing")
        else:
            expected = _config_value(config, field)
            if not _same_typed_value(checkpoint[field], expected):
                errors.append(
                    f"{field}: checkpoint={type(checkpoint[field]).__name__} "
                    f"{checkpoint[field]!r}, YAML={type(expected).__name__} "
                    f"{expected!r}"
                )

    num_classes = checkpoint.get("NUM_CLASSES")
    if not _positive_int(num_classes):
        errors.append("NUM_CLASSES: expected a positive integer")

    if errors:
        raise CheckpointCompatibilityError(
            "Schema-v1 checkpoint is incompatible:\n- " + "\n- ".join(errors)
        )
    return LoadedCheckpoint(
        checkpoint["model_state_dict"],
        num_classes,
        False,
        checkpoint,
    )


def load_checkpoint(
    path: str | Path,
    config: Any,
    *,
    allow_legacy_checkpoint: bool = False,
) -> LoadedCheckpoint:
    """Load on CPU and validate before callers construct a model."""
    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        raise RuntimeError(
            "Safe checkpoint loading requires PyTorch >= 2.1 with "
            "torch.load(..., weights_only=True) support; upgrade PyTorch."
        ) from exc
    return validate_checkpoint(
        checkpoint,
        config,
        allow_legacy_checkpoint=allow_legacy_checkpoint,
    )


def load_model_state(model: Any, checkpoint: LoadedCheckpoint) -> Any:
    """Strictly load a previously validated checkpoint into ``model``."""
    return model.load_state_dict(checkpoint.state_dict, strict=True)
