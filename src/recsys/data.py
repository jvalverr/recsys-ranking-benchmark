"""Shared event representation and source-preserving transformations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


EVENT_COLUMNS = (
    "user_id",
    "item_id",
    "timestamp",
    "event_type",
    "event_value",
    "dataset",
    "source_row",
)


@dataclass
class DatasetBundle:
    """Bundles keep standardized events, auxiliary tables, and source provenance together."""

    name: str
    events: pd.DataFrame
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    sources: dict[str, Path] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def require_columns(
    df: pd.DataFrame,
    required: set[str],
    source: str,
) -> None:
    """Validators reject incompatible sources before downstream transformations begin."""

    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            f"{source}: missing required columns: {missing}"
        )


def require_file(path: Path) -> Path:
    """Loaders fail explicitly when expected raw files are absent."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Required raw file not found: {path}"
        )

    return path


def standardize_events(
    raw: pd.DataFrame,
    *,
    dataset: str,
    user_col: str,
    item_col: str,
    timestamp_col: str,
    timestamp_unit: str | None,
    event_type: str | None = None,
    event_col: str | None = None,
    value_col: str | None = None,
) -> pd.DataFrame:
    """
    Adapters normalize essential fields while retaining source fields
    as raw_* columns.
    """

    if (event_type is None) == (event_col is None):
        raise ValueError(
            "Specify exactly one of event_type or event_col"
        )

    required = {
        user_col,
        item_col,
        timestamp_col,
    }

    if event_col:
        required.add(event_col)

    if value_col:
        required.add(value_col)

    require_columns(raw, required, dataset)

    raw = raw.reset_index(drop=True)

    # Adapters preserve source columns for auditing; researchers can
    # later introduce dataset-specific relevance labels without
    # losing original values.
    original = raw.rename(
        columns=lambda name: f"raw_{name}"
    )

    # Timestamp converters use the source-specific unit rather than
    # inferring time units from numeric magnitudes.
    if timestamp_unit is None:
        timestamp = pd.to_datetime(
            raw[timestamp_col],
            utc=True,
            errors="coerce",
        )
    else:
        numeric_time = pd.to_numeric(
            raw[timestamp_col],
            errors="coerce",
        )

        timestamp = pd.to_datetime(
            numeric_time,
            unit=timestamp_unit,
            utc=True,
            errors="coerce",
        )

    # Event types retain their original semantics across datasets.
    event_types = (
        raw[event_col]
        .astype("string")
        .str.strip()
        .str.lower()
        if event_col
        else pd.Series(
            event_type,
            index=raw.index,
            dtype="string",
        )
    )

    # Explicit ratings remain numerical, while implicit events
    # retain missing values until target construction is defined.
    values = (
        pd.to_numeric(
            raw[value_col],
            errors="coerce",
        ).astype("Float64")
        if value_col
        else pd.Series(
            pd.NA,
            index=raw.index,
            dtype="Float64",
        )
    )

    canonical = pd.DataFrame({
        "user_id": (
            raw[user_col]
            .astype("string")
            .str.strip()
        ),
        "item_id": (
            raw[item_col]
            .astype("string")
            .str.strip()
        ),
        "timestamp": timestamp,
        "event_type": event_types,
        "event_value": values,
        "dataset": pd.Series(
            dataset,
            index=raw.index,
            dtype="string",
        ),
        "source_row": np.arange(
            len(raw),
            dtype=np.int64,
        ),
    })

    return pd.concat(
        [canonical, original],
        axis=1,
    )
