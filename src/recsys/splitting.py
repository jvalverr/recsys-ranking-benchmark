"""They construct globally chronological, non-overlapping evaluation windows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from recsys.preprocessing import ModelingConfig
from recsys.utils import sql_literal


@dataclass(frozen=True)
class TemporalWindows:
    """
    They represent the boundaries separating training,
    validation, and test periods.
    """

    train_end: datetime
    validation_end: datetime
    strategy: str

    def as_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "train_end_utc": self.train_end.isoformat(),
            "validation_end_utc": (
                self.validation_end.isoformat()
            ),
        }


def _utc(
    value: datetime | str,
) -> datetime:
    """
    They normalize explicit and inferred cutoffs to UTC.
    Naive timestamps are rejected.
    """

    candidate = (
        datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
        if isinstance(value, str)
        else value
    )

    if candidate.tzinfo is None:
        raise ValueError(
            "Chronological cutoffs must include a timezone"
        )

    return candidate.astimezone(
        timezone.utc
    )


def resolve_windows(
    con: duckdb.DuckDBPyConnection,
    source: Path,
    config: ModelingConfig,
) -> TemporalWindows:
    """
    They resolve global temporal boundaries using either
    explicit UTC dates or discrete event-time quantiles.
    """

    if config.train_end_utc is not None:

        first = _utc(
            config.train_end_utc
        )

        second = _utc(
            config.validation_end_utc
        )

        strategy = "explicit_utc_cutoffs"

    else:

        # They select existing timestamps rather than interpolated
        # timestamps, keeping equal-time events within one period.
        first, second = con.execute(
            """
            SELECT
                QUANTILE_DISC(timestamp, ?),
                QUANTILE_DISC(timestamp, ?)

            FROM read_parquet(?)
            """,
            [
                config.train_fraction,
                (
                    config.train_fraction
                    + config.validation_fraction
                ),
                str(source),
            ],
        ).fetchone()

        if first is None or second is None:
            raise ValueError(
                "The source dataset has no usable timestamps"
            )

        first = _utc(first)
        second = _utc(second)

        strategy = "discrete_event_time_quantiles"

    if not first < second:
        raise ValueError(
            "Temporal cutoffs overlap; use explicit dates "
            "or a different dataset"
        )

    return TemporalWindows(
        first,
        second,
        strategy,
    )


def partition_predicates(
    windows: TemporalWindows,
) -> dict[str, str]:
    """
    They define disjoint time intervals with strict boundaries.
    """

    first = (
        "CAST("
        f"{sql_literal(windows.train_end.isoformat())}"
        " AS TIMESTAMPTZ)"
    )

    second = (
        "CAST("
        f"{sql_literal(windows.validation_end.isoformat())}"
        " AS TIMESTAMPTZ)"
    )

    return {
        "train": (
            f"timestamp < {first}"
        ),

        "validation": (
            f"timestamp >= {first} "
            f"AND timestamp < {second}"
        ),

        "test": (
            f"timestamp >= {second}"
        ),
    }


def split_source(
    con: duckdb.DuckDBPyConnection,
    source: Path,
    directory: Path,
    windows: TemporalWindows,
    feedback_sql: str,
) -> dict[str, Path]:
    """
    They partition source events and derive feedback labels
    using the selected dataset policy.
    """

    source_sql = (
        f"read_parquet({sql_literal(source)})"
    )

    paths: dict[str, Path] = {}

    for split_name, predicate in (
        partition_predicates(windows).items()
    ):

        target = (
            directory
            / f"base_{split_name}.parquet"
        )

        statement = f"""
            COPY (
                SELECT
                    e.*,
                    {feedback_sql},
                    {sql_literal(split_name)} AS split

                FROM {source_sql} AS e

                WHERE {predicate}
            )
            TO {sql_literal(target)}
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """

        con.execute(statement)

        paths[split_name] = target

    return paths


def validate_source(
    con: duckdb.DuckDBPyConnection,
    source: Path,
    dataset: str,
) -> int:
    """
    They check essential modeling-input invariants before
    assigning chronological windows.
    """

    row = con.execute(
        f"""
        SELECT
            COUNT(*) AS n,

            COUNT(*) FILTER (
                WHERE
                    user_id IS NULL
                    OR item_id IS NULL
                    OR timestamp IS NULL
                    OR TRIM(user_id) = ''
                    OR TRIM(item_id) = ''
                    OR dataset IS NULL
                    OR dataset <> ?
            ) AS invalid_identity,

            COUNT(*) FILTER (
                WHERE
                    event_type IS NULL
                    OR TRIM(event_type) = ''
            ) AS invalid_event,

            COUNT(*) FILTER (
                WHERE
                    event_value IS NULL
                    AND event_type = 'rating'
            ) AS invalid_rating

        FROM read_parquet(
            {sql_literal(source)}
        )
        """,
        [dataset],
    ).fetchone()

    if not row[0] or any(row[1:]):
        raise ValueError(
            "The source failed modeling-input "
            f"validation: {row}"
        )

    return int(row[0])


def verify_splits(
    con: duckdb.DuckDBPyConnection,
    paths: dict[str, Path],
    original_total: int,
) -> dict:
    """
    They verify event conservation, nonempty windows,
    and strict chronological separation.
    """

    stats = {}

    for name, path in paths.items():

        n, first, last = con.execute(
            "SELECT "
            "COUNT(*), "
            "MIN(timestamp), "
            "MAX(timestamp) "
            f"FROM read_parquet({sql_literal(path)})"
        ).fetchone()

        stats[name] = {
            "events": int(n),
            "first_utc": (
                first.isoformat()
                if first is not None
                else None
            ),
            "last_utc": (
                last.isoformat()
                if last is not None
                else None
            ),
        }

    # They ensure that the raw partitioning preserves exactly
    # the number of source events.
    if (
        sum(
            result["events"]
            for result in stats.values()
        )
        != original_total
    ):
        raise AssertionError(
            "The split process lost or duplicated "
            "source events"
        )

    if any(
        result["events"] == 0
        for result in stats.values()
    ):
        raise ValueError(
            "All three chronological windows "
            "must contain events"
        )

    train_last = datetime.fromisoformat(
        stats["train"]["last_utc"]
    )

    val_first = datetime.fromisoformat(
        stats["validation"]["first_utc"]
    )

    val_last = datetime.fromisoformat(
        stats["validation"]["last_utc"]
    )

    test_first = datetime.fromisoformat(
        stats["test"]["first_utc"]
    )

    # They reject overlapping windows, including timestamp
    # ties that would otherwise compromise temporal independence.
    if not (
        train_last < val_first
        and val_last < test_first
    ):
        raise AssertionError(
            "Chronological split boundaries overlap"
        )

    return stats
