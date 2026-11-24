"""Descriptive analyses on validated Parquet datasets without full-table pandas loads."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd


DATASET_NAMES = frozenset({
    "movielens-1m",
    "retailrocket",
    "amazon-video-games",
})

REQUIRED_COLUMNS = {
    "user_id",
    "item_id",
    "timestamp",
    "event_type",
    "event_value",
    "dataset",
}


@dataclass(frozen=True)
class EDADataset:
    """Researchers access verified datasets through their artifacts and validation reports."""

    name: str
    path: Path
    validation: dict


def load_eda_dataset(
    name: str,
    root: Path | str = "data/processed",
) -> EDADataset:
    """Researchers reject missing or unvalidated datasets before drawing conclusions."""

    if name not in DATASET_NAMES:
        raise ValueError(f"Unsupported dataset: {name}")

    directory = Path(root) / name
    path = directory / "events.parquet"
    report_path = directory / "validation.json"

    for required in (path, report_path):
        if not required.is_file():
            raise FileNotFoundError(
                f"Missing prepared artifact: {required}"
            )

    validation = json.loads(
        report_path.read_text(encoding="utf-8")
    )

    if (
        validation.get("valid") is not True
        or validation.get("dataset") != name
    ):
        raise ValueError(
            "The dataset does not have a matching valid report"
        )

    # Researchers inspect only the Parquet schema rather than
    # materializing full event records in memory.
    with duckdb.connect(":memory:") as con:
        columns = {
            column[0]
            for column in con.execute(
                "SELECT * FROM read_parquet(?) LIMIT 0",
                [str(path)],
            ).description
        }

    missing = REQUIRED_COLUMNS - columns

    if missing:
        raise ValueError(
            f"Missing canonical Parquet columns: {sorted(missing)}"
        )

    return EDADataset(
        name=name,
        path=path,
        validation=validation,
    )


def _query(
    dataset: EDADataset,
    sql: str,
    params: tuple = (),
) -> pd.DataFrame:
    """You can aggregate directly in DuckDB and transfer only results to pandas."""

    with duckdb.connect(":memory:") as con:
        con.execute("SET TimeZone='UTC'")

        return con.execute(
            sql,
            [str(dataset.path), *params],
        ).df()


def overview(dataset: EDADataset) -> pd.DataFrame:
    """You can summarize the validated population without recomputing it."""

    stats = dataset.validation["statistics"]

    fields = (
        ("events", "Events"),
        ("users", "Users"),
        ("items", "Items"),
        ("unique_user_item_pairs", "Unique user-item pairs"),
        ("repeat_user_item_events", "Repeated user-item events"),
        ("duplicate_event_observations", "Potential duplicate observations"),
        ("interaction_sparsity", "Observed matrix sparsity"),
        ("first_event_utc", "Earliest event (UTC)"),
        ("last_event_utc", "Latest event (UTC)"),
    )

    return pd.DataFrame([
        {
            "metric": title,
            "value": stats.get(key),
        }
        for key, title in fields
    ])


def compare_datasets(
    root: Path | str = "data/processed",
) -> pd.DataFrame:
    """You can contrast source populations without equating feedback semantics."""

    rows = []

    for name in sorted(DATASET_NAMES):
        report_path = (
            Path(root) / name / "validation.json"
        )

        if not report_path.is_file():
            continue

        report = json.loads(
            report_path.read_text(encoding="utf-8")
        )

        if (
            report.get("valid") is not True
            or report.get("dataset") != name
        ):
            continue

        stats = report["statistics"]

        rows.append({
            "dataset": name,
            "events": stats.get("events"),
            "users": stats.get("users"),
            "items": stats.get("items"),
            "sparsity": stats.get("interaction_sparsity"),
            "first_event": stats.get("first_event_utc"),
            "last_event": stats.get("last_event_utc"),
        })

    return pd.DataFrame(rows)


def activity_summary(
    dataset: EDADataset,
    entity: str,
) -> pd.DataFrame:
    """You can measure user/item activity and approximate tail quantiles at scale."""

    if entity not in {"user_id", "item_id"}:
        raise ValueError(
            "entity must be user_id or item_id"
        )

    sql = f"""
    WITH freq AS (
        SELECT {entity} AS entity_id, COUNT(*) AS n
        FROM read_parquet(?)
        GROUP BY 1
    )
    SELECT
        COUNT(*) AS entities,
        MIN(n) AS min_events,
        AVG(n) AS mean_events,
        APPROX_QUANTILE(n, 0.5) AS approx_median,
        APPROX_QUANTILE(n, 0.95) AS approx_p95,
        APPROX_QUANTILE(n, 0.99) AS approx_p99,
        MAX(n) AS max_events,
        SUM(
            CASE WHEN n = 1 THEN 1 ELSE 0 END
        ) AS singletons
    FROM freq
    """

    return _query(dataset, sql)


def activity_bins(
    dataset: EDADataset,
    entity: str,
) -> pd.DataFrame:
    """You can aggregate logarithmic bins without exporting millions of identifiers."""

    if entity not in {"user_id", "item_id"}:
        raise ValueError(
            "entity must be user_id or item_id"
        )

    sql = f"""
    WITH freq AS (
        SELECT {entity} AS entity_id, COUNT(*) AS n
        FROM read_parquet(?)
        GROUP BY 1
    )
    SELECT
        CAST(FLOOR(LOG2(n)) AS INTEGER) AS bin_power,
        COUNT(*) AS entities,
        SUM(n) AS events
    FROM freq
    GROUP BY 1
    ORDER BY 1
    """

    return _query(dataset, sql)


def event_distribution(
    dataset: EDADataset,
) -> pd.DataFrame:
    """You can preserve the observed semantics of explicit and implicit events."""

    sql = """
    WITH counts AS (
        SELECT event_type, COUNT(*) AS events
        FROM read_parquet(?)
        GROUP BY 1
    )
    SELECT
        event_type,
        events,
        100.0 * events / SUM(events) OVER () AS percent
    FROM counts
    ORDER BY events DESC
    """

    return _query(dataset, sql)


def rating_distribution(
    dataset: EDADataset,
) -> pd.DataFrame:
    """You can inspect explicit ratings without assigning implicit relevance labels."""

    sql = """
    SELECT
        CAST(event_value AS DOUBLE) AS rating,
        COUNT(*) AS events
    FROM read_parquet(?)
    WHERE event_type = 'rating'
      AND event_value IS NOT NULL
    GROUP BY 1
    ORDER BY 1
    """

    return _query(dataset, sql)


def monthly_events(
    dataset: EDADataset,
) -> pd.DataFrame:
    """You can measure monthly event volume using canonical UTC timestamps."""

    sql = """
    SELECT
        DATE_TRUNC('month', timestamp) AS month,
        event_type,
        COUNT(*) AS events,
        COUNT(DISTINCT user_id) AS active_users,
        COUNT(DISTINCT item_id) AS active_items
    FROM read_parquet(?)
    GROUP BY 1, 2
    ORDER BY 1, 2
    """

    return _query(dataset, sql)


def top_items(
    dataset: EDADataset,
    limit: int = 15,
) -> pd.DataFrame:
    """You can identify popular items without interpreting frequency as relevance."""

    if not 1 <= limit <= 100:
        raise ValueError(
            "limit must be between 1 and 100"
        )

    sql = """
    SELECT
        item_id,
        COUNT(*) AS events,
        COUNT(DISTINCT user_id) AS distinct_users
    FROM read_parquet(?)
    GROUP BY 1
    ORDER BY events DESC, item_id ASC
    LIMIT ?
    """

    return _query(dataset, sql, (limit,))


def item_concentration(
    dataset: EDADataset,
) -> pd.DataFrame:
    """You can quantify how much activity is concentrated among popular items."""

    sql = """
    WITH item_counts AS (
        SELECT
            item_id,
            COUNT(*) AS events
        FROM read_parquet(?)
        GROUP BY 1
    ),
    ranked AS (
        SELECT
            events,
            ROW_NUMBER() OVER (
                ORDER BY events DESC, item_id
            ) AS rank_num,
            COUNT(*) OVER () AS n_items
        FROM item_counts
    )
    SELECT
        MAX(n_items) AS n_items,
        SUM(events) AS total_events,

        SUM(
            CASE
                WHEN rank_num <= GREATEST(
                    1, CEIL(0.01 * n_items)
                )
                THEN events
                ELSE 0
            END
        ) * 1.0 / SUM(events) AS top_1pct_event_share,

        SUM(
            CASE
                WHEN rank_num <= GREATEST(
                    1, CEIL(0.10 * n_items)
                )
                THEN events
                ELSE 0
            END
        ) * 1.0 / SUM(events) AS top_10pct_event_share

    FROM ranked
    """

    return _query(dataset, sql)


def cold_start_snapshot(
    dataset: EDADataset,
    history_fraction: float = 0.8,
) -> tuple[pd.Timestamp, pd.DataFrame]:
    """
    You can inspect previously unseen users and items after
    a global time cutoff, without defining the final modeling split.
    """

    if not 0.0 < history_fraction < 1.0:
        raise ValueError(
            "history_fraction must be between 0 and 1"
        )

    # Researchers select the cutoff from chronological event-time
    # quantiles rather than from the original source-file order.
    cutoff_df = _query(
        dataset,
        """
        WITH times AS (
            SELECT EPOCH(timestamp) AS ts
            FROM read_parquet(?)
        )
        SELECT QUANTILE_CONT(ts, ?) AS cutoff
        FROM times
        """,
        (history_fraction,),
    )

    cutoff_epoch = float(
        cutoff_df.loc[0, "cutoff"]
    )

    # Researchers classify later events using only the existence
    # of users and items before the selected temporal cutoff.
    sql = """
    WITH base AS (
        SELECT
            user_id,
            item_id,
            event_type,
            EPOCH(timestamp) AS ts
        FROM read_parquet(?)
    ),
    historical_users AS (
        SELECT DISTINCT user_id
        FROM base
        WHERE ts <= ?
    ),
    historical_items AS (
        SELECT DISTINCT item_id
        FROM base
        WHERE ts <= ?
    ),
    future AS (
        SELECT *
        FROM base
        WHERE ts > ?
    ),
    classified AS (
        SELECT
            f.event_type,
            CASE
                WHEN u.user_id IS NOT NULL
                 AND i.item_id IS NOT NULL
                    THEN 'warm'
                WHEN u.user_id IS NULL
                 AND i.item_id IS NOT NULL
                    THEN 'new_user'
                WHEN u.user_id IS NOT NULL
                 AND i.item_id IS NULL
                    THEN 'new_item'
                ELSE 'new_user_and_item'
            END AS segment
        FROM future AS f
        LEFT JOIN historical_users AS u
            USING (user_id)
        LEFT JOIN historical_items AS i
            USING (item_id)
    )
    SELECT
        event_type,
        segment,
        COUNT(*) AS events
    FROM classified
    GROUP BY 1, 2
    ORDER BY 1, 2
    """

    table = _query(
        dataset,
        sql,
        (cutoff_epoch,) * 3,
    )

    if table.empty:
        raise ValueError(
            "No events remain beyond the temporal diagnostic cutoff"
        )

    table["percent_within_type"] = (
        100.0
        * table["events"]
        / table.groupby("event_type")["events"].transform("sum")
    )

    cutoff = pd.to_datetime(
        cutoff_epoch,
        unit="s",
        utc=True,
    )

    return cutoff, table
