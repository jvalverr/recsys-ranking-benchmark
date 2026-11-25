"""They define reproducible feedback policies and train-only support filters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
import tomllib

import duckdb

from recsys.utils import sql_literal


DATASETS = {
    "movielens-1m",
    "retailrocket",
    "amazon-video-games",
}


@dataclass(frozen=True)
class ModelingConfig:
    """
    They expose policy choices explicitly instead of hiding
    dataset-specific assumptions in processing functions.
    """

    train_fraction: float = 0.70
    validation_fraction: float = 0.15

    min_train_user_events: int = 1
    min_train_item_events: int = 1

    rating_positive_threshold: float = 4.0
    implicit_target: str = "any_event"

    train_end_utc: str | None = None
    validation_end_utc: str | None = None

    def __post_init__(self) -> None:

        if not (0 < self.train_fraction < 1):
            raise ValueError(
                "train_fraction must be between 0 and 1"
            )

        if not (
            0 < self.validation_fraction
            < 1 - self.train_fraction
        ):
            raise ValueError(
                "validation_fraction must leave a "
                "nonempty test fraction"
            )

        if (
            self.min_train_user_events < 1
            or self.min_train_item_events < 1
        ):
            raise ValueError(
                "Train minimum interaction counts must be >= 1"
            )

        if not (
            1 <= self.rating_positive_threshold <= 5
        ):
            raise ValueError(
                "rating_positive_threshold must be between 1 and 5"
            )

        if self.implicit_target not in {
            "any_event",
            "transaction_only",
        }:
            raise ValueError(
                "implicit_target must be any_event "
                "or transaction_only"
            )

        if (
            (self.train_end_utc is None)
            != (self.validation_end_utc is None)
        ):
            raise ValueError(
                "Both explicit cutoff timestamps "
                "must be supplied together"
            )

    def as_dict(self) -> dict:
        return asdict(self)


def load_modeling_config(
    path: Path,
    dataset: str,
) -> ModelingConfig:
    """
    They validate dataset-specific settings and reject unknown
    configuration fields before starting expensive processing.
    """

    if dataset not in DATASETS:
        raise ValueError(
            f"Unsupported dataset: {dataset}"
        )

    with path.open("rb") as handle:
        document = tomllib.load(handle)

    options = {
        **document.get("defaults", {}),
        **document.get(dataset, {}),
    }

    supported = {
        entry.name
        for entry in fields(ModelingConfig)
    }

    extra = set(options) - supported

    if extra:
        raise ValueError(
            "Unknown modeling configuration fields: "
            f"{sorted(extra)}"
        )

    config = ModelingConfig(**options)

    if (
        dataset != "retailrocket"
        and config.implicit_target != "any_event"
    ):
        raise ValueError(
            "implicit_target applies only to RetailRocket"
        )

    return config


def feedback_projection(
    dataset: str,
    config: ModelingConfig,
) -> str:
    """
    They derive dataset-specific labels while retaining every
    original source event and its associated attributes.
    """

    if dataset not in DATASETS:
        raise ValueError(
            f"Unsupported dataset: {dataset}"
        )

    if dataset == "retailrocket":

        positive = (
            "event_type IN ('view', 'addtocart', 'transaction')"
            if config.implicit_target == "any_event"
            else "event_type = 'transaction'"
        )

        # They preserve non-target observations as unlabeled.
        # Missing interactions are never fabricated as negative
        # examples during preprocessing.
        return (
            f"CASE WHEN {positive} THEN 1 ELSE NULL END"
            "::INTEGER AS relevance_label, "
            f"CASE WHEN {positive} "
            "THEN 'observed_positive' "
            "ELSE 'unlabeled' END AS label_status"
        )

    threshold = repr(
        float(config.rating_positive_threshold)
    )

    # They distinguish explicit below-threshold ratings from
    # unobserved user-item pairs, which remain unknown.
    return (
        "CASE WHEN CAST(event_value AS DOUBLE) "
        f">= {threshold} THEN 1 ELSE 0 END"
        "::INTEGER AS relevance_label, "
        "CASE WHEN CAST(event_value AS DOUBLE) "
        f">= {threshold} "
        "THEN 'observed_positive' "
        "ELSE 'observed_below_threshold' END AS label_status"
    )


def count_rows(
    con: duckdb.DuckDBPyConnection,
    path: Path,
) -> int:
    """
    They obtain record counts from Parquet without loading
    entire datasets into pandas.
    """

    return int(
        con.execute(
            "SELECT COUNT(*) "
            f"FROM read_parquet({sql_literal(path)})"
        ).fetchone()[0]
    )


def filter_train_by_support(
    con: duckdb.DuckDBPyConnection,
    source: Path,
    workdir: Path,
    config: ModelingConfig,
) -> tuple[Path, dict]:
    """
    They iteratively filter training interactions until all
    retained users and items meet the configured minimum support.
    """

    before = count_rows(
        con,
        source,
    )

    if (
        config.min_train_user_events
        == config.min_train_item_events
        == 1
    ):
        return source, {
            "before": before,
            "after": before,
            "removed": 0,
            "iterations": 0,
        }

    current = source
    previous_count = before

    # They use iterative filtering because removing low-activity
    # users can also reduce item support, and vice versa.
    for iteration in range(1, 51):

        next_path = (
            workdir
            / f"eligible_train_iter_{iteration}.parquet"
        )

        source_sql = (
            f"read_parquet({sql_literal(current)})"
        )

        query = f"""
            WITH supported_users AS (
                SELECT user_id
                FROM {source_sql}
                GROUP BY user_id
                HAVING COUNT(*) >=
                    {config.min_train_user_events}
            ),

            supported_items AS (
                SELECT item_id
                FROM {source_sql}
                GROUP BY item_id
                HAVING COUNT(*) >=
                    {config.min_train_item_events}
            )

            SELECT e.*
            FROM {source_sql} AS e

            INNER JOIN supported_users AS u
                USING (user_id)

            INNER JOIN supported_items AS i
                USING (item_id)
        """

        con.execute(
            f"COPY ({query}) "
            f"TO {sql_literal(next_path)} "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )

        current_count = count_rows(
            con,
            next_path,
        )

        if not current_count:
            raise ValueError(
                "Train support filtering removed "
                "every interaction"
            )

        if current_count == previous_count:

            return current, {
                "before": before,
                "after": current_count,
                "removed": before - current_count,
                "iterations": iteration,
            }

        current = next_path
        previous_count = current_count

    raise RuntimeError(
        "Train support filtering did not converge "
        "after 50 iterations"
    )


def write_model_history(
    con: duckdb.DuckDBPyConnection,
    train: Path,
    directory: Path,
) -> None:
    """
    They construct user/item histories exclusively from
    retained training interactions.
    """

    source = (
        f"read_parquet({sql_literal(train)})"
    )

    for entity, filename in (
        ("user_id", "train_user_counts.parquet"),
        ("item_id", "train_item_counts.parquet"),
    ):

        # They preserve historical interaction counts for
        # downstream eligibility checks and future baselines.
        con.execute(
            f"COPY ("
            f"SELECT {entity}, COUNT(*) AS train_events "
            f"FROM {source} "
            f"GROUP BY {entity}"
            f") TO {sql_literal(directory / filename)} "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )


def write_train(
    con: duckdb.DuckDBPyConnection,
    filtered: Path,
    output: Path,
) -> None:
    """
    They retain a consistent schema without attaching future
    cold-start labels to training observations.
    """

    null_fields = ", ".join([
        "CAST(NULL AS BOOLEAN) AS known_user_to_model",
        "CAST(NULL AS BOOLEAN) AS known_item_to_model",
        "CAST(NULL AS VARCHAR) AS model_cold_start_segment",
        "CAST(NULL AS BOOLEAN) AS known_user_at_window_start",
        "CAST(NULL AS BOOLEAN) AS known_item_at_window_start",
        "CAST(NULL AS VARCHAR) AS window_cold_start_segment",
    ])

    con.execute(
        f"COPY ("
        f"SELECT e.*, {null_fields} "
        f"FROM read_parquet({sql_literal(filtered)}) AS e"
        f") TO {sql_literal(output)} "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )


def write_evaluation_split(
    con: duckdb.DuckDBPyConnection,
    evaluation: Path,
    history: list[Path],
    directory: Path,
    output: Path,
) -> None:
    """
    They distinguish identifiers available to the fitted model
    from those observed before the evaluation-window boundary.
    """

    history_paths = (
        "["
        + ", ".join(
            sql_literal(path)
            for path in history
        )
        + "]"
    )

    sql = f"""
        WITH history_users AS (
            SELECT DISTINCT user_id
            FROM read_parquet({history_paths})
        ),

        history_items AS (
            SELECT DISTINCT item_id
            FROM read_parquet({history_paths})
        ),

        tagged AS (
            SELECT
                e.*,

                (mu.user_id IS NOT NULL)
                    AS known_user_to_model,

                (mi.item_id IS NOT NULL)
                    AS known_item_to_model,

                (hu.user_id IS NOT NULL)
                    AS known_user_at_window_start,

                (hi.item_id IS NOT NULL)
                    AS known_item_at_window_start

            FROM read_parquet(
                {sql_literal(evaluation)}
            ) AS e

            LEFT JOIN read_parquet(
                {sql_literal(directory / "train_user_counts.parquet")}
            ) AS mu
                ON e.user_id = mu.user_id

            LEFT JOIN read_parquet(
                {sql_literal(directory / "train_item_counts.parquet")}
            ) AS mi
                ON e.item_id = mi.item_id

            LEFT JOIN history_users AS hu
                ON e.user_id = hu.user_id

            LEFT JOIN history_items AS hi
                ON e.item_id = hi.item_id
        )

        SELECT
            * EXCLUDE (
                known_user_at_window_start,
                known_item_at_window_start
            ),

            CASE
                WHEN known_user_to_model
                 AND known_item_to_model
                    THEN 'warm'

                WHEN NOT known_user_to_model
                 AND known_item_to_model
                    THEN 'new_user'

                WHEN known_user_to_model
                 AND NOT known_item_to_model
                    THEN 'new_item'

                ELSE 'new_user_and_item'
            END AS model_cold_start_segment,

            known_user_at_window_start,
            known_item_at_window_start,

            CASE
                WHEN known_user_at_window_start
                 AND known_item_at_window_start
                    THEN 'warm'

                WHEN NOT known_user_at_window_start
                 AND known_item_at_window_start
                    THEN 'new_user'

                WHEN known_user_at_window_start
                 AND NOT known_item_at_window_start
                    THEN 'new_item'

                ELSE 'new_user_and_item'
            END AS window_cold_start_segment

        FROM tagged
    """

    # They keep every validation/test interaction, including
    # records involving users or items not observed in training.
    con.execute(
        f"COPY ({sql}) "
        f"TO {sql_literal(output)} "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
