"""Evaluate frozen retrieval indexes with strictly pre-window user histories."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from time import perf_counter

import duckdb
import numpy as np
import pandas as pd

from recsys.candidates import (
    generate_candidates,
    ranking_metrics,
)
from recsys.retrieval import RetrievalIndex


def _selected_positive_users(
    con: duckdb.DuckDBPyConnection,
    evaluation: Path,
    limit: int,
    seed: int,
) -> tuple[list[str], int]:
    """
    They sample distinct users with at least one positive
    held-out target using deterministic hashing.
    """

    total = int(
        con.execute(
            """
            SELECT COUNT(DISTINCT user_id)
            FROM read_parquet(?)
            WHERE relevance_label = 1
            """,
            [str(evaluation)],
        ).fetchone()[0]
    )

    if total == 0:
        raise ValueError(
            "The evaluation window contains no observed-positive targets"
        )

    selected = con.execute(
        """
        WITH positive_users AS (
            SELECT DISTINCT user_id
            FROM read_parquet(?)
            WHERE relevance_label = 1
        )

        SELECT user_id
        FROM positive_users
        ORDER BY
            md5(CAST(user_id AS VARCHAR) || ?),
            user_id
        LIMIT ?
        """,
        [
            str(evaluation),
            f":{seed}",
            limit,
        ],
    ).fetchall()

    return [
        str(row[0])
        for row in selected
    ], total


def _load_relevant_targets(
    con: duckdb.DuckDBPyConnection,
    evaluation: Path,
) -> dict[str, set[str]]:
    """
    They collapse repeated positive events into distinct
    user-item evaluation targets.
    """

    target_rows = con.execute(
        """
        SELECT DISTINCT
            e.user_id,
            e.item_id

        FROM read_parquet(?) AS e

        INNER JOIN selected_users AS s
            USING (user_id)

        WHERE e.relevance_label = 1
        """,
        [str(evaluation)],
    ).fetchall()

    positives = defaultdict(set)

    for user, item in target_rows:
        positives[str(user)].add(
            str(item)
        )

    return positives


def _load_pre_window_history(
    con: duckdb.DuckDBPyConnection,
    original_events: Path,
    cutoff_utc: str,
) -> pd.DataFrame:
    """
    They construct user histories from records strictly
    preceding the evaluation window.
    """

    return con.execute(
        """
        SELECT
            e.user_id,
            e.item_id,
            e.event_type,
            e.event_value

        FROM read_parquet(?) AS e

        INNER JOIN selected_users AS s
            USING (user_id)

        WHERE e.timestamp < CAST(? AS TIMESTAMPTZ)
        """,
        [
            str(original_events),
            cutoff_utc,
        ],
    ).df()


def _profile_sets(
    history: pd.DataFrame,
    dataset: str,
    modeling_manifest: dict,
) -> tuple[
    dict[str, set[str]],
    dict[str, set[str]],
]:
    """
    They separate historically observed items from
    positive user-profile seeds.
    """

    seen = defaultdict(set)
    positive = defaultdict(set)

    if dataset == "retailrocket":

        policy = modeling_manifest[
            "feedback_policy"
        ]

        if policy not in {
            "any_event",
            "transaction_only",
        }:
            raise ValueError(
                f"Unrecognized implicit-feedback target: {policy}"
            )

    else:

        threshold = float(
            modeling_manifest[
                "configuration"
            ]["rating_positive_threshold"]
        )

    for row in history.itertuples(
        index=False
    ):

        user = str(row.user_id)
        item = str(row.item_id)

        seen[user].add(item)

        if dataset == "retailrocket":

            is_positive = (
                policy == "any_event"
                or row.event_type == "transaction"
            )

        else:

            is_positive = (
                row.event_type == "rating"
                and pd.notna(row.event_value)
                and float(row.event_value) >= threshold
            )

        if is_positive:
            positive[user].add(item)

    return positive, seen


def evaluate_retrieval(
    index: RetrievalIndex,
    *,
    dataset: str,
    split: str,
    processed_dataset_dir: Path,
    modeling_manifest: dict,
    max_eval_users: int | None = None,
) -> tuple[
    dict,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    They evaluate fixed retrieval models without using
    future outcomes to construct candidate lists.
    """

    if split not in {
        "validation",
        "test",
    }:
        raise ValueError(
            "Evaluation split must be validation or test"
        )

    config = index.config

    limit = (
        max_eval_users
        if max_eval_users is not None
        else config.max_eval_users
    )

    if not isinstance(limit, int) or limit < 1:
        raise ValueError(
            "max_eval_users must be a positive integer"
        )

    evaluation = (
        processed_dataset_dir
        / "modeling"
        / f"{split}.parquet"
    )

    source = (
        processed_dataset_dir
        / "events.parquet"
    )

    train_user_counts = (
        processed_dataset_dir
        / "modeling"
        / "train_user_counts.parquet"
    )

    # They select the correct historical boundary.
    # Validation uses train history; test additionally
    # allows previously observed validation events.
    cutoff = modeling_manifest[
        "temporal_windows"
    ][
        "train_end_utc"
        if split == "validation"
        else "validation_end_utc"
    ]

    for path in (
        evaluation,
        source,
        train_user_counts,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    with duckdb.connect(":memory:") as con:

        con.execute(
            "SET TimeZone='UTC'"
        )

        selected_users, total_positive_users = (
            _selected_positive_users(
                con,
                evaluation,
                limit,
                config.seed,
            )
        )

        con.register(
            "selected_users",
            pd.DataFrame({
                "user_id": selected_users,
            }),
        )

        targets = _load_relevant_targets(
            con,
            evaluation,
        )

        history = _load_pre_window_history(
            con,
            source,
            cutoff,
        )

        trained_users = {
            str(row[0])
            for row in con.execute(
                """
                SELECT c.user_id

                FROM read_parquet(?) AS c

                INNER JOIN selected_users AS s
                    USING (user_id)
                """,
                [str(train_user_counts)],
            ).fetchall()
        }

    positive_history, seen_history = (
        _profile_sets(
            history,
            dataset,
            modeling_manifest,
        )
    )

    catalog = set(index.item_ids)

    user_rows = []
    candidate_rows = []

    for user_id in selected_users:

        relevant = targets[user_id]

        seeds = positive_history.get(
            user_id,
            set(),
        )

        seen = seen_history.get(
            user_id,
            set(),
        )

        excluded = (
            seen
            if config.seen_policy == "all"
            else set()
        )

        # They distinguish targets present in the frozen
        # catalog from those excluded by novelty policies.
        possible_catalog = (
            relevant & catalog
        )

        eligible = (
            possible_catalog - excluded
        )

        # They measure retrieval and fusion latency without
        # including data loading or evaluation computations.
        start = perf_counter()

        result = generate_candidates(
            index,
            seeds,
            seen,
            config,
        )

        generation_ms = (
            perf_counter() - start
        ) * 1000.0

        candidate_ids = [
            record["item_id"]
            for record in result.rows
        ]

        retrieved = len(
            relevant.intersection(
                candidate_ids
            )
        )

        # They persist candidate provenance without adding
        # future relevance labels to candidate records.
        for record in result.rows:

            candidate_rows.append({
                "user_id": user_id,
                "split": split,
                **record,
            })

        row = {
            "user_id": user_id,
            "split": split,

            "user_in_fitted_training": (
                user_id in trained_users
            ),

            "user_has_known_positive_seeds": (
                result.known_seed_items > 0
            ),

            "target_items": len(relevant),

            "target_items_in_training_catalog": (
                len(possible_catalog)
            ),

            "eligible_target_items": (
                len(eligible)
            ),

            "targets_excluded_by_seen_policy": (
                len(possible_catalog - eligible)
            ),

            "candidate_hits": retrieved,

            "candidate_recall_all": (
                retrieved / len(relevant)
            ),

            "candidate_recall_eligible": (
                retrieved / len(eligible)
                if eligible
                else float("nan")
            ),

            "candidate_pool_size": (
                len(candidate_ids)
            ),

            "candidate_generation_ms": (
                generation_ms
            ),

            "popularity_pool_size": (
                len(result.popularity_items)
            ),

            "knn_pool_size": (
                len(result.knn_items)
            ),
        }

        # They compare methods against identical targets,
        # distinguishing standalone baselines from fused ranking.
        for prefix, ranking in (
            (
                "popularity",
                result.popularity_items,
            ),
            (
                "knn",
                result.knn_items,
            ),
            (
                "fused",
                candidate_ids,
            ),
        ):

            for name, score in ranking_metrics(
                ranking,
                relevant,
                config.final_k,
            ).items():

                row[
                    f"{prefix}_{name}"
                ] = score

        user_rows.append(row)

    users = pd.DataFrame(
        user_rows
    )

    candidates = pd.DataFrame(
        candidate_rows,
        columns=[
            "user_id",
            "split",
            "item_id",
            "position",
            "fusion_score",
            "popularity_rank",
            "knn_rank",
            "popularity_score",
            "knn_score",
            "from_popularity",
            "from_knn",
        ],
    )

    metric_suffixes = (
        "observed_precision_at_k",
        "recall_at_k",
        "map_at_k",
        "ndcg_at_k",
    )

    numeric_metric_cols = [
        col
        for col in users.columns
        if any(
            col.endswith(suffix)
            for suffix in metric_suffixes
        )
    ]

    # They report micro and macro candidate recall separately
    # because users have different numbers of positive targets.
    summary = {
        "dataset": dataset,
        "split": split,

        "temporal_history_cutoff_utc": (
            cutoff
        ),

        "evaluation_policy": (
            "static_window_start_with_frozen_train_retrieval_model"
        ),

        "seen_policy": (
            config.seen_policy
        ),

        "selection": (
            "deterministic_uniform_hash_sample_of_positive_target_users"
        ),

        "positive_target_users_full_window": (
            total_positive_users
        ),

        "sampled_positive_target_users": (
            int(len(users))
        ),

        "all_relevant_user_item_targets_in_sample": (
            int(users["target_items"].sum())
        ),

        "targets_in_training_candidate_catalog": (
            int(
                users[
                    "target_items_in_training_catalog"
                ].sum()
            )
        ),

        "targets_eligible_after_seen_filter": (
            int(
                users[
                    "eligible_target_items"
                ].sum()
            )
        ),

        "candidate_hits_in_sample": (
            int(users["candidate_hits"].sum())
        ),

        "candidate_recall_micro_all": float(
            users["candidate_hits"].sum()
            / users["target_items"].sum()
        ),

        "candidate_recall_macro_all": float(
            users["candidate_recall_all"].mean()
        ),

        "candidate_recall_macro_eligible": (
            float(
                users[
                    "candidate_recall_eligible"
                ].mean()
            )
            if users[
                "candidate_recall_eligible"
            ].notna().any()
            else None
        ),

        "training_catalog_target_coverage": float(
            users[
                "target_items_in_training_catalog"
            ].sum()
            / users["target_items"].sum()
        ),

        "eligible_target_coverage": float(
            users[
                "eligible_target_items"
            ].sum()
            / users["target_items"].sum()
        ),

        "users_in_fitted_training_share": float(
            users[
                "user_in_fitted_training"
            ].mean()
        ),

        "users_with_known_positive_seeds_share": float(
            users[
                "user_has_known_positive_seeds"
            ].mean()
        ),

        "mean_candidate_pool_size": float(
            users[
                "candidate_pool_size"
            ].mean()
        ),

        "candidate_generation_latency_ms_p50": float(
            users[
                "candidate_generation_ms"
            ].quantile(0.5)
        ),

        "candidate_generation_latency_ms_p95": float(
            users[
                "candidate_generation_ms"
            ].quantile(0.95)
        ),

        "latency_scope": (
            "in-process per-user retrieval and fusion; "
            "excludes DuckDB, I/O, and API overhead"
        ),

        "ranking_metrics_macro": {
            col: float(
                users[col].mean()
            )
            for col in numeric_metric_cols
        },

        "interpretation": [
            "Only observed positive user-item targets are used as evaluation labels.",

            "A missing interaction does not establish a negative preference.",

            "Targets outside the fitted positive-item catalog cannot be retrieved.",

            "No events from the evaluation window enter user profiles or the fitted index.",

            "Precision@K is an observed-hit proxy, not a counterfactual click-through rate.",

            "Results from user sampling must not be presented as full-population metrics.",
        ],
    }

    # They report trained and previously unseen users separately,
    # preventing aggregate metrics from hiding cold-user failures.
    summary["by_fitted_user_status"] = {}

    for known, cohort in users.groupby(
        "user_in_fitted_training"
    ):

        label = (
            "trained_user"
            if known
            else "user_not_in_fitted_training"
        )

        summary[
            "by_fitted_user_status"
        ][label] = {
            "users": int(
                len(cohort)
            ),

            "candidate_recall_macro_all": float(
                cohort[
                    "candidate_recall_all"
                ].mean()
            ),

            "fused_ndcg_at_k": float(
                cohort[
                    "fused_ndcg_at_k"
                ].mean()
            ),

            "popularity_ndcg_at_k": float(
                cohort[
                    "popularity_ndcg_at_k"
                ].mean()
            ),

            "knn_ndcg_at_k": float(
                cohort[
                    "knn_ndcg_at_k"
                ].mean()
            ),
        }

    return summary, users, candidates

