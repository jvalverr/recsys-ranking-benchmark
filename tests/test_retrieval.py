"""Verify retrieval and evaluation using transparent synthetic user histories."""

from pathlib import Path
import json

import pandas as pd
import pytest

from recsys.retrieval import (
    RetrievalConfig,
    RetrievalIndex,
)

from recsys.candidates import (
    generate_candidates,
    ranking_metrics,
)

from scripts.run_retrieval import (
    fit_command,
    evaluate_command,
)


def positive_pairs():
    """They include a duplicate to verify binary user-item support."""

    return pd.DataFrame(
        [
            ("u1", "a", "2024-01-01"),
            ("u1", "b", "2024-01-02"),
            ("u1", "a", "2024-01-03"),
            ("u2", "a", "2024-01-04"),
            ("u2", "b", "2024-01-05"),
            ("u3", "a", "2024-01-06"),
            ("u3", "c", "2024-01-07"),
        ],
        columns=[
            "user_id",
            "item_id",
            "last_event",
        ],
    )


def basic_config(**overrides):
    """They keep synthetic tests independent of large-dataset settings."""

    params = dict(
        max_profile_items=5,
        min_common_users=1,
        shrinkage=0.0,
        topk_neighbors=5,
        max_pair_contributions=100,
        candidate_knn=3,
        candidate_popularity=3,
        final_k=2,
        max_eval_users=10,
        seen_policy="all",
    )

    return RetrievalConfig(
        **(params | overrides)
    )


def test_unique_positive_support_and_sparse_neighbors():
    """
    They verify that repeated observations do not inflate
    item popularity or co-occurrence counts.
    """

    index = RetrievalIndex.fit(
        positive_pairs(),
        basic_config(),
    )

    support = dict(
        zip(
            index.item_ids,
            index.popularity,
        )
    )

    assert support["a"] == 3

    assert index.neighbors.shape == (
        3,
        3,
    )

    assert (
        index.neighbors.diagonal().sum()
        == 0
    )

    assert (
        index.fit_stats[
            "positive_user_item_pairs"
        ]
        == 6
    )

    assert (
        index.knn_top(
            {"b"},
            k=2,
            exclude={"b"},
        )[0][0]
        == "a"
    )


def test_cold_user_falls_back_to_popularity():
    """
    They verify that users without positive historical seeds
    can receive non-personalized recommendations.
    """

    index = RetrievalIndex.fit(
        positive_pairs(),
        basic_config(),
    )

    result = generate_candidates(
        index,
        set(),
        set(),
        basic_config(),
    )

    assert result.knn_items == []

    assert (
        result.popularity_items[0]
        == "a"
    )

    assert (
        result.rows[0]["item_id"]
        == "a"
    )

    assert len({
        row["item_id"]
        for row in result.rows
    }) == len(result.rows)


def test_seen_item_policy_is_explicit():
    """
    They distinguish novel-item recommendation
    from repeated-interaction prediction.
    """

    index = RetrievalIndex.fit(
        positive_pairs(),
        basic_config(),
    )

    novel = generate_candidates(
        index,
        {"a"},
        {"a"},
        basic_config(),
    )

    assert "a" not in [
        row["item_id"]
        for row in novel.rows
    ]

    repeats = generate_candidates(
        index,
        {"a"},
        {"a"},
        basic_config(
            seen_policy="none"
        ),
    )

    assert "a" in [
        row["item_id"]
        for row in repeats.rows
    ]


def test_memory_budget_fails_before_cooccurrence():
    """
    They reject co-occurrence workloads that exceed
    the configured computational budget.
    """

    with pytest.raises(
        ValueError,
        match="pair contributions",
    ):

        RetrievalIndex.fit(
            positive_pairs(),
            basic_config(
                max_pair_contributions=1
            ),
        )


def test_rank_metrics_reject_duplicates_and_preserve_bounds():
    """
    They verify that ranking metrics remain well-defined
    and reject duplicated recommendations.
    """

    with pytest.raises(
        ValueError,
        match="Duplicate",
    ):

        ranking_metrics(
            ["a", "a"],
            {"a"},
            2,
        )

    values = ranking_metrics(
        ["a", "b"],
        {"a"},
        2,
    )

    assert values["recall_at_k"] == 1.0

    assert values["ndcg_at_k"] == 1.0

    assert (
        values["observed_precision_at_k"]
        == 0.5
    )


def test_fitted_index_roundtrip(tmp_path):
    """
    They verify that persisted sparse indexes reproduce
    the original retrieval scores.
    """

    index = RetrievalIndex.fit(
        positive_pairs(),
        basic_config(),
    )

    index.save(tmp_path)

    loaded = RetrievalIndex.load(
        tmp_path
    )

    assert loaded.knn_top(
        {"b"},
        2,
    ) == index.knn_top(
        {"b"},
        2,
    )

    assert loaded.popularity_top(
        3
    ) == index.popularity_top(
        3
    )


def write_fixture(root: Path):
    """
    They construct chronological training, validation,
    and test data with known and unknown items.
    """

    dataset_dir = (
        root
        / "data/processed/movielens-1m"
    )

    modeling = (
        dataset_dir / "modeling"
    )

    modeling.mkdir(
        parents=True
    )

    raw = pd.DataFrame(
        [
            ("u1", "a", "2024-01-01", 5),
            ("u1", "b", "2024-01-02", 5),
            ("u2", "a", "2024-01-03", 5),
            ("u2", "b", "2024-01-04", 5),
            ("u1", "c", "2024-01-05", 5),
            ("u3", "a", "2024-01-06", 5),
            ("u1", "c", "2024-01-07", 5),
            ("u4", "d", "2024-01-08", 5),
        ],
        columns=[
            "user_id",
            "item_id",
            "timestamp",
            "event_value",
        ],
    )

    raw["timestamp"] = pd.to_datetime(
        raw["timestamp"],
        utc=True,
    )

    raw["dataset"] = "movielens-1m"
    raw["event_type"] = "rating"
    raw["source_row"] = range(len(raw))

    raw.to_parquet(
        dataset_dir / "events.parquet",
        index=False,
    )

    labeled = raw.copy()
    labeled["relevance_label"] = 1

    labeled.iloc[:4].to_parquet(
        modeling / "train.parquet",
        index=False,
    )

    labeled.iloc[4:6].to_parquet(
        modeling / "validation.parquet",
        index=False,
    )

    labeled.iloc[6:].to_parquet(
        modeling / "test.parquet",
        index=False,
    )

    pd.DataFrame({
        "user_id": ["u1", "u2"],
        "train_events": [2, 2],
    }).to_parquet(
        modeling / "train_user_counts.parquet",
        index=False,
    )

    # They record the same temporal contract as Stage 03,
    # allowing the end-to-end retrieval tests to run independently.
    (
        modeling / "manifest.json"
    ).write_text(
        json.dumps({
            "dataset": "movielens-1m",

            "feedback_policy": (
                "explicit_rating_ge_4"
            ),

            "configuration": {
                "rating_positive_threshold": 4.0,
            },

            "temporal_windows": {
                "train_end_utc": (
                    "2024-01-05T00:00:00+00:00"
                ),

                "validation_end_utc": (
                    "2024-01-07T00:00:00+00:00"
                ),
            },
        }),
        encoding="utf-8",
    )

    cfg = root / "configs"
    cfg.mkdir()

    (
        cfg / "retrieval.toml"
    ).write_text(
        '[defaults]\n'
        'max_profile_items = 5\n'
        'min_common_users = 1\n'
        'shrinkage = 0.0\n'
        'max_pair_contributions = 100\n'
        'candidate_knn = 2\n'
        'candidate_popularity = 2\n'
        'final_k = 2\n'
        'max_eval_users = 10\n'
        '[movielens-1m]\n'
        'seen_policy = "all"\n',
        encoding="utf-8",
    )

    return dataset_dir


def test_full_fit_and_validation_no_future_leakage(tmp_path):
    """
    They verify that validation labels cannot enter
    retrieval training or candidate construction.
    """

    dataset_dir = write_fixture(
        tmp_path
    )

    fitted = fit_command(
        "movielens-1m",
        tmp_path,
        Path("configs/retrieval.toml"),
        False,
    )

    assert (
        fitted["fit_statistics"][
            "items_with_positive_train_events"
        ]
        == 2
    )

    summary = evaluate_command(
        "movielens-1m",
        "validation",
        tmp_path,
        10,
        False,
        False,
    )

    assert (
        summary["sampled_positive_target_users"]
        == 2
    )

    assert (
        summary[
            "all_relevant_user_item_targets_in_sample"
        ]
        == 2
    )

    # They identify target c as outside the training
    # catalog rather than injecting it into the index.
    assert (
        summary[
            "targets_in_training_candidate_catalog"
        ]
        == 1
    )

    rows = pd.read_parquet(
        dataset_dir
        / "retrieval/validation/candidates.parquet"
    )

    assert "c" not in set(
        rows["item_id"]
    )

    assert not {
        "relevance_label",
        "is_positive",
    }.intersection(
        rows.columns
    )


def test_final_test_requires_explicit_authorization(tmp_path):
    """
    They protect final test evaluation and distinguish
    historical validation events from frozen model training.
    """

    write_fixture(tmp_path)

    fit_command(
        "movielens-1m",
        tmp_path,
        Path("configs/retrieval.toml"),
        False,
    )

    with pytest.raises(
        ValueError,
        match="--final-test",
    ):

        evaluate_command(
            "movielens-1m",
            "test",
            tmp_path,
            10,
            False,
            False,
        )

    evaluate_command(
        "movielens-1m",
        "validation",
        tmp_path,
        10,
        False,
        False,
    )

    result = evaluate_command(
        "movielens-1m",
        "test",
        tmp_path,
        10,
        False,
        True,
    )

    # They exclude c because it appeared during validation,
    # while d remains unavailable in the frozen training catalog.
    assert (
        result[
            "targets_eligible_after_seen_filter"
        ]
        == 0
    )

    assert (
        result["candidate_hits_in_sample"]
        == 0
    )
