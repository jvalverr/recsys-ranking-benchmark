"""Synthetic contract tests for scalable exploratory analyses."""

import json

import pandas as pd
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

from recsys.eda import (
    activity_bins,
    activity_summary,
    cold_start_snapshot,
    event_distribution,
    item_concentration,
    load_eda_dataset,
    monthly_events,
    rating_distribution,
    top_items,
)


@pytest.fixture
def dataset(tmp_path):
    """Researchers construct a transparent time-ordered fixture with a cold-start event."""

    path = tmp_path / "movielens-1m"
    path.mkdir()

    records = [
        ("u1", "i1", "2024-01-01", 5),
        ("u1", "i2", "2024-01-02", 4),
        ("u2", "i1", "2024-01-03", 3),
        ("u3", "i3", "2024-01-04", 2),
        ("u1", "i1", "2024-01-05", 5),
        ("u4", "i4", "2024-02-06", 4),
    ]

    frame = pd.DataFrame(
        records,
        columns=[
            "user_id",
            "item_id",
            "timestamp",
            "event_value",
        ],
    )

    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        utc=True,
    )

    frame["event_type"] = "rating"
    frame["dataset"] = "movielens-1m"

    frame.to_parquet(
        path / "events.parquet",
        index=False,
    )

    # Researchers record the expected source population so that
    # validation and downstream statistics can be checked separately.
    (path / "validation.json").write_text(
        json.dumps({
            "dataset": "movielens-1m",
            "valid": True,
            "statistics": {
                "events": 6,
                "users": 4,
                "items": 4,
                "interaction_sparsity": 0.6875,
            },
        }),
        encoding="utf-8",
    )

    return load_eda_dataset(
        "movielens-1m",
        root=tmp_path,
    )


def test_eda_activity_and_event_counts(dataset):
    """Researchers verify aggregated frequencies against source counts."""

    users = activity_summary(
        dataset,
        "user_id",
    ).iloc[0]

    assert int(users["entities"]) == 4
    assert int(users["singletons"]) == 3

    assert int(
        activity_bins(
            dataset,
            "item_id",
        )["events"].sum()
    ) == 6

    assert int(
        event_distribution(dataset)["events"].sum()
    ) == 6

    assert int(
        rating_distribution(dataset)["events"].sum()
    ) == 6

    assert int(
        monthly_events(dataset)["events"].sum()
    ) == 6


def test_eda_concentration_and_top_items(dataset):
    """Researchers verify item concentration using known synthetic frequencies."""

    result = item_concentration(
        dataset
    ).iloc[0]

    assert result["top_1pct_event_share"] == pytest.approx(
        3 / 6
    )

    assert top_items(
        dataset
    ).iloc[0]["item_id"] == "i1"


def test_eda_temporal_cold_start(dataset):
    """Researchers verify that unseen future IDs are identified without hindsight."""

    cutoff, segments = cold_start_snapshot(
        dataset,
        history_fraction=0.8,
    )

    assert cutoff.year == 2024

    assert int(
        segments["events"].sum()
    ) == 1

    assert (
        segments.iloc[0]["segment"]
        == "new_user_and_item"
    )

    assert (
        segments.iloc[0]["percent_within_type"]
        == pytest.approx(100.0)
    )


def test_reject_invalid_temporal_fraction(dataset):
    """Researchers reject ambiguous temporal diagnostics before querying data."""

    with pytest.raises(ValueError):
        cold_start_snapshot(
            dataset,
            history_fraction=1.0,
        )
