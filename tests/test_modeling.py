"""They verify chronological partitioning and feedback semantics using synthetic events."""

import json

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from recsys.preprocessing import (
    ModelingConfig,
    filter_train_by_support,
)

from scripts.prepare_modeling import (
    prepare_modeling,
)


def fixture_project(
    root: Path,
    dataset: str,
    *,
    retail: bool = False,
) -> Path:
    """
    They create a controlled event sequence containing
    known and previously unseen users and items.
    """

    directory = (
        root / "data" / "processed" / dataset
    )

    directory.mkdir(
        parents=True,
    )

    users = [
        "u1", "u1", "u2", "u2",
        "u3", "u3", "u4", "u4",
        "u1", "u5", "u1", "u6",
    ]

    items = [
        "i1", "i2", "i1", "i2",
        "i1", "i2", "i1", "i3",
        "i3", "i1", "i4", "i5",
    ]

    ratings = [
        5, 4, 2, 5,
        3, 5, 4, 5,
        4, 5, 3, 5,
    ]

    types = (
        [
            "view",
            "transaction",
            "view",
            "addtocart",
        ] * 3
        if retail
        else ["rating"] * 12
    )

    data = pd.DataFrame({
        "user_id": users,
        "item_id": items,

        "timestamp": pd.date_range(
            "2024-01-01",
            periods=12,
            tz="UTC",
            freq="D",
        ),

        "event_type": types,

        "event_value": (
            [None] * 12
            if retail
            else ratings
        ),

        "dataset": [dataset] * 12,
        "source_row": range(12),
    })

    data.to_parquet(
        directory / "events.parquet",
        index=False,
    )

    # They reproduce the minimal verified-input contract
    # expected from the previous ingestion stage.
    (directory / "validation.json").write_text(
        json.dumps({
            "valid": True,
            "dataset": dataset,
        }),
        encoding="utf-8",
    )

    return directory


def config_file(
    root: Path,
    *,
    retail: bool = False,
) -> Path:
    """
    They use deterministic fractions to make the expected
    chronology easy to inspect.
    """

    config = root / "modeling.toml"

    text = (
        "[defaults]\n"
        "train_fraction = 0.50\n"
        "validation_fraction = 0.25\n"
    )

    if retail:
        text += (
            "[retailrocket]\n"
            'implicit_target = "transaction_only"\n'
        )

    config.write_text(
        text,
        encoding="utf-8",
    )

    return config


def test_explicit_rating_labels_and_chronology(
    tmp_path,
):
    """
    They verify source conservation, chronological ordering,
    and binary labels derived from explicit ratings.
    """

    fixture_project(
        tmp_path,
        "movielens-1m",
    )

    cfg = config_file(tmp_path)

    manifest = prepare_modeling(
        "movielens-1m",
        tmp_path,
        cfg,
    )

    directory = (
        tmp_path
        / "data/processed/movielens-1m/modeling"
    )

    train = pd.read_parquet(
        directory / "train.parquet"
    )

    validation = pd.read_parquet(
        directory / "validation.parquet"
    )

    test = pd.read_parquet(
        directory / "test.parquet"
    )

    assert (
        len(train)
        + len(validation)
        + len(test)
    ) == 12

    assert len(train) > 0
    assert len(validation) > 0
    assert len(test) > 0

    assert (
        train.timestamp.max()
        < validation.timestamp.min()
    )

    assert (
        validation.timestamp.max()
        < test.timestamp.min()
    )

    assert (
        train.loc[
            train.event_value == 2,
            "relevance_label",
        ].iloc[0]
        == 0
    )

    assert (
        train.loc[
            train.event_value == 5,
            "relevance_label",
        ].iloc[0]
        == 1
    )

    assert (
        manifest["train_support_filter"]["removed"]
        == 0
    )


def test_model_fit_and_window_history_differ(
    tmp_path,
):
    """
    They distinguish identifiers known to the fitted model
    from identifiers observed in later historical windows.
    """

    fixture_project(
        tmp_path,
        "movielens-1m",
    )

    prepare_modeling(
        "movielens-1m",
        tmp_path,
        config_file(tmp_path),
    )

    directory = (
        tmp_path
        / "data/processed/movielens-1m/modeling"
    )

    test = pd.read_parquet(
        directory / "test.parquet"
    )

    # They verify that item i3 remains unknown to the model
    # fitted on training, even though it first appeared
    # during validation.
    example = test.loc[
        (test.user_id == "u1")
        & (test.item_id == "i3")
    ].iloc[0]

    assert (
        example.model_cold_start_segment
        == "new_item"
    )

    assert (
        example.window_cold_start_segment
        == "warm"
    )


def test_unfiltered_evaluation_windows_with_train_support(
    tmp_path,
):
    """
    They verify that training support filters do not exclude
    difficult validation or test observations.
    """

    fixture_project(
        tmp_path,
        "movielens-1m",
    )

    config = tmp_path / "strict.toml"

    config.write_text(
        "[defaults]\n"
        "train_fraction = 0.50\n"
        "validation_fraction = 0.25\n"
        "min_train_user_events = 2\n"
        "min_train_item_events = 2\n",
        encoding="utf-8",
    )

    report = prepare_modeling(
        "movielens-1m",
        tmp_path,
        config,
    )

    outputs = report["label_statistics"]
    original = report["raw_split_statistics"]

    assert (
        outputs["validation"]["events"]
        == original["validation"]["events"]
    )

    assert (
        outputs["test"]["events"]
        == original["test"]["events"]
    )

    assert (
        outputs["train"]["events"]
        <= original["train"]["events"]
    )


def test_transaction_only_keeps_other_events_unlabeled(
    tmp_path,
):
    """
    They avoid treating views or cart additions as confirmed
    negative purchase outcomes.
    """

    fixture_project(
        tmp_path,
        "retailrocket",
        retail=True,
    )

    report = prepare_modeling(
        "retailrocket",
        tmp_path,
        config_file(
            tmp_path,
            retail=True,
        ),
    )

    counts = report["label_statistics"].values()

    assert sum(
        row["observed_below_threshold"]
        for row in counts
    ) == 0

    assert sum(
        row["unlabeled"]
        for row in counts
    ) > 0

    assert sum(
        row["observed_positive"]
        for row in counts
    ) > 0


def test_timestamps_with_identical_values_share_one_window(
    tmp_path,
):
    """
    They verify that timestamp ties never cross chronological
    partitions, even if target proportions change.
    """

    directory = fixture_project(
        tmp_path,
        "movielens-1m",
    )

    path = directory / "events.parquet"

    df = pd.read_parquet(path)

    df.loc[4:6, "timestamp"] = pd.Timestamp(
        "2024-01-06",
        tz="UTC",
    )

    df.to_parquet(
        path,
        index=False,
    )

    report = prepare_modeling(
        "movielens-1m",
        tmp_path,
        config_file(tmp_path),
    )

    assert sum(
        part["events"]
        for part in report["raw_split_statistics"].values()
    ) == 12

    output = directory / "modeling"

    groups = [
        pd.read_parquet(
            output / f"{part}.parquet"
        )
        for part in (
            "train",
            "validation",
            "test",
        )
    ]

    for first, second in (
        (0, 1),
        (0, 2),
        (1, 2),
    ):
        assert not set(
            groups[first].timestamp
        ).intersection(
            set(groups[second].timestamp)
        )


def test_invalid_cutoff_configuration():
    """
    They reject invalid fractions and incomplete explicit
    temporal configurations.
    """

    with pytest.raises(ValueError):
        ModelingConfig(
            train_fraction=0.95,
            validation_fraction=0.1,
        )

    with pytest.raises(ValueError):
        ModelingConfig(
            train_end_utc="2024-01-01T00:00:00Z"
        )


def test_overwrite_requires_explicit_permission(
    tmp_path,
):
    """
    They protect completed modeling artifacts against
    accidental replacement.
    """

    fixture_project(
        tmp_path,
        "movielens-1m",
    )

    cfg = config_file(tmp_path)

    prepare_modeling(
        "movielens-1m",
        tmp_path,
        cfg,
    )

    with pytest.raises(FileExistsError):
        prepare_modeling(
            "movielens-1m",
            tmp_path,
            cfg,
        )

    result = prepare_modeling(
        "movielens-1m",
        tmp_path,
        cfg,
        overwrite=True,
    )

    assert (
        result["dataset"]
        == "movielens-1m"
    )


def test_iterative_support_preserves_joint_minimums(
    tmp_path,
):
    """
    They verify that retained users and items jointly
    satisfy the configured minimum interaction counts.
    """

    frame = pd.DataFrame({
        "user_id": [
            "u1", "u1", "u2", "u2", "u3",
        ],

        "item_id": [
            "i1", "i2", "i1", "i2", "i2",
        ],
    })

    source = tmp_path / "train.parquet"

    frame.to_parquet(
        source,
        index=False,
    )

    config = ModelingConfig(
        min_train_user_events=2,
        min_train_item_events=2,
    )

    with duckdb.connect(":memory:") as con:
        filtered, report = (
            filter_train_by_support(
                con,
                source,
                tmp_path,
                config,
            )
        )

    survivors = pd.read_parquet(
        filtered
    )

    assert len(survivors) == 4
    assert report["removed"] == 1

    assert (
        survivors.groupby("user_id").size().min()
        >= 2
    )

    assert (
        survivors.groupby("item_id").size().min()
        >= 2
    )
