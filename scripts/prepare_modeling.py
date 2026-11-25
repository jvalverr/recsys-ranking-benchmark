"""They prepare chronological modeling datasets with explicit feedback policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile

from pathlib import Path

import duckdb

from recsys.preprocessing import (
    DATASETS,
    feedback_projection,
    filter_train_by_support,
    load_modeling_config,
    write_evaluation_split,
    write_model_history,
    write_train,
)

from recsys.splitting import (
    resolve_windows,
    split_source,
    validate_source,
    verify_splits,
)

from recsys.utils import (
    sql_literal,
    write_json,
)


def sha256_file(path: Path) -> str:
    """
    They record source-file identities to improve experiment
    traceability and reproducibility.
    """

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def label_counts(
    con: duckdb.DuckDBPyConnection,
    path: Path,
) -> dict:
    """
    They distinguish observed positives, below-threshold
    ratings, and unlabeled behavioral events.
    """

    total, positives, zeros, unlabeled = con.execute(
        f"""
        SELECT
            COUNT(*),

            COUNT(*) FILTER (
                WHERE relevance_label = 1
            ),

            COUNT(*) FILTER (
                WHERE relevance_label = 0
            ),

            COUNT(*) FILTER (
                WHERE relevance_label IS NULL
            )

        FROM read_parquet(
            {sql_literal(path)}
        )
        """
    ).fetchone()

    return {
        "events": int(total),
        "observed_positive": int(positives),
        "observed_below_threshold": int(zeros),
        "unlabeled": int(unlabeled),
    }


def segment_counts(
    con: duckdb.DuckDBPyConnection,
    path: Path,
    column: str,
) -> dict:
    """
    They summarize cold-start scenarios independently
    for each evaluation-history definition.
    """

    if column not in {
        "model_cold_start_segment",
        "window_cold_start_segment",
    }:
        raise ValueError(
            "Unsupported segment column"
        )

    rows = con.execute(
        f"SELECT {column}, COUNT(*) "
        f"FROM read_parquet({sql_literal(path)}) "
        f"GROUP BY {column} "
        f"ORDER BY {column}"
    ).fetchall()

    return {
        str(label): int(count)
        for label, count in rows
    }


def prepare_modeling(
    dataset: str,
    project_root: Path,
    config_path: Path,
    *,
    overwrite: bool = False,
) -> dict:
    """
    They build reproducible modeling artifacts while
    preserving the original validated Parquet dataset.
    """

    if dataset not in DATASETS:
        raise ValueError(
            f"Unsupported dataset: {dataset}"
        )

    root = project_root.resolve()

    config_file = (
        config_path
        if config_path.is_absolute()
        else root / config_path
    )

    config = load_modeling_config(
        config_file,
        dataset,
    )

    dataset_dir = (
        root
        / "data"
        / "processed"
        / dataset
    )

    source = dataset_dir / "events.parquet"
    validation_file = dataset_dir / "validation.json"
    final = dataset_dir / "modeling"

    # They require a successful ingestion report before
    # creating datasets intended for model development.
    if (
        not source.is_file()
        or not validation_file.is_file()
    ):
        raise FileNotFoundError(
            "The validated Stage 01 Parquet data "
            "and report are required"
        )

    source_report = json.loads(
        validation_file.read_text(
            encoding="utf-8"
        )
    )

    if (
        source_report.get("dataset") != dataset
        or source_report.get("valid") is not True
    ):
        raise ValueError(
            "The Stage 01 validation report "
            "does not match this dataset"
        )

    if final.exists() and not overwrite:
        raise FileExistsError(
            f"Modeling artifacts already exist: {final}; "
            "pass --overwrite"
        )

    # They generate intermediate files in a temporary
    # directory and publish only completed, checked outputs.
    with tempfile.TemporaryDirectory(
        prefix=".modeling-build-",
        dir=dataset_dir,
    ) as temporary:

        workdir = Path(temporary)

        output = workdir / "final"
        output.mkdir()

        with duckdb.connect(":memory:") as con:

            con.execute(
                "SET TimeZone='UTC'"
            )

            # They verify the input before any transformation
            # and establish global chronological boundaries.
            n_original = validate_source(
                con,
                source,
                dataset,
            )

            boundaries = resolve_windows(
                con,
                source,
                config,
            )

            paths = split_source(
                con,
                source,
                workdir,
                boundaries,
                feedback_projection(
                    dataset,
                    config,
                ),
            )

            raw_splits = verify_splits(
                con,
                paths,
                n_original,
            )

            # They estimate minimum user/item support using
            # training interactions only.
            eligible_train, filtering = (
                filter_train_by_support(
                    con,
                    paths["train"],
                    workdir,
                    config,
                )
            )

            write_train(
                con,
                eligible_train,
                output / "train.parquet",
            )

            write_model_history(
                con,
                output / "train.parquet",
                output,
            )

            # They keep the fitted model's history fixed.
            # The observed window-start history advances
            # chronologically without implying model retraining.
            write_evaluation_split(
                con,
                paths["validation"],
                [paths["train"]],
                output,
                output / "validation.parquet",
            )

            write_evaluation_split(
                con,
                paths["test"],
                [
                    paths["train"],
                    paths["validation"],
                ],
                output,
                output / "test.parquet",
            )

            files = {
                split: output / f"{split}.parquet"
                for split in (
                    "train",
                    "validation",
                    "test",
                )
            }

            results = {
                name: label_counts(con, file)
                for name, file in files.items()
            }

            # They verify that optional filtering modifies
            # only the training partition.
            for name in (
                "validation",
                "test",
            ):
                if (
                    results[name]["events"]
                    != raw_splits[name]["events"]
                ):
                    raise AssertionError(
                        f"{name} events were unexpectedly filtered"
                    )

            if (
                results["train"]["events"]
                != filtering["after"]
            ):
                raise AssertionError(
                    "Training support-filter counts "
                    "are inconsistent"
                )

            cold = {
                name: {
                    "model_fit_history": segment_counts(
                        con,
                        files[name],
                        "model_cold_start_segment",
                    ),

                    "window_start_history": segment_counts(
                        con,
                        files[name],
                        "window_cold_start_segment",
                    ),
                }
                for name in (
                    "validation",
                    "test",
                )
            }

        # They record the complete preprocessing policy,
        # source identity, temporal boundaries and counts.
        manifest = {
            "dataset": dataset,

            "stage": (
                "03_data_preprocessing_"
                "and_temporal_splitting"
            ),

            "source": {
                "path": str(
                    source.relative_to(root)
                ),
                "sha256": sha256_file(source),
                "validation_report": str(
                    validation_file.relative_to(root)
                ),
                "events": n_original,
            },

            "configuration": config.as_dict(),

            "feedback_policy": (
                f"explicit_rating_ge_"
                f"{config.rating_positive_threshold:g}"
                if dataset != "retailrocket"
                else config.implicit_target
            ),

            "temporal_windows": (
                boundaries.as_dict()
            ),

            "raw_split_statistics": raw_splits,

            "train_support_filter": filtering,

            "label_statistics": results,

            "cold_start_attribution": cold,

            "outputs": [
                "train.parquet",
                "validation.parquet",
                "test.parquet",
                "train_user_counts.parquet",
                "train_item_counts.parquet",
                "manifest.json",
            ],

            "methodological_notes": [
                "Only source events are labeled; missing "
                "user-item pairs are never confirmed negatives.",

                "The source dataset retains repeated "
                "events and remains unchanged.",

                "Optional minimum-support filtering "
                "uses training events only.",

                "The validation and test windows retain "
                "cold-start events.",

                "model_fit_history uses the filtered "
                "training set for both evaluation windows.",

                "window_start_history uses raw train for "
                "validation and raw train plus validation "
                "for test.",

                "The window-start tags are fixed at each "
                "boundary, not updated after every event.",

                "No model fitting, candidate sampling, "
                "or metadata joins occur in this stage.",
            ],
        }

        write_json(
            manifest,
            output / "manifest.json",
        )

        # They replace generated outputs only after all
        # consistency checks and report generation succeed.
        if final.exists():
            shutil.rmtree(final)

        shutil.move(
            str(output),
            str(final),
        )

        return manifest


def main() -> None:
    """
    They expose the modeling workflow through a consistent
    dataset-specific command-line interface.
    """

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASETS),
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/modeling.toml"),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    report = prepare_modeling(
        args.dataset,
        args.project_root,
        args.config,
        overwrite=args.overwrite,
    )

    print(
        json.dumps(
            {
                "dataset": report["dataset"],

                "temporal_windows": (
                    report["temporal_windows"]
                ),

                "raw_split_statistics": (
                    report["raw_split_statistics"]
                ),

                "train_support_filter": (
                    report["train_support_filter"]
                ),

                "label_statistics": (
                    report["label_statistics"]
                ),

                "cold_start_attribution": (
                    report["cold_start_attribution"]
                ),

                "status": "completed",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
