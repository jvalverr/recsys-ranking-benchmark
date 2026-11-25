"""Fit frozen retrieval indexes and evaluate candidate generation on held-out windows."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from time import perf_counter

import duckdb

from recsys.retrieval import (
    DATASETS,
    RetrievalIndex,
    load_retrieval_config,
)

from recsys.retrieval_evaluation import (
    evaluate_retrieval,
)


def sha256_file(path: Path) -> str:
    """
    They fingerprint modeling inputs so evaluations cannot
    silently reuse stale retrieval models.
    """

    digest = hashlib.sha256()

    with path.open("rb") as stream:

        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def write_json(
    data: dict,
    path: Path,
) -> None:
    """
    They persist human-readable experiment manifests
    using UTF-8 encoding.
    """

    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )


def _paths(
    root: Path,
    dataset: str,
) -> tuple[Path, Path]:
    """
    They separate processed datasets from regenerable
    fitted retrieval artifacts.
    """

    if dataset not in DATASETS:
        raise ValueError(
            f"Unsupported dataset: {dataset}"
        )

    return (
        root / "data/processed" / dataset,
        root / "artifacts/retrieval" / dataset,
    )


def fit_command(
    dataset: str,
    root: Path,
    config_path: Path,
    overwrite: bool,
) -> dict:
    """
    They train both retrieval baselines exclusively
    on positive training interactions.
    """

    root = root.resolve()

    dataset_dir, model_dir = _paths(
        root,
        dataset,
    )

    train = (
        dataset_dir / "modeling/train.parquet"
    )

    manifest_path = (
        dataset_dir / "modeling/manifest.json"
    )

    for required in (
        train,
        manifest_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(
                f"Run Stage 03 first: {required}"
            )

    modeling = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    if modeling.get("dataset") != dataset:
        raise ValueError(
            "The modeling manifest refers to another dataset"
        )

    config = load_retrieval_config(
        (
            config_path
            if config_path.is_absolute()
            else root / config_path
        ),
        dataset,
    )

    if model_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Model exists: {model_dir}. "
            "Use --overwrite to refit."
        )

    began = perf_counter()

    with duckdb.connect(":memory:") as con:

        # They read distinct positive user-item pairs
        # exclusively from the training partition.
        pairs = con.execute(
            """
            SELECT
                user_id,
                item_id,
                MAX(timestamp) AS last_event

            FROM read_parquet(?)

            WHERE relevance_label = 1

            GROUP BY
                user_id,
                item_id
            """,
            [str(train)],
        ).df()

    index = RetrievalIndex.fit(
        pairs,
        config,
    )

    model_dir.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # They stage model artifacts in a temporary directory
    # before publishing a complete fitted index.
    with tempfile.TemporaryDirectory(
        prefix=".retrieval-",
        dir=model_dir.parent,
    ) as temporary:

        temp = (
            Path(temporary) / "index"
        )

        index.save(temp)

        manifest = {
            "dataset": dataset,

            "stage": (
                "04_retrieval_and_candidate_generation"
            ),

            "training_input": (
                "modeling/train.parquet: observed-positive "
                "user-item pairs only"
            ),

            "source_train_sha256": (
                sha256_file(train)
            ),

            "source_modeling_manifest_sha256": (
                sha256_file(manifest_path)
            ),

            "config": asdict(config),

            "fit_statistics": (
                index.fit_stats
            ),

            "fit_elapsed_seconds": round(
                perf_counter() - began,
                4,
            ),

            "artifacts": [
                "catalog.json",
                "neighbors.npz",
                "index.json",
            ],

            "limitations": [
                "The item catalog includes only items with positive train interactions.",

                "A per-user history cap reduces co-occurrence cost and changes its statistics.",

                "Offline retrieval does not establish online relevance or conversion gains.",
            ],
        }

        write_json(
            manifest,
            temp / "fit_manifest.json",
        )

        if model_dir.exists():
            shutil.rmtree(model_dir)

        shutil.move(
            str(temp),
            str(model_dir),
        )

    return manifest


def evaluate_command(
    dataset: str,
    split: str,
    root: Path,
    max_users: int | None,
    overwrite: bool,
    final_test: bool,
) -> dict:
    """
    They evaluate immutable retrieval artifacts against
    later observed-positive interactions.
    """

    # They require explicit authorization before evaluating
    # the final test window.
    if split == "test" and not final_test:
        raise ValueError(
            "Final test evaluation requires "
            "the explicit --final-test flag"
        )

    root = root.resolve()

    dataset_dir, model_dir = _paths(
        root,
        dataset,
    )

    manifest_file = (
        dataset_dir / "modeling/manifest.json"
    )

    train = (
        dataset_dir / "modeling/train.parquet"
    )

    model_fit = (
        model_dir / "fit_manifest.json"
    )

    for required in (
        manifest_file,
        train,
        model_fit,
    ):
        if not required.is_file():
            raise FileNotFoundError(
                required
            )

    # They require prior validation and verify that it
    # refers to the same fitted retrieval index.
    if split == "test":

        validation_report = (
            dataset_dir
            / "retrieval/validation/summary.json"
        )

        if not validation_report.is_file():
            raise ValueError(
                "Run and review validation retrieval "
                "before final test evaluation"
            )

        previous = json.loads(
            validation_report.read_text(
                encoding="utf-8"
            )
        )

        if previous.get("index_sha256") != sha256_file(
            model_dir / "neighbors.npz"
        ):
            raise ValueError(
                "Validation used a different retrieval index: "
                "re-evaluate it first"
            )

    fit_manifest = json.loads(
        model_fit.read_text(
            encoding="utf-8"
        )
    )

    if fit_manifest.get("dataset") != dataset:
        raise ValueError(
            "Loaded retrieval index belongs to another dataset"
        )

    # They prevent evaluation with a retriever fitted on
    # outdated or modified training artifacts.
    if (
        fit_manifest["source_train_sha256"]
        != sha256_file(train)
    ):
        raise ValueError(
            "Training Parquet changed: "
            "refit retrieval before evaluating"
        )

    if (
        fit_manifest[
            "source_modeling_manifest_sha256"
        ]
        != sha256_file(manifest_file)
    ):
        raise ValueError(
            "Modeling manifest changed: "
            "refit retrieval before evaluating"
        )

    out = (
        dataset_dir
        / "retrieval"
        / split
    )

    if out.exists() and not overwrite:
        raise FileExistsError(
            f"Evaluation results exist: {out}. "
            "Use --overwrite."
        )

    index = RetrievalIndex.load(
        model_dir
    )

    modeling = json.loads(
        manifest_file.read_text(
            encoding="utf-8"
        )
    )

    started = perf_counter()

    summary, user_metrics, candidates = (
        evaluate_retrieval(
            index,
            dataset=dataset,
            split=split,
            processed_dataset_dir=dataset_dir,
            modeling_manifest=modeling,
            max_eval_users=max_users,
        )
    )

    summary["index_sha256"] = (
        sha256_file(
            model_dir / "neighbors.npz"
        )
    )

    summary[
        "total_evaluation_elapsed_seconds"
    ] = round(
        perf_counter() - started,
        4,
    )

    summary["artifacts"] = [
        "summary.json",
        "user_metrics.parquet",
        "candidates.parquet",
    ]

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(
        prefix=".candidates-",
        dir=out.parent,
    ) as temporary:

        temp = (
            Path(temporary) / "results"
        )

        temp.mkdir()

        user_metrics.to_parquet(
            temp / "user_metrics.parquet",
            index=False,
        )

        # They do not include held-out relevance labels
        # in the candidate artifact.
        candidates.to_parquet(
            temp / "candidates.parquet",
            index=False,
        )

        write_json(
            summary,
            temp / "summary.json",
        )

        if out.exists():
            shutil.rmtree(out)

        shutil.move(
            str(temp),
            str(out),
        )

    return summary


def main() -> None:
    """
    They expose fitting and evaluation as separate
    reproducible command-line operations.
    """

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    commands = parser.add_subparsers(
        dest="action",
        required=True,
    )

    for action in (
        "fit",
        "evaluate",
    ):

        sub = commands.add_parser(
            action
        )

        sub.add_argument(
            "--dataset",
            required=True,
            choices=sorted(DATASETS),
        )

        sub.add_argument(
            "--project-root",
            type=Path,
            default=Path("."),
        )

        sub.add_argument(
            "--overwrite",
            action="store_true",
        )

        if action == "fit":

            sub.add_argument(
                "--config",
                type=Path,
                default=Path(
                    "configs/retrieval.toml"
                ),
            )

        else:

            sub.add_argument(
                "--split",
                choices=(
                    "validation",
                    "test",
                ),
                default="validation",
            )

            sub.add_argument(
                "--max-users",
                type=int,
                default=None,
            )

            sub.add_argument(
                "--final-test",
                action="store_true",
            )

    args = parser.parse_args()

    if args.action == "fit":

        report = fit_command(
            args.dataset,
            args.project_root,
            args.config,
            args.overwrite,
        )

    else:

        report = evaluate_command(
            args.dataset,
            args.split,
            args.project_root,
            args.max_users,
            args.overwrite,
            args.final_test,
        )

    print(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
