"""Validate source datasets and optionally export canonical Parquet events."""

import argparse
import json

from pathlib import Path

from recsys.datasets import (
    load_amazon,
    load_movielens,
    load_retailrocket,
)

from recsys.utils import write_json
from recsys.validation import validate_bundle


def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=(
            "movielens-1m",
            "retailrocket",
            "amazon-video-games",
        ),
    )

    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw"),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed"),
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--metadata-preview-rows",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
    )

    args = parser.parse_args()

    if args.max_rows is not None and args.max_rows <= 0:
        parser.error(
            "--max-rows must be positive"
        )

    if args.metadata_preview_rows < 0:
        parser.error(
            "--metadata-preview-rows cannot be negative"
        )

    roots = {
        "movielens-1m": (
            args.raw_root / "movielens-1m"
        ),
        "retailrocket": (
            args.raw_root / "retailrocket"
        ),
        "amazon-video-games": (
            args.raw_root / "amazon-video-games"
        ),
    }

    root = roots[args.dataset]

    # Each adapter follows the same output contract despite
    # differences in source schemas and metadata structures.
    if args.dataset == "movielens-1m":

        bundle = load_movielens(
            root,
            limit=args.max_rows,
        )

    elif args.dataset == "retailrocket":

        bundle = load_retailrocket(
            root,
            limit=args.max_rows,
            metadata_preview=args.metadata_preview_rows,
        )

    else:

        bundle = load_amazon(
            root,
            limit=args.max_rows,
            metadata_preview=args.metadata_preview_rows,
        )

    # Validation reports expose anomalies before any generated
    # data are considered reliable for downstream experiments.
    report = validate_bundle(bundle)

    print(
        json.dumps(
            report.as_dict(),
            indent=2,
            ensure_ascii=False,
        )
    )

    if not report.ok:
        raise SystemExit(2)

    if args.validate_only:
        return

    output = (
        args.output_root / args.dataset
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Exporters write standardized event logs and manageable
    # reference tables. They never overwrite raw source files
    # or export metadata previews as complete catalogs.
    bundle.events.to_parquet(
        output / "events.parquet",
        index=False,
    )

    for name, table in bundle.tables.items():

        if not name.endswith("_preview"):

            table.to_parquet(
                output / f"{name}.parquet",
                index=False,
            )

    write_json(
        report.as_dict(),
        output / "validation.json",
    )

    print(
        "Exported validated events to "
        f"{output / 'events.parquet'}"
    )


if __name__ == "__main__":
    main()
