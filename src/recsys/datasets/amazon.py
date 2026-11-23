"""Amazon Video Games adapter: ratings and product metadata remain distinct."""

import gzip
import json

from pathlib import Path

import pandas as pd

from recsys.data import (
    DatasetBundle,
    require_file,
    standardize_events,
)


def _metadata_preview(
    path: Path,
    maximum: int,
) -> pd.DataFrame:
    """
    Readers inspect bounded samples of compressed JSONL without
    extracting full catalogs.
    """

    records: list[dict] = []

    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
    ) as handle:

        for line in handle:
            if not line.strip():
                continue

            record = json.loads(line)

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected metadata JSON objects in {path}"
                )

            if "parent_asin" not in record:
                raise ValueError(
                    f"Metadata record lacks parent_asin in {path}"
                )

            records.append(record)

            if len(records) >= maximum:
                break

    return pd.DataFrame(records)


def load_amazon(
    root: Path,
    *,
    limit: int | None = None,
    metadata_preview: int = 25,
) -> DatasetBundle:
    """
    Adapters select version-compatible product keys and
    preserve raw review fields.
    """

    reviews_path = require_file(
        root / "Video_Games.csv"
    )

    header = pd.read_csv(
        reviews_path,
        nrows=0,
    ).columns.tolist()

    # Schema detection prevents product identifiers and timestamp
    # units from being silently mixed across dataset generations.
    if {
        "user_id",
        "parent_asin",
        "rating",
        "timestamp",
    } <= set(header):

        user_col = "user_id"
        item_col = "parent_asin"
        time_col = "timestamp"
        unit = "ms"
        version = "2023"

    elif {
        "reviewerID",
        "asin",
        "overall",
        "unixReviewTime",
    } <= set(header):

        user_col = "reviewerID"
        item_col = "asin"
        time_col = "unixReviewTime"
        unit = "s"
        version = "legacy"

    else:
        raise ValueError(
            f"Unsupported Amazon review schema: {header}. "
            "Expected 2023 rating-only or legacy reviews columns."
        )

    reviews = pd.read_csv(
        reviews_path,
        nrows=limit,
        dtype={
            user_col: "string",
            item_col: "string",
        },
    )

    rating_col = (
        "rating"
        if version == "2023"
        else "overall"
    )

    events = standardize_events(
        reviews,
        dataset="amazon-video-games",
        user_col=user_col,
        item_col=item_col,
        timestamp_col=time_col,
        timestamp_unit=unit,
        event_type="rating",
        value_col=rating_col,
    )

    sources = {
        reviews_path.name: reviews_path,
    }

    tables: dict[str, pd.DataFrame] = {}

    notes = [
        f"Detected {version} review schema; "
        "the loader does not infer 5-core membership.",
        "2023 product metadata must be joined on "
        "parent_asin, not child asin.",
    ]

    metadata_path = (
        root / "meta_Video_Games.jsonl.gz"
    )

    if metadata_path.is_file():
        sources[metadata_path.name] = metadata_path

        # The legacy identifier convention is not assumed to
        # match the product hierarchy used by the 2023 catalog.
        if version != "2023":
            notes.append(
                "Legacy asin identifiers must not be assumed "
                "equivalent to 2023 parent_asin."
            )

        if metadata_preview > 0:
            tables["metadata_preview"] = (
                _metadata_preview(
                    metadata_path,
                    metadata_preview,
                )
            )

            notes.append(
                "Metadata preview is a sample, "
                "not a full product catalog."
            )

    return DatasetBundle(
        name="amazon-video-games",
        events=events,
        tables=tables,
        sources=sources,
        notes=notes,
    )
