"""RetailRocket adapter: event streams and time-dependent catalog metadata."""

from pathlib import Path

import pandas as pd

from recsys.data import (
    DatasetBundle,
    require_columns,
    require_file,
    standardize_events,
)


def load_retailrocket(
    root: Path,
    *,
    limit: int | None = None,
    metadata_preview: int = 25,
) -> DatasetBundle:
    """
    Adapters load events and inspect property snapshots without
    joining future data.
    """

    events_path = require_file(
        root / "events.csv"
    )

    events_raw = pd.read_csv(
        events_path,
        nrows=limit,
        dtype={
            "visitorid": "string",
            "itemid": "string",
            "transactionid": "string",
        },
    )

    # Behavioral event types remain distinct because views,
    # cart additions, and purchases imply different user intents.
    events = standardize_events(
        events_raw,
        dataset="retailrocket",
        user_col="visitorid",
        item_col="itemid",
        timestamp_col="timestamp",
        timestamp_unit="ms",
        event_col="event",
    )

    sources = {
        "events.csv": events_path,
    }

    tables: dict[str, pd.DataFrame] = {}

    # Category hierarchy data remain separate from interactions
    # until the modeling stage defines their analytical purpose.
    category_path = root / "category_tree.csv"

    if category_path.is_file():
        categories = pd.read_csv(
            category_path,
            dtype={
                "categoryid": "string",
                "parentid": "string",
            },
        )

        require_columns(
            categories,
            {
                "categoryid",
                "parentid",
            },
            str(category_path),
        )

        tables["categories"] = categories
        sources[category_path.name] = category_path

    # Property snapshots can be very large. Readers inspect their
    # schemas and retain source paths until time-aware joins
    # are implemented.
    for filename in (
        "item_properties_part1.csv",
        "item_properties_part2.csv",
    ):
        path = root / filename

        if not path.is_file():
            continue

        header = pd.read_csv(
            path,
            nrows=0,
        )

        require_columns(
            header,
            {
                "timestamp",
                "itemid",
                "property",
                "value",
            },
            str(path),
        )

        sources[filename] = path

        if metadata_preview > 0:
            tables[f"{path.stem}_preview"] = (
                pd.read_csv(
                    path,
                    nrows=metadata_preview,
                    dtype={
                        "itemid": "string",
                    },
                )
            )

    return DatasetBundle(
        name="retailrocket",
        events=events,
        tables=tables,
        sources=sources,
        notes=[
            "Views, carts, and transactions remain "
            "separate behavioral events.",
            "Item-property snapshots are not joined: "
            "attributes require as-of-time joins.",
        ],
    )
