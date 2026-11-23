"""Dataset quality checks and descriptive statistics without silent cleaning."""

from dataclasses import dataclass, field

import pandas as pd

from recsys.data import DatasetBundle, EVENT_COLUMNS


@dataclass
class ValidationReport:
    dataset: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    statistics: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "valid": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "statistics": self.statistics,
        }


def validate_bundle(
    bundle: DatasetBundle,
) -> ValidationReport:
    """
    Validators report anomalies while preserving all source records.
    """

    report = ValidationReport(
        dataset=bundle.name,
        warnings=list(bundle.notes),
    )

    df = bundle.events

    # Validators check the shared schema before executing statistical
    # operations that depend on standardized columns.
    missing = set(EVENT_COLUMNS) - set(df.columns)

    if missing:
        report.errors.append(
            f"Missing canonical columns: {sorted(missing)}"
        )
        return report

    if df.empty:
        report.errors.append(
            "No interaction records were loaded"
        )
        return report

    # Identifier checks prevent missing or empty IDs from becoming
    # artificial users or items in later recommendation models.
    for field_name in (
        "user_id",
        "item_id",
        "event_type",
    ):
        invalid = (
            df[field_name].isna()
            | df[field_name]
            .astype("string")
            .str.strip()
            .eq("")
        )

        invalid_count = int(
            invalid.fillna(True).sum()
        )

        if invalid_count:
            report.errors.append(
                f"{field_name}: {invalid_count} "
                "missing or empty values"
            )

    # Timestamp validation prevents malformed dates from entering
    # temporal train/validation/test partitions.
    missing_times = int(
        df["timestamp"].isna().sum()
    )

    if missing_times:
        report.errors.append(
            f"timestamp: {missing_times} "
            "missing or invalid values"
        )

    if not pd.api.types.is_datetime64_any_dtype(
        df["timestamp"]
    ):
        report.errors.append(
            "timestamp must be a datetime column"
        )

    # Different datasets retain different feedback semantics.
    # Behavioral events are not interpreted as numerical ratings.
    if bundle.name == "retailrocket":

        observed = set(
            df["event_type"]
            .dropna()
            .astype(str)
        )

        unexpected = sorted(
            observed - {
                "view",
                "addtocart",
                "transaction",
            }
        )

        if unexpected:
            report.errors.append(
                f"Unexpected RetailRocket event types: {unexpected}"
            )

    else:
        if df["event_value"].isna().any():
            report.errors.append(
                "Ratings contain missing or nonnumeric values"
            )

        invalid_rating = (
            df["event_value"].notna()
            & ~df["event_value"].between(1, 5)
        )

        if invalid_rating.any():
            report.errors.append(
                "Ratings outside [1, 5]: "
                f"{int(invalid_rating.sum())}"
            )

    # Validators distinguish potentially duplicated observations
    # from repeated user-item behavior, since repeated interactions
    # can represent meaningful engagement.
    event_keys = [
        "user_id",
        "item_id",
        "timestamp",
        "event_type",
        "event_value",
    ]

    if "raw_transactionid" in df.columns:
        event_keys.append("raw_transactionid")

    if "raw_asin" in df.columns:
        event_keys.append("raw_asin")

    exact_duplicate_count = int(
        df.duplicated(
            subset=event_keys
        ).sum()
    )

    unique_pairs = int(
        df[
            ["user_id", "item_id"]
        ].drop_duplicates().shape[0]
    )

    n_users = int(
        df["user_id"].nunique(dropna=True)
    )

    n_items = int(
        df["item_id"].nunique(dropna=True)
    )

    n_events = len(df)

    ts = df["timestamp"].dropna()

    # Distribution summaries reveal dataset sparsity and differences
    # in interaction frequency before any modeling decisions.
    interactions_per_user = (
        df.groupby(
            "user_id",
            dropna=True,
        ).size()
    )

    interactions_per_item = (
        df.groupby(
            "item_id",
            dropna=True,
        ).size()
    )

    report.statistics = {
        "events": n_events,
        "users": n_users,
        "events_per_user_median": float(
            interactions_per_user.median()
        ),
        "events_per_user_p95": float(
            interactions_per_user.quantile(0.95)
        ),
        "events_per_item_median": float(
            interactions_per_item.median()
        ),
        "items": n_items,
        "unique_user_item_pairs": unique_pairs,
        "repeat_user_item_events": (
            n_events - unique_pairs
        ),
        "duplicate_event_observations": (
            exact_duplicate_count
        ),
        "missing_timestamps": missing_times,
        "first_event_utc": (
            ts.min().isoformat()
            if not ts.empty
            else None
        ),
        "last_event_utc": (
            ts.max().isoformat()
            if not ts.empty
            else None
        ),
        "event_type_counts": {
            str(k): int(v)
            for k, v in (
                df["event_type"]
                .value_counts()
                .items()
            )
        },
        "interaction_sparsity": (
            round(
                1.0 - unique_pairs / (n_users * n_items),
                8,
            )
            if n_users and n_items
            else None
        ),
        "auxiliary_tables": {
            k: len(v)
            for k, v in bundle.tables.items()
        },
        "source_files": {
            k: str(v)
            for k, v in bundle.sources.items()
        },
    }

    if exact_duplicate_count:
        report.warnings.append(
            f"{exact_duplicate_count} repeated event "
            "observations detected; none removed."
        )

    # Metadata checks identify unmatched user or item identifiers
    # without silently discarding event records.
    if bundle.name == "movielens-1m":

        for table, event_key, metadata_key in (
            ("users", "user_id", "user_id"),
            ("movies", "item_id", "movie_id"),
        ):
            known = set(
                bundle.tables[table][metadata_key]
                .dropna()
                .astype(str)
            )

            unmatched = int(
                (~df[event_key].isin(known)).sum()
            )

            if unmatched:
                report.warnings.append(
                    f"{unmatched} interactions have no "
                    f"match in {table} metadata."
                )

    # Auxiliary catalogs have different identifiers and null
    # conventions; validators inspect their primary keys
    # without treating all optional fields as errors.
    auxiliary_keys = {
        "movies": "movie_id",
        "users": "user_id",
        "categories": "categoryid",
        "metadata_preview": "parent_asin",
        "item_properties_part1_preview": "itemid",
        "item_properties_part2_preview": "itemid",
    }

    for table_name, metadata in bundle.tables.items():

        key = auxiliary_keys.get(table_name)

        if key is None:
            continue

        if key not in metadata.columns:
            report.errors.append(
                f"{table_name}: missing identifier {key}"
            )
            continue

        invalid = (
            metadata[key].isna()
            | metadata[key]
            .astype("string")
            .str.strip()
            .eq("")
        )

        invalid_count = int(
            invalid.fillna(True).sum()
        )

        if invalid_count:
            report.errors.append(
                f"{table_name}.{key}: "
                f"{invalid_count} missing IDs"
            )

        if table_name in {
            "movies",
            "users",
            "categories",
        }:
            duplicates = int(
                metadata[key].duplicated().sum()
            )

            if duplicates:
                report.errors.append(
                    f"{table_name}.{key}: "
                    f"{duplicates} duplicate IDs"
                )

        elif (
            table_name == "metadata_preview"
            and metadata.empty
        ):
            report.warnings.append(
                "Amazon metadata preview is empty."
            )

        # Property timestamps receive additional checks because
        # future model features will require as-of-time joins.
        if table_name.startswith("item_properties_"):

            numeric = pd.to_numeric(
                metadata["timestamp"],
                errors="coerce",
            )

            invalid_times = pd.to_datetime(
                numeric,
                unit="ms",
                utc=True,
                errors="coerce",
            ).isna().sum()

            if invalid_times:
                report.errors.append(
                    f"{table_name}: "
                    f"{int(invalid_times)} "
                    "invalid property timestamps"
                )

    return report
