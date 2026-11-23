"""MovieLens 1M adapter: explicit ratings and separate user/movie metadata."""

from pathlib import Path

import pandas as pd

from recsys.data import (
    DatasetBundle,
    require_file,
    standardize_events,
)


def load_movielens(
    root: Path,
    *,
    limit: int | None = None,
) -> DatasetBundle:
    """
    Loaders read original ::-separated files without modifying
    the sources.
    """

    directory = (
        root / "ml-1m"
        if (root / "ml-1m").is_dir()
        else root
    )

    paths = {
        name: require_file(directory / name)
        for name in (
            "ratings.dat",
            "movies.dat",
            "users.dat",
        )
    }

    # MovieLens sources contain no header; adapters assign the
    # documented column names and preserve text using a compatible
    # single-byte encoding.
    ratings = pd.read_csv(
        paths["ratings.dat"],
        sep="::",
        engine="python",
        header=None,
        names=[
            "user_id",
            "movie_id",
            "rating",
            "timestamp",
        ],
        dtype={
            "user_id": "string",
            "movie_id": "string",
        },
        nrows=limit,
    )

    movies = pd.read_csv(
        paths["movies.dat"],
        sep="::",
        engine="python",
        header=None,
        names=[
            "movie_id",
            "title",
            "genres",
        ],
        dtype={
            "movie_id": "string",
        },
        encoding="latin-1",
    )

    users = pd.read_csv(
        paths["users.dat"],
        sep="::",
        engine="python",
        header=None,
        names=[
            "user_id",
            "gender",
            "age_group",
            "occupation",
            "zip_code",
        ],
        dtype={
            "user_id": "string",
            "zip_code": "string",
        },
    )

    # The standardized event log retains the original rating,
    # while movie and user attributes remain in separate tables.
    events = standardize_events(
        ratings,
        dataset="movielens-1m",
        user_col="user_id",
        item_col="movie_id",
        timestamp_col="timestamp",
        timestamp_unit="s",
        event_type="rating",
        value_col="rating",
    )

    return DatasetBundle(
        name="movielens-1m",
        events=events,
        tables={
            "movies": movies,
            "users": users,
        },
        sources=paths,
        notes=[
            "Ratings remain explicit feedback; "
            "relevance thresholds are not applied."
        ],
    )
