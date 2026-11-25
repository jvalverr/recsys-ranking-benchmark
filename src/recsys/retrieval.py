"""Fit and serve sparse, train-only popularity and item-item retrieval baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
import json
import tomllib

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, load_npz


DATASETS = {
    "movielens-1m",
    "retailrocket",
    "amazon-video-games",
}


@dataclass(frozen=True)
class RetrievalConfig:
    """They record retrieval choices in a reproducible contract."""

    max_profile_items: int = 30
    topk_neighbors: int = 100
    min_common_users: int = 2
    shrinkage: float = 10.0
    max_pair_contributions: int = 30_000_000

    candidate_knn: int = 100
    candidate_popularity: int = 100
    final_k: int = 10
    rrf_constant: int = 60

    max_eval_users: int = 1000
    seed: int = 42
    seen_policy: str = "all"

    def __post_init__(self) -> None:

        integer_fields = (
            "max_profile_items",
            "topk_neighbors",
            "min_common_users",
            "max_pair_contributions",
            "candidate_knn",
            "candidate_popularity",
            "final_k",
            "rrf_constant",
            "max_eval_users",
        )

        if any(
            not isinstance(getattr(self, key), int)
            or getattr(self, key) < 1
            for key in integer_fields
        ):
            raise ValueError(
                "Retrieval count parameters must be positive integers"
            )

        if not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")

        if (
            not np.isfinite(self.shrinkage)
            or self.shrinkage < 0
        ):
            raise ValueError(
                "shrinkage must be finite and nonnegative"
            )

        if self.seen_policy not in {"none", "all"}:
            raise ValueError(
                "seen_policy must be 'all' or 'none'"
            )


def load_retrieval_config(
    path: Path,
    dataset: str,
) -> RetrievalConfig:
    """
    They merge shared defaults with dataset-specific policies.
    Unknown configuration fields are rejected explicitly.
    """

    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")

    with Path(path).open("rb") as handle:
        document = tomllib.load(handle)

    options = {
        **document.get("defaults", {}),
        **document.get(dataset, {}),
    }

    supported = {
        field.name
        for field in fields(RetrievalConfig)
    }

    unknown = set(options) - supported

    if unknown:
        raise ValueError(
            f"Unknown retrieval configuration keys: {sorted(unknown)}"
        )

    return RetrievalConfig(**options)


@dataclass
class RetrievalIndex:
    """
    They retain a closed training catalog, popularity support,
    and a sparse directed item-neighbor graph.
    """

    item_ids: list[str]
    popularity: np.ndarray
    neighbors: csr_matrix
    config: RetrievalConfig
    fit_stats: dict

    def __post_init__(self) -> None:

        self._positions = {
            item: i
            for i, item in enumerate(self.item_ids)
        }

        n = len(self.item_ids)

        if n == 0 or len(self._positions) != n:
            raise ValueError(
                "The training catalog requires distinct item IDs"
            )

        if (
            self.popularity.shape != (n,)
            or self.neighbors.shape != (n, n)
        ):
            raise ValueError(
                "Catalog, popularity and similarity dimensions disagree"
            )

        if (self.popularity <= 0).any():
            raise ValueError(
                "Catalog items must have positive training support"
            )

        self.neighbors = self.neighbors.tocsr()

    @classmethod
    def fit(
        cls,
        positive_pairs: pd.DataFrame,
        config: RetrievalConfig,
    ) -> "RetrievalIndex":
        """
        They fit retrieval models exclusively on observed-positive
        training interactions.
        """

        required = {
            "user_id",
            "item_id",
            "last_event",
        }

        if not required.issubset(positive_pairs.columns):
            raise ValueError(
                f"Training pairs require {sorted(required)}"
            )

        if (
            positive_pairs.empty
            or positive_pairs[
                ["user_id", "item_id"]
            ].isna().any().any()
        ):
            raise ValueError(
                "Training positives require valid user and item IDs"
            )

        # They collapse repeated positive events into binary
        # user-item support while retaining the latest timestamp.
        pairs = (
            positive_pairs
            .groupby(
                ["user_id", "item_id"],
                as_index=False,
                sort=False,
            )
            .agg(last_event=("last_event", "max"))
        )

        support = (
            pairs.groupby("item_id")["user_id"]
            .nunique()
            .rename("users")
        )

        # They establish a deterministic catalog ordered by
        # descending positive-user support and then by item ID.
        catalog = (
            support.reset_index()
            .sort_values(
                ["users", "item_id"],
                ascending=[False, True],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )

        item_ids = (
            catalog["item_id"]
            .astype(str)
            .tolist()
        )

        popularity = catalog["users"].to_numpy(
            dtype=np.int64
        )

        item_to_idx = {
            item: i
            for i, item in enumerate(item_ids)
        }

        # They cap each user's most recent positive items
        # before constructing item co-occurrences.
        capped = (
            pairs.sort_values(
                ["user_id", "last_event", "item_id"],
                ascending=[True, False, True],
                kind="mergesort",
            )
            .groupby("user_id", sort=False)
            .head(config.max_profile_items)
        )

        user_idx, user_labels = pd.factorize(
            capped["user_id"],
            sort=False,
        )

        item_idx = (
            capped["item_id"]
            .map(item_to_idx)
            .to_numpy(dtype=np.int64)
        )

        # They estimate pairwise work before allocating
        # the sparse co-occurrence matrix.
        degrees = np.bincount(
            user_idx
        ).astype(np.int64)

        pair_budget = int(
            np.dot(
                degrees,
                degrees - 1,
            )
        )

        if (
            pair_budget
            > config.max_pair_contributions
        ):
            raise ValueError(
                f"Estimated co-occurrence pair contributions "
                f"{pair_budget:,} exceed the configured budget "
                f"{config.max_pair_contributions:,}. "
                "Reduce max_profile_items or partition the experiment."
            )

        # They construct a binary user-item CSR matrix
        # without allocating a dense user-item matrix.
        X = csr_matrix(
            (
                np.ones(
                    len(item_idx),
                    dtype=np.float32,
                ),
                (user_idx, item_idx),
            ),
            shape=(
                len(user_labels),
                len(item_ids),
            ),
            dtype=np.float32,
        )

        X.sum_duplicates()
        X.data[:] = 1.0

        # They compute sparse item co-occurrence counts
        # and remove self-similarity.
        cooccurrence = (
            X.T @ X
        ).tocsr()

        cooccurrence.setdiag(0)
        cooccurrence.eliminate_zeros()

        norms = np.sqrt(
            np.asarray(
                X.power(2).sum(axis=0)
            ).ravel()
        )

        rows = []
        cols = []
        values = []

        # They retain only supported, shrinkage-adjusted
        # neighbors, avoiding a dense item-item similarity matrix.
        for i in range(len(item_ids)):

            a, b = cooccurrence.indptr[
                i:i + 2
            ]

            js = cooccurrence.indices[a:b]
            count = cooccurrence.data[a:b]

            allowed = (
                count >= config.min_common_users
            )

            js = js[allowed]
            count = count[allowed]

            if len(js) == 0:
                continue

            similarity = (
                count
                / (norms[i] * norms[js])
            )

            if config.shrinkage:
                similarity *= (
                    count
                    / (count + config.shrinkage)
                )

            # They sort ties by catalog index to ensure
            # deterministic neighbor selection.
            order = np.lexsort(
                (js, -similarity)
            )[:config.topk_neighbors]

            js = js[order]
            similarity = similarity[order]

            rows.append(
                np.full(
                    len(js),
                    i,
                    dtype=np.int64,
                )
            )

            cols.append(
                js.astype(np.int64)
            )

            values.append(
                similarity.astype(np.float32)
            )

        if rows:

            graph = csr_matrix(
                (
                    np.concatenate(values),
                    (
                        np.concatenate(rows),
                        np.concatenate(cols),
                    ),
                ),
                shape=(
                    len(item_ids),
                    len(item_ids),
                ),
                dtype=np.float32,
            )

        else:

            graph = csr_matrix(
                (
                    len(item_ids),
                    len(item_ids),
                ),
                dtype=np.float32,
            )

        graph.sort_indices()

        # They record the effects of history truncation
        # and graph sparsification for later analysis.
        stats = {
            "positive_user_item_pairs": int(
                len(pairs)
            ),
            "users_with_positive_train_events": int(
                pairs["user_id"].nunique()
            ),
            "items_with_positive_train_events": int(
                len(item_ids)
            ),
            "pairs_used_for_knn": int(
                len(capped)
            ),
            "capped_user_histories": int(
                (
                    pairs.groupby("user_id").size()
                    > config.max_profile_items
                ).sum()
            ),
            "pair_contributions_upper_bound": (
                pair_budget
            ),
            "retained_neighbor_edges": int(
                graph.nnz
            ),
        }

        return cls(
            item_ids,
            popularity,
            graph,
            config,
            stats,
        )

    def popularity_top(
        self,
        k: int,
        exclude: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """
        They rank catalog items by distinct positive-user support.
        """

        excluded = exclude or set()
        result = []

        for i, item in enumerate(self.item_ids):

            if item not in excluded:

                result.append((
                    item,
                    float(self.popularity[i]),
                ))

                if len(result) == k:
                    break

        return result

    def knn_top(
        self,
        positive_history: set[str],
        k: int,
        exclude: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """
        They aggregate item-neighbor similarities using
        historically observed positive seed items.
        """

        seed_indices = sorted({
            self._positions[item]
            for item in positive_history
            if item in self._positions
        })

        if not seed_indices:
            return []

        scores = np.asarray(
            self.neighbors[
                seed_indices, :
            ].sum(axis=0)
        ).ravel()

        # They exclude historical items only when the
        # evaluation policy requires novel-item retrieval.
        for item in exclude or ():

            idx = self._positions.get(item)

            if idx is not None:
                scores[idx] = 0.0

        valid = np.flatnonzero(
            scores > 0
        )

        order = np.lexsort(
            (valid, -scores[valid])
        )[:k]

        return [
            (
                self.item_ids[i],
                float(scores[i]),
            )
            for i in valid[order]
        ]

    def save(
        self,
        directory: Path,
    ) -> None:
        """
        They persist the catalog and sparse graph for reuse
        without fitting models on evaluation data.
        """

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        catalog = {
            "items": self.item_ids,
            "popularity": self.popularity.tolist(),
        }

        (
            directory / "catalog.json"
        ).write_text(
            json.dumps(
                catalog,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        save_npz(
            directory / "neighbors.npz",
            self.neighbors,
        )

        (
            directory / "index.json"
        ).write_text(
            json.dumps(
                {
                    "config": asdict(self.config),
                    "fit_statistics": self.fit_stats,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        directory: Path,
    ) -> "RetrievalIndex":
        """
        They restore the fitted index without recomputing
        training statistics or similarities.
        """

        catalog = json.loads(
            (
                directory / "catalog.json"
            ).read_text(encoding="utf-8")
        )

        metadata = json.loads(
            (
                directory / "index.json"
            ).read_text(encoding="utf-8")
        )

        return cls(
            catalog["items"],
            np.asarray(
                catalog["popularity"],
                dtype=np.int64,
            ),
            load_npz(
                directory / "neighbors.npz"
            ),
            RetrievalConfig(
                **metadata["config"]
            ),
            metadata["fit_statistics"],
        )
