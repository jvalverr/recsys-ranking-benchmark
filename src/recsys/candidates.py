"""Fuse retrieval lists and compute positive-hit ranking diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from math import log2

from recsys.retrieval import (
    RetrievalConfig,
    RetrievalIndex,
)


@dataclass
class CandidateSet:
    """They keep retrieval provenance separate from outcome labels."""

    rows: list[dict]
    popularity_items: list[str]
    knn_items: list[str]
    known_seed_items: int


def generate_candidates(
    index: RetrievalIndex,
    positive_history: set[str],
    seen_items: set[str],
    config: RetrievalConfig,
) -> CandidateSet:
    """
    They combine independently retrieved candidates without
    consulting held-out targets.
    """

    excluded = (
        seen_items
        if config.seen_policy == "all"
        else set()
    )

    pop = index.popularity_top(
        config.candidate_popularity,
        excluded,
    )

    knn = index.knn_top(
        positive_history,
        config.candidate_knn,
        excluded,
    )

    merged: dict[str, dict] = {}

    # They combine rankings through reciprocal positions,
    # avoiding comparisons between incompatible raw scores.
    for origin, ranking in (
        ("popularity", pop),
        ("knn", knn),
    ):

        for rank, (item, score) in enumerate(
            ranking,
            start=1,
        ):

            row = merged.setdefault(
                item,
                {
                    "item_id": item,
                    "fusion_score": 0.0,
                    "popularity_rank": None,
                    "knn_rank": None,
                    "popularity_score": None,
                    "knn_score": None,
                },
            )

            row["fusion_score"] += (
                1.0
                / (config.rrf_constant + rank)
            )

            row[f"{origin}_rank"] = rank
            row[f"{origin}_score"] = score

    # They establish a deterministic order and preserve
    # the contribution of each retrieval method.
    rows = sorted(
        merged.values(),
        key=lambda row: (
            -row["fusion_score"],
            row["item_id"],
        ),
    )

    for position, row in enumerate(
        rows,
        start=1,
    ):

        row["position"] = position

        row["from_popularity"] = (
            row["popularity_rank"] is not None
        )

        row["from_knn"] = (
            row["knn_rank"] is not None
        )

    return CandidateSet(
        rows=rows,
        popularity_items=[
            item for item, _ in pop
        ],
        knn_items=[
            item for item, _ in knn
        ],
        known_seed_items=sum(
            item in index._positions
            for item in positive_history
        ),
    )


def ranking_metrics(
    items: list[str],
    relevant: set[str],
    k: int,
) -> dict[str, float]:
    """
    They evaluate observed-positive hits while rejecting
    duplicate IDs that invalidate ranking metrics.
    """

    if k < 1:
        raise ValueError(
            "k must be positive"
        )

    top = items[:k]

    if len(top) != len(set(top)):
        raise ValueError(
            "Duplicate recommendations invalidate top-K metrics"
        )

    if not relevant:
        raise ValueError(
            "At least one held-out observed-positive target is required"
        )

    hits = 0
    ap_sum = 0.0
    dcg = 0.0

    for rank, item in enumerate(
        top,
        start=1,
    ):

        if item in relevant:

            hits += 1
            ap_sum += hits / rank
            dcg += (
                1.0
                / log2(rank + 1)
            )

    ideal = sum(
        1.0 / log2(rank + 1)
        for rank in range(
            1,
            min(k, len(relevant)) + 1,
        )
    )

    return {
        "observed_precision_at_k": (
            hits / k
        ),
        "recall_at_k": (
            hits / len(relevant)
        ),
        "map_at_k": (
            ap_sum / min(k, len(relevant))
        ),
        "ndcg_at_k": (
            dcg / ideal
        ),
    }
