"""Reusable visualizations for exploratory recommendation analyses."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _finish(
    fig,
    output: str | Path | None,
):
    """You can optionally publish selected figures while preserving notebook interactivity."""

    fig.tight_layout()

    if output is not None:
        target = Path(output)

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            target,
            dpi=160,
            bbox_inches="tight",
        )

    return fig


def plot_activity_bins(
    bins: pd.DataFrame,
    entity: str,
    output: str | Path | None = None,
):
    """You visualize long-tailed activity using logarithmic entity frequencies."""

    # Researchers represent activity using powers-of-two intervals,
    # preserving distributional information without plotting every ID.
    labels = [
        str(2 ** int(p))
        if int(p) == 0
        else f"{2 ** int(p)}–{2 ** (int(p) + 1) - 1}"
        for p in bins["bin_power"]
    ]

    fig, ax = plt.subplots(
        figsize=(9, 4.5)
    )

    ax.bar(
        labels,
        bins["entities"],
    )

    ax.set_yscale("log")

    ax.set_xlabel(
        "Number of events per entity (log2 intervals)"
    )

    ax.set_ylabel(
        f"Number of {entity} (log scale)"
    )

    ax.set_title(
        f"{entity.capitalize()} activity distribution"
    )

    ax.tick_params(
        axis="x",
        rotation=45,
    )

    return _finish(fig, output)


def plot_event_types(
    counts: pd.DataFrame,
    output: str | Path | None = None,
):
    """You compare observed event categories without redefining relevance."""

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.bar(
        counts["event_type"],
        counts["percent"],
    )

    ax.set_ylabel(
        "Share of events (%)"
    )

    ax.set_title(
        "Observed feedback composition"
    )

    return _finish(fig, output)


def plot_ratings(
    ratings: pd.DataFrame,
    output: str | Path | None = None,
):
    """You display explicit rating frequencies before binary-label conversion."""

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.bar(
        ratings["rating"].astype(str),
        ratings["events"],
    )

    ax.set_xlabel("Rating")
    ax.set_ylabel("Events")

    ax.set_title(
        "Explicit rating distribution"
    )

    return _finish(fig, output)


def plot_monthly_events(
    monthly: pd.DataFrame,
    output: str | Path | None = None,
    log_scale: bool = False,
):
    """You examine temporal activity separately for each event category."""

    fig, ax = plt.subplots(
        figsize=(10, 4.5)
    )

    for event_type, frame in monthly.groupby(
        "event_type",
        sort=True,
    ):
        frame = frame.sort_values("month")

        ax.plot(
            frame["month"],
            frame["events"],
            marker=".",
            label=event_type,
        )

    if log_scale:
        ax.set_yscale("log")

    ax.set_xlabel("Month (UTC)")

    ax.set_ylabel(
        "Events" + (
            " (log scale)"
            if log_scale
            else ""
        )
    )

    ax.set_title(
        "Monthly interaction volume"
    )

    ax.legend()

    return _finish(fig, output)


def plot_cold_start(
    cold_start: pd.DataFrame,
    output: str | Path | None = None,
):
    """You display disjoint user/item cold-start categories in future events."""

    order = [
        "warm",
        "new_user",
        "new_item",
        "new_user_and_item",
    ]

    counts = (
        cold_start
        .groupby("segment")["events"]
        .sum()
    )

    total = counts.sum()

    shares = [
        100.0
        * counts.get(segment, 0)
        / total
        for segment in order
    ]

    fig, ax = plt.subplots(
        figsize=(9, 4)
    )

    ax.bar(
        order,
        shares,
    )

    ax.set_ylabel(
        "Share of future events (%)"
    )

    ax.set_xlabel(
        "Availability in the historical window"
    )

    ax.set_title(
        "Temporal cold-start diagnostic"
    )

    ax.tick_params(
        axis="x",
        rotation=15,
    )

    return _finish(fig, output)
