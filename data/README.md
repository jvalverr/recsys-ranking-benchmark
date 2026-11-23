# Dataset Management

## Overview

The dataset adapters preserve original source files and generate standardized event logs for reproducible recommendation experiments.

They maintain separate representations of raw interactions, auxiliary metadata, and validated outputs.

## Supported Datasets

| Dataset | Primary source | Feedback |
|---|---|---|
| MovieLens 1M | `movielens-1m/ml-1m/ratings.dat` | Explicit ratings |
| RetailRocket | `retailrocket/events.csv` | Implicit events |
| Amazon Video Games | `amazon-video-games/Video_Games.csv` | Explicit ratings |

## Processing Principles

The adapters follow five principles:

1. They preserve original interaction records.
2. They normalize timestamps to UTC.
3. They retain source attributes with `raw_` prefixes.
4. They report data-quality problems without silently deleting observations.
5. They keep auxiliary metadata separate until the experimental methodology defines valid joins.

## Metadata Handling

### MovieLens

The loaders retain movie and user metadata in separate tables.

### RetailRocket

The loaders preserve access to item-property snapshots and the category hierarchy.

They do not join item properties to interactions during ingestion because valid feature construction requires time-aware joins.

### Amazon

The loaders identify the review schema and use `parent_asin` for 2023 product-level identifiers.

They inspect bounded metadata samples without claiming to validate the complete product catalog.

## Generated Outputs

Successful preparation generates:

- `data/processed/<dataset>/events.parquet`
- `data/processed/<dataset>/validation.json`
- Additional small reference tables when available

The generated files remain outside Git version control.

## Data Quality

Validation reports include:

- Schema and identifier checks
- Timestamp checks
- Potential duplicate observations
- Interaction counts
- Unique users and items
- Interaction sparsity
- Temporal coverage
- Event-type distributions

Reports distinguish blocking errors from nonblocking observations.

## Reproducibility

The original datasets remain unchanged.

Source files and generated Parquet artifacts are not redistributed through this repository.
