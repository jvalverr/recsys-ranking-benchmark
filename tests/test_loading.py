"""Small, synthetic fixtures verify ingestion independently of external data."""

import gzip
import json

from recsys.datasets import (
    load_amazon,
    load_movielens,
    load_retailrocket,
)

from recsys.validation import validate_bundle


def test_movielens_preserves_ratings_and_metadata(tmp_path):

    folder = tmp_path / "ml-1m"
    folder.mkdir()

    (folder / "ratings.dat").write_text(
        "1::10::5::978300760\n"
        "1::11::3::978302109\n"
    )

    (folder / "movies.dat").write_bytes(
        b"10::A Film::Action\n"
        b"11::B Film::Drama\n"
    )

    (folder / "users.dat").write_text(
        "1::F::25::4::01234\n"
    )

    bundle = load_movielens(tmp_path)
    report = validate_bundle(bundle)

    assert report.ok, report.errors
    assert len(bundle.events) == 2

    assert bundle.events["event_value"].tolist() == [
        5.0,
        3.0,
    ]

    assert bundle.events["raw_rating"].tolist() == [
        5,
        3,
    ]

    assert (
        str(bundle.events["timestamp"].dtype)
        == "datetime64[ns, UTC]"
    )

    assert (
        bundle.tables["users"].iloc[0]["zip_code"]
        == "01234"
    )


def test_retailrocket_preserves_repeated_events(tmp_path):

    (tmp_path / "events.csv").write_text(
        "timestamp,visitorid,event,itemid,transactionid\n"
        "1430000000000,u1,view,i1,\n"
        "1430000000000,u1,view,i1,\n"
        "1430000001000,u1,transaction,i1,T1\n"
    )

    (tmp_path / "category_tree.csv").write_text(
        "categoryid,parentid\n"
        "A,\n"
    )

    (tmp_path / "item_properties_part1.csv").write_text(
        "timestamp,itemid,property,value\n"
        "1430000001000,i1,categoryid,A\n"
    )

    bundle = load_retailrocket(
        tmp_path,
        metadata_preview=1,
    )

    report = validate_bundle(bundle)

    assert report.ok, report.errors
    assert len(bundle.events) == 3

    assert (
        report.statistics["duplicate_event_observations"]
        == 1
    )

    assert len(
        bundle.tables["item_properties_part1_preview"]
    ) == 1

    assert set(bundle.events["event_type"]) == {
        "view",
        "transaction",
    }


def test_amazon_parent_id_and_compressed_metadata(tmp_path):

    (tmp_path / "Video_Games.csv").write_text(
        "user_id,parent_asin,rating,timestamp,asin\n"
        "u1,P1,4,1680000000000,CHILD1\n"
    )

    with gzip.open(
        tmp_path / "meta_Video_Games.jsonl.gz",
        "wt",
    ) as f:

        f.write(
            json.dumps({
                "parent_asin": "P1",
                "title": "Game",
            }) + "\n"
        )

    bundle = load_amazon(
        tmp_path,
        metadata_preview=1,
    )

    report = validate_bundle(bundle)

    assert report.ok, report.errors

    assert (
        bundle.events.iloc[0]["item_id"]
        == "P1"
    )

    assert (
        bundle.events.iloc[0]["raw_asin"]
        == "CHILD1"
    )

    assert (
        bundle.tables["metadata_preview"]
        .iloc[0]["title"]
        == "Game"
    )


def test_invalid_timestamp_is_reported_not_dropped(tmp_path):

    (tmp_path / "events.csv").write_text(
        "timestamp,visitorid,event,itemid\n"
        "invalid,u1,view,i1\n"
    )

    bundle = load_retailrocket(tmp_path)
    report = validate_bundle(bundle)

    assert not report.ok

    assert any(
        "timestamp" in message
        for message in report.errors
    )

    assert len(bundle.events) == 1


def test_legacy_amazon_schema_is_explicit(tmp_path):

    (tmp_path / "Video_Games.csv").write_text(
        "reviewerID,asin,overall,unixReviewTime\n"
        "u1,X1,3,1450000000\n"
    )

    bundle = load_amazon(tmp_path)

    assert validate_bundle(bundle).ok
    assert "legacy" in bundle.notes[0]


def test_invalid_user_id_is_reported(tmp_path):

    (tmp_path / "events.csv").write_text(
        "timestamp,visitorid,event,itemid\n"
        "1430000000000,,view,i1\n"
    )

    report = validate_bundle(
        load_retailrocket(tmp_path)
    )

    assert not report.ok

    assert any(
        "user_id" in message
        for message in report.errors
    )


def test_metadata_missing_parent_id_is_rejected(tmp_path):

    (tmp_path / "Video_Games.csv").write_text(
        "user_id,parent_asin,rating,timestamp\n"
        "u1,P1,4,1680000000000\n"
    )

    with gzip.open(
        tmp_path / "meta_Video_Games.jsonl.gz",
        "wt",
    ) as f:

        f.write(
            json.dumps({
                "title": "No product identifier",
            }) + "\n"
        )

    try:
        load_amazon(
            tmp_path,
            metadata_preview=1,
        )

    except ValueError as exc:
        assert "parent_asin" in str(exc)

    else:
        raise AssertionError(
            "Metadata without parent_asin was accepted"
        )
