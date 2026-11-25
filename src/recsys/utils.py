"""Cross-cutting helpers shared by independent experiment components."""

import json

from pathlib import Path


def write_json(
    payload: dict,
    path: Path,
) -> None:
    """
    Writers persist UTF-8 JSON for local experiment artifacts.
    """

    # Writers create output directories only for generated
    # artifacts; they leave all original dataset files unchanged.
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )





def sql_literal(value: str | Path) -> str:
    """
    Users safely represent local paths as SQL string literals
    when constructing DuckDB COPY statements.
    """

    # They escape embedded apostrophes to prevent malformed
    # SQL when project directories contain special characters.
    return "'" + str(value).replace("'", "''") + "'"

