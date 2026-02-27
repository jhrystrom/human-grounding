import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import polars as pl

from human_grounding.data_models import Opinion


def read_json(path: Path) -> list[dict] | dict:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def invert_dict(d: dict[str, list[str]]) -> dict[str, str]:
    return {value: key for key, values in d.items() for value in values}


def read_keymap(
    path: Path, key_name: str = "party", value_name: str = "ideology"
) -> pl.DataFrame:
    # Stances
    raw_dict = read_json(path)
    keymap = invert_dict(raw_dict)
    return pl.DataFrame(
        {key_name: key, value_name: value} for key, value in keymap.items()
    )


def write_json(data: dict, output_path: Path) -> None:
    output_path.write_text(json.dumps(data, ensure_ascii=False))


def read_opinions(path: Path) -> list[Opinion]:
    return read_dataclasses(path, Opinion)


def read_dataclasses(path: Path, dataclass) -> list:  # noqa: ANN001
    data = read_jsonl(path)
    return [dataclass(**d) for d in data]


def read_multi_dataclasses(paths: list[Path], datacls: Any) -> list[Any]:
    return [
        cls_instance
        for path in paths
        for cls_instance in read_dataclasses(path, datacls)
    ]


def write_dataclasses(dataclasses: list, output_path: Path) -> None:
    _write_jsonl([asdict(datacls) for datacls in dataclasses], output_path=output_path)


def _write_jsonl(data: list[dict], output_path: Path) -> None:
    if not output_path.suffix.endswith(".jsonl"):
        raise ValueError(f"Output path must have a .jsonl extension: {output_path}")
    output_path.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in data))
