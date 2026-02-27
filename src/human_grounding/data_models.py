from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol

import polars as pl

TOPICS = ["climate_change", "mental_health"]

Stance = Literal["left", "right"]
PoliticalParty = Literal["A", "B", "I", "M", "Ø", "Æ"]

Embedder = Callable[[list[str]], list[list[float]]]


def dataclasses_to_dataframe(dataclasses: list) -> pl.DataFrame:
    other_attributes = [asdict(dataclass) for dataclass in dataclasses]
    results = pl.DataFrame(other_attributes)
    return results


class SentenceEmbedder(Protocol):
    def encode(self, text: list[str]) -> list[list[float]]: ...


@dataclass
class PoliticalText:
    party: str
    text: str
    topic: str
    source: str

    @classmethod
    def from_raw_path(cls: "PoliticalText", path: Path) -> "PoliticalText":
        from loguru import logger

        logger.debug(f"Reading political text from {path}")
        topic = path.parent.parent.stem
        party = path.parent.stem
        source = path.stem
        text = path.read_text()
        logger.debug(f"Found: {party} on {topic}")
        return PoliticalText(party=party, text=text, topic=topic, source=source)


@dataclass
class Opinion:
    text: str
    stance: Stance


@dataclass
class PerspectivePrompt:
    prompt: str
    topic: str
    ideology: Stance | None = None
    party: PoliticalParty | None = None

    def __str__(self) -> str:
        META_PROMPT = "{solution} for at løse {topic}."
        return META_PROMPT.format(solution=self.prompt, topic=self.topic)


@dataclass
class Perspective:
    prompt: PerspectivePrompt
    statement: str


@dataclass
class StatementEmbedding:
    affirmative: list[float]
    negative: list[float]
