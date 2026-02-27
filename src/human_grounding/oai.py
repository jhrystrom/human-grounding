import os
from dataclasses import dataclass

from dotenv import load_dotenv
from instructor import Instructor
from openai import AsyncOpenAI, OpenAI

from human_grounding.structured import ModelConfig

GEMINI_MODELS = {"gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.0-flash-lite"}


@dataclass
class OpenAIConfig:
    client: Instructor | None = None
    model: str = "gpt-4o-mini"
    temperature: float = 0.7


def _initialize_instructor(client: OpenAI | AsyncOpenAI) -> Instructor:
    import instructor

    load_dotenv()
    return instructor.from_openai(client=client)


def initialize_oai_client(is_async: bool = False) -> OpenAI | AsyncOpenAI:
    load_dotenv(override=True)
    return OpenAI() if not is_async else AsyncOpenAI()


def initialize_gemini_client(is_async: bool = False) -> OpenAI | AsyncOpenAI:
    load_dotenv()
    kwargs = {
        "api_key": os.getenv("GEMINI_API_KEY"),
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    }
    return OpenAI(**kwargs) if not is_async else AsyncOpenAI(**kwargs)


def get_default_config(
    model: str = "gpt-4o-mini", use_async: bool = False
) -> ModelConfig:
    return ModelConfig(
        model=model,
        client=initialize_oai_client(is_async=use_async)
        if model not in GEMINI_MODELS
        else initialize_gemini_client(is_async=use_async),
    )
