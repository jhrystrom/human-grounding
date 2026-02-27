import asyncio
import base64
import json
from dataclasses import dataclass
from hashlib import md5, sha256
from pathlib import Path
from typing import TypedDict

from anyio.functools import cache
from instructor import Instructor
from loguru import logger
from openai import AsyncOpenAI, OpenAI
from PIL import Image
from pydantic import ValidationError

import human_grounding.fileio
from human_grounding.directories import CACHE_DIR
from human_grounding.oai_schemas import OpenAISchema, StructuredSchema


@dataclass
class ModelConfig:
    model: str
    client: Instructor | AsyncOpenAI | OpenAI
    temperature: float = 0.7


class Message(TypedDict):
    role: str
    content: str | list[dict[str, str]]


def encode_image(image: Path) -> str:
    base64_image = base64.b64encode(image.read_bytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_image}"


def _create_messages(
    prompt: str, system_message: str | None, image: Path | None = None
) -> list[Message]:
    messages_to_send = (
        [Message(role="system", content=system_message)]
        if system_message is not None
        else []
    )
    user_message = (
        Message(role="user", content=prompt)
        if image is None
        else Message(
            role="user",
            content=[
                {"type": "input_text", "text": "prompt"},
                {"type": "input_image", "image_url": encode_image(image=image)},
            ],
        )
    )
    messages_to_send.append(user_message)
    return messages_to_send


def generate_cache_name(
    messages: list[Message],
    model_name: str,
    schema: type[StructuredSchema],
    image: Path | None = None,
) -> Path:
    """Hash a name for the above"""
    messages_str = str(messages)
    schema_str = str(schema.openai_schema)
    cache_string = f"{messages_str}{model_name}{schema_str}"
    if image is not None:
        # If an image is provided, include its hash in the cache name
        cache_string += str(image)
    return (CACHE_DIR / sha256(cache_string.encode()).hexdigest()).with_suffix(".json")


async def agenerate_structured(
    prompt: str,
    schema: type[StructuredSchema],
    model_config: ModelConfig,
    system_message: str | None = None,
    use_cache: bool = True,
) -> OpenAISchema:
    client = model_config.client
    messages_to_send = _create_messages(prompt, system_message)
    if use_cache:
        cache_name = generate_cache_name(
            messages=messages_to_send, model_name=model_config.model, schema=schema
        )
        if cache_name.exists():
            data = human_grounding.fileio.read_json(cache_name)
            try:
                return schema.model_validate(data)
            except ValidationError:
                # Remove the cache file and re-run
                cache_name.unlink()
                logger.warning(
                    f"Cache file {cache_name} was invalid and has been removed. Regenerating response."
                )
                return await agenerate_structured(
                    prompt=prompt,
                    schema=schema,
                    model_config=model_config,
                    system_message=system_message,
                    use_cache=True,
                )

    response = await client.responses.parse(
        input=messages_to_send,
        text_format=schema,
        model=model_config.model,
        temperature=model_config.temperature,
    )

    if use_cache:
        human_grounding.fileio.write_json(
            data=response.output_parsed.model_dump(mode="json"), output_path=cache_name
        )
    return response.output_parsed


async def agenerate_structured_img(
    prompt: str,
    image: Path,
    schema: type[StructuredSchema],
    model_config: ModelConfig,
    system_message: str | None = None,
    use_cache: bool = True,
) -> OpenAISchema:
    client = model_config.client
    messages_to_send = _create_messages(prompt, system_message, image=image)
    if use_cache:
        cache_name = generate_cache_name(
            messages=messages_to_send, model_name=model_config.model, schema=schema
        )
        if cache_name.exists():
            data = human_grounding.fileio.read_json(cache_name)
            return schema.model_validate(data)
    if not isinstance(model_config.client, AsyncOpenAI):
        raise ValueError(
            "agenerate_structured_img requires an AsyncOpenAI client, "
            "but the provided client is not an instance of AsyncOpenAI."
        )

    raw_response = await client.responses.parse(
        input=messages_to_send,
        text_format=schema,
        model=model_config.model,
        temperature=model_config.temperature,
    )
    response = schema.model_validate(json.loads(raw_response.output[0].content[0].text))

    if use_cache:
        human_grounding.fileio.write_json(
            data=response.model_dump(mode="json"), output_path=cache_name
        )
    return response


async def agenerate_structured_multi(
    prompts: list[str],
    schema: type[OpenAISchema],
    model_config: ModelConfig,
    system_message: str | None = None,
    concurrency_limit: int = 50,
) -> list[OpenAISchema]:
    if model_config.client is None:
        raise ValueError("ModelConfig must have a client")
    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(concurrency_limit)

    async def limited_generate(prompt: str) -> OpenAISchema:
        # Use the semaphore to limit concurrent executions
        async with semaphore:
            return await agenerate_structured(
                prompt=prompt,
                schema=schema,
                system_message=system_message,
                model_config=model_config,
            )

    all_results = await asyncio.gather(
        *[limited_generate(prompt=prompt) for prompt in prompts]
    )
    return all_results


async def agenerate_structured_img_multi(
    prompts: list[str] | str,
    images: list[Path],
    schema: type[OpenAISchema],
    model_config: ModelConfig,
    system_message: str | None = None,
    concurrency_limit: int = 50,
) -> list[OpenAISchema]:
    if model_config.client is None:
        raise ValueError("ModelConfig must have a client")
    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(concurrency_limit)
    if isinstance(prompts, str):
        prompts = [prompts] * len(images)

    async def limited_generate(prompt: str, image: Path) -> OpenAISchema:
        # Use the semaphore to limit concurrent executions
        async with semaphore:
            return await agenerate_structured_img(
                prompt=prompt,
                image=image,
                schema=schema,
                system_message=system_message,
                model_config=model_config,
            )

    all_results = await asyncio.gather(
        *[
            limited_generate(prompt=prompt, image=image)
            for prompt, image in zip(prompts, images, strict=True)
        ]
    )
    return all_results


def generate_structured(
    prompt: str,
    schema: type[OpenAISchema],
    model_config: ModelConfig,
    system_message: str | None = None,
) -> OpenAISchema:
    client = model_config.client
    messages_to_send = _create_messages(prompt, system_message)
    response = client.chat.completions.create(
        messages=messages_to_send,
        response_model=schema,
        model=model_config.model,
        temperature=model_config.temperature,
    )
    return response


async def agenerate_from_inputs(
    values: list[str],
    schema: type[StructuredSchema],
    model_config: ModelConfig,
    system_message: str | None = None,
) -> list[OpenAISchema]:
    formatted = [schema.format_prompt(val) for val in values]
    return await agenerate_structured_multi(
        prompts=formatted,
        schema=schema,
        model_config=model_config,
        system_message=system_message,
    )


async def agenerate_complex_from_inputs(
    complex_values: list[dict[str, str | int]],
    schema: type[StructuredSchema],
    model_config: ModelConfig,
    system_message: str | None = None,
) -> list[OpenAISchema]:
    formatted = [schema.format_prompt(**val) for val in complex_values]
    return await agenerate_structured_multi(
        prompts=formatted,
        schema=schema,
        model_config=model_config,
        system_message=system_message,
    )
