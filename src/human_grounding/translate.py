import human_grounding.structured as wfs
from human_grounding.oai_schemas import EnglishTranslation


def _format_translation(text: str) -> str:
    return f"Translate the following text into English: '{text}'."


async def atranslate_texts(
    texts: list[str], model_config: wfs.ModelConfig, language: str = "English"
) -> list[str]:
    if language != "English":
        raise ValueError("Only English translations are supported.")
    prompts = [_format_translation(text) for text in texts]
    translations: list[EnglishTranslation] = await wfs.agenerate_structured_multi(
        prompts, schema=EnglishTranslation, model_config=model_config
    )
    return [translation.translation for translation in translations]
