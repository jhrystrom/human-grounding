from typing import Literal

from instructor import OpenAISchema
from pydantic import Field


class StructuredSchema(OpenAISchema):
    @staticmethod
    def format_prompt(*args, **kwargs):  # noqa: ANN002, ANN003
        raise NotImplementedError


class ImageInformation(StructuredSchema):
    """Extracts information from a given image of an experiment - output None for the ones that can't be seen"""

    seed: int | None = Field(..., description="the seed on the white slip")
    dataset: Literal["rai", "welfare"] | None = Field(
        ..., description="the dataset used for the experiment (white slip)"
    )
    number_of_samples: int | None = Field(
        20, description="the number of samples in the experiment (white slip)"
    )
    participant_id: str | None = Field(
        ..., description="Single letter participant ID (blue slip)"
    )

    def format_prompt() -> str:
        return (
            "Please extract the information from the image of the experiment. "
            "The information includes the seed on the white slip, the dataset used for the experiment, "
            "the number of samples in the experiment, and the participant ID on the blue slip. "
            "If any information is not visible, return None for that field."
        )

    def format_filestem(self) -> str:
        return f"{self.dataset}-n{self.number_of_samples}-seed{self.seed}-user_{self.participant_id}"


class OppositeStatement(StructuredSchema):
    """Provides a statement that takes the polar opposite stance of the input statement. Use the same language (Danish or English) as the input statement."""

    opposite: str = Field(
        ..., description="The opposite statement in the same language."
    )

    def format_prompt(statement: str, language: str) -> str:
        return f"Please generate the opposite stance of the following statement: {statement}. Write it in {language}."


class PoliticalStatements(OpenAISchema):
    """Provides an affirmative and a negative phrasing of a political position. Only one sentence each."""

    affirmative: str = Field(
        ...,
        description="Starts with an affirmative word ('ja', 'yes' etc) and a brief formulation of the stance",
    )
    negative: str = Field(
        ...,
        description="Starts with a negative word ('nej', 'yes' etc) and a brief formulation of the stance",
    )


class EnglishTranslation(StructuredSchema):
    """Translate the text into English. Match the tone and style of the original text closely."""

    translation: str = Field(..., description="The translated text.")


class StatementVagueness(OpenAISchema):
    """Classifies whether a given political statement is either vague (e.g., 'we should be greener`) or specific (e.g., 'we should build more wind turbines')."""

    classification: Literal["vague", "specific"] = Field(
        ..., description="The classification of the statement."
    )


class NeutralStatementTopic(OpenAISchema):
    """
    A completely neutral single-word summary of the topic of the statement
    The statement should not show a direct opinion. E.g.,
    """

    topic: str = Field(
        ...,
        description="A neutral topic description of the given statement.",
        example="CO2-tax",
    )


class StatementVariations(StructuredSchema):
    """Generate variations echoing the same position as the original statement. Each variation should have different structure, style, and wording. Keep the language the same as the original (Danish or English)."""

    variations: list[str] = Field(
        ..., description="Variations of the original statement."
    )

    def format_prompt(statement: str, num_variations: int) -> str:
        return f"Please produce {num_variations} variations of the following statement: {statement}. Imagine the variations are diverse people making the same point in very different ways. The language should be the same as the original statement (Danish/English)."


class StatementContinuation(StructuredSchema):
    """Generate a continuation of the provided statement, elaborating on the core message. The continuation should be in the same language as the original statement."""

    elaboration: str = Field(
        ..., description="2-3 sentences elaborating on the statement."
    )

    @staticmethod
    def format_prompt(statement: str, language: str) -> str:
        return f"Please elaborate on the following statement: {statement}. Write in {language}."


class StatementValidation(StructuredSchema):
    """Validate that the variation lives up to the specified constraints based on the perturbation type and the original statement."""

    same_topic: bool = Field(
        ...,
        description="The variation should talk about the same subtopic as the original statement",
    )
    applies_perturbation: bool = Field(
        ..., description="Whether the variation applies the specified perturbation."
    )
    natural: bool = Field(
        ..., description="Whether the variation sounds natural in the given language."
    )

    @staticmethod
    def format_prompt(statement: str, original: str, perturbation_type: str) -> str:
        return f"Please validate the following variation: '{statement}'. Ensure it has the same meaning as the original statement: '{original}'. It should have applied the perturbation: '{perturbation_type}'."
