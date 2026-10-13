"""Tests for parsing Gemini quota retry delays."""
from types import SimpleNamespace
from unittest.mock import MagicMock

from config.settings import Settings
import pytest
from google.genai import types
from google.genai.errors import APIError
from services.gemini_service import (
    GeminiService,
    GeminiServiceError,
    _format_retry_delay,
    _retry_delay_seconds,
)


def test_extract_retry_delay_from_google_error_details():
    error = "{'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '35012s'}"

    assert _retry_delay_seconds(error) == 35012


def test_missing_retry_delay_returns_none():
    assert _retry_delay_seconds("RESOURCE_EXHAUSTED") is None


def test_format_retry_delay_as_hours_and_minutes():
    assert _format_retry_delay(35012) == "9 hours 44 minutes"


def test_generation_uses_configured_model_and_temperature():
    settings = Settings(
        gemini_api_key="test",
        gemini_model="gemini-3.8-flash",
        llm_temperature=0.1,
        max_image_size_mb=10,
        max_history_messages=12,
    )
    service = GeminiService(settings)
    service._client = MagicMock()
    service._client.models.generate_content.return_value.text = "Generated answer"

    answer = service.generate_content("What is in the image?")

    assert answer == "Generated answer"
    call = service._client.models.generate_content.call_args.kwargs
    assert call["model"] == "gemini-3.8-flash"
    assert call["config"].temperature == 0.1


def test_stream_content_yields_text_deltas():
    settings = Settings(
        gemini_api_key="test",
        gemini_model="gemini-3.8-flash",
        llm_temperature=0.1,
        max_image_size_mb=10,
        max_history_messages=12,
    )
    service = GeminiService(settings)
    service._client = MagicMock()
    service._client.models.generate_content_stream.return_value = iter(
        [
            SimpleNamespace(text="First "),
            SimpleNamespace(text="part."),
        ]
    )

    assert list(service.stream_content("Hello")) == ["First ", "part."]
    call = service._client.models.generate_content_stream.call_args.kwargs
    assert call["model"] == "gemini-3.8-flash"
    assert call["config"].temperature == 0.1


def test_generation_sends_inline_image():
    settings = Settings(
        gemini_api_key="test",
        gemini_model="gemini-3.8-flash",
        llm_temperature=0.1,
        max_image_size_mb=10,
        max_history_messages=12,
    )
    service = GeminiService(settings)
    service._client = MagicMock()
    service._client.models.generate_content.return_value.text = "A photo summary"
    image_part = types.Part.from_bytes(
        data=b"image bytes",
        mime_type="image/jpeg",
    )
    contents = [
        types.Content(
            role="user",
            parts=[
                image_part,
                types.Part.from_text(text="Describe this image."),
            ],
        )
    ]

    assert service.generate_content(contents) == "A photo summary"
    call = service._client.models.generate_content.call_args.kwargs
    assert call["model"] == "gemini-3.8-flash"
    assert call["contents"] == contents


def test_chat_history_preserves_user_and_assistant_roles():
    history = [
        types.Content(role="user", parts=[types.Part.from_text(text="Question")]),
        types.Content(role="model", parts=[types.Part.from_text(text="Answer")]),
    ]

    converted = GeminiService._to_interactions_input(history)

    assert converted == [
        {"type": "user_input", "content": [{"type": "text", "text": "Question"}]},
        {"type": "model_output", "content": [{"type": "text", "text": "Answer"}]},
    ]


def test_stream_without_text_raises_instead_of_silently_returning_empty():
    settings = Settings(
        gemini_api_key="test",
        gemini_model="gemini-3.8-flash",
        llm_temperature=0.1,
        max_image_size_mb=10,
        max_history_messages=12,
    )
    service = GeminiService(settings)
    service._client = MagicMock()
    service._client.models.generate_content_stream.return_value = iter([])

    with pytest.raises(GeminiServiceError, match="without returning any text"):
        list(service.stream_content("Hello"))


def test_unavailable_model_error_is_actionable():
    settings = Settings(
        gemini_api_key="test",
        gemini_model="gemini-2.5-flash",
        llm_temperature=0.1,
        max_image_size_mb=10,
        max_history_messages=12,
    )
    service = GeminiService(settings)
    service._client = MagicMock()
    service._client.models.generate_content.side_effect = APIError(
        code=404,
        response_json={
            "error": {
                "code": 404,
                "message": (
                    "This model is no longer available to new users. "
                    "Please update your code to use models/gemini-3.8-flash."
                ),
                "status": "NOT_FOUND",
            }
        },
    )

    with pytest.raises(GeminiServiceError, match="unavailable") as error:
        service.generate_content("Describe this photo.")

    assert "gemini-3.8-flash" in error.value.message
