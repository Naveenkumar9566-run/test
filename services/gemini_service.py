"""Gemini API client wrapper using official google-genai SDK."""
import base64
import logging
import math
import re
from typing import Iterator, List, Optional, Union
from google import genai
from google.genai import types
from google.genai.errors import APIError

from config.settings import Settings
from utils.prompts import SYSTEM_INSTRUCTION

logger = logging.getLogger(__name__)


def _retry_delay_seconds(error_text: str) -> Optional[int]:
    """Extract Google's retry delay from a serialized API error, if present."""
    match = re.search(r"""retryDelay['"]?\s*:\s*['"](\d+(?:\.\d+)?)s""", error_text)
    if match is None:
        return None
    return math.ceil(float(match.group(1)))


def _format_retry_delay(seconds: int) -> str:
    """Format a retry delay as a concise, human-readable duration."""
    minutes = math.ceil(seconds / 60)
    hours, remaining_minutes = divmod(minutes, 60)
    if hours and remaining_minutes:
        return f"{hours} hours {remaining_minutes} minutes"
    if hours:
        return f"{hours} hours"
    if minutes:
        return f"{minutes} minutes"
    return "less than a minute"


class GeminiServiceError(Exception):
    """Custom exception for Gemini API errors with clean user-friendly messaging."""

    def __init__(
        self,
        message: str,
        technical_details: Optional[str] = None,
        is_retryable: bool = False,
        retry_after_seconds: Optional[int] = None,
    ):
        super().__init__(message)
        self.message = message
        self.technical_details = technical_details
        self.is_retryable = is_retryable
        self.retry_after_seconds = retry_after_seconds


class GeminiService:
    """Encapsulates interaction with the Google Gen AI SDK."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client: Optional[genai.Client] = None

    def _get_client(self) -> genai.Client:
        """Initialize or return existing genai.Client."""
        if not self.settings.has_api_key:
            raise GeminiServiceError(
                "Gemini API key is missing. Please set GEMINI_API_KEY in your .env file or environment variables.",
                technical_details="Missing GEMINI_API_KEY configuration.",
                is_retryable=False,
            )

        if self._client is None:
            try:
                self._client = genai.Client(api_key=self.settings.gemini_api_key)
            except Exception as e:
                logger.error("Failed to initialize Google Gen AI Client: %s", str(e))
                raise GeminiServiceError(
                    "Failed to initialize Gemini AI client. Check your API configuration.",
                    technical_details=str(e),
                    is_retryable=False,
                )
        return self._client

    @staticmethod
    def _to_interactions_input(
        contents: Union[str, List[Union[types.Content, types.Part, str]]],
    ) -> Union[str, List[dict[str, object]]]:
        """Convert SDK content into role-aware Interactions API input steps."""
        if isinstance(contents, str):
            return contents

        steps: List[dict[str, object]] = []
        loose_content: List[dict[str, str]] = []

        def append_part(part: types.Part) -> None:
            if part.text:
                loose_content.append({"type": "text", "text": part.text})
            if part.inline_data:
                loose_content.append(
                    {
                        "type": "image",
                        "data": base64.b64encode(part.inline_data.data).decode("ascii"),
                        "mime_type": part.inline_data.mime_type,
                    }
                )

        def flush_loose_content() -> None:
            if loose_content:
                steps.append(
                    {
                        "type": "user_input",
                        "content": loose_content.copy(),
                    }
                )
                loose_content.clear()

        for item in contents:
            if isinstance(item, types.Content):
                flush_loose_content()
                content: List[dict[str, str]] = []
                for part in item.parts or []:
                    if part.text:
                        content.append({"type": "text", "text": part.text})
                    if part.inline_data:
                        content.append(
                            {
                                "type": "image",
                                "data": base64.b64encode(
                                    part.inline_data.data
                                ).decode("ascii"),
                                "mime_type": part.inline_data.mime_type,
                            }
                        )
                steps.append(
                    {
                        "type": (
                            "model_output"
                            if item.role == "model"
                            else "user_input"
                        ),
                        "content": content,
                    }
                )
            elif isinstance(item, types.Part):
                append_part(item)
            elif isinstance(item, str):
                loose_content.append({"type": "text", "text": item})
            else:
                raise TypeError(f"Unsupported Gemini input content: {type(item).__name__}")

        flush_loose_content()
        if not steps:
            raise ValueError("Gemini input must contain text or image content.")
        return steps

    def generate_content(
        self,
        contents: Union[str, List[Union[types.Content, types.Part, str]]],
        system_instruction: Optional[str] = SYSTEM_INSTRUCTION,
    ) -> str:
        """
        Generate content using the configured Gemini model.

        Args:
            contents: Single string or list of types.Content/Part/str objects.
            system_instruction: Optional system instruction prompt.

        Returns:
            Extracted text response.

        Raises:
            GeminiServiceError: On any client or API failure.
        """
        client = self._get_client()
        model = self.settings.gemini_model

        try:
            config = types.GenerateContentConfig(
                temperature=self.settings.llm_temperature,
                system_instruction=system_instruction if system_instruction else None,
            )
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )

            if response and response.text:
                return response.text.strip()
            return "No response text was generated by the model."

        except APIError as api_err:
            raise self._to_service_error(api_err, model) from api_err

        except Exception as e:
            raise self._to_unexpected_error(e) from e

    def stream_content(
        self,
        contents: Union[str, List[Union[types.Content, types.Part, str]]],
        system_instruction: Optional[str] = SYSTEM_INSTRUCTION,
    ) -> Iterator[str]:
        """Yield Gemini response text as streamed output deltas arrive."""
        client = self._get_client()
        model = self.settings.gemini_model
        try:
            config = types.GenerateContentConfig(
                temperature=self.settings.llm_temperature,
                system_instruction=system_instruction if system_instruction else None,
            )
            stream = client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config,
            )
            received_text = False
            for chunk in stream:
                text = getattr(chunk, "text", "") or ""
                if text:
                    received_text = True
                    yield text
            if not received_text:
                raise GeminiServiceError(
                    "Gemini finished the request without returning any text. "
                    "Check model access and API quota, then try again."
                )
        except APIError as api_err:
            raise self._to_service_error(api_err, model) from api_err
        except GeminiServiceError:
            raise
        except Exception as e:
            raise self._to_unexpected_error(e) from e

    @staticmethod
    def _to_service_error(api_err: APIError, model: str) -> GeminiServiceError:
        status_code = getattr(api_err, "code", None)
        err_msg = str(api_err)
        logger.error("Gemini API Error (status %s): %s", status_code, err_msg)

        if status_code in (401, 403) or "API_KEY_INVALID" in err_msg or "PERMISSION_DENIED" in err_msg:
            return GeminiServiceError(
                "Invalid or unauthorized Gemini API key. Please verify GEMINI_API_KEY in .env.",
                technical_details=err_msg,
            )
        if status_code == 404 or "NOT_FOUND" in err_msg or "models/" in err_msg:
            recommended_model = re.search(r"models/([A-Za-z0-9._-]+)", err_msg)
            recommendation = (
                f" Google recommends '{recommended_model.group(1)}'."
                if recommended_model
                else ""
            )
            return GeminiServiceError(
                f"Gemini model '{model}' is unavailable for this API account. "
                f"Check model access in Google AI Studio.{recommendation}",
                technical_details=err_msg,
            )
        if status_code == 429 or "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower():
            retry_after_seconds = _retry_delay_seconds(err_msg)
            if "GenerateRequestsPerDayPerProjectPerModel" in err_msg:
                message = f"Gemini's daily request quota for '{model}' is exhausted."
                if retry_after_seconds is not None:
                    message += (
                        " Google says to retry in about "
                        f"{_format_retry_delay(retry_after_seconds)}."
                    )
                message += " Check your quota and billing in Google AI Studio."
            else:
                message = "Gemini API rate limit or quota exceeded."
                if retry_after_seconds is not None:
                    message += (
                        " Google says to retry in about "
                        f"{_format_retry_delay(retry_after_seconds)}."
                    )
                else:
                    message += " Check your quota in Google AI Studio before retrying."
            return GeminiServiceError(
                message,
                technical_details=err_msg,
                is_retryable=True,
                retry_after_seconds=retry_after_seconds,
            )
        if status_code in (500, 502, 503, 504) or "UNAVAILABLE" in err_msg:
            return GeminiServiceError(
                "Google Gemini service is temporarily unavailable. Please try again shortly.",
                technical_details=err_msg,
                is_retryable=True,
            )
        return GeminiServiceError(
            f"Gemini API request failed: {getattr(api_err, 'message', err_msg)}",
            technical_details=err_msg,
        )

    @staticmethod
    def _to_unexpected_error(error: Exception) -> GeminiServiceError:
        error_text = str(error)
        logger.error("Unexpected error during Gemini content generation: %s", error_text)
        if "timeout" in error_text.lower() or "connection" in error_text.lower():
            return GeminiServiceError(
                "Network connection to Google Gemini API timed out or failed. "
                "Please check your internet connection.",
                technical_details=error_text,
                is_retryable=True,
            )
        return GeminiServiceError(
            f"An unexpected error occurred while communicating with the AI model: {error_text}",
            technical_details=error_text,
        )
