"""Chat service managing multi-turn conversation and image context."""
import logging
from dataclasses import dataclass, field
from typing import Iterator, List, Optional
from google.genai import types

from services.gemini_service import GeminiService
from services.image_service import ProcessedImage
from utils.validators import validate_user_prompt, ValidationError
from utils.prompts import SYSTEM_INSTRUCTION

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Represents a chat turn in session history."""
    role: str  # "user" or "assistant"
    content: str
    has_image: bool = False


class ChatService:
    """Coordinates conversational requests with image and history context."""

    def __init__(self, gemini_service: GeminiService, max_history_messages: int = 12):
        self.gemini_service = gemini_service
        self.max_history_messages = max_history_messages

    def build_contents(
        self,
        history: List[ChatMessage],
        latest_question: str,
        active_image: Optional[ProcessedImage] = None,
    ) -> List[types.Content]:
        """
        Build the list of types.Content objects expected by Gemini generate_content.
        
        Args:
            history: Previous ChatMessage history in chronological order.
            latest_question: Current user question.
            active_image: Current uploaded image, if any.
            
        Returns:
            List of types.Content objects structured for multi-turn chat.
        """
        # Trim history to max_history_messages (ensuring we keep the most recent context)
        trimmed_history = history[-self.max_history_messages:] if history else []

        contents: List[types.Content] = []

        # If there is an active image, determine if it should be attached to the first turn
        # or directly to the latest turn.
        # Attaching the image Part to the earliest user turn or current query provides full vision context.
        image_part: Optional[types.Part] = None
        if active_image:
            image_part = types.Part.from_bytes(
                data=active_image.data,
                mime_type=active_image.mime_type,
            )

        # Build turns from history
        image_already_attached = False

        # If history begins with an assistant turn (e.g. from an analysis result added to chat),
        # insert an initial user prompt to ensure the conversation starts with role 'user'.
        if trimmed_history and trimmed_history[0].role in ("assistant", "model"):
            init_parts: List[types.Part] = []
            if image_part:
                init_parts.append(image_part)
                image_already_attached = True
            init_parts.append(types.Part.from_text(text="Please analyze this photo."))
            contents.append(types.Content(role="user", parts=init_parts))

        for msg in trimmed_history:
            role = "user" if msg.role == "user" else "model"
            parts: List[types.Part] = []
            
            # Attach active image to the very first user turn in history if active_image is present
            if role == "user" and image_part and not image_already_attached:
                parts.append(image_part)
                image_already_attached = True

            parts.append(types.Part.from_text(text=msg.content))
            contents.append(types.Content(role=role, parts=parts))

        # Build latest turn
        latest_parts: List[types.Part] = []
        if image_part and not image_already_attached:
            latest_parts.append(image_part)
        latest_parts.append(types.Part.from_text(text=latest_question))

        contents.append(types.Content(role="user", parts=latest_parts))
        return contents

    def ask_question(
        self,
        question: str,
        history: List[ChatMessage],
        active_image: Optional[ProcessedImage] = None,
    ) -> str:
        """
        Validate question, assemble context, and retrieve model response.
        
        Args:
            question: User's input text.
            history: Existing conversation list.
            active_image: Current photo if uploaded.
            
        Returns:
            Assistant's textual answer.
            
        Raises:
            ValidationError: If question is empty or invalid.
            GeminiServiceError: If API call fails.
        """
        clean_question = validate_user_prompt(question)
        contents = self.build_contents(
            history=history,
            latest_question=clean_question,
            active_image=active_image,
        )

        return self.gemini_service.generate_content(
            contents=contents,
            system_instruction=SYSTEM_INSTRUCTION,
        )

    def stream_question(
        self,
        question: str,
        history: List[ChatMessage],
        active_image: Optional[ProcessedImage] = None,
    ) -> Iterator[str]:
        """Stream the model response to a chat question."""
        clean_question = validate_user_prompt(question)
        contents = self.build_contents(
            history=history,
            latest_question=clean_question,
            active_image=active_image,
        )
        return self.gemini_service.stream_content(
            contents=contents,
            system_instruction=SYSTEM_INSTRUCTION,
        )

    def analyze_image(
        self,
        active_image: ProcessedImage,
        prompt: str,
    ) -> str:
        """
        Run a dedicated single-shot analysis prompt against an active image.
        
        Args:
            active_image: The uploaded photo.
            prompt: Structured analysis prompt instructions.
            
        Returns:
            Structured textual analysis response.
        """
        if not active_image:
            raise ValidationError("No image is currently uploaded for analysis.")

        image_part = types.Part.from_bytes(
            data=active_image.data,
            mime_type=active_image.mime_type,
        )
        text_part = types.Part.from_text(text=prompt)

        content = types.Content(role="user", parts=[image_part, text_part])
        return self.gemini_service.generate_content(
            contents=[content],
            system_instruction=SYSTEM_INSTRUCTION,
        )

    def stream_image_analysis(
        self,
        active_image: ProcessedImage,
        prompt: str,
    ) -> Iterator[str]:
        """Stream a single-shot analysis of an uploaded image."""
        if not active_image:
            raise ValidationError("No image is currently uploaded for analysis.")
        content = types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(
                    data=active_image.data,
                    mime_type=active_image.mime_type,
                ),
                types.Part.from_text(text=prompt),
            ],
        )
        return self.gemini_service.stream_content(
            contents=[content],
            system_instruction=SYSTEM_INSTRUCTION,
        )
