"""Unit tests for prompt construction, history limits, and ChatService."""
import pytest
from unittest.mock import MagicMock
from google.genai import types

from config.settings import Settings
from services.chat_service import ChatService, ChatMessage
from services.image_service import ProcessedImage
from services.gemini_service import GeminiService
from utils.prompts import SYSTEM_INSTRUCTION, IMAGE_ANALYSIS_PROMPT
from utils.validators import ValidationError


@pytest.fixture
def mock_gemini_service():
    """Create a mock GeminiService to avoid real API calls and costs."""
    settings = Settings(
        gemini_api_key="test_api_key",
        gemini_model="gemini-3.8-flash",
        llm_temperature=0.1,
        max_image_size_mb=10,
        max_history_messages=4,
    )
    service = GeminiService(settings=settings)
    service.generate_content = MagicMock(return_value="Mocked AI response text")
    return service


@pytest.fixture
def sample_processed_image():
    """Create a sample ProcessedImage for testing."""
    return ProcessedImage(
        filename="scenery.jpg",
        data=b"\xff\xd8\xff\xe0testjpegdata",
        mime_type="image/jpeg",
        width=800,
        height=600,
        format="JPEG",
    )


class TestChatService:
    """Test ChatService logic and history bounds."""

    def test_empty_question_raises_validation_error(self, mock_gemini_service):
        chat_service = ChatService(gemini_service=mock_gemini_service, max_history_messages=4)
        with pytest.raises(ValidationError):
            chat_service.ask_question("   ", history=[])

    def test_text_only_question_without_image(self, mock_gemini_service):
        chat_service = ChatService(gemini_service=mock_gemini_service, max_history_messages=4)
        resp = chat_service.ask_question("What is machine learning?", history=[])

        assert resp == "Mocked AI response text"
        mock_gemini_service.generate_content.assert_called_once()
        call_args = mock_gemini_service.generate_content.call_args[1]
        contents = call_args["contents"]
        assert len(contents) == 1
        assert contents[0].role == "user"
        # Only text part, no inline_data
        assert contents[0].parts[0].text == "What is machine learning?"
        assert contents[0].parts[0].inline_data is None

    def test_question_with_active_image(self, mock_gemini_service, sample_processed_image):
        chat_service = ChatService(gemini_service=mock_gemini_service, max_history_messages=4)
        resp = chat_service.ask_question(
            "What color is the car?",
            history=[],
            active_image=sample_processed_image,
        )

        assert resp == "Mocked AI response text"
        call_args = mock_gemini_service.generate_content.call_args[1]
        contents = call_args["contents"]
        assert len(contents) == 1
        # The single user turn should contain image part and question part
        parts = contents[0].parts
        assert len(parts) == 2
        assert parts[0].inline_data is not None
        assert parts[0].inline_data.mime_type == "image/jpeg"
        assert parts[1].text == "What color is the car?"

    def test_multi_turn_history_limits(self, mock_gemini_service, sample_processed_image):
        # Configure max_history_messages to 4 turns
        chat_service = ChatService(gemini_service=mock_gemini_service, max_history_messages=4)

        long_history = [
            ChatMessage(role="user", content="Turn 1"),
            ChatMessage(role="assistant", content="Turn 2"),
            ChatMessage(role="user", content="Turn 3"),
            ChatMessage(role="assistant", content="Turn 4"),
            ChatMessage(role="user", content="Turn 5"),
            ChatMessage(role="assistant", content="Turn 6"),
        ]

        contents = chat_service.build_contents(
            history=long_history,
            latest_question="Turn 7",
            active_image=sample_processed_image,
        )

        # 4 trimmed turns + 1 latest turn = 5 total Content items
        assert len(contents) == 5
        # The trimmed history should start from Turn 3
        assert contents[0].parts[-1].text == "Turn 3"
        assert contents[1].parts[-1].text == "Turn 4"
        assert contents[2].parts[-1].text == "Turn 5"
        assert contents[3].parts[-1].text == "Turn 6"
        assert contents[4].parts[-1].text == "Turn 7"

        # The image part should be attached to the first user turn in the context (Turn 3)
        assert contents[0].parts[0].inline_data is not None

    def test_analyze_image_prompt(self, mock_gemini_service, sample_processed_image):
        chat_service = ChatService(gemini_service=mock_gemini_service, max_history_messages=4)
        result = chat_service.analyze_image(
            active_image=sample_processed_image,
            prompt=IMAGE_ANALYSIS_PROMPT,
        )
        assert result == "Mocked AI response text"
        mock_gemini_service.generate_content.assert_called_once()
        call_args = mock_gemini_service.generate_content.call_args[1]
        assert call_args["system_instruction"] == SYSTEM_INSTRUCTION
