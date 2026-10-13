"""Tests for text-only and image-context chat submission."""
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from services.gemini_service import GeminiService


@pytest.mark.parametrize("with_image", [False, True])
def test_chat_submission_returns_answer(with_image):
    image_setup = """
from services.image_service import ProcessedImage
st.session_state["active_image"] = ProcessedImage(
    filename="photo.jpg",
    data=b"test-image",
    mime_type="image/jpeg",
    width=10,
    height=10,
    format="JPEG",
)
""" if with_image else 'st.session_state["active_image"] = None'

    app_script = f"""
import streamlit as st

from config.settings import Settings
from services.chat_service import ChatService
from services.gemini_service import GeminiService
from components.chat_interface import render_chat_interface

st.session_state["chat_history"] = []
{image_setup}
settings = Settings(
    gemini_api_key="test",
    gemini_model="gemini-3.8-flash",
    llm_temperature=0.1,
    max_image_size_mb=10,
    max_history_messages=12,
)
render_chat_interface(ChatService(GeminiService(settings)))
"""

    with patch.object(
        GeminiService,
        "stream_content",
        return_value=iter(["Mocked ", "AI answer"]),
    ) as stream_content:
        app = AppTest.from_string(app_script).run()
        assert not app.exception
        app.chat_input[0].set_value("What is in this photo?").run()

    assert not app.exception
    assert any("Mocked AI answer" in block.value for block in app.markdown)
    stream_content.assert_called_once()
    content = stream_content.call_args.kwargs["contents"]
    latest_parts = content[-1].parts
    assert latest_parts[-1].text == "What is in this photo?"
    assert (latest_parts[0].inline_data is not None) is with_image


def test_chat_with_prior_analysis_context():
    app_script = """
import streamlit as st
from config.settings import Settings
from services.chat_service import ChatService
from services.gemini_service import GeminiService
from services.image_service import ProcessedImage
from components.chat_interface import render_chat_interface

st.session_state["active_image"] = ProcessedImage(
    filename="photo.jpg",
    data=b"test-bytes",
    mime_type="image/jpeg",
    width=10,
    height=10,
    format="JPEG",
)
st.session_state["chat_history"] = [
    {"role": "assistant", "content": "**[Quick Summary]** A red flower", "has_image": True}
]
settings = Settings(
    gemini_api_key="test",
    gemini_model="gemini-3.8-flash",
    llm_temperature=0.1,
    max_image_size_mb=10,
    max_history_messages=12,
)
render_chat_interface(ChatService(GeminiService(settings)))
"""
    with patch.object(
        GeminiService,
        "stream_content",
        return_value=iter(["Flower ", "details"]),
    ) as stream_content:
        app = AppTest.from_string(app_script).run()
        app.chat_input[0].set_value("What species is it?").run()

    assert not app.exception
    stream_content.assert_called_once()
    contents = stream_content.call_args.kwargs["contents"]
    # First turn must be user to satisfy Gemini API requirements
    assert contents[0].role == "user"
    assert contents[0].parts[0].inline_data is not None
    # Second turn must be model with the prior summary
    assert contents[1].role == "model"
    # Third turn must be user with the question
    assert contents[2].role == "user"
    assert contents[2].parts[0].text == "What species is it?"
