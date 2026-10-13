"""Tests for automatic analysis after a newly uploaded image."""
import io
from unittest.mock import patch

from PIL import Image
from streamlit.testing.v1 import AppTest

from services.gemini_service import GeminiService, GeminiServiceError
from utils.prompts import IMAGE_SUMMARY_PROMPT


def test_pending_upload_runs_one_quick_summary():
    image_buffer = io.BytesIO()
    Image.new("RGB", (10, 10), color="red").save(image_buffer, format="PNG")
    image_bytes = image_buffer.getvalue()
    app_script = f"""
import streamlit as st

from config.settings import Settings
from services.chat_service import ChatService
from services.gemini_service import GeminiService
from services.image_service import ImageService
from components.image_upload import render_image_upload
from components.image_analysis import render_image_analysis

render_image_upload(ImageService())
gemini = GeminiService(Settings(
    gemini_api_key="test",
    gemini_model="gemini-3.8-flash",
    llm_temperature=0.1,
    max_image_size_mb=10,
    max_history_messages=12,
))
render_image_analysis(ChatService(gemini))
"""
    uploaded_file = type(
        "UploadedFile",
        (),
        {"name": "photo.png", "getvalue": lambda self: image_bytes},
    )()

    with (
        patch("streamlit.file_uploader", return_value=uploaded_file),
        patch.object(
            GeminiService,
            "stream_content",
            return_value=iter(["Photo ", "summary"]),
        ) as stream_content,
    ):
        app = AppTest.from_string(app_script).run(timeout=10)

    assert not app.exception
    assert app.session_state["auto_analysis_pending"] is False
    assert app.session_state["active_image"].filename == "photo.png"
    stream_content.assert_called_once()
    content = stream_content.call_args.kwargs["contents"][0]
    assert content.parts[0].inline_data is not None
    assert content.parts[1].text == IMAGE_SUMMARY_PROMPT
    assert any("Photo summary" in block.value for block in app.markdown)


def test_model_error_is_shown_in_analysis_screen():
    app_script = """
import streamlit as st

from config.settings import Settings
from services.chat_service import ChatService
from services.gemini_service import GeminiService
from services.image_service import ProcessedImage
from components.image_analysis import render_image_analysis

st.session_state["active_image"] = ProcessedImage(
    filename="photo.jpg",
    data=b"image-bytes",
    mime_type="image/jpeg",
    width=10,
    height=10,
    format="JPEG",
)
gemini = GeminiService(Settings(
    gemini_api_key="test",
    gemini_model="gemini-3.8-flash",
    llm_temperature=0.1,
    max_image_size_mb=10,
    max_history_messages=12,
))
render_image_analysis(ChatService(gemini))
"""

    with patch.object(
        GeminiService,
        "stream_content",
        side_effect=GeminiServiceError(
            "Gemini model 'gemini-2.5-flash' is unavailable. "
            "Google's error recommends gemini-3.8-flash."
        ),
    ):
        app = AppTest.from_string(app_script).run()
        app.button[1].click().run()

    assert not app.exception
    assert any("gemini-3.8-flash" in item.value for item in app.error)
