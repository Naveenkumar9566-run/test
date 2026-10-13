"""
Photo Analysis AI — Intelligent Image Chat Assistant
Main Streamlit application entry point.
"""
import streamlit as st

from config.settings import get_settings
from services.gemini_service import GeminiService
from services.image_service import ImageService
from services.chat_service import ChatService
from components.sidebar import render_sidebar
from components.image_upload import render_image_upload
from components.image_analysis import render_image_analysis
from components.chat_interface import render_chat_interface

# Page configuration
st.set_page_config(
    page_title="Lumen — photo intelligence",
    page_icon=":material/photo_camera:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# The CSS complements Streamlit's theme with a small set of stable app surfaces.
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at 86% 0%, rgba(217, 197, 170, 0.24), transparent 35%),
            #f7f5f0;
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    [data-testid="stAppViewContainer"] > .main .block-container {
        max-width: 1440px;
        padding-top: 2.25rem;
        padding-bottom: 4rem;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 20px;
        border-color: #e7e1d7;
        box-shadow: 0 10px 30px rgba(75, 61, 46, 0.035);
        background: #fffefb;
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid #e2ddd2;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button {
        border-radius: 12px;
    }
    [data-testid="stFileUploader"] section {
        border: 1px dashed #c7ad94;
        border-radius: 16px;
        background: #fbf9f4;
        transition: border-color 160ms ease, background 160ms ease;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: #a87552;
        background: #f7f0e7;
    }
    [data-testid="stChatMessage"] {
        border: 1px solid #ebe5dc;
        border-radius: 18px;
        padding: 0.8rem 1rem;
        background: #fffefb;
    }
    .studio-hero {
        position: relative;
        overflow: hidden;
        margin: 0 0 1.7rem;
        padding: clamp(1.5rem, 4vw, 3.2rem);
        border: 1px solid #e6d8c7;
        border-radius: 28px;
        color: #342f29;
        background:
            radial-gradient(circle at 82% 34%, rgba(208, 177, 142, 0.30), transparent 26%),
            linear-gradient(120deg, #f4f0e7 0%, #efe8dc 58%, #e9e2d6 100%);
        box-shadow: 0 18px 44px rgba(104, 78, 54, 0.09);
    }
    .studio-hero::after {
        position: absolute;
        top: -115px;
        right: -70px;
        width: 360px;
        height: 360px;
        border: 1px solid rgba(138, 93, 67, 0.15);
        border-radius: 50%;
        box-shadow: 0 0 0 34px rgba(138, 93, 67, 0.04), 0 0 0 68px rgba(138, 93, 67, 0.025);
        content: "";
        pointer-events: none;
    }
    .hero-eyebrow {
        margin-bottom: 1.1rem;
        color: #8a634c;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
    }
    .hero-title {
        max-width: 760px;
        margin: 0;
        color: #342f29;
        font-family: Georgia, "Times New Roman", serif;
        font-size: clamp(2.5rem, 5.2vw, 4.6rem);
        font-weight: 500;
        letter-spacing: -0.055em;
        line-height: 0.98;
    }
    .hero-title span {
        color: #986b4d;
        font-style: italic;
    }
    .hero-copy {
        max-width: 560px;
        margin: 1.15rem 0 0;
        color: #71675b;
        font-size: 1rem;
        line-height: 1.65;
    }
    .hero-foot {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 1.65rem;
    }
    .hero-tag {
        padding: 0.42rem 0.72rem;
        border: 1px solid #e1d2c0;
        border-radius: 999px;
        color: #76533c;
        background: rgba(255, 254, 251, 0.65);
        font-size: 0.76rem;
    }
    .section-kicker {
        margin: 0 0 0.25rem;
        color: #8a634c;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
    }
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 0.25rem 0 1.8rem;
    }
    .sidebar-mark {
        display: grid;
        width: 2.6rem;
        height: 2.6rem;
        place-items: center;
        border: 1px solid #e1d6c7;
        border-radius: 13px;
        color: #8a5d43;
        background: #e8e0d4;
        font-size: 1.2rem;
    }
    .sidebar-name {
        color: #342f29;
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        line-height: 1.15;
    }
    .sidebar-name span {
        display: block;
        margin-top: 0.2rem;
        color: #81796e;
        font-size: 0.62rem;
        font-weight: 600;
        letter-spacing: 0.16em;
    }
    .sidebar-label {
        margin: 1.1rem 0 0.55rem;
        color: #81796e;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.13em;
        text-transform: uppercase;
    }
    @media (max-width: 700px) {
        [data-testid="stAppViewContainer"] > .main .block-container {
            padding-top: 1.2rem;
        }
        .studio-hero {
            border-radius: 21px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state() -> None:
    """Initialize required session state keys if not already present."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "active_image" not in st.session_state:
        st.session_state.active_image = None
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "auto_analysis_pending" not in st.session_state:
        st.session_state.auto_analysis_pending = False
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0


def main() -> None:
    """Application main runner."""
    init_session_state()

    # Load configuration
    settings = get_settings()

    # Initialize services
    gemini_service = GeminiService(settings=settings)
    image_service = ImageService(max_size_bytes=settings.max_image_size_bytes)
    chat_service = ChatService(
        gemini_service=gemini_service,
        max_history_messages=settings.max_history_messages,
    )

    # Render sidebar
    render_sidebar(settings=settings)

    st.markdown(
        """
        <section class="studio-hero">
            <div class="hero-eyebrow">Lumen photo intelligence · visual studio 01</div>
            <h1 class="hero-title">See the story<br><span>inside every photo.</span></h1>
            <p class="hero-copy">
                A thoughtful space to explore the details, mood, and meaning in your images —
                with a little help from multimodal AI.
            </p>
            <div class="hero-foot">
                <span class="hero-tag">Gemini vision</span>
                <span class="hero-tag">Context-aware chat</span>
                <span class="hero-tag">Session-only privacy</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    # API Key warning if not configured
    if not settings.has_api_key:
        st.warning(
            "⚠️ **Gemini API Key Required:** Please configure `GEMINI_API_KEY` in your `.env` file to enable AI image analysis and chat. See README.md for setup instructions.",
            icon="🔑",
        )

    render_image_upload(image_service=image_service)

    render_image_analysis(chat_service=chat_service)

    render_chat_interface(chat_service=chat_service)


if __name__ == "__main__":
    main()
