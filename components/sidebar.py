"""Sidebar component for session controls, settings, and info."""
import streamlit as st
from config.settings import Settings
from utils.prompts import SUGGESTED_QUESTIONS


def render_sidebar(settings: Settings) -> None:
    """Render application sidebar with session controls, privacy info, and examples."""
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-mark">L</div>
                <div class="sidebar-name">Lumen<span>PHOTO INTELLIGENCE</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-label">Your workspace</div>', unsafe_allow_html=True)
        if st.button(
            "New analysis",
            icon=":material/add_photo_alternate:",
            width="stretch",
            help="Clear the current photo and start a fresh session",
        ):
            st.session_state.active_image = None
            st.session_state.analysis_result = None
            st.session_state.auto_analysis_pending = False
            st.session_state.chat_history = []
            st.session_state.uploader_key = st.session_state.get("uploader_key", 0) + 1
            st.rerun()

        if st.button(
            "Clear conversation",
            icon=":material/delete_sweep:",
            width="stretch",
            help="Clear the conversation while keeping the active photo",
        ):
            st.session_state.chat_history = []
            st.rerun()

        st.markdown('<div class="sidebar-label">Connection</div>', unsafe_allow_html=True)
        if settings.has_api_key:
            st.markdown(":green-badge[Gemini connected]")
        else:
            st.markdown(":orange-badge[API key needed]")
            st.caption("Add `GEMINI_API_KEY` to your `.env` file to enable analysis.")

        active_image = st.session_state.get("active_image")
        if active_image:
            st.caption(f"Photo in focus · {active_image.filename}")
        else:
            st.caption("Ready for your next photo")

        st.markdown('<div class="sidebar-label">Try asking</div>', unsafe_allow_html=True)
        for q in SUGGESTED_QUESTIONS:
            if st.button(q, key=f"sugg_{hash(q)}", width="stretch"):
                st.session_state["prefill_input"] = q
                st.rerun()

        with st.expander("Model & limits", icon=":material/tune:"):
            st.caption(f"Model · {settings.gemini_model}")
            st.caption(f"Temperature · {settings.llm_temperature:g}")
            st.caption(f"Maximum photo · {settings.max_image_size_mb} MB")
            st.caption(f"Conversation context · {settings.max_history_messages} turns")

        with st.expander("Privacy", icon=":material/lock:"):
            st.caption(
                "Photos and conversation are kept in memory for this session only. "
                "Photos are sent to the configured Google Gemini API for inference."
            )
