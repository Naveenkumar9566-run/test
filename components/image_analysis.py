"""Image analysis actions and structured result view."""
import streamlit as st
from services.chat_service import ChatService
from services.gemini_service import GeminiServiceError
from services.image_service import ProcessedImage
from utils.prompts import (
    IMAGE_ANALYSIS_PROMPT,
    IMAGE_SUMMARY_PROMPT,
    DETAILED_ANALYSIS_PROMPT,
    EXPLAIN_SIMPLY_PROMPT,
)


def render_image_analysis(chat_service: ChatService) -> None:
    """Render one-click analysis actions and structured analysis display."""
    active_img: ProcessedImage = st.session_state.get("active_image")
    if active_img is None:
        return

    st.markdown(
        """
        <div class="section-kicker">02 · Look closer</div>
        <h3 style="margin: 0 0 0.35rem;">Explore your image</h3>
        <p style="margin: 0 0 1rem; color: #68766e;">
            Choose a lens to get a quick read, a structured report, or a deeper visual breakdown.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Action buttons row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        btn_full = st.button(
            "Full analysis",
            icon=":material/analytics:",
            width="stretch",
            help="Full structured analysis",
        )
    with col2:
        btn_summary = st.button(
            "Quick summary",
            icon=":material/bolt:",
            width="stretch",
            type="primary",
            help="Concise 2-3 sentence overview",
        )
    with col3:
        btn_detailed = st.button(
            "Deep breakdown",
            icon=":material/center_focus_strong:",
            width="stretch",
            help="Detailed visual and spatial analysis",
        )
    with col4:
        btn_simple = st.button(
            "Explain simply",
            icon=":material/lightbulb:",
            width="stretch",
            help="Beginner-friendly explanation",
        )

    selected_prompt = None
    action_label = ""
    if st.session_state.get("auto_analysis_pending", False):
        st.session_state.auto_analysis_pending = False
        selected_prompt = IMAGE_SUMMARY_PROMPT
        action_label = "Quick Image Summary"
    elif btn_full:
        selected_prompt = IMAGE_ANALYSIS_PROMPT
        action_label = "Structured Image Analysis"
    elif btn_summary:
        selected_prompt = IMAGE_SUMMARY_PROMPT
        action_label = "Quick Image Summary"
    elif btn_detailed:
        selected_prompt = DETAILED_ANALYSIS_PROMPT
        action_label = "Deep Visual Breakdown"
    elif btn_simple:
        selected_prompt = EXPLAIN_SIMPLY_PROMPT
        action_label = "Simple Explanation"

    # Execute requested analysis action
    if selected_prompt:
        with st.container(border=True):
            st.markdown(f"#### {action_label}")
            with st.spinner(f"Generating {action_label}..."):
                try:
                    result_text = st.write_stream(chat_service.stream_image_analysis(
                        active_image=active_img,
                        prompt=selected_prompt,
                    ))
                    st.session_state.analysis_result = {
                        "title": action_label,
                        "content": result_text,
                    }
                except GeminiServiceError as g_err:
                    st.error(f"Analysis Failed: {g_err.message}", icon="⚠️")
                    if g_err.technical_details:
                        with st.expander("Technical details"):
                            st.code(g_err.technical_details)
                    if g_err.retry_after_seconds is not None:
                        st.info(
                            "Retry after the wait time shown above; submitting again sooner "
                            "may use a request without restoring quota.",
                            icon="🔄",
                        )
                    elif g_err.is_retryable:
                        st.info("The image is still loaded. Retry with Quick Summary or ask a question in chat.", icon="🔄")
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}", icon="❌")
    elif st.session_state.get("analysis_result"):
        analysis_data = st.session_state["analysis_result"]
        with st.container(border=True):
            st.markdown(f"#### {analysis_data['title']}")
            st.markdown(analysis_data["content"])

    # Render context action if an analysis result exists
    analysis_data = st.session_state.get("analysis_result")
    if analysis_data:
        col_copy, _ = st.columns([1, 4])
        with col_copy:
            if st.button(
                "Add to conversation",
                key="btn_add_to_chat",
                icon=":material/add_comment:",
                help="Include this analysis in chat history",
            ):
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"**[{analysis_data['title']}]**\n\n{analysis_data['content']}",
                    "has_image": True,
                })
                st.success("Added to conversation context!")
                st.rerun()
