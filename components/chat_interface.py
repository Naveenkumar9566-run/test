"""Intelligent text chat interface component."""
from typing import Optional

import streamlit as st
from services.chat_service import ChatService, ChatMessage
from services.gemini_service import GeminiServiceError
from services.image_service import ProcessedImage
from utils.validators import ValidationError


def render_chat_interface(chat_service: ChatService) -> None:
    """Render conversation history, input field, suggested questions, and answer generation."""
    st.markdown(
        """
        <div class="section-kicker">03 · Ask anything</div>
        <h3 style="margin: 0 0 0.35rem;">A conversation with your photo</h3>
        """,
        unsafe_allow_html=True,
    )

    active_img: Optional[ProcessedImage] = st.session_state.get("active_image")
    if active_img:
        st.caption(f"Photo in context · {active_img.filename} · {active_img.dimensions_str}")
    else:
        st.caption("No photo in context · Ask a general question or add a photo above.")

    # Display chronological messages
    chat_history_dicts = st.session_state.get("chat_history", [])
    
    # Message container
    msg_container = st.container()
    with msg_container:
        if not chat_history_dicts:
            with st.container(border=True):
                st.markdown("**Where should we begin?**")
                st.caption("Try a prompt like one of these, or write your own below.")
                st.markdown(
                    "- Describe the scene and the feeling it creates.\n"
                    "- Point out small details I might have missed.\n"
                    "- Tell me about the lighting, colors, and composition.\n"
                    "- Read any visible text in the image."
                )
        else:
            for idx, msg in enumerate(chat_history_dicts):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                with st.chat_message(role):
                    st.markdown(content)

    # Check if a prefill input was selected from suggested questions
    prefill = st.session_state.pop("prefill_input", None)

    # Chat input box
    prompt = st.chat_input("Ask a question about the photo or any topic...")
    
    if prefill and not prompt:
        prompt = prefill

    if prompt:
        # Prevent empty input
        stripped_prompt = prompt.strip()
        if not stripped_prompt:
            st.warning("Please enter a question.")
            return

        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(stripped_prompt)

        # Append user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": stripped_prompt,
            "has_image": bool(active_img),
        })

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing and generating response..."):
                try:
                    # Convert internal dicts to ChatMessage dataclass instances
                    history_objs = [
                        ChatMessage(role=m["role"], content=m["content"], has_image=m.get("has_image", False))
                        for m in st.session_state.chat_history[:-1]  # Exclude current question just appended
                    ]

                    response_text = st.write_stream(chat_service.stream_question(
                        question=stripped_prompt,
                        history=history_objs,
                        active_image=active_img,
                    ))

                    # Append assistant message to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response_text,
                        "has_image": bool(active_img),
                    })

                except ValidationError as val_err:
                    st.error(f"Input Error: {str(val_err)}", icon="⚠️")
                    # Pop the invalid user message if it failed validation
                    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
                        st.session_state.chat_history.pop()

                except GeminiServiceError as gem_err:
                    st.error(f"AI Service Error: {gem_err.message}", icon="⚠️")
                    if gem_err.technical_details:
                        with st.expander("Technical details"):
                            st.code(gem_err.technical_details)
                    if gem_err.retry_after_seconds is not None:
                        st.info(
                            "Retry after the wait time shown above; submitting again sooner "
                            "may use a request without restoring quota.",
                            icon="🔄",
                        )
                    elif gem_err.is_retryable:
                        st.info("This is a temporary error. You can retry your question.", icon="🔄")
                    # Keep question in history or remove depending on retry intent
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}", icon="❌")
