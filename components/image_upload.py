"""Image upload and preview component."""
import streamlit as st
from services.image_service import ImageService, ProcessedImage
from utils.validators import ValidationError


def render_image_upload(image_service: ImageService) -> None:
    """Render file uploader, image preview, metadata card, and remove button."""
    uploader_key = f"file_uploader_{st.session_state.get('uploader_key', 0)}"

    st.markdown(
        """
        <div class="section-kicker">01 · Bring a moment into focus</div>
        <h3 style="margin: 0 0 0.35rem;">Start with a photo</h3>
        <p style="margin: 0 0 1rem; color: #68766e;">
            Add an image and Lumen will find the details worth noticing.
        </p>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        uploaded_file = st.file_uploader(
            "Choose or drop a photo",
            type=["jpg", "jpeg", "png", "webp"],
            key=uploader_key,
            help=f"JPG, PNG, or WEBP · Up to {image_service.max_size_bytes // (1024 * 1024)} MB",
        )
        st.caption(
            f"JPG · PNG · WEBP  &nbsp; /  &nbsp; "
            f"Up to {image_service.max_size_bytes // (1024 * 1024)} MB · "
            "Your image is not saved to a database."
        )

    if uploaded_file is not None:
        # Check if this is a new file or already processed
        current_img: ProcessedImage = st.session_state.get("active_image")
        file_bytes = uploaded_file.getvalue()

        # If no active image, or a different file is uploaded, process it
        if current_img is None or current_img.filename != uploaded_file.name or current_img.data != file_bytes:
            try:
                processed = image_service.process_uploaded_file(
                    filename=uploaded_file.name,
                    file_bytes=file_bytes,
                )
                # Setting new image starts a fresh image session
                st.session_state.active_image = processed
                st.session_state.analysis_result = None
                st.session_state.auto_analysis_pending = True
                # Reset chat history so questions aren't confused with previous image
                st.session_state.chat_history = []
                st.toast(f"Loaded image: {uploaded_file.name}", icon="🖼️")
            except ValidationError as val_err:
                st.error(f"Image Error: {str(val_err)}", icon="❌")
                st.session_state.active_image = None
                st.session_state.auto_analysis_pending = False
                return
            except Exception as e:
                st.error(f"Unexpected error loading image: {str(e)}", icon="❌")
                st.session_state.active_image = None
                st.session_state.auto_analysis_pending = False
                return

    # If an active image exists, display preview & metadata
    active_img: ProcessedImage = st.session_state.get("active_image")
    if active_img is not None:
        st.markdown('<div class="section-kicker">Photo preview · ready to explore</div>', unsafe_allow_html=True)
        with st.container(border=True):
            col_img, col_meta = st.columns([1.35, 1], vertical_alignment="center")

            with col_img:
                st.image(
                    active_img.to_pil_image(),
                    caption=active_img.filename,
                    width="stretch",
                    alt=f"Uploaded photo: {active_img.filename}",
                )

            with col_meta:
                st.markdown("#### Image at a glance")
                st.caption("The photo is ready for analysis.")
                st.markdown(f"**File**  \n{active_img.filename}")
                st.markdown(f"**Dimensions**  \n{active_img.dimensions_str}")
                st.markdown(f"**Format · size**  \n{active_img.format} · {active_img.size_str}")
                st.caption(f"Content type · {active_img.mime_type}")

                if st.button(
                    "Remove photo",
                    key="btn_remove_image",
                    icon=":material/delete_outline:",
                    width="stretch",
                ):
                    st.session_state.active_image = None
                    st.session_state.analysis_result = None
                    st.session_state.auto_analysis_pending = False
                    st.session_state.chat_history = []
                    st.session_state.uploader_key = st.session_state.get("uploader_key", 0) + 1
                    st.rerun()
    else:
        st.caption("No photo yet — you can still use the chat below for general questions.")
