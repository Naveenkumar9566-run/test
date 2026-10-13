"""Unit tests for validation functions and image service."""
import io
import pytest
from PIL import Image

from utils.validators import validate_image_bytes, validate_user_prompt, ValidationError
from services.image_service import ImageService, ProcessedImage


def create_test_image_bytes(format_name: str = "JPEG", size=(100, 100), color=(255, 0, 0)) -> bytes:
    """Helper to generate in-memory test image bytes."""
    buf = io.BytesIO()
    # JPEG does not support RGBA, so use RGB
    mode = "RGB" if format_name.upper() in ("JPEG", "JPG") else "RGBA"
    img = Image.new(mode, size, color=color)
    img.save(buf, format=format_name)
    return buf.getvalue()


class TestValidators:
    """Tests for input validation."""

    def test_valid_jpeg_format(self):
        data = create_test_image_bytes(format_name="JPEG")
        mime_type, dims = validate_image_bytes(data, max_size_bytes=10 * 1024 * 1024)
        assert mime_type == "image/jpeg"
        assert dims == (100, 100)

    def test_valid_png_format(self):
        data = create_test_image_bytes(format_name="PNG")
        mime_type, dims = validate_image_bytes(data, max_size_bytes=10 * 1024 * 1024)
        assert mime_type == "image/png"
        assert dims == (100, 100)

    def test_valid_webp_format(self):
        data = create_test_image_bytes(format_name="WEBP")
        mime_type, dims = validate_image_bytes(data, max_size_bytes=10 * 1024 * 1024)
        assert mime_type == "image/webp"
        assert dims == (100, 100)

    def test_unsupported_image_format(self):
        # GIF is not in the supported formats list (JPG, JPEG, PNG, WEBP)
        buf = io.BytesIO()
        img = Image.new("RGB", (50, 50), color="blue")
        img.save(buf, format="GIF")
        data = buf.getvalue()

        with pytest.raises(ValidationError, match="Unsupported image format"):
            validate_image_bytes(data, max_size_bytes=10 * 1024 * 1024)

    def test_oversized_upload(self):
        data = create_test_image_bytes(format_name="PNG")
        # Set limit lower than actual byte length
        max_bytes = len(data) - 1
        with pytest.raises(ValidationError, match="exceeds maximum allowed limit"):
            validate_image_bytes(data, max_size_bytes=max_bytes)

    def test_empty_image_bytes(self):
        with pytest.raises(ValidationError, match="image file is empty"):
            validate_image_bytes(b"", max_size_bytes=1024)

    def test_corrupted_image_data(self):
        corrupt_data = b"NOT_AN_IMAGE_HEADER_RANDOM_GARBAGE_BYTES_12345"
        with pytest.raises(ValidationError, match="Invalid or corrupted"):
            validate_image_bytes(corrupt_data, max_size_bytes=1024 * 1024)

    def test_empty_user_prompt(self):
        with pytest.raises(ValidationError, match="Question cannot be empty"):
            validate_user_prompt("")

        with pytest.raises(ValidationError, match="Question cannot be empty"):
            validate_user_prompt("   \n\t  ")

    def test_valid_user_prompt(self):
        cleaned = validate_user_prompt("  What is in this photo?   ")
        assert cleaned == "What is in this photo?"


class TestImageService:
    """Tests for ImageService wrapper."""

    def test_process_uploaded_file_success(self):
        service = ImageService(max_size_bytes=5 * 1024 * 1024)
        data = create_test_image_bytes("JPEG", size=(200, 150))
        processed = service.process_uploaded_file("photo.jpg", data)

        assert isinstance(processed, ProcessedImage)
        assert processed.filename == "photo.jpg"
        assert processed.width == 200
        assert processed.height == 150
        assert processed.format == "JPEG"
        assert processed.mime_type == "image/jpeg"
        assert processed.dimensions_str == "200 × 150 px"
        assert processed.size_mb > 0
