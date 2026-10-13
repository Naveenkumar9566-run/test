"""Validation helpers for images and chat inputs."""
import io
from typing import Tuple
from PIL import Image, UnidentifiedImageError

SUPPORTED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
SUPPORTED_MIME_TYPES = {
    "image/jpeg": "JPEG",
    "image/jpg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


def validate_image_bytes(image_bytes: bytes, max_size_bytes: int) -> Tuple[str, Tuple[int, int]]:
    """
    Validate raw image bytes for size, format, and integrity.
    
    Returns:
        Tuple of (normalized_mime_type, (width, height))
        
    Raises:
        ValidationError: If the file is empty, too large, unsupported, or corrupt.
    """
    if not image_bytes:
        raise ValidationError("The uploaded image file is empty.")

    if len(image_bytes) > max_size_bytes:
        max_mb = max_size_bytes / (1024 * 1024)
        actual_mb = len(image_bytes) / (1024 * 1024)
        raise ValidationError(
            f"Image file size ({actual_mb:.2f} MB) exceeds maximum allowed limit of {max_mb:.1f} MB."
        )

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.verify()
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        raise ValidationError(f"Invalid or corrupted image file: {str(e)}")

    # Re-open after verify to inspect dimensions and format safely
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            format_name = (img.format or "").upper()
            dimensions = (img.width, img.height)
    except Exception as e:
        raise ValidationError(f"Unable to read image dimensions: {str(e)}")

    format_to_mime = {
        "JPEG": "image/jpeg",
        "JPG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
    }

    if format_name not in format_to_mime:
        raise ValidationError(
            f"Unsupported image format '{format_name}'. Supported formats: JPEG, PNG, WEBP."
        )

    return format_to_mime[format_name], dimensions


def validate_user_prompt(prompt: str) -> str:
    """
    Validate and clean user chat prompt.
    
    Returns:
        Stripped prompt string.
        
    Raises:
        ValidationError: If prompt is empty or pure whitespace.
    """
    if not prompt or not prompt.strip():
        raise ValidationError("Question cannot be empty.")
    return prompt.strip()
