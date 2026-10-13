"""Image processing and validation service."""
import io
from dataclasses import dataclass
from typing import Tuple, Optional
from PIL import Image
from utils.validators import validate_image_bytes, ValidationError


@dataclass
class ProcessedImage:
    """Represents a validated in-memory image."""
    filename: str
    data: bytes
    mime_type: str
    width: int
    height: int
    format: str

    @property
    def dimensions_str(self) -> str:
        return f"{self.width} × {self.height} px"

    @property
    def size_mb(self) -> float:
        return len(self.data) / (1024 * 1024)

    @property
    def size_str(self) -> str:
        return f"{self.size_mb:.2f} MB"

    def to_pil_image(self) -> Image.Image:
        """Return PIL Image object for Streamlit display without saving to disk."""
        return Image.open(io.BytesIO(self.data))


class ImageService:
    """Handles image validation, metadata extraction, and in-memory representation."""

    def __init__(self, max_size_bytes: int = 10 * 1024 * 1024):
        self.max_size_bytes = max_size_bytes

    def process_uploaded_file(self, filename: str, file_bytes: bytes) -> ProcessedImage:
        """
        Validate and package uploaded image file bytes.
        
        Args:
            filename: Original uploaded filename.
            file_bytes: Raw bytes from the uploaded file.
            
        Returns:
            ProcessedImage with verified metadata and bytes.
            
        Raises:
            ValidationError: If validation fails.
        """
        mime_type, (width, height) = validate_image_bytes(file_bytes, self.max_size_bytes)
        
        # Derive canonical format name (JPEG, PNG, WEBP)
        format_mapping = {
            "image/jpeg": "JPEG",
            "image/png": "PNG",
            "image/webp": "WEBP",
        }
        fmt = format_mapping.get(mime_type, "UNKNOWN")

        return ProcessedImage(
            filename=filename,
            data=file_bytes,
            mime_type=mime_type,
            width=width,
            height=height,
            format=fmt,
        )
