"""Kim Han-Nom Sentence Segmentation API package for processing folders of text files."""

from .api_client import (
    APIConfig,
    APIError,
    HanNomSegmentationClient,
    RateLimitError,
    SegmentationAPIConfig,
)
from .segmentation_runner import HanNomSegmenter
from .utils import concat_chunk_results, split_text_into_line_chunks

__all__ = [
    "APIConfig",
    "APIError",
    "HanNomSegmentationClient",
    "HanNomSegmenter",
    "RateLimitError",
    "SegmentationAPIConfig",
    "concat_chunk_results",
    "split_text_into_line_chunks",
]
