"""Capture room (PIVOT_ROADMAP §B.6/§B.18).

The Capture room turns a single camera frame into a German lesson.
Public surface stays small so the router and the demo are short.
"""

from backend.capture.service import (
    CaptureRequest,
    CaptureResult,
    list_recent_captures,
    process_capture,
)

__all__ = [
    "CaptureRequest",
    "CaptureResult",
    "list_recent_captures",
    "process_capture",
]
