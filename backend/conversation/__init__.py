"""Konversation room (PIVOT_ROADMAP §B.5).

Live voice loop: browser MediaRecorder → backend STT → Gemma 4 → Piper.
Public surface is intentionally tiny so the router and the milestone
demo both stay one-line callers.
"""

from backend.conversation.runner import (
    ConversationTurnRequest,
    ConversationTurnResult,
    TranscriptEntry,
    list_recent_transcripts,
    run_turn,
    stream_turn,
)

__all__ = [
    "ConversationTurnRequest",
    "ConversationTurnResult",
    "TranscriptEntry",
    "list_recent_transcripts",
    "run_turn",
    "stream_turn",
]
