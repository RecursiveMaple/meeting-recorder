"""
Session store for persisting transcription and summary data across WebSocket connections.

This module provides a global session store that can be used to:
1. Export transcription data as JSONL
2. Retry summary generation for specific segments
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class SegmentRecord:
    """A record of a transcribed segment with its summary."""

    id: int  # Segment ID within the session
    session_id: str
    start: Optional[float] = None
    end: Optional[float] = None
    text: Optional[str] = None
    speaker: Optional[str] = None
    summary: Optional[str] = None
    summary_status: str = "pending"  # pending, processing, ready, error, timeout
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "speaker": self.speaker,
            "summary": self.summary,
            "summary_status": self.summary_status,
            "created_at": self.created_at,
        }


@dataclass
class Session:
    """A transcription session with its segments."""

    id: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    segments: Dict[int, SegmentRecord] = field(default_factory=dict)
    segment_counter: int = 0
    summary_template: str = "meeting_minutes"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "created_at": self.created_at,
            "segment_count": len(self.segments),
            "summary_template": self.summary_template,
        }


class SessionStore:
    """
    Global session store for transcription data.

    This is a singleton that persists across WebSocket connections,
    allowing for JSONL export and summary retry functionality.
    """

    _instance: Optional["SessionStore"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __new__(cls) -> "SessionStore":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._sessions: Dict[str, Session] = {}
            cls._instance._active_processors: Dict[str, Any] = {}  # session_id -> AudioProcessor
        return cls._instance

    @classmethod
    def get_instance(cls) -> "SessionStore":
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def create_session(self, summary_template: str = "meeting_minutes") -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid4())[:8]  # Short ID for readability
        session = Session(id=session_id, summary_template=summary_template)
        self._sessions[session_id] = session
        logger.debug(f"Created session {session_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def add_segment(
        self,
        session_id: str,
        text: str,
        start: Optional[float] = None,
        end: Optional[float] = None,
        speaker: Optional[str] = None,
    ) -> int:
        """Add a segment to a session and return its ID."""
        session = self._sessions.get(session_id)
        if not session:
            logger.warning(f"Session {session_id} not found")
            return -1

        session.segment_counter += 1
        segment_id = session.segment_counter

        segment = SegmentRecord(
            id=segment_id,
            session_id=session_id,
            text=text,
            start=start,
            end=end,
            speaker=speaker,
        )
        session.segments[segment_id] = segment
        logger.debug(f"Added segment {segment_id} to session {session_id}")
        return segment_id

    def update_segment_summary(
        self,
        session_id: str,
        segment_id: int,
        summary: Optional[str] = None,
        status: Optional[str] = None,
    ) -> bool:
        """Update the summary for a segment."""
        session = self._sessions.get(session_id)
        if not session:
            return False

        segment = session.segments.get(segment_id)
        if not segment:
            return False

        if summary is not None:
            segment.summary = summary
        if status is not None:
            segment.summary_status = status

        return True

    def get_segment(self, session_id: str, segment_id: int) -> Optional[SegmentRecord]:
        """Get a specific segment from a session."""
        session = self._sessions.get(session_id)
        if not session:
            return None
        return session.segments.get(segment_id)

    def get_all_sessions(self) -> List[Session]:
        """Get all sessions."""
        return list(self._sessions.values())

    def get_session_segments(self, session_id: str) -> List[SegmentRecord]:
        """Get all segments for a session."""
        session = self._sessions.get(session_id)
        if not session:
            return []
        return list(session.segments.values())

    def to_jsonl(self, session_id: Optional[str] = None) -> str:
        """
        Export sessions as JSONL format.

        Each line is a JSON object:
        - For segments: {"type": "segment", ...}
        - For summaries: {"type": "summary", ...}

        If session_id is provided, only export that session.
        Otherwise, export all sessions.
        """
        import json

        lines = []

        sessions = (
            [self._sessions[session_id]]
            if session_id and session_id in self._sessions
            else list(self._sessions.values())
        )

        for session in sessions:
            # Session metadata line
            lines.append(
                json.dumps(
                    {
                        "type": "session",
                        "session_id": session.id,
                        "created_at": session.created_at,
                        "summary_template": session.summary_template,
                    }
                )
            )

            for segment in session.segments.values():
                # Segment line
                lines.append(
                    json.dumps(
                        {
                            "type": "segment",
                            "session_id": session.id,
                            "segment_id": segment.id,
                            "start": segment.start,
                            "end": segment.end,
                            "text": segment.text,
                            "speaker": segment.speaker,
                            "created_at": segment.created_at,
                        }
                    )
                )

                # Summary line (if available)
                if segment.summary:
                    lines.append(
                        json.dumps(
                            {
                                "type": "summary",
                                "session_id": session.id,
                                "segment_id": segment.id,
                                "summary": segment.summary,
                                "status": segment.summary_status,
                            }
                        )
                    )

        return "\n".join(lines)

    def register_processor(self, session_id: str, processor: Any) -> None:
        """Register an AudioProcessor for a session (for retry functionality)."""
        self._active_processors[session_id] = processor

    def unregister_processor(self, session_id: str) -> None:
        """Unregister an AudioProcessor when the session ends."""
        self._active_processors.pop(session_id, None)

    def get_processor(self, session_id: str) -> Optional[Any]:
        """Get the AudioProcessor for a session."""
        return self._active_processors.get(session_id)

    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """Remove sessions older than max_age_hours."""
        from datetime import datetime, timedelta

        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        removed = 0

        for session_id, session in list(self._sessions.items()):
            created = datetime.fromisoformat(session.created_at)
            if created < cutoff:
                del self._sessions[session_id]
                self._active_processors.pop(session_id, None)
                removed += 1

        if removed:
            logger.info(f"Cleaned up {removed} old sessions")
        return removed


# Global instance
session_store = SessionStore.get_instance()
