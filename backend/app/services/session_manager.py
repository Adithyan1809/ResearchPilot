"""Session and conversational memory management service.

Maintains multi-turn chat history for research sessions, providing thread-safe
storage, bounded window retention, and conversational context formatting.
"""

from datetime import datetime, timezone
import threading
from typing import Dict, List, Optional
import uuid

from backend.app.models.schemas import ChatMessage, SessionInfo
from backend.app.utils.logger import setup_logger

logger = setup_logger(__name__)


def _utc_now() -> datetime:
    """Return timezone-aware current UTC time."""
    return datetime.now(timezone.utc)


class SessionManagerService:
    """Thread-safe in-memory session manager for conversational multi-turn RAG."""

    def __init__(self, max_history_messages: int = 20) -> None:
        """Initialise the session manager.

        Parameters
        ----------
        max_history_messages:
            Maximum number of messages (user + assistant) to retain per session.
            Default is 20 (10 conversational turns).
        """
        self._max_messages = max_history_messages
        self._sessions: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        logger.info(
            "SessionManagerService initialised (max_history_messages=%d)",
            max_history_messages,
        )

    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new conversational session.

        Parameters
        ----------
        session_id:
            Optional custom session ID. If not provided, a UUID4 string is generated.

        Returns
        -------
        str
            The unique session identifier.
        """
        sid = session_id or str(uuid.uuid4())
        now = _utc_now()
        with self._lock:
            if sid not in self._sessions:
                self._sessions[sid] = {
                    "session_id": sid,
                    "messages": [],
                    "created_at": now,
                    "updated_at": now,
                }
                logger.info("Created new session: %s", sid)
        return sid

    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        """Retrieve existing session or create a new one."""
        if not session_id:
            return self.create_session()
        with self._lock:
            if session_id not in self._sessions:
                now = _utc_now()
                self._sessions[session_id] = {
                    "session_id": session_id,
                    "messages": [],
                    "created_at": now,
                    "updated_at": now,
                }
                logger.info("Initialized session on first use: %s", session_id)
        return session_id

    def get_history(self, session_id: str) -> List[ChatMessage]:
        """Retrieve all recorded messages for a session.

        Parameters
        ----------
        session_id:
            The session identifier.

        Returns
        -------
        List[ChatMessage]
            Chronological list of chat messages.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return []
            return list(session["messages"])

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Append a message to session history with bounded window trimming."""
        now = _utc_now()
        message = ChatMessage(role=role, content=content, timestamp=now)
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = {
                    "session_id": session_id,
                    "messages": [],
                    "created_at": now,
                    "updated_at": now,
                }
            session = self._sessions[session_id]
            session["messages"].append(message)
            session["updated_at"] = now

            # Trim to max window
            if len(session["messages"]) > self._max_messages:
                session["messages"] = session["messages"][-self._max_messages :]

    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        """Record a full conversation turn (question + answer)."""
        self.add_message(session_id, role="user", content=question)
        self.add_message(session_id, role="assistant", content=answer)

    def clear_session(self, session_id: str) -> bool:
        """Clear conversation history for a session."""
        with self._lock:
            if session_id in self._sessions:
                now = _utc_now()
                self._sessions[session_id]["messages"] = []
                self._sessions[session_id]["updated_at"] = now
                logger.info("Cleared history for session: %s", session_id)
                return True
            return False

    def delete_session(self, session_id: str) -> bool:
        """Permanently remove a session."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info("Deleted session: %s", session_id)
                return True
            return False

    def list_sessions(self) -> List[SessionInfo]:
        """Return a list of all active sessions and their metadata."""
        with self._lock:
            results = []
            for sid, data in self._sessions.items():
                results.append(
                    SessionInfo(
                        session_id=sid,
                        message_count=len(data["messages"]),
                        created_at=data["created_at"],
                        updated_at=data["updated_at"],
                    )
                )
            # Sort by last updated descending
            results.sort(key=lambda s: s.updated_at, reverse=True)
            return results

    def get_session_info(self, session_id: str) -> Optional[SessionInfo]:
        """Return metadata for a single session if present."""
        with self._lock:
            data = self._sessions.get(session_id)
            if not data:
                return None
            return SessionInfo(
                session_id=session_id,
                message_count=len(data["messages"]),
                created_at=data["created_at"],
                updated_at=data["updated_at"],
            )
