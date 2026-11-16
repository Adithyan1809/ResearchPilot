"""Unit tests for SessionManagerService (conversational memory)."""

from datetime import datetime
import pytest

from backend.app.models.schemas import ChatMessage
from backend.app.services.session_manager import SessionManagerService


def test_session_creation_and_retrieval():
    """Test generating session IDs and retrieving empty history."""
    sm = SessionManagerService(max_history_messages=10)
    sid = sm.create_session()
    assert sid is not None
    assert isinstance(sid, str)
    assert len(sid) > 0

    history = sm.get_history(sid)
    assert history == []

    # Custom session ID
    custom_sid = sm.get_or_create_session("my-custom-session")
    assert custom_sid == "my-custom-session"
    assert sm.get_history("my-custom-session") == []


def test_add_message_and_turn():
    """Test adding messages and full turns."""
    sm = SessionManagerService(max_history_messages=10)
    sid = sm.create_session()

    sm.add_turn(sid, question="What is Transformer?", answer="An attention-based model.")

    history = sm.get_history(sid)
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].content == "What is Transformer?"
    assert history[1].role == "assistant"
    assert history[1].content == "An attention-based model."


def test_bounded_window_trimming():
    """Test that session history is trimmed to max_history_messages."""
    sm = SessionManagerService(max_history_messages=4)
    sid = sm.create_session()

    # Add 3 turns (6 messages)
    sm.add_turn(sid, "Q1", "A1")
    sm.add_turn(sid, "Q2", "A2")
    sm.add_turn(sid, "Q3", "A3")

    history = sm.get_history(sid)
    assert len(history) == 4
    # The oldest turn (Q1, A1) should have been pruned
    assert history[0].content == "Q2"
    assert history[1].content == "A2"
    assert history[2].content == "Q3"
    assert history[3].content == "A3"


def test_clear_and_delete_session():
    """Test clearing messages and deleting a session."""
    sm = SessionManagerService()
    sid = sm.create_session()
    sm.add_turn(sid, "Q1", "A1")

    assert len(sm.get_history(sid)) == 2

    # Clear
    assert sm.clear_session(sid) is True
    assert sm.get_history(sid) == []

    # Non-existent
    assert sm.clear_session("non-existent") is False

    # Delete
    assert sm.delete_session(sid) is True
    assert sm.get_session_info(sid) is None


def test_list_sessions():
    """Test listing active sessions."""
    sm = SessionManagerService()
    sid1 = sm.create_session("sess-1")
    sid2 = sm.create_session("sess-2")

    sm.add_turn(sid1, "Hello", "Hi")

    sessions = sm.list_sessions()
    assert len(sessions) >= 2
    s1_info = next((s for s in sessions if s.session_id == "sess-1"), None)
    assert s1_info is not None
    assert s1_info.message_count == 2
