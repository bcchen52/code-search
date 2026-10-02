"""An in-memory session store for tests and the command-line demo."""

from __future__ import annotations

from datetime import datetime, timezone

from .models import Session, User


class MemoryStore:
    def __init__(self):
        self.sessions: dict[str, Session] = {}
        self.users: dict[str, User] = {}

    def save(self, session: Session) -> None:
        self.sessions[session.id] = session

    def get(self, session_id: str) -> Session | None:
        return self.sessions.get(session_id)

    def get_user(self, user_id: str) -> User | None:
        return self.users.get(user_id)

    def active_for_user(self, user_id: str) -> list[Session]:
        return [s for s in self.sessions.values() if s.user_id == user_id and s.revoked_at is None]

    def revoke(self, session_id: str) -> None:
        session = self.sessions.get(session_id)
        if session is not None and session.revoked_at is None:
            session.revoked_at = datetime.now(timezone.utc)

    def touch(self, session: Session) -> None:
        self.sessions[session.id] = session
