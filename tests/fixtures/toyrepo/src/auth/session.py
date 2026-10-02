"""Session tokens: issue, validate, refresh, and revoke."""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone

import jwt

from .errors import AuthError
from .models import Session, User

SESSION_TTL = timedelta(hours=12)
REFRESH_WINDOW = timedelta(minutes=30)
MAX_SESSIONS_PER_USER = 5


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class SessionManager:
    """Issues signed session tokens and checks them against the session store."""

    algorithm = "HS256"

    def __init__(self, store, secret: str, ttl: timedelta = SESSION_TTL):
        self.store = store
        self.secret = secret
        self.ttl = ttl

    def create_session(self, user: User) -> str:
        """Start a session for `user` and return its signed token."""
        self._enforce_limit(user)
        now = _now()
        session = Session(
            id=secrets.token_hex(16),
            user_id=user.id,
            created_at=now,
            expires_at=now + self.ttl,
        )
        token = self._encode(session)
        session.token_hash = _hash_token(token)
        self.store.save(session)
        return token

    def _encode(self, session: Session) -> str:
        claims = {
            "sid": session.id,
            "sub": session.user_id,
            "typ": "session",
            "iat": int(session.created_at.timestamp()),
            "exp": int(session.expires_at.timestamp()),
        }
        return jwt.encode(claims, self.secret, algorithm=self.algorithm)

    def _enforce_limit(self, user: User) -> None:
        """Revoke the oldest sessions so a user never holds more than the limit."""
        active = self.store.active_for_user(user.id)
        excess = len(active) - MAX_SESSIONS_PER_USER + 1
        for session in sorted(active, key=lambda s: s.created_at)[: max(excess, 0)]:
            self.store.revoke(session.id)

    def list_sessions(self, user: User) -> list[Session]:
        """Active sessions for `user`, newest first."""
        sessions = self.store.active_for_user(user.id)
        return sorted(sessions, key=lambda s: s.created_at, reverse=True)

    def _decode(self, token: str) -> dict:
        """Check the signature and required claims; raise AuthError on any failure."""
        try:
            return jwt.decode(
                token,
                self.secret,
                algorithms=[self.algorithm],
                options={"require": ["exp", "iat", "sid", "sub"]},
            )
        except jwt.ExpiredSignatureError as exc:
            raise AuthError("token expired") from exc
        except jwt.InvalidTokenError as exc:
            raise AuthError("invalid token") from exc

    def validate_token(self, token: str) -> User:
        """Return the user for a valid session token, or raise AuthError.

        A token is valid when its signature checks out, it has not expired,
        its session still exists and is not revoked, and the stored hash
        matches, so a leaked token stops working once its session is rotated.
        """
        if not token:
            raise AuthError("missing token")
        claims = self._decode(token)

        session = self.store.get(claims["sid"])
        if session is None:
            raise AuthError("unknown session")
        if session.revoked_at is not None:
            raise AuthError("session revoked")
        if session.token_hash != _hash_token(token):
            raise AuthError("token does not match session")

        now = _now()
        if session.expires_at <= now:
            self.store.revoke(session.id)
            raise AuthError("session expired")

        user = self.store.get_user(session.user_id)
        if user is None or not user.is_active:
            raise AuthError("user inactive")

        if session.expires_at - now < REFRESH_WINDOW:
            session.needs_refresh = True

        session.last_seen_at = now
        self.store.touch(session)
        return user

    def refresh(self, token: str) -> str:
        """Swap a valid token that is close to expiry for a fresh one."""
        user = self.validate_token(token)
        claims = self._decode(token)
        self.store.revoke(claims["sid"])
        return self.create_session(user)

    def revoke(self, token: str) -> None:
        """End the session behind `token`; unknown or invalid tokens are ignored."""
        try:
            claims = self._decode(token)
        except AuthError:
            return
        self.store.revoke(claims["sid"])


if __name__ == "__main__":
    import sys

    from .store import MemoryStore

    manager = SessionManager(MemoryStore(), secret="dev-only")
    print(manager.validate_token(sys.argv[1]))
