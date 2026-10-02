"""Session and user records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class User:
    id: str
    email: str
    is_active: bool = True


@dataclass
class Session:
    id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    token_hash: str = ""
    revoked_at: datetime | None = None
    last_seen_at: datetime | None = None
    needs_refresh: bool = False
