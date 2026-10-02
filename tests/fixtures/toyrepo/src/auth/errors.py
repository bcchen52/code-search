"""Errors raised by the auth package."""


class AuthError(Exception):
    """A token or session failed a check; the message says which one."""
