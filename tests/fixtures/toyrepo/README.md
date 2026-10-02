# toy auth

A tiny session-token package used as a test fixture for cqa.

## Layout

- `src/auth/session.py`: issue, validate, refresh, and revoke tokens
- `src/auth/store.py`: an in-memory session store
- `config/settings.yaml`: session limits

## Running the demo

    python -m auth.session <token>
