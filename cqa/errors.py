"""Exceptions raised by cqa."""


class CqaError(Exception):
    """Base class for errors raised by cqa."""


class ConfigError(CqaError):
    """A configuration file is invalid, or its ``extends`` chain cannot be resolved."""


class IndexNotReadyError(CqaError):
    """The requested index does not exist or has not finished building."""


class IndexBuildError(CqaError):
    """Building an index failed. The index version is marked failed and can be rebuilt."""


class PromptNotFoundError(CqaError):
    """No prompt template exists for the requested version."""


class JudgeResponseError(CqaError):
    """The judge model's reply could not be parsed as a verdict."""


class InvalidRequestError(CqaError):
    """A request is malformed, for example a commit that is not a full SHA."""


class NotFoundError(CqaError):
    """A requested repository, index, file, job, query, or run does not exist."""


class GuardrailError(CqaError):
    """A request was refused by a serving guardrail."""


class RateLimitedError(GuardrailError):
    """The client exceeded its request rate."""


class SpendCapReachedError(GuardrailError):
    """Today's model spend has reached the configured cap."""


class RepoNotAllowedError(GuardrailError):
    """The repository is not on the serving allowlist."""
