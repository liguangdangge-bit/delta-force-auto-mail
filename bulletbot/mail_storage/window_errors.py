"""Typed failures which may be transient while the game starts."""


class WindowActivationError(RuntimeError):
    def __init__(self, message: str, details: dict):
        super().__init__(message)
        self.details = details


class WindowUnavailableError(RuntimeError):
    pass
