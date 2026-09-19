from __future__ import annotations

import threading
from enum import StrEnum


class InputOwner(StrEnum):
    TRADING = "trading"
    MAIL_STORAGE = "mail_storage"


class InputOwnershipError(RuntimeError):
    pass


class InputLease:
    def __init__(
        self,
        manager: "InputOwnershipManager",
        owner: InputOwner,
        token: object,
    ) -> None:
        self._manager = manager
        self.owner = owner
        self._token = token
        self._released = False
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        with self._lock:
            if self._released:
                return False
            return self._manager._is_active(self.owner, self._token)

    def assert_active(self, expected_owner: InputOwner | None = None) -> None:
        if expected_owner is not None and self.owner != expected_owner:
            raise InputOwnershipError(
                f"输入租约属于 {self.owner.value}，不能由 {expected_owner.value} 使用"
            )
        if not self.active:
            raise InputOwnershipError(f"{self.owner.value} 的输入租约已经失效")

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._manager._release(self.owner, self._token)
            self._released = True

    def __enter__(self) -> "InputLease":
        self.assert_active()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.release()


class InputOwnershipManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._owner: InputOwner | None = None
        self._token: object | None = None

    @property
    def owner(self) -> InputOwner | None:
        with self._lock:
            return self._owner

    def acquire(self, owner: InputOwner) -> InputLease:
        token = object()
        with self._lock:
            if self._owner is not None:
                raise InputOwnershipError(
                    f"输入控制权当前由 {self._owner.value} 持有，{owner.value} 无法接管"
                )
            self._owner = owner
            self._token = token
        return InputLease(self, owner, token)

    def _is_active(self, owner: InputOwner, token: object) -> bool:
        with self._lock:
            return self._owner == owner and self._token is token

    def _release(self, owner: InputOwner, token: object) -> None:
        with self._lock:
            if self._owner != owner or self._token is not token:
                raise InputOwnershipError("无法释放不属于当前调用方的输入租约")
            self._owner = None
            self._token = None


GLOBAL_INPUT_OWNERSHIP = InputOwnershipManager()
