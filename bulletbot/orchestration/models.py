from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any


CHECKPOINT_VERSION = 1


def utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat()


class OrchestrationState(StrEnum):
    IDLE = "idle"
    TRADING_STARTING = "trading_starting"
    TRADING_ACTIVE = "trading_active"
    MAIL_PENDING = "mail_pending"
    TRADING_DRAINING = "trading_draining"
    MAIL_STARTING = "mail_starting"
    MAIL_RUNNING = "mail_running"
    MAIL_VERIFYING = "mail_verifying"
    TRADING_REINITIALIZING = "trading_reinitializing"
    SESSION_STOPPED = "session_stopped"
    SESSION_ERROR = "session_error"


class MailTriggerKind(StrEnum):
    QUANTITY_THRESHOLD = "quantity_threshold"
    STORAGE_FALLBACK = "storage_fallback"


@dataclass(frozen=True)
class PurchaseCountUpdate:
    quantity_added: int
    confirmed_quantity: int
    quantity_threshold: int
    mail_requested: bool


@dataclass(frozen=True)
class SessionCheckpoint:
    product_name: str
    normalized_product_name: str
    quantity_threshold: int
    state: OrchestrationState = OrchestrationState.IDLE
    confirmed_quantity: int = 0
    trigger_kind: MailTriggerKind | None = None
    trigger_reason: str | None = None
    mail_stop_requested: bool = False
    last_mail_result: dict[str, Any] | None = None
    error: str | None = None
    updated_at: str = field(default_factory=utc_now_text)
    version: int = CHECKPOINT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "product_name": self.product_name,
            "normalized_product_name": self.normalized_product_name,
            "quantity_threshold": self.quantity_threshold,
            "state": self.state.value,
            "confirmed_quantity": self.confirmed_quantity,
            "trigger_kind": self.trigger_kind.value if self.trigger_kind else None,
            "trigger_reason": self.trigger_reason,
            "mail_stop_requested": self.mail_stop_requested,
            "last_mail_result": self.last_mail_result,
            "error": self.error,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SessionCheckpoint":
        version = int(value.get("version", 0))
        if version != CHECKPOINT_VERSION:
            raise ValueError(f"不支持的会话检查点版本：{version}")
        product_name = str(value.get("product_name", "")).strip()
        normalized_name = str(value.get("normalized_product_name", "")).strip()
        threshold = int(value.get("quantity_threshold", 0))
        confirmed = int(value.get("confirmed_quantity", 0))
        if not product_name or not normalized_name:
            raise ValueError("会话检查点缺少商品名称")
        if threshold <= 0 or confirmed < 0:
            raise ValueError("会话检查点包含无效数量")
        trigger_value = value.get("trigger_kind")
        mail_result = value.get("last_mail_result")
        if mail_result is not None and not isinstance(mail_result, dict):
            raise ValueError("会话检查点的邮件结果格式无效")
        return cls(
            version=version,
            product_name=product_name,
            normalized_product_name=normalized_name,
            quantity_threshold=threshold,
            state=OrchestrationState(str(value.get("state", ""))),
            confirmed_quantity=confirmed,
            trigger_kind=(
                MailTriggerKind(str(trigger_value)) if trigger_value else None
            ),
            trigger_reason=(
                str(value["trigger_reason"])
                if value.get("trigger_reason") is not None
                else None
            ),
            mail_stop_requested=bool(value.get("mail_stop_requested", False)),
            last_mail_result=mail_result,
            error=str(value["error"]) if value.get("error") is not None else None,
            updated_at=str(value.get("updated_at", "")) or utc_now_text(),
        )
