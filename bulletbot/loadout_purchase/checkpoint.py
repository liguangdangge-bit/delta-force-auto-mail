from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
import json
import os
from pathlib import Path
from typing import Any

from .warehouse_sell_session import PendingWarehouseListing


LOADOUT_CHECKPOINT_VERSION = 1


class LoadoutCheckpointStage(StrEnum):
    PURCHASE_SCAN = "purchase_scan"
    PURCHASE_CONFIRM_PENDING = "purchase_confirm_pending"
    PURCHASE_CONFIRMED = "purchase_confirmed"
    BLANK_APPLIED = "blank_applied"
    WAREHOUSE_SCAN = "warehouse_scan"
    LISTING_CONFIRM_PENDING = "listing_confirm_pending"
    LISTING_CONFIRMED = "listing_confirmed"
    WAITING_SALE = "waiting_sale"
    SELL_SLOT_WAIT = "sell_slot_wait"
    SELL_UNLIST_ALL = "sell_unlist_all"
    UNLIST_CONFIRM_PENDING = "unlist_confirm_pending"
    MAIL_STORAGE = "mail_storage"
    RECOVERY_WAIT = "recovery_wait"
    RETURN_HOME_PENDING = "return_home_pending"


class LoadoutResumeTarget(StrEnum):
    """Purpose to continue after the game process has been restarted."""

    LOADOUT_SCAN = "loadout_scan"
    RETURN_HOME_THEN_SCAN = "return_home_then_scan"
    WAREHOUSE_SELL = "warehouse_sell"
    MAIL_STORAGE = "mail_storage"


@dataclass(frozen=True)
class LoadoutPurchaseCheckpoint:
    """Durable state for one loadout-purchase round and its older listing."""

    round_id: int
    stage: LoadoutCheckpointStage = LoadoutCheckpointStage.PURCHASE_SCAN
    scheme_index: int | None = None
    ammo_name: str = ""
    sale_gate_passed: bool = False
    listed_batch_count: int = 0
    listed_quantity_total: int = 0
    sale_proceeds_claim_attempted: bool = False
    sale_proceeds_claimed: bool = False
    pending_batch_quantity: int = 0
    pending_batch_price: int | None = None
    first_lowest_price: int | None = None
    purchase_balance_before: int | None = None
    purchase_listed_price: int | None = None
    purchase_was_partial: bool = False
    pending_scheme_index: int | None = None
    pending_ammo_name: str = ""
    pending_additional_ammo_names: tuple[str, ...] = ()
    pending_listed_quantity: int = 0
    pending_listing_started_at: float | None = None
    pending_listing_deadline: float | None = None
    last_confirmed_page: str | None = None
    last_safe_action: str = ""
    resume_target: LoadoutResumeTarget | None = None
    pending_action: str = ""
    recovery_reason: str = ""
    workflow_version: int = LOADOUT_CHECKPOINT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_version": self.workflow_version,
            "round_id": self.round_id,
            "scheme_index": self.scheme_index,
            "ammo_name": self.ammo_name,
            "stage": self.stage.value,
            "sale_gate_passed": self.sale_gate_passed,
            "listed_batch_count": self.listed_batch_count,
            "listed_quantity_total": self.listed_quantity_total,
            "sale_proceeds_claim_attempted": self.sale_proceeds_claim_attempted,
            "sale_proceeds_claimed": self.sale_proceeds_claimed,
            "pending_batch_quantity": self.pending_batch_quantity,
            "pending_batch_price": self.pending_batch_price,
            "first_lowest_price": self.first_lowest_price,
            "purchase_balance_before": self.purchase_balance_before,
            "purchase_listed_price": self.purchase_listed_price,
            "purchase_was_partial": self.purchase_was_partial,
            "pending_scheme_index": self.pending_scheme_index,
            "pending_ammo_name": self.pending_ammo_name,
            "pending_additional_ammo_names": list(
                self.pending_additional_ammo_names
            ),
            "pending_listed_quantity": self.pending_listed_quantity,
            "pending_listing_started_at": self.pending_listing_started_at,
            "pending_listing_deadline": self.pending_listing_deadline,
            "last_confirmed_page": self.last_confirmed_page,
            "last_safe_action": self.last_safe_action,
            "resume_target": (
                self.resume_target.value if self.resume_target is not None else None
            ),
            "pending_action": self.pending_action,
            "recovery_reason": self.recovery_reason,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LoadoutPurchaseCheckpoint":
        version = int(value.get("workflow_version", 0))
        if version != LOADOUT_CHECKPOINT_VERSION:
            raise ValueError(f"不支持的配装买入检查点版本：{version}")
        checkpoint = cls(
            workflow_version=version,
            round_id=int(value.get("round_id", 0)),
            scheme_index=_optional_int(value.get("scheme_index")),
            ammo_name=str(value.get("ammo_name", "")).strip(),
            stage=LoadoutCheckpointStage(str(value.get("stage", ""))),
            sale_gate_passed=bool(value.get("sale_gate_passed", False)),
            listed_batch_count=int(value.get("listed_batch_count", 0)),
            listed_quantity_total=int(value.get("listed_quantity_total", 0)),
            sale_proceeds_claim_attempted=bool(
                value.get("sale_proceeds_claim_attempted", False)
            ),
            sale_proceeds_claimed=bool(
                value.get("sale_proceeds_claimed", False)
            ),
            pending_batch_quantity=int(value.get("pending_batch_quantity", 0)),
            pending_batch_price=_optional_int(value.get("pending_batch_price")),
            first_lowest_price=_optional_int(value.get("first_lowest_price")),
            purchase_balance_before=_optional_int(
                value.get("purchase_balance_before")
            ),
            purchase_listed_price=_optional_int(value.get("purchase_listed_price")),
            purchase_was_partial=bool(value.get("purchase_was_partial", False)),
            pending_scheme_index=_optional_int(value.get("pending_scheme_index")),
            pending_ammo_name=str(value.get("pending_ammo_name", "")).strip(),
            pending_additional_ammo_names=_string_tuple(
                value.get("pending_additional_ammo_names", ())
            ),
            pending_listed_quantity=int(value.get("pending_listed_quantity", 0)),
            pending_listing_started_at=_optional_float(
                value.get("pending_listing_started_at")
            ),
            pending_listing_deadline=_optional_float(
                value.get("pending_listing_deadline")
            ),
            last_confirmed_page=(
                str(value["last_confirmed_page"])
                if value.get("last_confirmed_page") is not None
                else None
            ),
            last_safe_action=str(value.get("last_safe_action", "")),
            resume_target=(
                LoadoutResumeTarget(str(value["resume_target"]))
                if value.get("resume_target")
                else None
            ),
            pending_action=str(value.get("pending_action", "")),
            recovery_reason=str(value.get("recovery_reason", "")),
        )
        checkpoint.validate()
        return checkpoint

    def validate(self) -> None:
        if self.round_id < 0:
            raise ValueError("配装买入检查点轮次不能为负数")
        if self.scheme_index is not None and self.scheme_index not in {1, 2, 3}:
            raise ValueError("配装买入检查点方案编号无效")
        if self.pending_scheme_index is not None and self.pending_scheme_index not in {
            1,
            2,
            3,
        }:
            raise ValueError("待售挂单检查点方案编号无效")
        for label, number in (
            ("已上架批次", self.listed_batch_count),
            ("已上架数量", self.listed_quantity_total),
            ("待确认批次数量", self.pending_batch_quantity),
            ("待售数量", self.pending_listed_quantity),
        ):
            if number < 0:
                raise ValueError(f"配装买入检查点的{label}不能为负数")
        for label, number in (
            ("待确认挂牌价", self.pending_batch_price),
            ("首批最低价", self.first_lowest_price),
            ("购买前余额", self.purchase_balance_before),
            ("达标整套金额", self.purchase_listed_price),
        ):
            if number is not None and number <= 0:
                raise ValueError(f"配装买入检查点的{label}必须大于 0")
        pending_values = (
            self.pending_scheme_index,
            self.pending_ammo_name,
            self.pending_listing_started_at,
            self.pending_listing_deadline,
        )
        has_pending = any(value not in {None, ""} for value in pending_values) or bool(
            self.pending_additional_ammo_names
        )
        if has_pending:
            if (
                self.pending_scheme_index is None
                or not self.pending_ammo_name
                or self.pending_listing_started_at is None
                or self.pending_listing_deadline is None
            ):
                raise ValueError("待售挂单检查点字段不完整")
            if self.pending_listing_deadline < self.pending_listing_started_at:
                raise ValueError("待售挂单检查点截止时间早于开始时间")
        if any(not name.strip() for name in self.pending_additional_ammo_names):
            raise ValueError("待售挂单检查点包含空子弹名称")
        if len(set(self.pending_additional_ammo_names)) != len(
            self.pending_additional_ammo_names
        ):
            raise ValueError("待售挂单检查点包含重复子弹名称")
        if self.pending_ammo_name in self.pending_additional_ammo_names:
            raise ValueError("待售挂单检查点主子弹名称重复")

    def pending_listing(self) -> PendingWarehouseListing | None:
        if (
            self.pending_scheme_index is None
            or not self.pending_ammo_name
            or self.pending_listing_started_at is None
            or self.pending_listing_deadline is None
        ):
            return None
        return PendingWarehouseListing(
            scheme_index=self.pending_scheme_index,
            ammunition_name=self.pending_ammo_name,
            listed_quantity=self.pending_listed_quantity,
            started_at=self.pending_listing_started_at,
            deadline=self.pending_listing_deadline,
            additional_ammunition_names=self.pending_additional_ammo_names,
        )

    def with_pending(
        self,
        pending: PendingWarehouseListing | None,
    ) -> "LoadoutPurchaseCheckpoint":
        if pending is None:
            return replace(
                self,
                pending_scheme_index=None,
                pending_ammo_name="",
                pending_additional_ammo_names=(),
                pending_listed_quantity=0,
                pending_listing_started_at=None,
                pending_listing_deadline=None,
            )
        return replace(
            self,
            pending_scheme_index=pending.scheme_index,
            pending_ammo_name=pending.ammunition_name,
            pending_additional_ammo_names=tuple(
                name
                for name in pending.ammunition_names
                if name != pending.ammunition_name
            ),
            pending_listed_quantity=pending.listed_quantity,
            pending_listing_started_at=pending.started_at,
            pending_listing_deadline=pending.deadline,
        )


class LoadoutPurchaseCheckpointStore:
    """Persist one loadout checkpoint using atomic replacement."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> LoadoutPurchaseCheckpoint | None:
        if not self.path.exists():
            return None
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"无法读取配装买入检查点：{exc}") from exc
        if not isinstance(value, dict):
            raise ValueError("配装买入检查点根节点必须是对象")
        return LoadoutPurchaseCheckpoint.from_dict(value)

    def save(self, checkpoint: LoadoutPurchaseCheckpoint) -> None:
        checkpoint.validate()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(checkpoint.to_dict(), handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        except OSError:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def delete(self) -> None:
        self.path.unlink(missing_ok=True)


def _optional_int(value: object) -> int | None:
    return None if value is None or value == "" else int(value)


def _optional_float(value: object) -> float | None:
    return None if value is None or value == "" else float(value)


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("待售挂单检查点的附加子弹名称必须是列表")
    return tuple(str(item).strip() for item in value)
