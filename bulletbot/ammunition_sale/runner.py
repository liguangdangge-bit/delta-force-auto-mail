from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time

from bulletbot.loadout_purchase.warehouse_sell_session import (
    WarehouseSellOutcome,
    WarehouseSellResult,
)

from .settings import AmmunitionSaleRule, AmmunitionSaleSettings, AmmunitionSaleSource
from .session import (
    AmmunitionMailClaimResult,
    AmmunitionSaleOutcome,
    AmmunitionSaleResult,
    AmmunitionSaleRuleOutcome,
    AmmunitionSaleRuleResult,
    AmmunitionSaleRuleState,
    AmmunitionSaleStatus,
)


@dataclass
class _PendingRule:
    index: int
    rule: AmmunitionSaleRule
    low_price_attempts: int = 0
    eligible_at: float = 0.0
    listed_quantity: int = 0
    listed_batches: int = 0
    claimed_mails: int = 0
    awaiting_claimed_inventory: bool = False


class AmmunitionSaleRunner:
    """Schedule independent warehouse sales without changing loadout-sale rules."""

    def __init__(
        self,
        sell: Callable[[AmmunitionSaleRule], WarehouseSellResult],
        *,
        wait: Callable[[float], None],
        stopped: Callable[[], bool],
        claim_from_mail: Callable[
            [AmmunitionSaleRule], AmmunitionMailClaimResult
        ] | None = None,
        clock: Callable[[], float] = time.monotonic,
        on_status: Callable[[AmmunitionSaleStatus], None] | None = None,
        finish_requested: Callable[[], bool] | None = None,
        finish_current_inventory: bool = False,
    ) -> None:
        self._sell = sell
        self._wait = wait
        self._finish_requested = finish_requested or (lambda: False)
        self._timed = finish_requested is not None
        self._finish_current_inventory = finish_current_inventory
        self._stopped = stopped
        self._claim_from_mail = claim_from_mail
        self._clock = clock
        self._on_status = on_status or (lambda _status: None)

    def run(self, settings: AmmunitionSaleSettings) -> AmmunitionSaleResult:
        settings.validate(require_enabled=True)
        queue = [
            _PendingRule(index=index, rule=rule)
            for index, rule in enumerate(settings.rules)
            if rule.enabled
        ]
        results: list[AmmunitionSaleRuleResult] = []
        for pending in queue:
            self._status(pending, AmmunitionSaleRuleState.QUEUED, "等待处理")

        while queue:
            if self._finish_requested():
                for pending in queue:
                    if self._finish_current_inventory and pending.awaiting_claimed_inventory:
                        continue
                    if pending.listed_quantity or pending.claimed_mails:
                        results.append(AmmunitionSaleRuleResult(
                            pending.index, pending.rule.ammunition_name,
                            (AmmunitionSaleRuleOutcome.LISTED if pending.listed_quantity
                             else AmmunitionSaleRuleOutcome.MAIL_PENDING),
                            listed_quantity=pending.listed_quantity,
                            listed_batches=pending.listed_batches,
                            claimed_mails=pending.claimed_mails,
                            message="当前操作已确认，定时任务结束，剩余库存不再上架"))
                queue = [pending for pending in queue
                         if self._finish_current_inventory and pending.awaiting_claimed_inventory]
                if not queue:
                    break
            if self._stopped():
                return AmmunitionSaleResult(
                    AmmunitionSaleOutcome.STOPPED,
                    tuple(results),
                    "领邮件自动卖已由用户停止",
                )
            now = self._clock()
            current_index = next(
                (
                    index
                    for index, pending in enumerate(queue)
                    if pending.eligible_at <= now
                ),
                None,
            )
            if current_index is None:
                next_pending = min(queue, key=lambda pending: pending.eligible_at)
                remaining = max(0.0, next_pending.eligible_at - now)
                self._status(
                    next_pending,
                    AmmunitionSaleRuleState.COOLING_DOWN,
                    f"价格未达到条件，约 {max(1, round(remaining))} 秒后复查",
                    next_retry_seconds=remaining,
                )
                self._wait_for_retry(remaining, next_pending.awaiting_claimed_inventory)
                continue

            pending = queue.pop(current_index)
            self._status(
                pending,
                AmmunitionSaleRuleState.SCANNING,
                "正在扫描仓库并检查交易行价格",
                attempt=pending.low_price_attempts + 1,
            )
            sale = self._sell(pending.rule)
            pending.listed_quantity += sale.listed_quantity
            pending.listed_batches += len(sale.batches)
            if sale.outcome is WarehouseSellOutcome.FINISHED:
                queue.insert(0, pending)
                continue
            if sale.outcome in (
                WarehouseSellOutcome.LISTED,
                WarehouseSellOutcome.INVENTORY_NOT_FOUND,
            ):
                listed_any = (
                    pending.listed_quantity > 0
                    or sale.outcome is WarehouseSellOutcome.LISTED
                )
                if (
                    sale.outcome is WarehouseSellOutcome.INVENTORY_NOT_FOUND
                    and pending.awaiting_claimed_inventory
                ):
                    message = (
                        f"已从 {pending.claimed_mails} 封邮件领取目标子弹，"
                        "但返回仓库后仍未识别到；已停止以避免继续领取"
                    )
                    results.append(
                        AmmunitionSaleRuleResult(
                            pending.index,
                            pending.rule.ammunition_name,
                            AmmunitionSaleRuleOutcome.FAILED,
                            listed_quantity=pending.listed_quantity,
                            listed_batches=pending.listed_batches,
                            claimed_mails=pending.claimed_mails,
                            message=message,
                        )
                    )
                    self._status(
                        pending,
                        AmmunitionSaleRuleState.FAILED,
                        message,
                    )
                    return AmmunitionSaleResult(
                        AmmunitionSaleOutcome.FAILED,
                        tuple(results),
                        message,
                        error=message,
                    )

                if (
                    sale.outcome is WarehouseSellOutcome.LISTED
                    and pending.awaiting_claimed_inventory
                ):
                    pending.awaiting_claimed_inventory = False

                should_scan_mail = (
                    not self._finish_requested()
                    and pending.rule.source is AmmunitionSaleSource.WAREHOUSE_AND_MAIL
                )
                if should_scan_mail:
                    warehouse_state = (
                        f"仓库已挂牌 {pending.listed_quantity:,} 发"
                        if listed_any
                        else "仓库未找到"
                    )
                    if self._claim_from_mail is None:
                        outcome = AmmunitionSaleRuleOutcome.MAIL_PENDING
                        state = AmmunitionSaleRuleState.MAIL_PENDING
                        message = f"{warehouse_state}；邮件领取流程尚未接入"
                    else:
                        self._status(
                            pending,
                            AmmunitionSaleRuleState.MAIL_SCANNING,
                            f"{warehouse_state}；继续扫描邮件，本轮最多领取 "
                            f"{pending.rule.mail_claim_limit} 封",
                        )
                        try:
                            claimed = self._claim_from_mail(pending.rule)
                        except Exception as exc:
                            if self._stopped():
                                return AmmunitionSaleResult(
                                    AmmunitionSaleOutcome.STOPPED,
                                    tuple(results),
                                    "领邮件自动卖已由用户停止",
                                )
                            message = f"扫描或领取邮件失败：{exc}"
                            results.append(
                                AmmunitionSaleRuleResult(
                                    pending.index,
                                    pending.rule.ammunition_name,
                                    AmmunitionSaleRuleOutcome.FAILED,
                                    listed_quantity=pending.listed_quantity,
                                    listed_batches=pending.listed_batches,
                                    claimed_mails=pending.claimed_mails,
                                    message=message,
                                )
                            )
                            self._status(
                                pending,
                                AmmunitionSaleRuleState.FAILED,
                                message,
                            )
                            return AmmunitionSaleResult(
                                AmmunitionSaleOutcome.FAILED,
                                tuple(results),
                                message,
                                error=message,
                            )
                        if claimed.claimed_mails > 0:
                            pending.claimed_mails += claimed.claimed_mails
                            pending.awaiting_claimed_inventory = True
                            pending.eligible_at = 0.0
                            self._status(
                                pending,
                                AmmunitionSaleRuleState.QUEUED,
                                f"已从 {claimed.claimed_mails} 封邮件领取目标子弹；"
                                "返回仓库重新上架",
                            )
                            queue.insert(0, pending)
                            continue
                        mail_message = claimed.message or (
                            f"已扫描 {claimed.scanned_mails} 封邮件，"
                            "未找到目标子弹"
                        )
                        if listed_any:
                            outcome = AmmunitionSaleRuleOutcome.LISTED
                            state = AmmunitionSaleRuleState.LISTED
                            message = (
                                f"已挂牌 {pending.listed_quantity:,} 发；"
                                f"{mail_message}"
                            )
                        else:
                            outcome = AmmunitionSaleRuleOutcome.MAIL_NOT_FOUND
                            state = AmmunitionSaleRuleState.MAIL_NOT_FOUND
                            message = mail_message
                    results.append(
                        AmmunitionSaleRuleResult(
                            pending.index,
                            pending.rule.ammunition_name,
                            outcome,
                            listed_quantity=pending.listed_quantity,
                            listed_batches=pending.listed_batches,
                            claimed_mails=pending.claimed_mails,
                            message=message,
                        )
                    )
                    self._status(pending, state, message)
                    continue

                if listed_any:
                    message = sale.message or (
                        f"已挂牌 {pending.listed_quantity:,} 发；"
                        "仓库未发现剩余子弹"
                    )
                    outcome = AmmunitionSaleRuleOutcome.LISTED
                    state = AmmunitionSaleRuleState.LISTED
                else:
                    message = "仓库未找到该子弹，本次跳过"
                    outcome = AmmunitionSaleRuleOutcome.NOT_FOUND
                    state = AmmunitionSaleRuleState.NOT_FOUND
                results.append(
                    AmmunitionSaleRuleResult(
                        pending.index,
                        pending.rule.ammunition_name,
                        outcome,
                        listed_quantity=pending.listed_quantity,
                        listed_batches=pending.listed_batches,
                        claimed_mails=pending.claimed_mails,
                        message=message,
                    )
                )
                self._status(pending, state, message)
                continue
            if sale.outcome is WarehouseSellOutcome.BELOW_TRIGGER:
                pending.low_price_attempts += 1
                if pending.low_price_attempts < settings.low_price_attempts:
                    self._status(
                        pending,
                        AmmunitionSaleRuleState.LOW_PRICE,
                        f"价格低于条件，第 {pending.low_price_attempts}/"
                        f"{settings.low_price_attempts} 次；短暂等待后复查",
                        attempt=pending.low_price_attempts,
                        next_retry_seconds=settings.short_retry_seconds,
                    )
                    self._wait_for_retry(settings.short_retry_seconds, pending.awaiting_claimed_inventory)
                    queue.insert(0, pending)
                    continue
                pending.low_price_attempts = 0
                pending.eligible_at = self._clock() + settings.cooldown_seconds
                self._status(
                    pending,
                    AmmunitionSaleRuleState.COOLING_DOWN,
                    f"连续 {settings.low_price_attempts} 次低于条件；"
                    f"{round(settings.cooldown_seconds)} 秒后复查",
                    next_retry_seconds=settings.cooldown_seconds,
                )
                queue.append(pending)
                continue
            if sale.outcome is WarehouseSellOutcome.NO_SELL_SLOT:
                self._status(
                    pending,
                    AmmunitionSaleRuleState.WAITING_FOR_SLOT,
                    f"交易行售位已满；{settings.sell_slot_poll_seconds:g} 秒后复查",
                    next_retry_seconds=settings.sell_slot_poll_seconds,
                )
                self._wait_for_retry(settings.sell_slot_poll_seconds, pending.awaiting_claimed_inventory)
                queue.insert(0, pending)
                continue
            if sale.outcome is WarehouseSellOutcome.STOPPED:
                return AmmunitionSaleResult(
                    AmmunitionSaleOutcome.STOPPED,
                    tuple(results),
                    sale.message or "领邮件自动卖已由用户停止",
                )

            self._status(
                pending,
                AmmunitionSaleRuleState.FAILED,
                sale.message or "交易行直售需要安全恢复",
            )
            results.append(
                AmmunitionSaleRuleResult(
                    pending.index,
                    pending.rule.ammunition_name,
                    AmmunitionSaleRuleOutcome.FAILED,
                    listed_quantity=pending.listed_quantity,
                    listed_batches=pending.listed_batches,
                    claimed_mails=pending.claimed_mails,
                    message=sale.message or "交易行直售需要安全恢复",
                )
            )
            return AmmunitionSaleResult(
                AmmunitionSaleOutcome.FAILED,
                tuple(results),
                sale.message or "交易行直售需要安全恢复",
                error=sale.message or "交易行直售需要安全恢复",
            )

        listed = sum(result.listed_quantity for result in results)
        claimed_mails = sum(result.claimed_mails for result in results)
        return AmmunitionSaleResult(
            AmmunitionSaleOutcome.COMPLETED,
            tuple(results),
            f"领邮件自动卖本轮完成；共领取 {claimed_mails} 封邮件，"
            f"共挂牌 {listed:,} 发",
        )

    def _wait_for_retry(self, seconds, claimed_inventory=False):
        if not self._timed:
            self._wait(seconds)
            return
        remaining = seconds
        while remaining > 0 and not self._stopped():
            if self._finish_requested() and not (self._finish_current_inventory and claimed_inventory):
                return
            step = min(0.1, remaining)
            self._wait(step)
            remaining -= step

    def _status(
        self,
        pending: _PendingRule,
        state: AmmunitionSaleRuleState,
        message: str,
        *,
        attempt: int = 0,
        next_retry_seconds: float | None = None,
    ) -> None:
        self._on_status(
            AmmunitionSaleStatus(
                rule_index=pending.index,
                ammunition_name=pending.rule.ammunition_name,
                state=state,
                message=message,
                attempt=attempt,
                next_retry_seconds=next_retry_seconds,
            )
        )
