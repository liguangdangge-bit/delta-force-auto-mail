from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AmmunitionSaleOutcome(StrEnum):
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


class AmmunitionSaleRuleOutcome(StrEnum):
    LISTED = "listed"
    NOT_FOUND = "not_found"
    MAIL_PENDING = "mail_pending"
    MAIL_NOT_FOUND = "mail_not_found"
    FAILED = "failed"


class AmmunitionSaleRuleState(StrEnum):
    QUEUED = "queued"
    SCANNING = "scanning"
    LOW_PRICE = "low_price"
    COOLING_DOWN = "cooling_down"
    WAITING_FOR_SLOT = "waiting_for_slot"
    LISTED = "listed"
    NOT_FOUND = "not_found"
    MAIL_PENDING = "mail_pending"
    MAIL_SCANNING = "mail_scanning"
    MAIL_NOT_FOUND = "mail_not_found"
    FAILED = "failed"


@dataclass(frozen=True)
class AmmunitionMailClaimResult:
    claimed_mails: int = 0
    scanned_mails: int = 0
    reached_end: bool = False
    message: str = ""

    def __post_init__(self) -> None:
        if self.claimed_mails < 0 or self.scanned_mails < 0:
            raise ValueError("邮件扫描和领取数量不能小于 0")
        if self.claimed_mails > self.scanned_mails:
            raise ValueError("邮件领取数量不能超过扫描数量")


@dataclass(frozen=True)
class AmmunitionSaleStatus:
    rule_index: int
    ammunition_name: str
    state: AmmunitionSaleRuleState
    message: str
    attempt: int = 0
    next_retry_seconds: float | None = None


@dataclass(frozen=True)
class AmmunitionSaleRuleResult:
    rule_index: int
    ammunition_name: str
    outcome: AmmunitionSaleRuleOutcome
    listed_quantity: int = 0
    listed_batches: int = 0
    claimed_mails: int = 0
    message: str = ""


@dataclass(frozen=True)
class AmmunitionSaleResult:
    outcome: AmmunitionSaleOutcome
    rules: tuple[AmmunitionSaleRuleResult, ...] = ()
    message: str = ""
    error: str | None = None

    @property
    def listed_quantity(self) -> int:
        return sum(rule.listed_quantity for rule in self.rules)

    @property
    def listed_rules(self) -> int:
        return sum(
            rule.outcome is AmmunitionSaleRuleOutcome.LISTED
            for rule in self.rules
        )

    @property
    def claimed_mails(self) -> int:
        return sum(rule.claimed_mails for rule in self.rules)
