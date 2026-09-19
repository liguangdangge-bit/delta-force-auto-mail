from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
import re
import shutil
from statistics import median
import threading
import traceback
from typing import Any, Mapping
import unicodedata

from bulletbot.reporting.run_storage import cleanup_run_directories
from bulletbot.runtime.performance import TimingSummary


BEIJING_TIMEZONE = timezone(timedelta(hours=8), name="Asia/Shanghai")


REPORT_THEME_CSS = """
:root {
  color-scheme: dark;
  --bg: #060a0f;
  --surface: #0a1117;
  --surface-strong: #0d171d;
  --border: #1b2c33;
  --border-strong: #31505a;
  --text: #d7e4e1;
  --muted: #8da3a1;
  --green: #43edb8;
  --green-soft: #103d33;
  --cyan: #36b9dc;
  --amber: #e6b54d;
  --red: #ff8279;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  color: var(--text);
  background: var(--bg);
  font-family: "Microsoft YaHei UI", "Microsoft YaHei", Arial, sans-serif;
}
main { max-width: 1360px; margin: 0 auto; padding: 28px; }
.report-header {
  padding: 2px 0 18px;
  border-bottom: 1px solid var(--border-strong);
}
.eyebrow {
  color: var(--cyan);
  font-size: 12px;
  font-weight: 700;
  margin-bottom: 8px;
}
.report-head {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 18px;
}
.run-tag {
  flex: 0 0 auto;
  padding: 5px 9px;
  color: var(--amber);
  background: #211a0b;
  border: 1px solid #815f1a;
  font-size: 12px;
}
h1 { margin: 0 0 7px; color: var(--green); font-size: 30px; }
h2 { margin: 30px 0 10px; font-size: 20px; }
h3 { margin: 18px 0 8px; color: #bceade; font-size: 16px; }
.meta, .muted { color: var(--muted); }
.summary {
  display: grid;
  grid-template-columns: repeat(6, minmax(150px, 1fr));
  gap: 10px;
  margin-top: 18px;
}
.metric, .panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 4px;
}
.metric { min-height: 84px; padding: 13px 14px; color: var(--muted); }
.metric b {
  display: block;
  margin-top: 7px;
  color: var(--text);
  font-size: 19px;
  font-variant-numeric: tabular-nums;
}
.panel { margin-top: 12px; padding: 16px; overflow-x: auto; }
.notice { border-left: 3px solid var(--green); }
.session {
  margin-top: 34px;
  padding-top: 20px;
  border-top: 1px solid var(--border-strong);
}
.session-head {
  display: flex;
  justify-content: space-between;
  gap: 20px;
  align-items: flex-start;
  flex-wrap: wrap;
}
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td {
  padding: 9px 8px;
  text-align: left;
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
}
th { color: #8be9cf; background: var(--surface-strong); font-weight: 700; }
tbody tr:hover { background: #0e1b20; }
input {
  width: 112px;
  padding: 6px 8px;
  color: var(--text);
  background: #080e13;
  border: 1px solid var(--border-strong);
  border-radius: 3px;
}
input:focus { outline: 2px solid var(--green); outline-offset: 1px; }
.positive { color: var(--green); font-weight: 700; }
.negative { color: var(--red); font-weight: 700; }
.pill {
  display: inline-block;
  margin: 0 5px 5px 0;
  padding: 3px 7px;
  color: #aee9d8;
  background: var(--green-soft);
  border: 1px solid #27594d;
  border-radius: 3px;
}
svg { min-width: 760px; width: 100%; height: auto; margin: 8px 0 4px; }
.chart-grid { stroke: var(--border); stroke-width: 1; }
.chart-axis { stroke: var(--border-strong); stroke-width: 1; }
.chart-line { fill: none; stroke: var(--cyan); stroke-width: 1.8; }
.chart-label { fill: var(--muted); font-size: 11px; }
@media (max-width: 900px) {
  main { padding: 16px; }
  .summary { grid-template-columns: repeat(2, minmax(140px, 1fr)); }
}
@media (max-width: 540px) {
  main { padding: 11px; }
  .report-head { align-items: flex-start; flex-direction: column; }
  .summary { grid-template-columns: 1fr; }
  h1 { font-size: 24px; }
}
"""


class RunRecorder:
    """Persist one application process and its individual trading sessions."""

    _LOADOUT_PRICE_COUNT_FLUSH_INTERVAL = 32

    _PRICE_FIELDS = (
        "beijing_time",
        "run_id",
        "session_id",
        "cycle",
        "product",
        "page",
        "source",
        "price",
        "confidence",
    )
    _OPERATION_FIELDS = (
        "beijing_time",
        "run_id",
        "session_id",
        "type",
        "product",
        "quantity",
        "estimated_quantity",
        "estimated_unit_price",
        "quantity_source",
        "displayed_unit_price",
        "actual_unit_price",
        "total_amount",
        "balance_before",
        "balance_after",
        "included_in_cost",
        "result",
        "details",
    )
    _BALANCE_FIELDS = (
        "beijing_time",
        "run_id",
        "session_id",
        "label",
        "balance",
    )
    _PERFORMANCE_FIELDS = (
        "beijing_time",
        "run_id",
        "session_id",
        "scope",
        "phase",
        "calls",
        "total_ms",
        "average_ms",
        "last_ms",
        "max_ms",
    )
    _SESSION_FIELDS = (
        "run_id",
        "session_id",
        "session_type",
        "start_time_beijing",
        "end_time_beijing",
        "initial_balance",
        "final_balance",
        "balance_change",
        "fee_percent",
        "selected_products",
        "reason",
    )
    _SUMMARY_FIELDS = (
        "run_id",
        "session_id",
        "session_start_beijing",
        "session_end_beijing",
        "product",
        "buy_quantity",
        "buy_total_cost",
        "weighted_average_cost",
        "confirmed_buy_records",
        "estimated_partial_buy_quantity",
        "estimated_partial_buy_cost",
        "partial_buy_records",
        "buy_quantity_including_estimates",
        "buy_total_cost_including_estimates",
        "weighted_average_cost_including_estimates",
        "listed_quantity",
        "listing_expected_income",
        "confirmed_probe_sold_quantity",
        "default_expected_sell_price",
        "fee_percent",
        "estimated_net_unit_profit",
        "estimated_total_profit",
    )
    _BREAKDOWN_FIELDS = (
        "run_id",
        "session_id",
        "product",
        "actual_unit_price",
        "quantity",
        "total_cost",
        "confirmed_records",
        "estimated_records",
        "quantity_source",
        "first_buy_time_beijing",
        "last_buy_time_beijing",
    )
    _LOADOUT_PRICE_COUNT_FIELDS = (
        "run_id",
        "session_id",
        "scheme_index",
        "recognized_price",
        "count",
        "normal_count",
        "fast_count",
        "first_seen_beijing",
        "last_seen_beijing",
    )
    _SOURCE_COLORS = {
        "favorites": "#36b9dc",
        "favorites_price": "#36b9dc",
        "detail_lowest": "#43edb8",
        "detail_average": "#7bd79b",
        "listing_lowest": "#e6b54d",
        "confirmed_purchase": "#38d997",
        "confirmed_purchase_batch": "#1fab78",
        "loadout_low_price": "#e6b54d",
    }

    def __init__(self, runs_root: Path) -> None:
        self.started_at = self._now()
        self.run_id = self.started_at.strftime("%Y%m%d_%H%M%S_%f")
        self.directory = runs_root / self.run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        removed_runs = cleanup_run_directories(
            runs_root,
            protected=(self.directory,),
        )
        self.log_path = self.directory / "run.log"
        self.price_path = self.directory / "price_samples.csv"
        self.loadout_price_counts_path = (
            self.directory / "loadout_price_counts.csv"
        )
        self.operations_path = self.directory / "operations.csv"
        self.balance_path = self.directory / "balances.csv"
        self.performance_path = self.directory / "performance.csv"
        self.sessions_path = self.directory / "sessions.csv"
        self.summary_path = self.directory / "trade_summary.csv"
        self.breakdown_path = self.directory / "buy_price_breakdown.csv"
        self.report_path = self.directory / "report.html"
        self._lock = threading.RLock()
        self._loadout_price_counts: dict[tuple[str, int, int], dict[str, Any]] = {}
        self._loadout_price_count_updates = 0
        self._write_loadout_price_counts_locked()
        self._log_file = self.log_path.open("a", encoding="utf-8", buffering=1)
        self._price_file = self.price_path.open(
            "a", encoding="utf-8-sig", newline="", buffering=1
        )
        self._operations_file = self.operations_path.open(
            "a", encoding="utf-8-sig", newline="", buffering=1
        )
        self._balance_file = self.balance_path.open(
            "a", encoding="utf-8-sig", newline="", buffering=1
        )
        self._performance_file = self.performance_path.open(
            "a", encoding="utf-8-sig", newline="", buffering=1
        )
        self._price_writer = csv.DictWriter(self._price_file, fieldnames=self._PRICE_FIELDS)
        self._operation_writer = csv.DictWriter(
            self._operations_file, fieldnames=self._OPERATION_FIELDS
        )
        self._balance_writer = csv.DictWriter(
            self._balance_file, fieldnames=self._BALANCE_FIELDS
        )
        self._performance_writer = csv.DictWriter(
            self._performance_file, fieldnames=self._PERFORMANCE_FIELDS
        )
        self._price_writer.writeheader()
        self._operation_writer.writeheader()
        self._balance_writer.writeheader()
        self._performance_writer.writeheader()
        self._price_file.flush()
        self._operations_file.flush()
        self._balance_file.flush()
        self._performance_file.flush()
        self._latest_price: dict[str, Any] | None = None
        self._sessions: list[dict[str, Any]] = []
        self._rules: list[dict[str, Any]] = []
        self._binding: dict[str, Any] = {}
        self._active_session_id = ""
        self._active_products: dict[str, str] = {}
        self._session_counter = 0
        self._loadout_offer_counter = 0
        self._latest_balance: int | None = None
        self._last_reason = "程序运行中"
        self._closed = False
        self.log("运行记录目录已创建：" + str(self.directory))
        if removed_runs:
            self.log(f"已自动清理 {len(removed_runs)} 个旧运行目录。")

    @property
    def active_session_id(self) -> str:
        return self._active_session_id

    def log(self, message: str, level: str = "INFO") -> None:
        with self._lock:
            if self._closed:
                return
            timestamp = self._timestamp()
            normalized = str(message).replace("\r", " ").replace("\n", " | ")
            self._log_file.write(f"[{timestamp}] [{level}] {normalized}\n")
            self._log_file.flush()

    def record_exception(self, exc_type, exc_value, exc_traceback) -> None:
        details = "".join(
            traceback.format_exception(exc_type, exc_value, exc_traceback)
        ).rstrip()
        self.log(details, level="ERROR")
        self.generate_report("未处理异常")

    def set_binding(self, *, title: str, width: int, height: int) -> None:
        with self._lock:
            self._binding = {
                "title": title,
                "width": int(width),
                "height": int(height),
            }

    def set_rules(self, rules: list[dict[str, Any]]) -> None:
        with self._lock:
            self._rules = [dict(rule) for rule in rules]

    def begin_session(
        self,
        rules: list[dict[str, Any]],
        *,
        fee_percent: float = 13.0,
        initial_balance: int | None = None,
        session_type: str = "trading",
    ) -> str:
        with self._lock:
            if self._closed:
                return ""
            if self._active_session_id:
                self._end_session_locked("开始新的自动交易会话", self._latest_balance)
            self._session_counter += 1
            session_id = f"S{self._session_counter:03d}"
            snapshot = [dict(rule) for rule in rules]
            self._rules = snapshot
            self._active_session_id = session_id
            self._active_products = {
                self._normalize_product(str(rule.get("name", ""))): str(
                    rule.get("name", "")
                ).strip()
                for rule in snapshot
                if str(rule.get("name", "")).strip()
            }
            self._latest_balance = initial_balance
            self._loadout_offer_counter = 0
            self._sessions.append(
                {
                    "run_id": self.run_id,
                    "session_id": session_id,
                    "started_at": self._timestamp(),
                    "ended_at": "",
                    "initial_balance": initial_balance if initial_balance is not None else "",
                    "final_balance": "",
                    "fee_percent": float(fee_percent),
                    "session_type": session_type,
                    "rules": snapshot,
                    "reason": (
                        "配装买入法运行中"
                        if session_type == "loadout_purchase"
                        else "自动交易运行中"
                    ),
                }
            )
            self._write_sessions_locked()
        self.log(
            f"{'配装买入法' if session_type == 'loadout_purchase' else '自动交易'}"
            f"会话 {session_id} 开始；选中商品："
            + "、".join(self._active_products.values()),
            level="SESSION",
        )
        return session_id

    def set_session_initial_balance(self, balance: int) -> None:
        with self._lock:
            session = self._active_session_locked()
            if session is None:
                return
            value = int(balance)
            session["initial_balance"] = value
            self._latest_balance = value
            self._write_sessions_locked()

    def end_session(self, reason: str, final_balance: int | None = None) -> None:
        with self._lock:
            if not self._active_session_id:
                return
            session_id = self._active_session_id
            self._end_session_locked(reason, final_balance)
            self._write_loadout_price_counts_locked()
            self._loadout_price_count_updates = 0
            self._write_sessions_locked()
        self.log(f"自动交易会话 {session_id} 结束：{reason}", level="SESSION")

    def _end_session_locked(self, reason: str, final_balance: int | None) -> None:
        session = self._active_session_locked()
        if session is None:
            return
        resolved_balance = self._latest_balance if final_balance is None else int(final_balance)
        session["ended_at"] = self._timestamp()
        session["final_balance"] = (
            resolved_balance if resolved_balance is not None else ""
        )
        session["reason"] = reason
        self._last_reason = reason
        self._active_session_id = ""
        self._active_products = {}

    def record_balance(self, *, label: str, balance: int) -> None:
        with self._lock:
            if self._closed or not self._active_session_id:
                return
            value = int(balance)
            self._latest_balance = value
            session = self._active_session_locked()
            if session is not None and session["initial_balance"] == "":
                session["initial_balance"] = value
            row = {
                "beijing_time": self._timestamp(),
                "run_id": self.run_id,
                "session_id": self._active_session_id,
                "label": label,
                "balance": value,
            }
            self._balance_writer.writerow(row)
            self._balance_file.flush()

    def record_performance(
        self,
        snapshot: Mapping[str, TimingSummary],
        *,
        scope: str = "monitor",
    ) -> None:
        """Persist one timing snapshot as a long-form CSV sample set."""

        with self._lock:
            if self._closed:
                return
            timestamp = self._timestamp()
            for phase, summary in snapshot.items():
                if not isinstance(summary, TimingSummary):
                    continue
                self._performance_writer.writerow(
                    {
                        "beijing_time": timestamp,
                        "run_id": self.run_id,
                        "session_id": self._active_session_id,
                        "scope": scope,
                        "phase": phase,
                        "calls": summary.calls,
                        "total_ms": f"{summary.total_ms:.3f}",
                        "average_ms": f"{summary.average_ms:.3f}",
                        "last_ms": f"{summary.last_ms:.3f}",
                        "max_ms": f"{summary.max_ms:.3f}",
                    }
                )
            self._performance_file.flush()

    def record_price(
        self,
        *,
        cycle: int,
        product: str,
        page: str,
        source: str,
        price: int,
        confidence: float | None = None,
    ) -> None:
        canonical_product = self._canonical_active_product(product)
        if canonical_product is None or price <= 0:
            return
        row = {
            "beijing_time": self._timestamp(),
            "run_id": self.run_id,
            "session_id": self._active_session_id,
            "cycle": max(0, int(cycle)),
            "product": canonical_product,
            "page": page,
            "source": source,
            "price": int(price),
            "confidence": "" if confidence is None else f"{confidence:.6f}",
        }
        with self._lock:
            if self._closed or not self._active_session_id:
                return
            previous = self._latest_price
            if previous and all(
                previous.get(key) == row.get(key)
                for key in (
                    "session_id",
                    "cycle",
                    "product",
                    "page",
                    "source",
                    "price",
                )
            ):
                return
            self._latest_price = dict(row)
            self._price_writer.writerow(row)
            self._price_file.flush()

    def record_loadout_price_recognition(
        self,
        *,
        scheme_index: int,
        price: int,
        mode: str,
    ) -> None:
        """Aggregate every recognized loadout price without writing to run.log."""

        value = int(price)
        if value <= 0:
            return
        mode_key = (
            "fast"
            if str(mode).strip().casefold() in {"fast", "极速", "极速配装"}
            else "normal"
        )
        with self._lock:
            if self._closed or not self._active_session_id:
                return
            session_id = self._active_session_id
            scheme = max(0, int(scheme_index))
            timestamp = self._timestamp()
            key = (session_id, scheme, value)
            row = self._loadout_price_counts.get(key)
            if row is None:
                row = {
                    "run_id": self.run_id,
                    "session_id": session_id,
                    "scheme_index": scheme,
                    "recognized_price": value,
                    "count": 0,
                    "normal_count": 0,
                    "fast_count": 0,
                    "first_seen_beijing": timestamp,
                    "last_seen_beijing": timestamp,
                }
                self._loadout_price_counts[key] = row
            row["count"] += 1
            row[f"{mode_key}_count"] += 1
            row["last_seen_beijing"] = timestamp
            self._loadout_price_count_updates += 1
            if (
                self._loadout_price_count_updates
                >= self._LOADOUT_PRICE_COUNT_FLUSH_INTERVAL
            ):
                self._write_loadout_price_counts_locked()
                self._loadout_price_count_updates = 0

    def record_operation(
        self,
        *,
        operation_type: str,
        product: str,
        quantity: int,
        unit_price: float | None,
        total_amount: int,
        estimated_quantity: int = 0,
        estimated_unit_price: float | None = None,
        quantity_source: str = "",
        displayed_unit_price: float | None = None,
        balance_before: int | None = None,
        balance_after: int | None = None,
        included_in_cost: bool | None = None,
        result: str = "成功",
        details: str = "",
    ) -> None:
        canonical_product = self._canonical_active_product(product)
        if canonical_product is None:
            return
        if included_in_cost is None:
            included_in_cost = operation_type == "BUY_CONFIRMED"
        row = {
            "beijing_time": self._timestamp(),
            "run_id": self.run_id,
            "session_id": self._active_session_id,
            "type": operation_type,
            "product": canonical_product,
            "quantity": max(0, int(quantity)),
            "estimated_quantity": max(0, int(estimated_quantity)),
            "estimated_unit_price": (
                ""
                if estimated_unit_price is None
                else f"{float(estimated_unit_price):.4f}"
            ),
            "quantity_source": str(
                quantity_source
                or (
                    "余额反推（估算）"
                    if operation_type in {"LOADOUT_BUY_PARTIAL", "BUY_UNCERTAIN"}
                    else "确认"
                )
            ),
            "displayed_unit_price": (
                "" if displayed_unit_price is None else f"{float(displayed_unit_price):.4f}"
            ),
            "actual_unit_price": (
                "" if unit_price is None else f"{float(unit_price):.4f}"
            ),
            "total_amount": int(total_amount),
            "balance_before": "" if balance_before is None else int(balance_before),
            "balance_after": "" if balance_after is None else int(balance_after),
            "included_in_cost": "是" if included_in_cost else "否",
            "result": result,
            "details": details,
        }
        with self._lock:
            if self._closed or not self._active_session_id:
                return
            self._operation_writer.writerow(row)
            self._operations_file.flush()
        self.log(
            f"{operation_type} {canonical_product} quantity={quantity} "
            f"unit_price="
            f"{'--' if unit_price is None else f'{unit_price:.2f}'} "
            f"total={total_amount} result={result}",
            level="OPERATION",
        )

    def record_loadout_offer(
        self,
        *,
        scheme_index: int,
        quantity: int,
        total_price: int,
        displayed_unit_price: float,
        max_total_price: int,
    ) -> None:
        product = f"配装方案 {int(scheme_index)}"
        with self._lock:
            if self._closed or not self._active_session_id:
                return
            self._loadout_offer_counter += 1
            cycle = self._loadout_offer_counter
        self.record_price(
            cycle=cycle,
            product=product,
            page="配装方案列表",
            source="loadout_low_price",
            price=max(1, round(float(displayed_unit_price))),
        )
        self.record_operation(
            operation_type="LOADOUT_LOW_PRICE",
            product=product,
            quantity=quantity,
            unit_price=None,
            displayed_unit_price=displayed_unit_price,
            total_amount=total_price,
            included_in_cost=False,
            result="发现低价",
            details=f"整套低价 {total_price:,}；允许上限 {max_total_price:,}",
        )

    def record_loadout_purchase_result(
        self,
        *,
        scheme_index: int,
        outcome: str,
        quantity: int,
        listed_price: int,
        displayed_unit_price: float,
        balance_before: int | None,
        balance_after: int | None,
        spent: int | None,
        actual_average: int | None,
        included_in_average: bool,
    ) -> None:
        product = f"配装方案 {int(scheme_index)}"
        if balance_before is not None:
            self.record_balance(label=f"{product}购买前", balance=balance_before)
        if balance_after is not None:
            self.record_balance(label=f"{product}购买后", balance=balance_after)
        operation_types = {
            "full": "BUY_CONFIRMED",
            "partial": "LOADOUT_BUY_PARTIAL",
            "none": "LOADOUT_BUY_FAILED",
            "unknown": "LOADOUT_BUY_UNKNOWN",
        }
        result_labels = {
            "full": "全部成交",
            "partial": "部分成交",
            "none": "完全没有成交",
            "unknown": "成交状态未知",
        }
        resolved_spent = int(spent) if spent is not None and spent > 0 else 0
        if (
            resolved_spent <= 0
            and balance_before is not None
            and balance_after is not None
            and balance_before > balance_after
        ):
            resolved_spent = int(balance_before - balance_after)
        estimated_quantity = 0
        estimated_unit_price: float | None = None
        if outcome == "partial" and resolved_spent > 0 and displayed_unit_price > 0:
            # A partial loadout does not expose the bullet count.  The offer's
            # displayed average price is the best available unit-price sample;
            # infer the count from the amount actually deducted.  Keep this in
            # separate estimated columns so confirmed quantities remain intact.
            estimated_quantity = max(
                0,
                int(round(resolved_spent / float(displayed_unit_price))),
            )
            target_quantity = (
                int(round(float(listed_price) / float(displayed_unit_price)))
                if displayed_unit_price > 0
                else 0
            )
            if target_quantity > 0:
                estimated_quantity = min(estimated_quantity, target_quantity)
            estimated_unit_price = float(displayed_unit_price)
        unit_price = (
            float(actual_average)
            if outcome == "full"
            and actual_average is not None
            and actual_average > 0
            else None
        )
        details = f"达标时整套金额 {int(listed_price):,}"
        if outcome == "partial":
            details += (
                f"；按余额反推约 {estimated_quantity:,} 发，"
                f"推算单价 {float(displayed_unit_price):,.2f}"
            )
        elif outcome == "unknown":
            details += "；前后余额未完整识别，不计入单发平均价"
        self.record_operation(
            operation_type=operation_types.get(outcome, "LOADOUT_BUY_UNKNOWN"),
            product=product,
            quantity=quantity if outcome == "full" else 0,
            estimated_quantity=estimated_quantity,
            estimated_unit_price=estimated_unit_price,
            quantity_source="余额反推（估算）" if outcome == "partial" else "确认",
            unit_price=unit_price,
            displayed_unit_price=displayed_unit_price,
            total_amount=resolved_spent,
            balance_before=balance_before,
            balance_after=balance_after,
            included_in_cost=bool(included_in_average and outcome == "full"),
            result=result_labels.get(outcome, "成交状态未知"),
            details=details,
        )

    def record_loadout_post_purchase_result(
        self,
        *,
        scheme_index: int,
        outcome: str,
        quantity: int = 0,
        batch_count: int = 0,
        ammunition_name: str = "",
        unlisted_orders: int = 0,
    ) -> None:
        labels = {
            "listed": (
                "LOADOUT_LISTED",
                "已上架（未确认实际售出）",
                f"已确认上架 {max(0, int(batch_count))} 批",
            ),
            "slot_available": (
                "LOADOUT_SLOT_RELEASED",
                "售位释放（推定售出，非成交确认）",
                "按用户确认的独占售位前提推定上一轮挂单售出",
            ),
            "unlisted": (
                "LOADOUT_UNLISTED",
                "已下架回库",
                f"已确认连续下架 {max(0, int(unlisted_orders))} 个挂单",
            ),
            "mail_stored": (
                "LOADOUT_MAIL_STORED",
                "已卡邮件",
                "卡邮件流程完成且 PAK 恢复由工作流统一验证",
            ),
        }
        resolved = labels.get(outcome)
        if resolved is None:
            return
        operation_type, result, details = resolved
        if ammunition_name.strip():
            details = f"{ammunition_name.strip()}；{details}"
        self.record_operation(
            operation_type=operation_type,
            product=f"配装方案 {int(scheme_index)}",
            quantity=max(0, int(quantity)),
            unit_price=None,
            total_amount=0,
            included_in_cost=False,
            result=result,
            details=details,
        )

    def snapshot_for_export(self, destination: Path) -> tuple[str, ...]:
        """Copy recorder-owned files while writers are briefly excluded."""
        with self._lock:
            if not self._closed:
                self.generate_report()
                for handle in (self._log_file, self._price_file, self._operations_file,
                               self._balance_file, self._performance_file):
                    handle.flush()
            destination.mkdir(parents=True, exist_ok=True)
            paths = (self.log_path, self.price_path, self.operations_path,
                     self.balance_path, self.performance_path, self.sessions_path,
                     self.summary_path, self.breakdown_path, self.report_path,
                     self.loadout_price_counts_path)
            for path in paths:
                if not path.is_file():
                    raise FileNotFoundError(f"诊断包缺少必需文件：{path.name}")
                shutil.copyfile(path, destination / path.name)
            return tuple(path.name for path in paths)

    def generate_report(self, reason: str | None = None) -> Path:
        with self._lock:
            if reason:
                self._last_reason = reason
            self._write_loadout_price_counts_locked()
            self._loadout_price_count_updates = 0
            self._price_file.flush()
            self._operations_file.flush()
            prices = self._read_csv(self.price_path)
            operations = self._read_csv(self.operations_path)
            sessions = [self._copy_session(row) for row in self._sessions]
            binding = dict(self._binding)
            last_reason = self._last_reason
            summary_rows = self._build_summary_rows(sessions, operations)
            breakdown_rows = self._build_breakdown_rows(operations)
            self._write_csv_atomic(self.summary_path, self._SUMMARY_FIELDS, summary_rows)
            self._write_csv_atomic(
                self.breakdown_path, self._BREAKDOWN_FIELDS, breakdown_rows
            )
            self._write_sessions_locked()
        html = self._build_report(
            prices,
            operations,
            sessions,
            summary_rows,
            breakdown_rows,
            binding,
            last_reason,
        )
        temporary = self.report_path.with_suffix(".tmp")
        temporary.write_text(html, encoding="utf-8")
        temporary.replace(self.report_path)
        return self.report_path

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, Any]]:
        with path.open(encoding="utf-8-sig", newline="") as source:
            return [dict(row) for row in csv.DictReader(source)]

    def _write_loadout_price_counts_locked(self) -> None:
        rows = sorted(
            self._loadout_price_counts.values(),
            key=lambda row: (
                str(row["session_id"]),
                int(row["scheme_index"]),
                int(row["recognized_price"]),
            ),
        )
        self._write_csv_atomic(
            self.loadout_price_counts_path,
            self._LOADOUT_PRICE_COUNT_FIELDS,
            rows,
        )

    def close(self, reason: str = "应用正常关闭") -> None:
        with self._lock:
            if self._closed:
                return
        self.end_session(reason)
        self.log(reason, level="STOP")
        self.generate_report(reason)
        with self._lock:
            self._closed = True
            self._log_file.close()
            self._price_file.close()
            self._operations_file.close()
            self._balance_file.close()
            self._performance_file.close()

    def _build_summary_rows(
        self,
        sessions: list[dict[str, Any]],
        operations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for session in sessions:
            session_id = str(session["session_id"])
            session_operations = [
                row for row in operations if row["session_id"] == session_id
            ]
            rules = session.get("rules", [])
            product_names = [
                str(rule.get("name", "")).strip()
                for rule in rules
                if str(rule.get("name", "")).strip()
            ]
            fee_percent = float(session.get("fee_percent", 13.0))
            for product in product_names:
                product_operations = [
                    row for row in session_operations if row["product"] == product
                ]
                buys = [
                    row
                    for row in product_operations
                    if row["type"] == "BUY_CONFIRMED"
                    and row["included_in_cost"] == "是"
                    and int(row["quantity"]) > 0
                    and int(row["total_amount"]) > 0
                ]
                partial_buys = [
                    row
                    for row in product_operations
                    if row["type"] in {"LOADOUT_BUY_PARTIAL", "BUY_UNCERTAIN"}
                    and int(row.get("total_amount") or 0) > 0
                    and self._operation_estimated_quantity(row) > 0
                ]
                listings = [
                    row
                    for row in product_operations
                    if row["type"] in {"LISTING_CONFIRMED", "LOADOUT_LISTED"}
                ]
                probe_sales = [
                    row
                    for row in product_operations
                    if row["type"] == "SELL_PROBE_CONFIRMED"
                ]
                quantity = sum(int(row["quantity"]) for row in buys)
                total_cost = sum(int(row["total_amount"]) for row in buys)
                weighted_cost = total_cost / quantity if quantity else 0.0
                estimated_quantity = sum(
                    self._operation_estimated_quantity(row) for row in partial_buys
                )
                estimated_cost = sum(
                    int(row["total_amount"]) for row in partial_buys
                )
                quantity_with_estimates = quantity + estimated_quantity
                cost_with_estimates = total_cost + estimated_cost
                weighted_cost_with_estimates = (
                    cost_with_estimates / quantity_with_estimates
                    if quantity_with_estimates
                    else 0.0
                )
                rule = next(
                    (row for row in rules if str(row.get("name", "")) == product),
                    {},
                )
                expected_sell_price = int(rule.get("sell_trigger_price") or 0)
                net_unit_profit = (
                    expected_sell_price * (1.0 - fee_percent / 100.0)
                    - weighted_cost_with_estimates
                    if expected_sell_price > 0 and quantity_with_estimates > 0
                    else 0.0
                )
                result.append(
                    {
                        "run_id": self.run_id,
                        "session_id": session_id,
                        "session_start_beijing": session["started_at"],
                        "session_end_beijing": session["ended_at"],
                        "product": product,
                        "buy_quantity": quantity,
                        "buy_total_cost": total_cost,
                        "weighted_average_cost": f"{weighted_cost:.4f}",
                        "confirmed_buy_records": len(buys),
                        "estimated_partial_buy_quantity": estimated_quantity,
                        "estimated_partial_buy_cost": estimated_cost,
                        "partial_buy_records": len(partial_buys),
                        "buy_quantity_including_estimates": quantity_with_estimates,
                        "buy_total_cost_including_estimates": cost_with_estimates,
                        "weighted_average_cost_including_estimates": (
                            f"{weighted_cost_with_estimates:.4f}"
                        ),
                        "listed_quantity": sum(int(row["quantity"]) for row in listings),
                        "listing_expected_income": sum(
                            int(row["total_amount"]) for row in listings
                        ),
                        "confirmed_probe_sold_quantity": sum(
                            int(row["quantity"]) for row in probe_sales
                        ),
                        "default_expected_sell_price": expected_sell_price,
                        "fee_percent": f"{fee_percent:.2f}",
                        "estimated_net_unit_profit": f"{net_unit_profit:.4f}",
                        "estimated_total_profit": f"{net_unit_profit * quantity_with_estimates:.2f}",
                    }
                )
        return result

    @staticmethod
    def _operation_estimated_quantity(row: Mapping[str, Any]) -> int:
        """Return a persisted or legacy-compatible inferred bullet count."""

        value = row.get("estimated_quantity", "")
        try:
            quantity = int(float(value)) if value not in (None, "") else 0
        except (TypeError, ValueError):
            quantity = 0
        if quantity > 0:
            return quantity
        # Reports generated before the estimate columns were introduced may
        # still contain a partial row.  Reconstruct the estimate from its
        # balance delta and displayed offer price when possible.
        if row.get("type") == "BUY_UNCERTAIN":
            try:
                return max(0, int(float(row.get("quantity", 0) or 0)))
            except (TypeError, ValueError):
                return 0
        if row.get("type") != "LOADOUT_BUY_PARTIAL":
            return 0
        try:
            before = int(float(row.get("balance_before", "")))
            after = int(float(row.get("balance_after", "")))
            unit_price = float(row.get("displayed_unit_price", ""))
        except (TypeError, ValueError):
            return 0
        if before <= after or unit_price <= 0:
            return 0
        return max(0, int(round((before - after) / unit_price)))

    def _build_breakdown_rows(
        self, operations: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for row in operations:
            is_confirmed = (
                row["type"] == "BUY_CONFIRMED"
                and row["included_in_cost"] == "是"
                and int(row["quantity"]) > 0
            )
            is_partial = (
                row["type"] in {"LOADOUT_BUY_PARTIAL", "BUY_UNCERTAIN"}
                and self._operation_estimated_quantity(row) > 0
            )
            if not (is_confirmed or is_partial):
                continue
            source_price = (
                row.get("actual_unit_price")
                or row.get("estimated_unit_price")
                or row.get("displayed_unit_price")
            )
            price_key = f"{float(source_price or 0):.2f}"
            key = (str(row["session_id"]), str(row["product"]), price_key)
            groups.setdefault(key, []).append(row)
        result: list[dict[str, Any]] = []
        for (session_id, product, price), rows in sorted(groups.items()):
            result.append(
                {
                    "run_id": self.run_id,
                    "session_id": session_id,
                    "product": product,
                    "actual_unit_price": price,
                    "estimated_records": sum(
                        1
                        for row in rows
                        if row["type"] in {"LOADOUT_BUY_PARTIAL", "BUY_UNCERTAIN"}
                    ),
                    "quantity": sum(
                        (
                            int(row["quantity"])
                            if row["type"] == "BUY_CONFIRMED"
                            else self._operation_estimated_quantity(row)
                        )
                        for row in rows
                    ),
                    "total_cost": sum(int(row["total_amount"]) for row in rows),
                    "confirmed_records": sum(
                        1 for row in rows if row["type"] == "BUY_CONFIRMED"
                    ),
                    "quantity_source": (
                        "余额反推（估算）"
                        if any(
                            row["type"] in {"LOADOUT_BUY_PARTIAL", "BUY_UNCERTAIN"}
                            for row in rows
                        )
                        else "确认"
                    ),
                    "first_buy_time_beijing": rows[0]["beijing_time"],
                    "last_buy_time_beijing": rows[-1]["beijing_time"],
                }
            )
        return result

    def _build_report(
        self,
        prices: list[dict[str, Any]],
        operations: list[dict[str, Any]],
        sessions: list[dict[str, Any]],
        summary_rows: list[dict[str, Any]],
        breakdown_rows: list[dict[str, Any]],
        binding: dict[str, Any],
        reason: str,
    ) -> str:
        now = self._now()
        duration = now - self.started_at
        confirmed_buys = [
            row
            for row in operations
            if row["type"] == "BUY_CONFIRMED" and row["included_in_cost"] == "是"
        ]
        buy_quantity = sum(int(row["quantity"]) for row in confirmed_buys)
        buy_cost = sum(int(row["total_amount"]) for row in confirmed_buys)
        partial_buys = [
            row
            for row in operations
            if row["type"] in {"LOADOUT_BUY_PARTIAL", "BUY_UNCERTAIN"}
        ]
        estimated_partial_quantity = sum(
            self._operation_estimated_quantity(row) for row in partial_buys
        )
        estimated_partial_cost = sum(
            int(row["total_amount"]) for row in partial_buys
        )
        total_buy_quantity = buy_quantity + estimated_partial_quantity
        total_buy_cost = buy_cost + estimated_partial_cost
        listing_quantity = sum(
            int(row["quantity"])
            for row in operations
            if row["type"] in {"LISTING_CONFIRMED", "LOADOUT_LISTED"}
        )
        binding_text = (
            f"{escape(str(binding.get('title', '未绑定')))} "
            f"{binding.get('width', '--')} × {binding.get('height', '--')}"
        )
        loadout_session_ids = {
            str(session["session_id"])
            for session in sessions
            if session.get("session_type") == "loadout_purchase"
        }
        trading_summary_rows = [
            row
            for row in summary_rows
            if str(row["session_id"]) not in loadout_session_ids
        ]
        overall_rows = self._overall_summary_rows(trading_summary_rows)
        session_overview_html = self._session_overview_table(sessions, operations)
        calculator_fee = (
            float(sessions[-1].get("fee_percent", 13.0)) if sessions else 13.0
        )
        overall_summary_html = (
            '<h2>自动交易汇总</h2>'
            '<div class="panel">'
            '<b>预期利润计算器</b>'
            '<span class="muted">　手续费：</span>'
            f'<input id="fee-global" type="number" value="{calculator_fee:g}" min="0" max="99" step="0.1"> %'
            '<div class="muted" style="margin-top:8px">交易行挂牌价与预计到手价会自动换算；默认到手比例为 87%。两者都可以手工输入，所有利润均为预计值，不代表挂单已经成交。</div>'
            '</div>'
            + self._summary_table(overall_rows, "ALL")
            if overall_rows
            else '<input id="fee-global" type="hidden" value="13">'
        )
        session_sections = "".join(
            self._session_section(
                session,
                [row for row in prices if row["session_id"] == session["session_id"]],
                [
                    row
                    for row in operations
                    if row["session_id"] == session["session_id"]
                ],
                [
                    row
                    for row in summary_rows
                    if row["session_id"] == session["session_id"]
                ],
                [
                    row
                    for row in breakdown_rows
                    if row["session_id"] == session["session_id"]
                ],
            )
            for session in sessions
        ) or '<div class="panel muted">尚未启动自动任务，本报告暂无会话数据。</div>'
        return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>天才交易员运行报告 {escape(self.run_id)}</title>
<style>
{REPORT_THEME_CSS}
</style>
</head>
<body><main>
<header class="report-header">
  <div class="eyebrow">三角洲行动 · 交易运行记录</div>
  <div class="report-head"><div>
    <h1>天才交易员</h1>
    <div class="meta">所有时间均为北京时间（UTC+8） | {binding_text}</div>
  </div><div class="run-tag">RUN {escape(self.run_id)}</div></div>
</header>
<div class="summary">
  <div class="metric">程序启动<b>{self._display_time(self.started_at.isoformat())}</b></div>
  <div class="metric">运行时长<b>{str(duration).split('.')[0]}</b></div>
  <div class="metric">自动任务会话<b>{len(sessions)}</b></div>
  <div class="metric">确认买入<b>{buy_quantity:,} 发</b></div>
  <div class="metric">成交总量（含估算）<b>{total_buy_quantity:,} 发</b></div>
  <div class="metric">确认上架<b>{listing_quantity:,} 发</b></div>
</div>
<div class="panel notice"><b>最近状态：</b>{escape(reason)}　<b>确认买入总成本：</b>{buy_cost:,}　<b>成交总成本（含估算）：</b>{total_buy_cost:,}　<b>部分成交估算：</b>{estimated_partial_quantity:,} 发 / {estimated_partial_cost:,}</div>
{session_overview_html}
{overall_summary_html}
{session_sections}
<p class="muted">“售卖上架”仅表示程序确认挂牌成功；只有“试单成交”才是程序确认的实际卖出。部分成交数量和综合成本按余额与购买前显示单价反推，仅作估算并单独标记，不等同于完全成交确认。</p>
</main>
<script>
function fmt(value, digits=2) {{
  if (!Number.isFinite(value)) return '--';
  return value.toLocaleString('zh-CN', {{minimumFractionDigits:digits, maximumFractionDigits:digits}});
}}
function recalcRow(row, changed) {{
  const globalFee = Number(document.getElementById('fee-global').value || 0);
  const quantity = Number(row.dataset.quantity || 0);
  const costInput = row.querySelector('.cost-price');
  const cost = Number(costInput.value || row.dataset.cost || 0);
  const listingInput = row.querySelector('.listing-price');
  const netInput = row.querySelector('.net-price');
  const fee = globalFee / 100;
  let listing = Number(listingInput.value || 0);
  let net = Number(netInput.value || 0);
  if (changed === 'net') {{
    listing = fee < 1 ? net / (1-fee) : 0;
    listingInput.value = listing > 0 ? Math.ceil(listing) : 0;
  }} else {{
    net = listing * (1-fee);
    netInput.value = net > 0 ? net.toFixed(1) : 0;
  }}
  const breakEven = cost > 0 && fee < 1 ? cost / (1-fee) : 0;
  const unitProfit = net - cost;
  const totalProfit = unitProfit * quantity;
  const roi = cost > 0 ? unitProfit / cost * 100 : 0;
  row.querySelector('.break-even').textContent = fmt(breakEven, 1);
  const unitCell = row.querySelector('.unit-profit');
  const totalCell = row.querySelector('.total-profit');
  unitCell.textContent = fmt(unitProfit, 1);
  totalCell.textContent = fmt(totalProfit, 0);
  row.querySelector('.profit-rate').textContent = fmt(roi, 1) + '%';
  [unitCell,totalCell].forEach(cell => {{ cell.className = cell.className.replace(/ positive| negative/g,'') + (unitProfit >= 0 ? ' positive' : ' negative'); }});
}}
function recalcAll() {{
  document.querySelectorAll('tr.calc-row').forEach(row => recalcRow(row, 'listing'));
}}
document.getElementById('fee-global').addEventListener('input', recalcAll);
document.querySelectorAll('.listing-price').forEach(input => input.addEventListener('input', event => recalcRow(event.target.closest('tr'), 'listing')));
document.querySelectorAll('.net-price').forEach(input => input.addEventListener('input', event => recalcRow(event.target.closest('tr'), 'net')));
document.querySelectorAll('.cost-price').forEach(input => input.addEventListener('input', event => recalcRow(event.target.closest('tr'), 'listing')));
recalcAll();
</script>
</body></html>"""

    def _session_overview_table(
        self,
        sessions: list[dict[str, Any]],
        operations: list[dict[str, Any]],
    ) -> str:
        """Render one compact row per session plus a cross-session total."""

        rows: list[dict[str, Any]] = []
        for session in sessions:
            session_id = str(session["session_id"])
            scoped = [
                row for row in operations if str(row["session_id"]) == session_id
            ]
            confirmed_quantity = sum(
                int(row["quantity"])
                for row in scoped
                if row["type"] == "BUY_CONFIRMED"
                and row["included_in_cost"] == "是"
            )
            confirmed_cost = sum(
                int(row["total_amount"])
                for row in scoped
                if row["type"] == "BUY_CONFIRMED"
                and row["included_in_cost"] == "是"
            )
            partial_quantity = sum(
                self._operation_estimated_quantity(row)
                for row in scoped
                if row["type"] in {"LOADOUT_BUY_PARTIAL", "BUY_UNCERTAIN"}
            )
            partial_cost = sum(
                int(row["total_amount"])
                for row in scoped
                if row["type"] in {"LOADOUT_BUY_PARTIAL", "BUY_UNCERTAIN"}
            )
            listed_quantity = sum(
                int(row["quantity"])
                for row in scoped
                if row["type"] in {"LISTING_CONFIRMED", "LOADOUT_LISTED"}
            )
            rows.append(
                {
                    "session_id": session_id,
                    "session_type": (
                        "配装买入" if session.get("session_type") == "loadout_purchase"
                        else "自动交易"
                    ),
                    "confirmed_quantity": confirmed_quantity,
                    "partial_quantity": partial_quantity,
                    "total_quantity": confirmed_quantity + partial_quantity,
                    "confirmed_cost": confirmed_cost,
                    "partial_cost": partial_cost,
                    "total_cost": confirmed_cost + partial_cost,
                    "listed_quantity": listed_quantity,
                    "partial_records": sum(
                        1
                        for row in scoped
                        if row["type"] in {"LOADOUT_BUY_PARTIAL", "BUY_UNCERTAIN"}
                    ),
                    "balance_change": (
                        int(session["final_balance"]) - int(session["initial_balance"])
                        if session.get("initial_balance", "") != ""
                        and session.get("final_balance", "") != ""
                        else None
                    ),
                    "start": session.get("started_at", ""),
                    "end": session.get("ended_at", ""),
                }
            )
        if not rows:
            return '<h2>全部会话总览</h2><div class="panel muted">尚未产生会话。</div>'

        total = {
            key: sum(int(row[key]) for row in rows)
            for key in (
                "confirmed_quantity",
                "partial_quantity",
                "total_quantity",
                "confirmed_cost",
                "partial_cost",
                "total_cost",
                "listed_quantity",
                "partial_records",
            )
        }
        total["balance_change"] = sum(
            int(row["balance_change"] or 0) for row in rows
        )
        body = "".join(
            "<tr>"
            f'<td>{escape(str(row["session_id"]))}</td>'
            f'<td>{escape(str(row["session_type"]))}</td>'
            f'<td>{escape(self._display_time(str(row["start"])))}</td>'
            f'<td>{escape(self._display_time(str(row["end"]))) if row["end"] else "运行中"}</td>'
            f'<td>{int(row["confirmed_quantity"]):,}</td>'
            f'<td>{int(row["partial_quantity"]):,}</td>'
            f'<td><b>{int(row["total_quantity"]):,}</b></td>'
            f'<td>{int(row["total_cost"]):,}</td>'
            f'<td>{int(row["listed_quantity"]):,}</td>'
            f'<td>{int(row["partial_records"]):,}</td>'
            f'<td>{self._signed_number(row["balance_change"])}</td>'
            "</tr>"
            for row in rows
        )
        body += (
            "<tr>"
            '<td><b>全部会话</b></td><td>合计</td><td>--</td><td>--</td>'
            f'<td><b>{total["confirmed_quantity"]:,}</b></td>'
            f'<td><b>{total["partial_quantity"]:,}</b></td>'
            f'<td><b>{total["total_quantity"]:,}</b></td>'
            f'<td><b>{total["total_cost"]:,}</b></td>'
            f'<td><b>{total["listed_quantity"]:,}</b></td>'
            f'<td>{total["partial_records"]:,}</td>'
            f'<td>{self._signed_number(total["balance_change"])}</td>'
            "</tr>"
        )
        return (
            '<h2>全部会话总览</h2><div class="panel"><table><thead><tr>'
            '<th>会话</th><th>类型</th><th>开始</th><th>结束</th>'
            '<th>确认成交量</th><th>部分成交估算量</th><th>成交总量（含估算）</th>'
            '<th>成交总成本（含估算）</th><th>已上架量</th><th>部分成交记录</th><th>余额变化</th>'
            f'</tr></thead><tbody>{body}</tbody></table>'
            '<div class="muted" style="margin-top:10px">总览覆盖本次程序运行产生的所有会话；部分成交量由余额扣款 ÷ 购买前显示的平均单价反推。</div></div>'
        )

    def _overall_summary_rows(
        self, summary_rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in summary_rows:
            groups.setdefault(str(row["product"]), []).append(row)
        result: list[dict[str, Any]] = []
        for product, rows in sorted(groups.items()):
            quantity = sum(int(row["buy_quantity"]) for row in rows)
            total_cost = sum(int(row["buy_total_cost"]) for row in rows)
            weighted = total_cost / quantity if quantity else 0.0
            latest = rows[-1]
            result.append(
                {
                    **latest,
                    "session_id": "全部",
                    "product": product,
                    "buy_quantity": quantity,
                    "buy_total_cost": total_cost,
                    "weighted_average_cost": weighted,
                    "confirmed_buy_records": sum(
                        int(row["confirmed_buy_records"]) for row in rows
                    ),
                    "estimated_partial_buy_quantity": sum(
                        int(row.get("estimated_partial_buy_quantity", 0) or 0)
                        for row in rows
                    ),
                    "estimated_partial_buy_cost": sum(
                        int(row.get("estimated_partial_buy_cost", 0) or 0)
                        for row in rows
                    ),
                    "partial_buy_records": sum(
                        int(row.get("partial_buy_records", 0) or 0) for row in rows
                    ),
                    "buy_quantity_including_estimates": sum(
                        int(row.get("buy_quantity_including_estimates", 0) or 0)
                        for row in rows
                    ),
                    "buy_total_cost_including_estimates": sum(
                        int(row.get("buy_total_cost_including_estimates", 0) or 0)
                        for row in rows
                    ),
                    "weighted_average_cost_including_estimates": (
                        f"{(
                            sum(int(row.get('buy_total_cost_including_estimates', 0) or 0)
                                for row in rows)
                            / sum(int(row.get('buy_quantity_including_estimates', 0) or 0)
                                  for row in rows)
                        ):.4f}"
                        if sum(
                            int(row.get("buy_quantity_including_estimates", 0) or 0)
                            for row in rows
                        )
                        else "0.0000"
                    ),
                    "listed_quantity": sum(int(row["listed_quantity"]) for row in rows),
                    "listing_expected_income": sum(
                        int(row["listing_expected_income"]) for row in rows
                    ),
                    "confirmed_probe_sold_quantity": sum(
                        int(row["confirmed_probe_sold_quantity"]) for row in rows
                    ),
                }
            )
        return result

    def _summary_table(self, rows: list[dict[str, Any]], scope: str) -> str:
        summary_body = []
        calculator_body = []
        for row in rows:
            cost = float(
                row.get(
                    "weighted_average_cost_including_estimates",
                    row["weighted_average_cost"],
                )
            )
            confirmed_quantity = int(row["buy_quantity"])
            estimated_quantity = int(
                row.get("estimated_partial_buy_quantity", 0) or 0
            )
            total_quantity = int(
                row.get("buy_quantity_including_estimates", confirmed_quantity)
                or 0
            )
            total_cost = int(
                row.get("buy_total_cost_including_estimates", row["buy_total_cost"])
                or 0
            )
            expected_sell_price = int(row["default_expected_sell_price"])
            input_value = expected_sell_price or (round(cost) if cost > 0 else 0)
            summary_body.append(
                "<tr>"
                f'<td>{escape(str(row["product"]))}</td>'
                f'<td>{confirmed_quantity:,}</td>'
                f'<td>{estimated_quantity:,}</td>'
                f'<td>{int(row.get("estimated_partial_buy_cost", 0) or 0):,}</td>'
                f'<td><b>{total_quantity:,}</b></td>'
                f'<td>{total_cost:,}</td>'
                f'<td><b>{cost:,.2f}</b></td>'
                f'<td>{int(row["confirmed_buy_records"]):,}</td>'
                f'<td>{int(row.get("partial_buy_records", 0) or 0):,}</td>'
                f'<td>{int(row["listed_quantity"]):,}</td>'
                f'<td>{int(row["listing_expected_income"]):,}</td>'
                f'<td>{int(row["confirmed_probe_sold_quantity"]):,}</td>'
                "</tr>"
            )
            calculator_body.append(
                f'<tr class="calc-row" data-quantity="{total_quantity}" data-cost="{cost:.8f}">'
                f'<td>{escape(str(row["product"]))}</td>'
                f'<td><input class="cost-price" aria-label="{escape(str(row["product"]))}成本价" type="number" min="0" step="0.1" value="{cost:.2f}"></td>'
                f'<td><input class="listing-price" aria-label="{escape(str(row["product"]))}交易行挂牌价" type="number" min="0" step="1" value="{input_value}"></td>'
                '<td><input class="net-price" aria-label="预计到手价" type="number" min="0" step="0.1" value="0"></td>'
                '<td class="break-even">--</td>'
                '<td class="unit-profit">--</td><td class="total-profit">--</td><td class="profit-rate">--</td>'
                "</tr>"
            )
        if not summary_body:
            summary_body.append('<tr><td colspan="12" class="muted">本范围没有选中商品或成交记录。</td></tr>')
            calculator_body.append('<tr><td colspan="8" class="muted">暂无可试算商品。</td></tr>')
        return (
            f'<div class="panel" data-scope="{escape(scope)}"><table><thead><tr>'
            '<th>子弹类型</th><th>确认买入量</th><th>部分成交估算量</th><th>部分成交扣款</th>'
            '<th>成交总量（含估算）</th><th>成交总成本（含估算）</th>'
            '<th>加权平均成本（含估算）</th><th>确认买入记录</th><th>部分成交记录</th>'
            '<th>已上架量</th><th>挂牌预计到手金额</th><th>确认试单卖出</th>'
            "</tr></thead><tbody>" + "".join(summary_body) + "</tbody></table></div>"
            f'<div class="panel calculator" data-scope="{escape(scope)}"><table><thead><tr>'
            '<th>子弹类型</th><th>成本价（可改）</th><th>交易行挂牌价</th><th>预计到手价</th>'
            '<th>保本挂牌价</th><th>预计单发利润</th><th>预计总利润</th><th>成本利润率</th>'
            "</tr></thead><tbody>" + "".join(calculator_body) + "</tbody></table></div>"
        )

    def _session_section(
        self,
        session: dict[str, Any],
        prices: list[dict[str, Any]],
        operations: list[dict[str, Any]],
        summary_rows: list[dict[str, Any]],
        breakdown_rows: list[dict[str, Any]],
    ) -> str:
        is_loadout_purchase = session.get("session_type") == "loadout_purchase"
        initial = session.get("initial_balance", "")
        final = session.get("final_balance", "")
        balance_change = (
            int(final) - int(initial) if initial != "" and final != "" else None
        )
        rule_pills = "".join(
            f'<span class="pill">{escape(str(rule.get("name", "")))}</span>'
            for rule in session.get("rules", [])
        )
        product_sections = "".join(
            self._product_section(
                str(row["product"]),
                [price for price in prices if price["product"] == row["product"]],
                [op for op in operations if op["product"] == row["product"]],
                [detail for detail in breakdown_rows if detail["product"] == row["product"]],
            )
            for row in summary_rows
        )
        operation_rows = "".join(
            "<tr>"
            f'<td>{escape(self._display_time(str(row["beijing_time"])))}</td>'
            f'<td>{self._operation_label(str(row["type"]))}</td>'
            f'<td>{escape(str(row["product"]))}</td>'
            f'<td>{self._number(row["quantity"] if row["type"] == "BUY_CONFIRMED" else 0)}</td>'
            f'<td>{self._number(self._operation_estimated_quantity(row) if row["type"] in {"LOADOUT_BUY_PARTIAL", "BUY_UNCERTAIN"} else row.get("estimated_quantity"))}</td>'
            f'<td>{self._number(row.get("estimated_unit_price"))}</td>'
            f'<td>{escape(str(row.get("quantity_source", "确认")))}</td>'
            f'<td>{self._number(row["displayed_unit_price"])}</td>'
            f'<td>{self._number(row["actual_unit_price"])}</td>'
            f'<td>{self._number(row["total_amount"])}</td>'
            f'<td>{self._number(row["balance_before"])}</td>'
            f'<td>{self._number(row["balance_after"])}</td>'
            f'<td>{self._signed_number(self._balance_change(row))}</td>'
            f'<td>{escape(str(row["included_in_cost"]))}</td>'
            f'<td>{escape(str(row["result"]))}</td>'
            f'<td>{escape(str(row["details"]))}</td>'
            "</tr>"
            for row in operations
        ) or '<tr><td colspan="16" class="muted">本会话没有买入或上架操作。</td></tr>'
        summary_html = (
            self._loadout_summary_table(session, operations)
            if is_loadout_purchase
            else self._summary_table(summary_rows, str(session["session_id"]))
        )
        session_title = "配装买入法会话" if is_loadout_purchase else "会话"
        summary_title = (
            "本会话配装买入汇总"
            if is_loadout_purchase
            else "本会话汇总与利润计算"
        )
        detail_title = "低价与时间明细" if is_loadout_purchase else "价位与时间明细"
        return (
            '<section class="session">'
            '<div class="session-head"><div>'
            f'<h2 style="margin:0">{session_title} {escape(str(session["session_id"]))}</h2>'
            f'<div class="meta">{self._display_time(str(session["started_at"]))} → '
            f'{self._display_time(str(session["ended_at"])) if session["ended_at"] else "运行中"}</div>'
            f'<div style="margin-top:8px">{rule_pills}</div></div>'
            '<div>'
            f'<b>初始余额：</b>{self._number(initial)}　'
            f'<b>最终/最近余额：</b>{self._number(final)}　'
            f'<b>余额变化：</b>{self._signed_number(balance_change)}<br>'
            f'<span class="muted">结束状态：{escape(str(session["reason"]))}</span>'
            "</div></div>"
            f'<h3>{summary_title}</h3>'
            + summary_html
            + f'<h3>{detail_title}</h3>'
            + (product_sections or '<div class="muted">暂无价格样本或确认买入。</div>')
            + '<h3>按北京时间排序的操作记录</h3>'
            + '<div class="panel"><table><thead><tr><th>北京时间</th><th>类型</th><th>商品</th><th>确认数量</th>'
            '<th>估算数量</th><th>估算单价</th><th>数量来源</th><th>下单前显示价</th><th>余额反算均价</th><th>金额</th><th>交易前余额</th><th>交易后余额</th>'
            '<th>余额变化</th><th>计入成本</th><th>结果</th><th>详情</th>'
            f'</tr></thead><tbody>{operation_rows}</tbody></table></div></section>'
        )

    def _loadout_summary_table(
        self,
        session: dict[str, Any],
        operations: list[dict[str, Any]],
    ) -> str:
        products = [
            str(rule.get("name", "")).strip()
            for rule in session.get("rules", [])
            if str(rule.get("name", "")).strip()
        ]

        def summarize(product: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
            offers = [row for row in rows if row["type"] == "LOADOUT_LOW_PRICE"]
            full = [row for row in rows if row["type"] == "BUY_CONFIRMED"]
            included = [row for row in full if row["included_in_cost"] == "是"]
            partial = [row for row in rows if row["type"] == "LOADOUT_BUY_PARTIAL"]
            failed = [row for row in rows if row["type"] == "LOADOUT_BUY_FAILED"]
            listed = [row for row in rows if row["type"] == "LOADOUT_LISTED"]
            inferred_sales = [
                row for row in rows if row["type"] == "LOADOUT_SLOT_RELEASED"
            ]
            unlisted = [row for row in rows if row["type"] == "LOADOUT_UNLISTED"]
            mailed = [row for row in rows if row["type"] == "LOADOUT_MAIL_STORED"]
            quantity = sum(int(row["quantity"]) for row in included)
            cost = sum(int(row["total_amount"]) for row in included)
            partial_quantity = sum(
                self._operation_estimated_quantity(row) for row in partial
            )
            partial_cost = sum(int(row["total_amount"]) for row in partial)
            total_quantity = quantity + partial_quantity
            total_cost = cost + partial_cost
            return {
                "product": product,
                "offers": len(offers),
                "lowest": min((int(row["total_amount"]) for row in offers), default=0),
                "full": len(full),
                "partial": len(partial),
                "failed": len(failed),
                "quantity": quantity,
                "cost": cost,
                "average": cost / quantity if quantity else 0.0,
                "partial_quantity": partial_quantity,
                "partial_cost": partial_cost,
                "total_quantity": total_quantity,
                "total_cost": total_cost,
                "total_average": total_cost / total_quantity if total_quantity else 0.0,
                "listed_quantity": sum(int(row["quantity"]) for row in listed),
                "inferred_sold_quantity": sum(
                    int(row["quantity"]) for row in inferred_sales
                ),
                "unlisted": len(unlisted),
                "mailed": len(mailed),
            }

        summaries = [
            summarize(
                product,
                [row for row in operations if row["product"] == product],
            )
            for product in products
        ]
        if len(summaries) > 1:
            summaries.append(summarize("全部方案", operations))
        rows = "".join(
            "<tr>"
            f'<td>{escape(str(row["product"]))}</td>'
            f'<td>{int(row["offers"]):,}</td>'
            f'<td>{self._number(row["lowest"]) if row["lowest"] else "--"}</td>'
            f'<td>{int(row["full"]):,}</td>'
            f'<td>{int(row["partial"]):,}</td>'
            f'<td>{int(row["failed"]):,}</td>'
            f'<td>{int(row["quantity"]):,}</td>'
            f'<td>{int(row["cost"]):,}</td>'
            f'<td><b>{float(row["average"]):,.2f}</b></td>'
            f'<td>{int(row["partial_quantity"]):,}</td>'
            f'<td>{int(row["partial_cost"]):,}</td>'
            f'<td><b>{int(row["total_quantity"]):,}</b></td>'
            f'<td>{int(row["total_cost"]):,}</td>'
            f'<td><b>{float(row["total_average"]):,.2f}</b></td>'
            f'<td>{int(row["listed_quantity"]):,}</td>'
            f'<td>{int(row["inferred_sold_quantity"]):,}</td>'
            f'<td>{int(row["unlisted"]):,}</td>'
            f'<td>{int(row["mailed"]):,}</td>'
            "</tr>"
            for row in summaries
        ) or '<tr><td colspan="18" class="muted">本会话尚无配装买入记录。</td></tr>'
        return (
            '<div class="panel"><table><thead><tr><th>方案</th><th>发现低价次数</th>'
            '<th>最低整套价</th><th>全部成交</th><th>部分成交</th><th>完全未成交</th>'
            '<th>完整成交数量</th><th>完整成交成本</th><th>单发加权平均价</th>'
            '<th>部分成交估算数量</th><th>部分成交扣款</th>'
            '<th>成交总量（含估算）</th><th>成交总成本（含估算）</th>'
            '<th>综合单发平均价</th>'
            '<th>已上架数量</th><th>售位释放推定售出数量（非成交确认）</th>'
            '<th>下架次数</th><th>卡邮件次数</th>'
            f'</tr></thead><tbody>{rows}</tbody></table>'
            '<div class="muted" style="margin-top:10px">部分成交数量按“购买前显示的整套平均单价 × 余额实际扣款”反推，仅为估算；完整成交数量仍以确认记录为准。</div></div>'
        )

    def _product_section(
        self,
        product: str,
        rows: list[dict[str, Any]],
        operations: list[dict[str, Any]],
        breakdown_rows: list[dict[str, Any]],
    ) -> str:
        breakdown_html = "".join(
            "<tr>"
            f'<td>{self._number(row["actual_unit_price"])}</td>'
            f'<td>{int(row["quantity"]):,}</td>'
            f'<td>{int(row["total_cost"]):,}</td>'
            f'<td>{int(row["confirmed_records"]):,}</td>'
            f'<td>{int(row.get("estimated_records", 0) or 0):,}</td>'
            f'<td>{escape(str(row.get("quantity_source", "确认")))}</td>'
            f'<td>{escape(self._display_time(str(row["first_buy_time_beijing"])))}</td>'
            f'<td>{escape(self._display_time(str(row["last_buy_time_beijing"])))}</td>'
            "</tr>"
            for row in breakdown_rows
        ) or '<tr><td colspan="8" class="muted">没有可计入成本的确认买入。</td></tr>'
        chart = self._price_svg(rows, operations) if rows else ""
        price_summary = ""
        if rows:
            values = [int(row["price"]) for row in rows]
            price_summary = (
                f'<div class="muted">价格样本 {len(values)} | 低 {min(values):,} | '
                f'高 {max(values):,} | 中位 {median(values):,.0f}</div>'
            )
        return (
            '<div class="panel">'
            f'<h3 style="margin-top:0">{escape(product)}</h3>{price_summary}{chart}'
            '<table><thead><tr><th>实际/估算平均价位</th><th>成交数量</th><th>总成本</th>'
            '<th>确认记录数</th><th>估算记录数</th><th>数量来源</th>'
            '<th>首次买入（北京时间）</th><th>最后买入（北京时间）</th>'
            f'</tr></thead><tbody>{breakdown_html}</tbody></table></div>'
        )

    def _price_svg(
        self,
        rows: list[dict[str, Any]],
        operations: list[dict[str, Any]],
    ) -> str:
        width, height = 960, 260
        left, right, top, bottom = 64, 22, 24, 46
        plot_width = width - left - right
        plot_height = height - top - bottom
        times = [self._parse_time(str(row["beijing_time"])) for row in rows]
        prices = [int(row["price"]) for row in rows]
        priced_operations = [
            row
            for row in operations
            if float(
                row.get("actual_unit_price")
                or row.get("estimated_unit_price")
                or 0
            )
            > 0
        ]
        operation_times = [
            self._parse_time(str(row["beijing_time"])) for row in priced_operations
        ]
        operation_prices = [
            float(
                row.get("actual_unit_price")
                or row.get("estimated_unit_price")
                or 0
            )
            for row in priced_operations
        ]
        min_time, max_time = min(times + operation_times), max(times + operation_times)
        all_prices = [float(value) for value in prices] + operation_prices
        min_price, max_price = min(all_prices), max(all_prices)
        if min_time == max_time:
            max_time = min_time + 1.0
        if min_price == max_price:
            padding = max(1, round(min_price * 0.02))
            min_price -= padding
            max_price += padding
        else:
            padding = max(1, round((max_price - min_price) * 0.08))
            min_price = max(0, min_price - padding)
            max_price += padding

        def point(timestamp: float, price: float) -> tuple[float, float]:
            x = left + (timestamp - min_time) / (max_time - min_time) * plot_width
            y = top + (max_price - price) / (max_price - min_price) * plot_height
            return x, y

        all_points = [point(timestamp, price) for timestamp, price in zip(times, prices)]
        polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in all_points)
        grid: list[str] = []
        labels: list[str] = []
        for index in range(5):
            ratio = index / 4
            y = top + ratio * plot_height
            price = round(max_price - ratio * (max_price - min_price))
            grid.append(
                f'<line class="chart-grid" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" />'
            )
            labels.append(
                f'<text class="chart-label" x="{left-8}" y="{y+4:.1f}" text-anchor="end">{price:,}</text>'
            )
        circles = []
        for row, (x, y) in zip(rows, all_points):
            color = self._SOURCE_COLORS.get(str(row["source"]), "#66717e")
            circles.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{color}"><title>{escape(self._display_time(str(row["beijing_time"])))} {escape(str(row["source"]))}: {int(row["price"]):,}</title></circle>'
            )
        markers = []
        for operation in priced_operations:
            price = float(
                operation.get("actual_unit_price")
                or operation.get("estimated_unit_price")
                or 0
            )
            x, y = point(self._parse_time(str(operation["beijing_time"])), price)
            color = (
                "#43edb8"
                if operation["type"] == "BUY_CONFIRMED"
                else "#e6b54d"
            )
            markers.append(
                f'<path d="M {x:.1f} {y-8:.1f} L {x+7:.1f} {y+6:.1f} L {x-7:.1f} {y+6:.1f} Z" fill="{color}"><title>{self._operation_label(str(operation["type"]))} {(int(operation.get("quantity", 0) or 0) if operation["type"] == "BUY_CONFIRMED" else self._operation_estimated_quantity(operation)):,} 发</title></path>'
            )
        start_label = datetime.fromtimestamp(min_time, BEIJING_TIMEZONE).strftime("%H:%M:%S")
        end_label = datetime.fromtimestamp(max_time, BEIJING_TIMEZONE).strftime("%H:%M:%S")
        return (
            f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="价格趋势">'
            + "".join(grid)
            + "".join(labels)
            + f'<line class="chart-axis" x1="{left}" y1="{top+plot_height}" x2="{width-right}" y2="{top+plot_height}" />'
            + f'<polyline class="chart-line" points="{polyline}" />'
            + "".join(circles)
            + "".join(markers)
            + f'<text class="chart-label" x="{left}" y="{height-14}">{start_label}</text>'
            + f'<text class="chart-label" x="{width-right}" y="{height-14}" text-anchor="end">{end_label}</text>'
            + "</svg>"
        )

    def _canonical_active_product(self, product: str) -> str | None:
        normalized = self._normalize_product(product)
        with self._lock:
            if self._closed or not self._active_session_id:
                return None
            return self._active_products.get(normalized)

    def _active_session_locked(self) -> dict[str, Any] | None:
        return next(
            (
                session
                for session in reversed(self._sessions)
                if session["session_id"] == self._active_session_id
            ),
            None,
        )

    def _write_sessions_locked(self) -> None:
        rows = []
        for session in self._sessions:
            initial = session.get("initial_balance", "")
            final = session.get("final_balance", "")
            balance_change = (
                int(final) - int(initial) if initial != "" and final != "" else ""
            )
            rows.append(
                {
                    "run_id": self.run_id,
                    "session_id": session["session_id"],
                    "session_type": session.get("session_type", "trading"),
                    "start_time_beijing": session["started_at"],
                    "end_time_beijing": session["ended_at"],
                    "initial_balance": initial,
                    "final_balance": final,
                    "balance_change": balance_change,
                    "fee_percent": f"{float(session.get('fee_percent', 13.0)):.2f}",
                    "selected_products": "、".join(
                        str(rule.get("name", "")) for rule in session.get("rules", [])
                    ),
                    "reason": session["reason"],
                }
            )
        self._write_csv_atomic(self.sessions_path, self._SESSION_FIELDS, rows)

    @staticmethod
    def _write_csv_atomic(
        path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]
    ) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)

    @staticmethod
    def _copy_session(session: dict[str, Any]) -> dict[str, Any]:
        copied = dict(session)
        copied["rules"] = [dict(rule) for rule in session.get("rules", [])]
        return copied

    @staticmethod
    def _normalize_product(value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value).casefold()
        return re.sub(r"[^\w\u4e00-\u9fff]", "", normalized)

    @staticmethod
    def _parse_time(value: str) -> float:
        return datetime.fromisoformat(value).timestamp()

    @staticmethod
    def _now() -> datetime:
        return datetime.now(BEIJING_TIMEZONE)

    @classmethod
    def _timestamp(cls) -> str:
        return cls._now().isoformat(sep=" ", timespec="milliseconds")

    @staticmethod
    def _display_time(value: str) -> str:
        if not value:
            return "--"
        try:
            moment = datetime.fromisoformat(value)
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=BEIJING_TIMEZONE)
            return moment.astimezone(BEIJING_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return value

    @staticmethod
    def _number(value: Any) -> str:
        if value in (None, ""):
            return "--"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return escape(str(value))
        if numeric.is_integer():
            return f"{int(numeric):,}"
        return f"{numeric:,.2f}"

    @classmethod
    def _signed_number(cls, value: Any) -> str:
        if value in (None, ""):
            return "--"
        return f"{int(value):+,}"

    @staticmethod
    def _balance_change(row: Mapping[str, Any]) -> int | None:
        before = row.get("balance_before", "")
        after = row.get("balance_after", "")
        if before in (None, "") or after in (None, ""):
            return None
        return int(after) - int(before)

    @staticmethod
    def _operation_label(value: str) -> str:
        return {
            "BUY_CONFIRMED": "买入成交",
            "BUY_UNCERTAIN": "买入部分成交（余额估算）",
            "LISTING_CONFIRMED": "售卖上架",
            "SELL_PROBE_CONFIRMED": "试单成交",
            "LISTING_CANCELLED": "试单下架",
            "LOADOUT_LOW_PRICE": "配装低价",
            "LOADOUT_BUY_PARTIAL": "配装部分成交",
            "LOADOUT_BUY_FAILED": "配装未成交",
            "LOADOUT_BUY_UNKNOWN": "配装成交未知",
            "LOADOUT_LISTED": "配装上架（未确认售出）",
            "LOADOUT_SLOT_RELEASED": "售位释放（推定售出）",
            "LOADOUT_UNLISTED": "配装挂单下架",
            "LOADOUT_MAIL_STORED": "配装卡邮件",
        }.get(value, escape(value))
