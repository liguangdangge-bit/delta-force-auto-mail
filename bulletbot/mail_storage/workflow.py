from __future__ import annotations

import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from bulletbot.ammunition_sale import (
    AmmunitionMailClaimResult,
    AmmunitionMailClaimWorkflow,
    AmmunitionSaleOutcome,
    AmmunitionSaleRule,
    AmmunitionSaleResult,
    AmmunitionSaleRunner,
    AmmunitionSaleSettings,
    AmmunitionSaleStatus,
)
from bulletbot.ammunition_sale.mail_vision import selected_mail_tab
from bulletbot.navigation.adapters import (
    adapt_mail_observation,
    adapt_trading_observation,
)
from bulletbot.navigation.models import (
    GamePageId,
    NavigationStatus,
    UnifiedPageObservation,
)
from bulletbot.navigation.recognition import (
    PageStabilityTracker,
    resolve_page_observations,
)
from bulletbot.ocr.numbers import (
    has_valid_ocr_thousands_grouping,
    parse_ocr_integer,
)
from bulletbot.loadout_purchase import (
    ListingPriceMode,
    LoadoutCheckpointStage,
    LoadoutPurchaseCheckpoint,
    LoadoutPurchaseCheckpointStore,
    LoadoutResumeTarget,
    LoadoutPurchaseSettings,
    LoadoutSchemeRule,
    PARTIAL_TRANSFER_SCHEME_INDEX,
    PendingWarehouseListing,
    SellSlotRecoveryOutcome,
    SellSlotRecoveryResult,
    WarehouseListedBatch,
    WarehouseSellOutcome,
    WarehouseSellRequest,
    WarehouseSellResult,
)
from bulletbot.platform.input_ownership import InputLease
from bulletbot.platform.launcher_recovery import LauncherRecovery
from bulletbot.platform.game_process_control import (
    ProcessExitCleanup,
    ProcessRecord,
    ProcessTreeTermination,
    terminate_game_process_tree,
)
from bulletbot.vision.page_recognizer import PageRecognizer

from .capture import CapturedFrame, ScreenCapture
from .diagnostics import RunDiagnostics
from .input_control import InputController
from .navigation import MailUnifiedNavigation
from .models import (
    MailWorkflowOutcome,
    MailWorkflowResult,
    PageObservation,
    PageType,
    Rect,
    WorkflowEvent,
    WorkflowState,
    WindowInfo,
    page_type_label,
)
from .pak_transaction import PakTransaction, PakTransactionError
from .settings import AppSettings
from .vision import VisionEngine
from .windows import WindowBindingManager, enumerate_windows, matching_process_running


EventCallback = Callable[[WorkflowEvent], None]
FrameCallback = Callable[[np.ndarray, PageObservation, str], None]

LONGBOW_DOWNLOAD_TIMEOUT_SECONDS = 600
WARFARE_SETTLEMENT_TIMEOUT_SECONDS = 30
LOADOUT_SCHEME_PAGES = {
    PageType.LOADOUT_SCHEMES,
    PageType.MAIL_SCHEME_SELECTED,
}
INTRO_CONTINUE_PAGES = {
    PageType.GAME_TRANSITION,
    PageType.ACTIVITY_REMINDER,
}
INTRO_COMPLETE_PAGES = {
    PageType.GAME_HOME_PREPARE,
    PageType.GAME_HOME_READY,
    PageType.MAP_OVERVIEW,
    PageType.MAP_SELECTION,
    PageType.MAP_OTHER,
    PageType.MAP_ZERO_DAM_START,
    PageType.MAP_LONGBOW_START,
    PageType.MAP_LONGBOW_DOWNLOAD,
}
INTRO_PROGRESS_PAGES = INTRO_CONTINUE_PAGES | INTRO_COMPLETE_PAGES
UNIFIED_HOME_FAST_PATH_ANCHORS = {
    PageType.GAME_HOME_PREPARE: ("home_prepare_button",),
    PageType.GAME_HOME_READY: (
        "season_2026_home_depart_buttons",
        "season_2026_home_depart_buttons",
    ),
}
UNIFIED_HOME_FAST_PATH_MIN_CONFIDENCE = 0.85
LOADOUT_SCHEME_INITIAL_SETTLE_SECONDS = 0.08
LONGBOW_DOWNLOAD_OBSERVATION_INTERVAL_SECONDS = 3.0
LOADOUT_SCHEME_RETRY_INTERVAL_SECONDS = 0.005
LOADOUT_SCHEME_FALLBACK_RETRY_INTERVAL_SECONDS = 0.05
LOADOUT_HIGHLIGHT_UNCONFIRMED_RECOVERY_STREAK = 3
LOADOUT_HIGH_PRICE_LOG_INTERVAL_SECONDS = 10.0
MARKET_SELL_HOME_CLICK_DELAY_SECONDS = 2.0
LOADOUT_SCHEME_BORDER_BRIGHTNESS = 140
LOADOUT_SCHEME_BORDER_MIN_CONTINUITY = 0.65
# Fixed price text region shared by normal and fast loadout scanning.
LOADOUT_PRICE_OCR_RECT = (0.855, 0.788, 0.925, 0.83)
# Kept for callers that import the old setting; price scanning no longer uses
# the detector-based OCR path or this minimum-side-length value.
LOADOUT_PRICE_OCR_DETECTOR_MIN_SIDE = 320
LOADOUT_PRICE_OCR_MIN_CONFIDENCE = 0.90
FAST_LOADOUT_PRICE_RECT = LOADOUT_PRICE_OCR_RECT
FAST_LOADOUT_PRICE_SCALE = 3.0
FAST_LOADOUT_PRICE_FALLBACK_SCALES = (2.0, 2.75, 1.0)
FAST_LOADOUT_PRICE_MIN_CONFIDENCE = LOADOUT_PRICE_OCR_MIN_CONFIDENCE
FAST_LOADOUT_PRICE_MIN_VOTES = 2
FAST_LOADOUT_INVALID_OCR_RECOVERY_STREAK = 3
LOADOUT_PRICE_TRACE_POST_CLICK_SECONDS = 0.40
LOADOUT_PRICE_TRACE_INTERVAL_SECONDS = 0.10
LOADOUT_PRICE_MIN_RATIO = 0.50
LOADOUT_QUANTITY_INFERENCE_MIN_SAMPLES = 2
LOADOUT_QUANTITY_INFERENCE_MAX_SAMPLES = 5
LOADOUT_QUANTITY_INFERENCE_MAX_WARNINGS = 2
LOADOUT_QUANTITY_INFERENCE_TOLERANCE = 0.30
CURRENT_LOADOUT_CARD_RECT = (86, 275, 311, 336)
PARTIAL_TRANSFER_SAVE_SLOT_RECT = (966, 466, 1520, 555)
FIXED_CARD_BORDER_BRIGHTNESS = 140
FIXED_CARD_BORDER_MIN_RATIO = 0.15
LOADOUT_SCAN_TEMPLATE_NAMES = frozenset(
    {
        "real_loadout_schemes_header",
        "real_use_scheme_button",
        "real_loadout_purchase_confirm_header",
        "real_loadout_price_change_header",
    }
)

AGENT_SELECTION_TEMPLATE_NAMES = frozenset(
    {
        "real_agent_select_header",
        "real_agent_select_view_ability",
        "real_entry_warning_header",
        "real_entry_warning_continue",
    }
)
AGENT_SELECTION_RECOVERY_TEMPLATE_NAMES = (
    AGENT_SELECTION_TEMPLATE_NAMES | {"real_game_loading_art"}
)
AGENT_SELECTION_OCR_CROP = (
    50 / 1920,
    10 / 1080,
    480 / 1920,
    155 / 1080,
)

MAIL_ATTACHMENT_TEMPLATE_NAMES = frozenset(
    {
        "real_mail_partial_claim",
        "real_mail_partial_selected",
        "real_mail_claim_button",
        "real_mail_first_selected",
        "real_mail_second_selected",
    }
)
MAIL_ATTACHMENT_GEAR_KEYWORDS = ("胸挂", "战术背心", "战术包", "背包")
MAIL_ATTACHMENT_SCROLL_DELTA = -120
MAIL_ATTACHMENT_SCROLL_MAX_STEPS = 36
MAIL_ATTACHMENT_SCROLL_STABLE_FRAMES = 3
MAIL_ATTACHMENT_SCROLL_SETTLE_SECONDS = 0.25
MAIL_ATTACHMENT_STRIP_RECT = (570, 850, 1530, 1005)
MAIL_ATTACHMENT_CARD_TOP = 890
MAIL_ATTACHMENT_CARD_BOTTOM = 977
MAIL_ATTACHMENT_CARD_MIN_WIDTH = 68
MAIL_ATTACHMENT_CARD_MAX_WIDTH = 105
MAIL_ATTACHMENT_EDGE_GROUP_DISTANCE = 12
MAIL_ATTACHMENT_OUTLIER_MIN_SCORE = 12.0
MAIL_ATTACHMENT_OUTLIER_MIN_GAP = 5.0
MAIL_ATTACHMENT_OUTLIER_CONFIRMATIONS = 3
MAIL_ATTACHMENT_IDENTIFICATION_MAX_ATTEMPTS = 36
MAIL_ATTACHMENT_SELECTION_RETRY_SECONDS = 6.0
MAIL_ATTACHMENT_SAFE_CURSOR_POINT = (0.72, 0.72)


class WorkflowStopped(RuntimeError):
    pass


class ScheduledGameRestartRequested(RuntimeError):
    pass


class WorkflowTimeout(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        last_observation: PageObservation | None = None,
    ) -> None:
        super().__init__(message)
        self.last_observation = last_observation


class LoadoutSchemePurchaseRequired(RuntimeError):
    pass


class LauncherResourceAbsent(RuntimeError):
    pass


@dataclass(frozen=True)
class _StepOutcome:
    name: str
    pages: frozenset[PageType] = frozenset()
    templates: frozenset[str] = frozenset()
    predicate: Callable[[CapturedFrame, PageObservation], bool] | None = None
    single_ocr_confirmation_terms: frozenset[str] = frozenset()
    read_text: bool = False


@dataclass(frozen=True)
class _StepResult:
    outcome: str
    frame: CapturedFrame | None
    observation: PageObservation


@dataclass(frozen=True)
class _MailAttachmentCard:
    bounds: tuple[int, int, int, int]
    difference_score: float

    @property
    def center_normalized(self) -> tuple[float, float]:
        left, top, right, bottom = self.bounds
        return (
            ((left + right) / 2) / 1920,
            ((top + bottom) / 2) / 1080,
        )


class _AmmunitionMailClaimPort:
    _SYSTEM_TAB_POINT = (0.10, 0.07)
    _TAB_CONFIRM_ATTEMPTS = 8
    _MAIL_OPEN_TAB_CONFIRM_ATTEMPTS = 80
    _TAB_SETTLE_SECONDS = 0.35
    _MAIL_CLAIM_RETURN_TIMEOUT_SECONDS = 30.0
    _MAIL_CLAIM_RETURN_RETRY_DELAY_SECONDS = 2.0
    _MAIL_CLAIM_RETURN_SECOND_ATTEMPT_TIMEOUT_SECONDS = 5.0

    def __init__(self, controller: "WorkflowController") -> None:
        self._controller = controller

    def navigate_to_mail(self) -> bool:
        controller = self._controller
        game = controller._require_bound(controller.game, "游戏")
        _frame, current = controller._observe(
            game,
            "game",
            include_ocr=False,
            template_names=frozenset(
                {"real_mail_inbox_header", "real_mail_notification_icon"}
            ),
        )
        if current.page_type is PageType.MAIL_INBOX:
            return self._ensure_system_mail_tab()
        controller._prepare_game_for_trading_handoff()
        game = controller._require_bound(controller.game, "游戏")
        _frame, home = controller._observe(
            game,
            "game",
            include_ocr=False,
            template_names=frozenset(
                {"real_mail_inbox_header", "real_mail_notification_icon"}
            ),
        )
        if home.page_type is PageType.MAIL_INBOX:
            return self._ensure_system_mail_tab()
        controller._click_match(
            game,
            home,
            ("real_mail_notification_icon",),
            "mail_icon",
            "打开邮件",
            require_match=False,
        )
        if not self._ensure_system_mail_tab(wait_for_mail_open=True):
            return False
        controller._wait_page(
            controller.game,
            {PageType.MAIL_INBOX},
            30,
            "领邮件自动卖系统邮件附件界面",
            save_timeout=True,
        )
        return True

    def _ensure_system_mail_tab(self, *, wait_for_mail_open: bool = False) -> bool:
        controller = self._controller
        tab_names = ("系统", "通知", "交易行", "安全")
        clicked = False
        self.wait(self._TAB_SETTLE_SECONDS)
        attempts = (
            self._MAIL_OPEN_TAB_CONFIRM_ATTEMPTS
            if wait_for_mail_open
            else self._TAB_CONFIRM_ATTEMPTS
        )
        for _attempt in range(attempts):
            selected = selected_mail_tab(self.capture())
            if selected == 0:
                controller._emit("领邮件自动卖：已确认位于“系统”栏目")
                return True
            if selected is not None and not clicked:
                current = tab_names[selected]
                controller._emit(
                    f"领邮件自动卖：当前栏目为“{current}”，切换到“系统”"
                )
                self.click(self._SYSTEM_TAB_POINT, "切换邮件系统栏目")
                clicked = True
            self.wait(self._TAB_SETTLE_SECONDS)
        controller._emit(
            "领邮件自动卖：无法确认“系统”栏目高亮，停止邮件扫描",
            "error",
        )
        return False

    def capture(self) -> np.ndarray:
        controller = self._controller
        game = controller._require_bound(controller.game, "游戏")
        frame = controller.capture.capture(game)
        if not frame.healthy:
            raise RuntimeError("邮件页面截图无效")
        return frame.image

    def click(self, point: tuple[float, float], label: str) -> None:
        controller = self._controller
        game = controller._require_bound(controller.game, "游戏")
        controller._click_normalized(
            game,
            point,
            label,
            source="ammunition_mail_claim",
        )

    def scroll(
        self,
        point: tuple[float, float],
        delta: int,
        label: str,
    ) -> None:
        controller = self._controller
        game = controller._require_bound(controller.game, "游戏")
        controller._emit(f"滚动：{label} @ {point}，delta={delta}")
        controller.input.scroll_normalized(game, point, delta)
        controller.diagnostics.event(
            "input",
            action="scroll",
            label=label,
            point=point,
            delta=delta,
            hwnd=game.hwnd,
        )

    def close_attachment_detail(self) -> None:
        self.click((0.646, 0.403), "关闭邮件附件详情")

    def claim_all(self) -> bool:
        controller = self._controller
        game = controller._require_bound(controller.game, "游戏")
        controller._click(game, "mail_claim", "领取整封目标子弹邮件")
        try:
            controller._wait_page(
                controller.game,
                {PageType.MAIL_CLAIM_COMPLETE},
                30,
                "子弹邮件领取完成",
                save_timeout=True,
            )
        except WorkflowTimeout:
            return False

        def press_return(attempt: int) -> None:
            game = controller._require_bound(controller.game, "游戏")
            label = (
                "空格：领取完成后返回邮件"
                if attempt == 1
                else f"空格：领取完成后返回邮件（重试 {attempt}/3）"
            )
            controller._press_key(game, "space", label)

        def wait_for_inbox(timeout: float) -> bool:
            try:
                controller._wait_page(
                    controller.game,
                    {PageType.MAIL_INBOX},
                    timeout,
                    "领取后的邮件附件界面",
                    save_timeout=True,
                )
            except WorkflowTimeout:
                return False
            return True

        press_return(1)
        if wait_for_inbox(self._MAIL_CLAIM_RETURN_TIMEOUT_SECONDS):
            return True

        controller._emit(
            "领邮件自动卖：领取完成后返回邮件等待超时，"
            f"{self._MAIL_CLAIM_RETURN_RETRY_DELAY_SECONDS:g} 秒后第 2 次按空格重试",
            "warning",
        )
        self.wait(self._MAIL_CLAIM_RETURN_RETRY_DELAY_SECONDS)
        press_return(2)
        if wait_for_inbox(self._MAIL_CLAIM_RETURN_SECOND_ATTEMPT_TIMEOUT_SECONDS):
            return True

        controller._emit(
            "领邮件自动卖：第 2 次按空格后仍未返回邮件，"
            "5 秒后第 3 次按空格重试",
            "warning",
        )
        press_return(3)
        return wait_for_inbox(self._MAIL_CLAIM_RETURN_TIMEOUT_SECONDS)

    def wait(self, seconds: float) -> None:
        self._controller._sleep(seconds)

    def stopped(self) -> bool:
        return self._controller._stop.is_set()

    def safe_point(self) -> None:
        self._controller._pause_at_safe_point()


@dataclass
class StablePageTracker:
    required_observations: int = 2
    candidate: tuple[str, PageType] | None = None
    candidate_count: int = 0
    last_recorded: tuple[str, PageType] | None = None

    def update(self, target: str, page_type: PageType) -> bool:
        if page_type == PageType.INVALID_FRAME:
            self.candidate = None
            self.candidate_count = 0
            return False
        key = (target, page_type)
        if key == self.candidate:
            self.candidate_count += 1
        else:
            self.candidate = key
            self.candidate_count = 1
        if self.candidate_count < self.required_observations or key == self.last_recorded:
            return False
        self.last_recorded = key
        return True


@dataclass(frozen=True)
class LoadoutSchemeOffer:
    price: int | None
    usable: bool
    partially_sold_out: bool
    selection_confirmed: bool
    observation: PageObservation | None


@dataclass(frozen=True)
class _LoadoutSchemeRefresh:
    status: str
    selected_frame: CapturedFrame | None = None
    selected_region: Rect | None = None
    frame_size: tuple[int, int] | None = None
    observation: PageObservation | None = None
    reason: str = ""

    @property
    def confirmed(self) -> bool:
        return (
            self.status == "confirmed"
            and self.selected_frame is not None
            and self.selected_region is not None
            and self.frame_size is not None
        )


@dataclass(frozen=True)
class LoadoutPurchaseAttempt:
    purchased: bool
    price_changed: bool = False
    final_page: PageObservation | None = None

    def __bool__(self) -> bool:
        return self.purchased


@dataclass(frozen=True)
class LoadoutPurchaseSelection:
    rule: LoadoutSchemeRule
    listed_price: int
    actual_average: int | None
    partial_spent: int | None = None

    @property
    def partially_purchased(self) -> bool:
        return self.partial_spent is not None


@dataclass
class _LoadoutHighPriceSummary:
    started_at: float
    samples: int
    minimum: int
    maximum: int
    latest: int


@dataclass
class _LoadoutQuantityInference:
    samples: list[int] = field(default_factory=list)
    inferred_quantity: int | None = None
    candidates: tuple[int, ...] = ()
    last_guarded_price: int | None = None
    unresolved_warning_count: int = 0


class WorkflowController:
    def __init__(
        self,
        settings: AppSettings,
        game: WindowBindingManager,
        launcher: WindowBindingManager,
        launcher_settings: WindowBindingManager | None = None,
        on_event: EventCallback | None = None,
        on_frame: FrameCallback | None = None,
        input_lease: InputLease | None = None,
        diagnostics_root: Path | None = None,
        save_diagnostic_images: bool = True,
        loadout_checkpoint_store: LoadoutPurchaseCheckpointStore | None = None,
        loadout_purchase_settings_provider: Callable[[], LoadoutPurchaseSettings]
        | None = None,
    ) -> None:
        self.settings = settings
        self.game = game
        self.launcher = launcher
        self.launcher_settings = launcher_settings or WindowBindingManager(
            settings.launcher_settings_window,
            candidate_filter=lambda window: not (
                settings.launcher_window.title
                and window.title == settings.launcher_window.title
                and window.process_name.casefold()
                == settings.launcher_window.process_name.casefold()
            ),
        )
        self.on_event = on_event or (lambda _event: None)
        self.on_frame = on_frame or (lambda _image, _observation, _target: None)
        self.vision = VisionEngine()
        self._trading_page_recognizer = PageRecognizer()
        self._unified_page_stability = PageStabilityTracker()
        self.input = InputController(
            enabled=settings.allow_input,
            input_lease=input_lease,
        )
        self.capture = ScreenCapture(prepare_mss_window=self._prepare_mss_window)
        self.pak = PakTransaction(
            settings.effective_pak_paths(),
            settings.effective_staged_pak_paths(),
            enabled=settings.allow_file_operations,
        )
        self.diagnostics = RunDiagnostics(diagnostics_root, save_images=save_diagnostic_images)
        self._stop = threading.Event()
        self._finish_round_requested = threading.Event()
        self._scheduled_finish_requested = threading.Event()
        self._scheduled_end_at = None
        self._loadout_end_at: datetime | None = None
        self._loadout_round_in_flight = False
        self._pause_requested = threading.Event()
        self._paused = threading.Event()
        self._pause_deferral_depth = 0
        self._paused_duration_total = 0.0
        self._discard_loadout_checkpoint_on_exit = False
        self._unified_navigation = MailUnifiedNavigation(self)
        self._snapshot_requested = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = WorkflowState.IDLE
        self._result_lock = threading.Lock()
        self._result: MailWorkflowResult | AmmunitionSaleResult | None = None
        self._last_page: str | None = None
        self._last_unified_frame: CapturedFrame | None = None
        self._last_unified_mail_observation: PageObservation | None = None
        self._demo_step = 0
        # Business mode is independent of game-stall recovery eligibility.
        self._loadout_purchase_mode = False
        self._loadout_purchase_settings: LoadoutPurchaseSettings | None = None
        self._loadout_purchase_settings_provider = loadout_purchase_settings_provider
        self._loadout_purchase_fast_mode = False
        self._ammunition_sale_settings: AmmunitionSaleSettings | None = None
        self._fast_loadout_buy_locked = False
        self._last_fast_loadout_price_crop: np.ndarray | None = None
        self._last_fast_loadout_ocr_attempts: list[dict[str, object]] = []
        self._last_fast_loadout_ocr_rejection_reason = ""
        self._last_fast_loadout_ocr_expected_quantity: int | None = None
        self._pending_warehouse_listing: PendingWarehouseListing | None = None
        self._loadout_checkpoint_store = loadout_checkpoint_store
        self._loadout_checkpoint: LoadoutPurchaseCheckpoint | None = None
        self._loadout_round_id = 0
        self._resumed_prior_batches = ()
        self._warehouse_sell_ocr = None
        self._scheduled_game_restart_deadline: float | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def state(self) -> WorkflowState:
        return self._state

    @property
    def result(self) -> MailWorkflowResult | AmmunitionSaleResult | None:
        with self._result_lock:
            return self._result

    @property
    def pause_requested(self) -> bool:
        return self._pause_requested.is_set()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def start_observation(self) -> None:
        self._loadout_purchase_mode = False
        self._start(self._run_observation, "DeltaForceBulletBot-mail-observe")

    def start_demonstration(self) -> None:
        self._loadout_purchase_mode = False
        self._start(self._run_demonstration, "DeltaForceBulletBot-mail-demonstration")

    def start_workflow(self) -> None:
        self._loadout_purchase_mode = False
        self._start(self._run_workflow, "DeltaForceBulletBot-mail-workflow")

    def start_loadout_purchase(
        self,
        settings: LoadoutPurchaseSettings,
        *,
        fast_mode: bool = False,
        end_at: datetime | None = None,
    ) -> None:
        settings.validate(require_enabled=True)
        if fast_mode and len(settings.enabled_schemes) != 2:
            raise ValueError("极速配装模式必须启用且仅启用两个方案")
        self._loadout_purchase_mode = True
        self._loadout_purchase_settings = settings
        self._loadout_purchase_fast_mode = fast_mode
        self._loadout_end_at = end_at
        self._fast_loadout_buy_locked = False
        self._reset_scheduled_game_restart_deadline()
        self._load_loadout_checkpoint(settings)
        self._loadout_round_in_flight = bool(
            self._loadout_checkpoint is not None
            and self._loadout_checkpoint.stage != LoadoutCheckpointStage.PURCHASE_SCAN
        )
        self._start(
            self._run_loadout_purchase_workflow,
            "DeltaForceBulletBot-loadout-purchase",
        )

    def start_ammunition_sale(self, settings: AmmunitionSaleSettings) -> None:
        settings.validate(require_enabled=True)
        self._loadout_purchase_mode = False
        self._ammunition_sale_settings = settings
        self._start(
            self._run_ammunition_sale_workflow,
            "DeltaForceBulletBot-ammunition-sale",
        )

    def _load_loadout_checkpoint(self, settings: LoadoutPurchaseSettings) -> None:
        store = getattr(self, "_loadout_checkpoint_store", None)
        checkpoint = store.load() if store is not None else None
        self._loadout_checkpoint = checkpoint
        self._pending_warehouse_listing = None
        self._loadout_round_id = 0
        self._resumed_prior_batches = ()
        if checkpoint is None:
            return

        rules = {rule.scheme_index: rule for rule in settings.enabled_schemes}
        if checkpoint.scheme_index is not None:
            rule = rules.get(checkpoint.scheme_index)
            if rule is None:
                raise ValueError(
                    f"检查点中的配装方案 {checkpoint.scheme_index} 当前未启用"
                )
            if checkpoint.ammo_name and checkpoint.ammo_name != rule.ammunition_name:
                raise ValueError("检查点子弹名称与当前方案配置不一致")
        if checkpoint.pending_scheme_index is not None:
            pending_rule = rules.get(checkpoint.pending_scheme_index)
            if pending_rule is None:
                raise ValueError(
                    f"待售检查点中的配装方案 {checkpoint.pending_scheme_index} "
                    "当前未启用"
                )
            if checkpoint.pending_ammo_name != pending_rule.ammunition_name:
                raise ValueError("待售检查点子弹名称与当前方案配置不一致")
            configured_sell_names = {
                rule.ammunition_name
                for rule in settings.schemes
                if rule.warehouse_sell_enabled and rule.ammunition_name
            }
            unknown_pending_names = set(
                checkpoint.pending_additional_ammo_names
            ) - configured_sell_names
            if unknown_pending_names:
                raise ValueError(
                    "待售检查点包含当前直售配置中不存在的子弹："
                    + "、".join(sorted(unknown_pending_names))
                )
        self._loadout_round_id = checkpoint.round_id
        self._pending_warehouse_listing = checkpoint.pending_listing()
        if checkpoint.stage is LoadoutCheckpointStage.LISTING_CONFIRMED:
            self._resumed_prior_batches = self._prior_batches_from_checkpoint(
                checkpoint
            )

    def _save_loadout_checkpoint(
        self,
        stage: LoadoutCheckpointStage,
        **changes: object,
    ) -> LoadoutPurchaseCheckpoint:
        checkpoint = getattr(self, "_loadout_checkpoint", None)
        if checkpoint is None:
            checkpoint = LoadoutPurchaseCheckpoint(
                round_id=max(0, int(getattr(self, "_loadout_round_id", 0))),
                stage=stage,
            )
        checkpoint = replace(
            checkpoint,
            round_id=max(0, int(getattr(self, "_loadout_round_id", 0))),
            stage=stage,
            **changes,
        ).with_pending(getattr(self, "_pending_warehouse_listing", None))
        checkpoint.validate()
        self._loadout_checkpoint = checkpoint
        store = getattr(self, "_loadout_checkpoint_store", None)
        if store is not None:
            store.save(checkpoint)
        diagnostics = getattr(self, "diagnostics", None)
        if diagnostics is not None:
            diagnostics.event(
                "loadout_checkpoint",
                stage=checkpoint.stage.value,
                round_id=checkpoint.round_id,
                scheme_index=checkpoint.scheme_index,
                last_safe_action=checkpoint.last_safe_action,
                resume_target=(
                    checkpoint.resume_target.value
                    if checkpoint.resume_target is not None
                    else None
                ),
                pending_action=checkpoint.pending_action,
                recovery_reason=checkpoint.recovery_reason,
            )
        return checkpoint

    def _clear_loadout_checkpoint(self) -> None:
        store = getattr(self, "_loadout_checkpoint_store", None)
        if store is not None:
            store.delete()
        self._loadout_checkpoint = None

    def _complete_loadout_round_checkpoint(self) -> None:
        pending = getattr(self, "_pending_warehouse_listing", None)
        if pending is None:
            self._clear_loadout_checkpoint()
            return
        self._save_loadout_checkpoint(
            LoadoutCheckpointStage.WAITING_SALE,
            scheme_index=pending.scheme_index,
            ammo_name=pending.ammunition_name,
            sale_gate_passed=True,
            listed_batch_count=0,
            listed_quantity_total=pending.listed_quantity,
            sale_proceeds_claim_attempted=False,
            sale_proceeds_claimed=False,
            pending_batch_quantity=0,
            pending_batch_price=None,
            first_lowest_price=None,
            purchase_balance_before=None,
            purchase_listed_price=None,
            purchase_was_partial=False,
            last_confirmed_page=GamePageId.WAREHOUSE.value,
            last_safe_action="当前轮卡邮件完成，继续保留更早一轮待售挂单",
        )

    def _start_loadout_round_checkpoint(self) -> None:
        self._loadout_round_id = max(
            int(getattr(self, "_loadout_round_id", 0)),
            int(getattr(getattr(self, "_loadout_checkpoint", None), "round_id", 0)),
        ) + 1
        self._save_loadout_checkpoint(
            LoadoutCheckpointStage.PURCHASE_SCAN,
            scheme_index=None,
            ammo_name="",
            sale_gate_passed=False,
            listed_batch_count=0,
            listed_quantity_total=0,
            sale_proceeds_claim_attempted=False,
            sale_proceeds_claimed=False,
            pending_batch_quantity=0,
            pending_batch_price=None,
            first_lowest_price=None,
            purchase_balance_before=None,
            purchase_listed_price=None,
            purchase_was_partial=False,
            last_confirmed_page=None,
            last_safe_action="开始新一轮配装价格扫描",
            resume_target=None,
            pending_action="",
            recovery_reason="",
        )

    def _start(self, target: Callable[[], None], name: str) -> None:
        if self.running:
            raise RuntimeError("已有任务正在运行")
        self._stop.clear()
        self._finish_round_requested.clear()
        self._scheduled_finish_requested.clear()
        self._pause_requested.clear()
        self._paused.clear()
        self._pause_deferral_depth = 0
        self._paused_duration_total = 0.0
        self._discard_loadout_checkpoint_on_exit = False
        self._snapshot_requested.clear()
        with self._result_lock:
            self._result = None
        self._last_page = None
        self._thread = threading.Thread(target=target, name=name, daemon=False)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def request_finish_cycle(self) -> None:
        self._scheduled_finish_requested.set()
        self._pause_requested.clear()

    def _scheduled_finishing(self) -> bool:
        requested = getattr(self, "_scheduled_finish_requested", None)
        end_at = getattr(self, "_scheduled_end_at", None)
        return bool((requested is not None and requested.is_set()) or
                    (end_at is not None and datetime.now() >= end_at))

    def _scheduled_sale_finishing(self) -> bool:
        return getattr(self, "_scheduled_finish_mode", "safe") == "safe" and self._scheduled_finishing()

    def start_game_prepare(self, settings: LoadoutPurchaseSettings) -> None:
        self._loadout_purchase_settings = settings
        self._loadout_purchase_mode = False
        self._start(self._run_game_prepare, "DeltaForceBulletBot-scheduled-prepare")

    def start_game_restart(self, settings: LoadoutPurchaseSettings) -> None:
        self._loadout_purchase_settings = settings
        self._loadout_purchase_mode = False
        self._start(self._run_game_restart, "DeltaForceBulletBot-page-restart")

    def _run_game_restart(self) -> None:
        outcome, message, error = MailWorkflowOutcome.COMPLETED, "页面恢复重启完成", None
        try:
            self._precheck()
            self._restart_game_without_mail_resource_cycle(LoadoutResumeTarget.LOADOUT_SCAN)
        except WorkflowStopped:
            outcome, message = MailWorkflowOutcome.STOPPED, "页面恢复重启已停止"
        except Exception as exc:
            outcome, message, error = MailWorkflowOutcome.FAILED, f"页面恢复重启失败：{exc}", str(exc)
        finally:
            restored, restore_error = self._restore_on_exit()
            if not restored:
                outcome, message, error = MailWorkflowOutcome.FAILED, str(restore_error), restore_error
            with self._result_lock:
                self._result = MailWorkflowResult(outcome=outcome, completed_rounds=0,
                    pak_restored=restored, final_page=self._last_page, message=message, error=error)
            self._close_runtime_resources()

    def _run_game_prepare(self) -> None:
        outcome, message, error = MailWorkflowOutcome.COMPLETED, "游戏已就绪", None
        try:
            self._precheck()
            self._ensure_loadout_game_started()
            self._prepare_game_for_trading_handoff()
        except WorkflowStopped:
            outcome, message = MailWorkflowOutcome.STOPPED, "游戏启动已取消"
        except Exception as exc:
            outcome, message, error = MailWorkflowOutcome.FAILED, f"游戏启动失败：{exc}", str(exc)
        finally:
            restored, restore_error = self._restore_on_exit()
            if not restored:
                outcome, message, error = MailWorkflowOutcome.FAILED, str(restore_error), restore_error
            with self._result_lock:
                self._result = MailWorkflowResult(outcome=outcome, completed_rounds=0,
                    pak_restored=restored, final_page=self._last_page, message=message, error=error)
            self._close_runtime_resources()

    def request_finish_loadout_round(self) -> None:
        self._finish_round_requested.set()

    def _loadout_finishing(self) -> bool:
        requested = getattr(self, "_finish_round_requested", None)
        end_at = getattr(self, "_loadout_end_at", None)
        return bool((requested is not None and requested.is_set()) or (
            end_at is not None and datetime.now() >= end_at
        ))

    def _check_finish_loadout_round(self) -> None:
        if self._loadout_finishing() and not getattr(self, "_loadout_round_in_flight", False):
            raise WorkflowStopped()

    def _begin_loadout_purchase_action(self) -> None:
        # Check the wall-clock deadline on the worker immediately before locking
        # the round, even when the UI timer is delayed by OCR or another dialog.
        self._check_stop()
        self._loadout_round_in_flight = True

    def _ensure_loadout_game_started(self) -> None:
        if self.game.refresh() is not None:
            return
        settings = self._timing_settings()
        if not matching_process_running(self.game.fingerprint):
            self._wait_window(self.launcher, "启动器", settings.launcher_window_timeout_seconds)
            self._wait_page(self.launcher, {PageType.LAUNCHER_HOME},
                            settings.launcher_window_timeout_seconds, "启动器首页")
            self._check_stop()
            if self.game.refresh() is None and not matching_process_running(self.game.fingerprint):
                self._click(self._require_bound(self.launcher, "启动器"),
                            "launcher_start_game", "定时配装：开始游戏")
        self._wait_window(self.game, "游戏", settings.game_window_timeout_seconds)
        self._finish_scheduled_restart_startup()

    def start_game_close(self, settings: LoadoutPurchaseSettings) -> None:
        self._loadout_purchase_settings = settings
        self._loadout_purchase_mode = False
        self._loadout_end_at = None
        self._loadout_round_in_flight = False
        self._start(self._run_game_close, "DeltaForceBulletBot-scheduled-close")

    def _run_game_close(self) -> None:
        outcome = MailWorkflowOutcome.COMPLETED
        error = None
        message = "游戏已关闭，保持启动器界面"
        try:
            self._precheck()
            self._check_stop()
            game = self.game.refresh()
            if game is not None or matching_process_running(self.game.fingerprint):
                termination = self._terminate_game_process_tree(
                    game, reason="scheduled_close", label="定时关闭游戏")
                self._wait_window_gone(self.game, self._timing_settings().game_exit_timeout_seconds,
                                       expected_processes=termination.processes)
            self._wait_window(self.launcher, "启动器", self._timing_settings().launcher_window_timeout_seconds)
        except WorkflowStopped:
            outcome, message = MailWorkflowOutcome.STOPPED, "关闭游戏任务已停止"
        except Exception as exc:
            outcome, error = MailWorkflowOutcome.FAILED, str(exc)
            message = f"关闭游戏失败：{exc}"
        finally:
            restored, restore_error = self._restore_on_exit()
            if not restored:
                outcome, error = MailWorkflowOutcome.FAILED, restore_error
            with self._result_lock:
                self._result = MailWorkflowResult(
                    outcome=outcome, completed_rounds=0, pak_restored=restored,
                    final_page=self._last_page, message=message, error=error)
            self._emit(message, "error" if error else "info")
            self._close_runtime_resources()

    def request_stop(self, *, discard_loadout_checkpoint: bool = False) -> None:
        if discard_loadout_checkpoint:
            self._discard_loadout_checkpoint_on_exit = True
        self._pause_requested.clear()
        self.stop()

    def request_pause(self) -> bool:
        if not self.running or self._stop.is_set():
            return False
        self._pause_requested.set()
        return True

    def request_resume(self) -> bool:
        if not self.running or (
            not self._pause_requested.is_set() and not self._paused.is_set()
        ):
            return False
        self._pause_requested.clear()
        return True

    def request_snapshot(self) -> bool:
        if not self.running or self._state != WorkflowState.DEMONSTRATION:
            return False
        self._snapshot_requested.set()
        return True

    @property
    def run_directory(self) -> Path:
        return self.diagnostics.root

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    def _run_observation(self) -> None:
        self._transition(WorkflowState.IDLE, "观察模式已启动，不会发送输入或移动文件")
        try:
            while not self._stop.is_set():
                game_window = self.game.refresh()
                launcher_window = self.launcher.refresh()
                if game_window is not None:
                    self._observe(game_window, "game", include_ocr=True)
                elif launcher_window is not None:
                    self._observe(launcher_window, "launcher", include_ocr=True)
                else:
                    self._emit("等待已配置的游戏或启动器窗口")
                self._sleep(self.settings.observation_interval_seconds)
        except WorkflowStopped:
            pass
        except Exception as exc:
            self._transition(WorkflowState.ERROR, f"观察失败：{exc}", "error")
        finally:
            self._transition(WorkflowState.STOPPED, "观察模式已停止")
            self._close_runtime_resources()

    def _run_demonstration(self) -> None:
        self._transition(
            WorkflowState.DEMONSTRATION,
            f"演示采集已启动，请手动操作游戏；记录目录：{self.diagnostics.root}",
        )
        tracker = StablePageTracker()
        try:
            while not self._stop.is_set():
                game_window = self.game.refresh()
                launcher_window = self.launcher.refresh()
                if game_window is not None:
                    window, target = game_window, "game"
                elif launcher_window is not None:
                    window, target = launcher_window, "launcher"
                else:
                    self._emit("等待已配置的游戏或启动器窗口")
                    self._sleep(self.settings.observation_interval_seconds)
                    continue

                frame, observation = self._observe(
                    window,
                    target,
                    include_ocr=True,
                )
                if tracker.update(target, observation.page_type):
                    self._record_demo_step(frame, observation, target, "自动")
                if self._snapshot_requested.is_set():
                    self._snapshot_requested.clear()
                    self._record_demo_step(frame, observation, target, "手动")
                self._sleep(self.settings.observation_interval_seconds)
        except WorkflowStopped:
            pass
        except Exception as exc:
            self._transition(WorkflowState.ERROR, f"演示采集失败：{exc}", "error")
        finally:
            self._transition(
                WorkflowState.STOPPED,
                f"演示采集已停止，共记录 {self._demo_step} 个节点",
            )
            self._close_runtime_resources()

    def _record_demo_step(
        self,
        frame: CapturedFrame,
        observation: PageObservation,
        target: str,
        source: str,
    ) -> None:
        self._demo_step += 1
        target_label = "游戏" if target == "game" else "启动器"
        page_label = page_type_label(observation.page_type)
        screenshot = self.diagnostics.save_frame(
            f"演示_{self._demo_step:03d}_{target_label}_{page_label}",
            frame.image,
        )
        line = (
            f"第 {self._demo_step:03d} 步 | {target_label} | {page_label} | "
            f"置信度 {observation.confidence:.0%} | {source}记录 | {screenshot.name}"
        )
        index_path = self.diagnostics.append_demo_index(line)
        self.diagnostics.event(
            "demo_step",
            step=self._demo_step,
            target=target,
            page=observation.page_type.value,
            page_label=page_label,
            confidence=round(observation.confidence, 4),
            source=source,
            screenshot=str(screenshot),
            index=str(index_path),
        )
        self._emit(f"已记录第 {self._demo_step} 个节点：{target_label} - {page_label}")

    def _run_workflow(self) -> None:
        completed_rounds = 0
        outcome = MailWorkflowOutcome.FAILED
        terminal_state = WorkflowState.ERROR
        message = "流程未正常结束"
        error: str | None = None
        try:
            self._set_capture_backend("mss")
            if getattr(self, "_loadout_purchase_settings", None) is None:
                provider = getattr(self, "_loadout_purchase_settings_provider", None)
                self._loadout_purchase_settings = (
                    provider() if provider is not None else LoadoutPurchaseSettings()
                )
                self._loadout_purchase_settings.validate()
            while True:
                self._check_stop()
                if self._scheduled_finishing():
                    self._prepare_game_for_trading_handoff()
                    outcome = MailWorkflowOutcome.COMPLETED
                    terminal_state = WorkflowState.COMPLETE
                    message = f"当前卡邮件循环已结束；累计 {completed_rounds} 轮"
                    break
                round_number = completed_rounds + 1
                if not self._preflight_mail_scheme():
                    outcome = MailWorkflowOutcome.COMPLETED
                    terminal_state = WorkflowState.COMPLETE
                    message = (
                        "轮前检查确认目标配装方案需要购买，连续卡邮件已正常结束；"
                        "已返回主界面，未进入本轮关游戏、删除地图或下载流程；"
                        f"累计完成 {completed_rounds} 轮"
                    )
                    break
                if self._scheduled_finishing():
                    continue
                self._emit(f"========== 开始第 {round_number} 轮卡邮件 ==========")
                self._run_workflow_cycle(round_number)
                completed_rounds += 1
        except LoadoutSchemePurchaseRequired as exc:
            try:
                self._complete_purchase_required_mode_switch()
                self._prepare_game_for_trading_handoff()
            except WorkflowStopped:
                outcome = MailWorkflowOutcome.STOPPED
                terminal_state = WorkflowState.STOPPED
                message = (
                    "配装方案已经需要购买，但用户在完成长弓下载与放弃对局前停止了流程；"
                    f"已完成 {completed_rounds} 轮"
                )
            except Exception as handoff_exc:
                error = f"{type(handoff_exc).__name__}: {handoff_exc}"
                message = (
                    "配装方案已经需要购买，但未能完成长弓下载与放弃对局后的主界面交接："
                    f"{handoff_exc}"
                )
                self._save_failure_frame()
            else:
                completed_rounds += 1
                outcome = MailWorkflowOutcome.COMPLETED
                terminal_state = WorkflowState.COMPLETE
                message = (
                    f"检测到配装方案需要购买，连续循环已正常停止；"
                    f"未购买或应用方案，但已等待长弓下载完成、放弃重连对局并返回主界面；"
                    f"本轮计为完成，累计 {completed_rounds} 轮：{exc}"
                )
        except WorkflowStopped:
            outcome = MailWorkflowOutcome.STOPPED
            terminal_state = WorkflowState.STOPPED
            message = f"流程已由用户停止；已完成 {completed_rounds} 轮"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            message = f"流程失败：{exc}"
            self._save_failure_frame()
        finally:
            pak_restored, restore_error = self._restore_on_exit()
            if not pak_restored:
                outcome = MailWorkflowOutcome.FAILED
                terminal_state = WorkflowState.ERROR
                restore_message = restore_error or "PAK 未处于原始路径"
                message = f"{message}；PAK 恢复验证失败：{restore_message}"
                error = f"{error}; {restore_message}" if error else restore_message
            if getattr(self, "_discard_loadout_checkpoint_on_exit", False):
                self._clear_loadout_checkpoint()
                self.diagnostics.event(
                    "loadout_checkpoint_discarded",
                    reason="用户终止当前配装买入轮次",
                )

            result = MailWorkflowResult(
                outcome=outcome,
                completed_rounds=completed_rounds,
                pak_restored=pak_restored,
                final_page=self._last_page,
                message=message,
                error=error,
            )
            with self._result_lock:
                self._result = result
            self.diagnostics.event("workflow_result", **result.to_dict())
            level = (
                "warning"
                if outcome == MailWorkflowOutcome.COMPLETED
                else "error"
                if outcome == MailWorkflowOutcome.FAILED
                else "info"
            )
            self._transition(terminal_state, message, level)
            self._close_runtime_resources()

    def _resume_loadout_purchase_checkpoint(
        self,
        settings: LoadoutPurchaseSettings,
    ) -> tuple[LoadoutPurchaseSelection | None, bool]:
        checkpoint = getattr(self, "_loadout_checkpoint", None)
        if checkpoint is None:
            return None, False
        self._emit(
            f"发现配装买入检查点：第 {checkpoint.round_id} 轮，"
            f"阶段 {checkpoint.stage.value}；先恢复再产生新的购买输入",
            "warning",
        )
        if checkpoint.stage in {
            LoadoutCheckpointStage.PURCHASE_SCAN,
            LoadoutCheckpointStage.WAITING_SALE,
            LoadoutCheckpointStage.RETURN_HOME_PENDING,
        }:
            return None, False
        if checkpoint.stage is LoadoutCheckpointStage.RECOVERY_WAIT:
            raise RuntimeError("检查点仍处于恢复等待，无法证明上一危险动作结果")
        if checkpoint.scheme_index is None:
            raise RuntimeError("恢复检查点缺少当前配装方案编号")
        rule = next(
            (
                candidate
                for candidate in settings.enabled_schemes
                if candidate.scheme_index == checkpoint.scheme_index
            ),
            None,
        )
        if rule is None:
            raise RuntimeError(
                f"恢复检查点中的方案 {checkpoint.scheme_index} 当前不可用"
            )
        listed_price = max(1, int(checkpoint.purchase_listed_price or 1))

        if checkpoint.stage is LoadoutCheckpointStage.PURCHASE_CONFIRM_PENDING:
            balance_before = checkpoint.purchase_balance_before
            if balance_before is None:
                raise RuntimeError("购买确认检查点缺少购买前余额")
            self._prepare_game_for_trading_handoff()
            loadout = self._open_zero_dam_loadout()
            balance_after, balance_page = self._read_loadout_balance_with_page_recovery(
                "购买中断恢复",
                loadout,
            )
            spent = balance_before - balance_after
            if spent <= 0:
                self._save_loadout_checkpoint(
                    LoadoutCheckpointStage.PURCHASE_SCAN,
                    scheme_index=None,
                    ammo_name="",
                    purchase_balance_before=None,
                    purchase_listed_price=None,
                    purchase_was_partial=False,
                    last_confirmed_page=balance_page.page_type.value,
                    last_safe_action="中断恢复余额未减少，确认不重放购买点击",
                )
                self._ensure_loadout_scheme_list(balance_page)
                return None, False
            self._save_loadout_checkpoint(
                LoadoutCheckpointStage.PURCHASE_CONFIRMED,
                purchase_was_partial=True,
                last_confirmed_page=balance_page.page_type.value,
                last_safe_action="中断恢复确认余额已减少，按部分成交安全处理",
            )
            self._prepare_partial_loadout_for_mail(balance_page)
            return (
                LoadoutPurchaseSelection(
                    rule=rule,
                    listed_price=listed_price,
                    actual_average=None,
                    partial_spent=spent,
                ),
                False,
            )

        if checkpoint.stage is LoadoutCheckpointStage.PURCHASE_CONFIRMED:
            if checkpoint.purchase_was_partial:
                self._prepare_game_for_trading_handoff()
                loadout = self._open_zero_dam_loadout()
                balance_after, balance_page = (
                    self._read_loadout_balance_with_page_recovery(
                        "部分成交中断恢复",
                        loadout,
                    )
                )
                self._prepare_partial_loadout_for_mail(balance_page)
                before = checkpoint.purchase_balance_before
                spent = max(1, (before - balance_after) if before else 1)
                return (
                    LoadoutPurchaseSelection(
                        rule=rule,
                        listed_price=listed_price,
                        actual_average=None,
                        partial_spent=spent,
                    ),
                    False,
                )
            self._prepare_game_for_trading_handoff()
            self._apply_blank_loadout_scheme(settings.blank_scheme_index)
            self._save_loadout_checkpoint(
                LoadoutCheckpointStage.BLANK_APPLIED,
                last_confirmed_page=PageType.GAME_HOME_READY.value,
                last_safe_action=(
                    f"中断恢复后已应用空白方案 {settings.blank_scheme_index}"
                ),
            )

        checkpoint = getattr(self, "_loadout_checkpoint", checkpoint)
        if checkpoint.stage is LoadoutCheckpointStage.LISTING_CONFIRMED:
            self._resumed_prior_batches = self._prior_batches_from_checkpoint(
                checkpoint
            )
        resume_mail = checkpoint.stage is LoadoutCheckpointStage.MAIL_STORAGE
        return (
            LoadoutPurchaseSelection(
                rule=rule,
                listed_price=listed_price,
                actual_average=None,
            ),
            resume_mail,
        )

    def _run_loadout_purchase_workflow(self) -> None:
        settings = self._loadout_purchase_settings
        if settings is None:
            raise RuntimeError("配装买入配置尚未初始化")

        completed_rounds = 0
        outcome = MailWorkflowOutcome.FAILED
        terminal_state = WorkflowState.ERROR
        message = "配装买入法未正常结束"
        error: str | None = None
        try:
            self._page_recovery_enabled = True
            self._set_capture_backend("wgc")
            self._transition(WorkflowState.PRECHECK, "执行配装买入启动检查")
            self._precheck()
            self._transition(WorkflowState.WAIT_GAME, "等待并绑定游戏窗口")
            self._ensure_loadout_game_started()

            resumed_purchase, resume_mail_only = (
                self._resume_loadout_purchase_checkpoint(settings)
            )
            self._loadout_round_in_flight = resumed_purchase is not None

            while True:
                self._check_stop()
                try:
                    # A completed direct listing returns here before the next
                    # purchase round.  This is the first safe restart point:
                    # never interrupt purchase settlement or warehouse listing.
                    if resumed_purchase is None:
                        self._raise_if_scheduled_game_restart_due()
                    purchase = resumed_purchase or (
                        self._wait_and_buy_fast_loadout_scheme(settings)
                        if getattr(self, "_loadout_purchase_fast_mode", False)
                        else self._wait_and_buy_loadout_scheme(settings)
                    )
                except ScheduledGameRestartRequested:
                    resumed_purchase = None
                    self._restart_game_without_mail_resource_cycle()
                    continue
                resumed_purchase = None
                rule = purchase.rule
                average_text = (
                    f"，余额核算实际平均 {purchase.actual_average:,}/发"
                    if purchase.actual_average is not None
                    else "，余额核算暂不可用"
                )
                if purchase.partially_purchased:
                    post_purchase = (
                        "开始交易行直售"
                        if rule.warehouse_sell_enabled
                        else "开始卡邮件清理"
                    )
                    self._emit(
                        f"配装方案 {rule.scheme_index} 疑似部分成交：余额减少 "
                        f"{purchase.partial_spent:,}；具体子弹数量未知，"
                        f"已停止继续买入并保留当前部分配装，{post_purchase}",
                        "warning",
                    )
                else:
                    post_purchase = (
                        "开始交易行直售"
                        if rule.warehouse_sell_enabled
                        else "开始卡邮件"
                    )
                    self._emit(
                        f"配装方案 {rule.scheme_index} 已按 "
                        f"{purchase.listed_price:,} 购买 {rule.total_quantity:,} 发"
                        f"{average_text}；{post_purchase}"
                    )
                if (
                    not resume_mail_only
                    and rule.warehouse_sell_enabled
                ):
                    sell_result = self._run_warehouse_sell_after_purchase(rule)
                    if sell_result.outcome is WarehouseSellOutcome.STOPPED:
                        raise WorkflowStopped()
                    if sell_result.outcome is WarehouseSellOutcome.LISTED:
                        completed_rounds += 1
                        self._emit(
                            f"配装方案 {rule.scheme_index} 的买入与交易行直售已完成；"
                            f"共上架 {sell_result.listed_quantity:,} 发，返回方案轮询"
                        )
                        self._return_home_after_warehouse_sell()
                        self._loadout_round_in_flight = False
                        continue
                    if sell_result.outcome in {
                        WarehouseSellOutcome.BELOW_TRIGGER,
                        WarehouseSellOutcome.INVENTORY_NOT_FOUND,
                    }:
                        self._emit(
                            f"交易行直售未执行最终上架：{sell_result.message}；"
                            "按本轮规则转入卡邮件",
                            "warning",
                        )
                    else:
                        raise RuntimeError(
                            f"交易行直售未完成：{sell_result.message}"
                        )
                elif (
                    not resume_mail_only
                    and
                    not purchase.partially_purchased
                    and getattr(self, "_pending_warehouse_listing", None) is not None
                ):
                    pending = self._pending_warehouse_listing
                    recovery = self._resolve_pending_listing_with_recovery()
                    if recovery.outcome is SellSlotRecoveryOutcome.STOPPED:
                        raise WorkflowStopped()
                    if recovery.outcome is SellSlotRecoveryOutcome.RECOVERY_REQUIRED:
                        raise RuntimeError(recovery.message)
                    if recovery.outcome is SellSlotRecoveryOutcome.UNLISTED:
                        pending_rule = next(
                            (
                                candidate
                                for candidate in settings.schemes
                                if candidate.scheme_index == pending.scheme_index
                                and candidate.warehouse_sell_enabled
                            ),
                            None,
                        )
                        if pending_rule is None:
                            raise RuntimeError(
                                "超时撤单后找不到上一轮子弹的直售配置，"
                                "无法安全执行降一柱重卖"
                            )
                        relist_result = self._run_warehouse_sell_after_purchase(
                            pending_rule,
                            forced_ammunition_names=pending.ammunition_names,
                        )
                        if relist_result.outcome is WarehouseSellOutcome.STOPPED:
                            raise WorkflowStopped()
                        if relist_result.outcome is not WarehouseSellOutcome.LISTED:
                            raise RuntimeError(
                                "超时撤单后的降一柱重卖未完成："
                                f"{relist_result.message}"
                            )
                self._save_loadout_checkpoint(
                    LoadoutCheckpointStage.MAIL_STORAGE,
                    scheme_index=rule.scheme_index,
                    ammo_name=rule.ammunition_name,
                    last_safe_action="进入卡邮件前已确认目标方案",
                )
                try:
                    self._page_recovery_enabled = False
                    self._set_capture_backend("mss")
                    self._run_workflow_cycle(
                        completed_rounds + 1,
                        mail_scheme_index=(
                            PARTIAL_TRANSFER_SCHEME_INDEX
                            if purchase.partially_purchased
                            else rule.scheme_index
                        ),
                    )
                except LoadoutSchemePurchaseRequired as exc:
                    self._complete_purchase_required_mode_switch()
                    self._prepare_game_for_trading_handoff()
                    if purchase.partially_purchased:
                        raise RuntimeError(
                            f"部分成交已保存到中转方案 "
                            f"{PARTIAL_TRANSFER_SCHEME_INDEX}，但卡邮件时仍显示需要购买：{exc}"
                        ) from exc
                    raise RuntimeError(
                        f"刚购买的配装方案 {rule.scheme_index} 在卡邮件阶段仍显示需要购买：{exc}"
                    ) from exc
                finally:
                    self._page_recovery_enabled = True
                self._set_capture_backend("wgc")
                completed_rounds += 1
                self._emit_structured(
                    "loadout_mail_storage_result",
                    f"配装方案 {rule.scheme_index} 已完成卡邮件",
                    scheme_index=rule.scheme_index,
                    ammunition_name=rule.ammunition_name,
                    outcome="mail_stored",
                )
                self._complete_loadout_round_checkpoint()
                self._loadout_round_in_flight = False
                resume_mail_only = False
                self._mark_loadout_price_refresh_resumed(rule)
                self._emit(
                    f"配装方案 {rule.scheme_index} 的买入与卡邮件已完成；"
                    "返回方案轮询"
                )
        except WorkflowStopped:
            outcome = MailWorkflowOutcome.STOPPED
            terminal_state = WorkflowState.STOPPED
            message = f"配装买入法已停止；已完成 {completed_rounds} 轮"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            message = f"配装买入法失败：{exc}"
            self._save_failure_frame()
        finally:
            pak_restored, restore_error = self._restore_on_exit()
            if not pak_restored:
                outcome = MailWorkflowOutcome.FAILED
                terminal_state = WorkflowState.ERROR
                restore_message = restore_error or "PAK 未处于原始路径"
                message = f"{message}；PAK 恢复验证失败：{restore_message}"
                error = f"{error}; {restore_message}" if error else restore_message
            if getattr(self, "_discard_loadout_checkpoint_on_exit", False):
                self._clear_loadout_checkpoint()
                self.diagnostics.event(
                    "loadout_checkpoint_discarded",
                    reason="用户终止当前配装买入轮次",
                )

            result = MailWorkflowResult(
                outcome=outcome,
                completed_rounds=completed_rounds,
                pak_restored=pak_restored,
                final_page=self._last_page,
                message=message,
                error=error,
            )
            with self._result_lock:
                self._result = result
            self.diagnostics.event("loadout_purchase_result", **result.to_dict())
            level = "error" if outcome == MailWorkflowOutcome.FAILED else "info"
            self._transition(terminal_state, message, level)
            self._close_runtime_resources()

    def _run_ammunition_sale_workflow(self) -> None:
        from bulletbot.loadout_purchase.warehouse_sell_runtime import (
            WarehouseSellRuntimePort,
        )
        from bulletbot.loadout_purchase.warehouse_sell_workflow import (
            WarehouseSellWorkflow,
        )

        settings = self._ammunition_sale_settings
        if settings is None:
            raise RuntimeError("领邮件自动卖配置尚未初始化")
        result = AmmunitionSaleResult(
            AmmunitionSaleOutcome.FAILED,
            message="领邮件自动卖未正常结束",
            error="领邮件自动卖未正常结束",
        )
        try:
            self._transition(
                WorkflowState.AMMUNITION_SALE,
                f"领邮件自动卖已启动，共 {len(settings.enabled_rules)} 条规则",
            )
            sell_ocr = self._get_warehouse_sell_ocr()
            port = WarehouseSellRuntimePort(
                window_provider=self.game.refresh,
                generation_provider=lambda: getattr(self.game, "generation", 0),
                capture=self.capture,
                input_controller=self.input,
                navigation=self._unified_navigation,
                stop_event=self._stop,
                page_recognizer=self._trading_page_recognizer,
                sell_ocr=sell_ocr,
            )

            def on_diagnostic_frame(label: str, image: Image.Image) -> None:
                frame = cv2.cvtColor(
                    np.asarray(image.convert("RGB")),
                    cv2.COLOR_RGB2BGR,
                )
                path = self.diagnostics.save_frame(label, frame)
                self.diagnostics.event(
                    "ammunition_sale_diagnostic_frame",
                    label=label,
                    screenshot=str(path),
                )

            warehouse_sell = WarehouseSellWorkflow(
                port,
                sell_ocr=sell_ocr,
                log=self._emit,
                on_diagnostic_frame=on_diagnostic_frame,
                pause_at_safe_point=self._pause_at_safe_point,
                finish_requested=self._scheduled_sale_finishing,
            )
            mail_claim = AmmunitionMailClaimWorkflow(
                _AmmunitionMailClaimPort(self),
                text_ocr=self.vision.ocr,
                numeric_ocr=self.vision.fast_price_ocr,
                log=self._emit,
                finish_requested=self._scheduled_sale_finishing,
            )

            def sell(rule) -> WarehouseSellResult:
                self._pause_at_safe_point()
                return warehouse_sell.run(rule.warehouse_sell_request())

            def claim_from_mail(
                rule: AmmunitionSaleRule,
            ) -> AmmunitionMailClaimResult:
                self._pause_at_safe_point()
                return mail_claim.claim(rule)

            def on_status(status: AmmunitionSaleStatus) -> None:
                self._emit_structured(
                    "ammunition_sale_status",
                    f"{status.ammunition_name}：{status.message}",
                    "warning"
                    if status.state.value in {"low_price", "cooling_down", "mail_pending"}
                    else "info",
                    rule_index=status.rule_index,
                    ammunition_name=status.ammunition_name,
                    display_message=status.message,
                    status=status.state.value,
                    attempt=status.attempt,
                    next_retry_seconds=status.next_retry_seconds,
                )

            runner = AmmunitionSaleRunner(
                sell,
                wait=self._sleep,
                stopped=self._stop.is_set,
                claim_from_mail=claim_from_mail,
                on_status=on_status,
                finish_requested=self._scheduled_finishing,
                finish_current_inventory=getattr(self, "_scheduled_finish_mode", "safe") == "cycle",
            )
            result = runner.run(settings)
        except WorkflowStopped:
            result = AmmunitionSaleResult(
                AmmunitionSaleOutcome.STOPPED,
                message="领邮件自动卖已由用户停止",
            )
        except Exception as exc:
            message = f"领邮件自动卖失败：{exc}"
            result = AmmunitionSaleResult(
                AmmunitionSaleOutcome.FAILED,
                message=message,
                error=f"{type(exc).__name__}: {exc}",
            )
            self._save_failure_frame()
        finally:
            with self._result_lock:
                self._result = result
            self.diagnostics.event(
                "ammunition_sale_result",
                outcome=result.outcome.value,
                listed_rules=result.listed_rules,
                listed_quantity=result.listed_quantity,
                claimed_mails=result.claimed_mails,
                message=result.message,
                error=result.error,
            )
            terminal_state = (
                WorkflowState.ERROR
                if result.outcome is AmmunitionSaleOutcome.FAILED
                else WorkflowState.STOPPED
                if result.outcome is AmmunitionSaleOutcome.STOPPED
                else WorkflowState.COMPLETE
            )
            self._transition(
                terminal_state,
                result.message,
                "error" if result.outcome is AmmunitionSaleOutcome.FAILED else "info",
            )
            self._close_runtime_resources()

    def _run_warehouse_sell_after_purchase(
        self,
        rule: LoadoutSchemeRule,
        *,
        forced_ammunition_names: tuple[str, ...] = (),
    ) -> WarehouseSellResult:
        from bulletbot.loadout_purchase.warehouse_sell_runtime import (
            WarehouseSellRuntimePort,
        )
        from bulletbot.loadout_purchase.warehouse_sell_workflow import (
            WarehouseSellWorkflow,
        )

        settings = self._loadout_purchase_settings
        if settings is None:
            raise RuntimeError("配装买入配置尚未初始化")
        forced_relist = bool(forced_ammunition_names)
        if forced_relist:
            tracked_ammunition_names = list(
                self._unique_ammunition_names(*forced_ammunition_names)
            )
            if not tracked_ammunition_names:
                raise RuntimeError("超时撤单后没有可重卖的子弹名称")
            request = self._forced_one_tick_lower_request(
                tracked_ammunition_names[0]
            )
            remaining_relist_names = tracked_ammunition_names[1:]
        else:
            request = rule.warehouse_sell_request()
            tracked_ammunition_names = [request.ammunition_name]
            remaining_relist_names: list[str] = []
        self._transition(
            WorkflowState.LOADOUT_WAREHOUSE_SELL,
            (
                "超时挂单已下架，强制按降一柱重卖："
                + "、".join(tracked_ammunition_names)
                if forced_relist
                else f"进入交易行出售页直售 {request.ammunition_name}"
            ),
            "warning" if forced_relist else "info",
        )
        sell_ocr = self._get_warehouse_sell_ocr()
        port = WarehouseSellRuntimePort(
            window_provider=self.game.refresh,
            generation_provider=lambda: getattr(self.game, "generation", 0),
            capture=self.capture,
            input_controller=self.input,
            navigation=self._unified_navigation,
            stop_event=self._stop,
            page_recognizer=self._trading_page_recognizer,
            sell_ocr=sell_ocr,
        )
        checkpoint = getattr(self, "_loadout_checkpoint", None)
        if (
            checkpoint is not None
            and checkpoint.stage is LoadoutCheckpointStage.LISTING_CONFIRM_PENDING
        ):
            self._resolve_uncertain_listing_checkpoint(port)
        pending = getattr(self, "_pending_warehouse_listing", None)
        if pending is not None and not forced_relist:
            pending_ammunition_names = pending.ammunition_names
            recovery = self._resolve_pending_listing_with_recovery(
                port=port,
                sell_ocr=sell_ocr,
            )
            if recovery.outcome is SellSlotRecoveryOutcome.STOPPED:
                return WarehouseSellResult(
                    outcome=WarehouseSellOutcome.STOPPED,
                    message=recovery.message,
                )
            if recovery.outcome is SellSlotRecoveryOutcome.RECOVERY_REQUIRED:
                return WarehouseSellResult(
                    outcome=WarehouseSellOutcome.RECOVERY_REQUIRED,
                    message=recovery.message,
                )
            if recovery.outcome is SellSlotRecoveryOutcome.UNLISTED:
                tracked_ammunition_names = list(
                    self._unique_ammunition_names(
                        request.ammunition_name,
                        *pending_ammunition_names,
                    )
                )
                forced_relist = True
                request = self._forced_one_tick_lower_request(
                    tracked_ammunition_names[0]
                )
                remaining_relist_names = tracked_ammunition_names[1:]
                self._transition(
                    WorkflowState.LOADOUT_WAREHOUSE_SELL,
                    "超时挂单已全部下架，强制按降一柱重卖："
                    + "、".join(tracked_ammunition_names),
                    "warning",
                )
            else:
                self._transition(
                    WorkflowState.LOADOUT_WAREHOUSE_SELL,
                    f"售位可用，继续从出售页上架 {request.ammunition_name}",
                )

        checkpoint = getattr(self, "_loadout_checkpoint", None)
        if checkpoint is None or checkpoint.stage not in {
            LoadoutCheckpointStage.LISTING_CONFIRMED,
            LoadoutCheckpointStage.LISTING_CONFIRM_PENDING,
        }:
            self._save_loadout_checkpoint(
                LoadoutCheckpointStage.WAREHOUSE_SCAN,
                scheme_index=rule.scheme_index,
                ammo_name=rule.ammunition_name,
                last_safe_action="开始从交易行出售库存扫描目标子弹",
            )

        def on_checkpoint(stage: str, data: dict[str, object]) -> None:
            resolved_stage = (
                LoadoutCheckpointStage.LISTING_CONFIRM_PENDING
                if stage == "listing_confirm_pending"
                else LoadoutCheckpointStage.LISTING_CONFIRMED
            )
            self._save_loadout_checkpoint(
                resolved_stage,
                scheme_index=rule.scheme_index,
                ammo_name=rule.ammunition_name,
                sale_gate_passed=True,
                listed_batch_count=int(data.get("listed_batch_count", 0)),
                listed_quantity_total=int(data.get("listed_quantity_total", 0)),
                pending_batch_quantity=int(data.get("pending_batch_quantity", 0)),
                pending_batch_price=(
                    int(data["pending_batch_price"])
                    if data.get("pending_batch_price") is not None
                    else None
                ),
                first_lowest_price=(
                    int(data["first_lowest_price"])
                    if data.get("first_lowest_price") is not None
                    else None
                ),
                last_confirmed_page=(
                    GamePageId.LISTING_EDITOR.value
                    if resolved_stage
                    is LoadoutCheckpointStage.LISTING_CONFIRM_PENDING
                    else GamePageId.MARKET_SELL.value
                ),
                last_safe_action=(
                    "最终上架按钮点击前已完成名称、数量和价格校验"
                    if resolved_stage
                    is LoadoutCheckpointStage.LISTING_CONFIRM_PENDING
                    else "最终上架已确认返回交易行出售页"
                ),
            )

        def on_diagnostic_frame(label: str, image: Image.Image) -> None:
            frame = cv2.cvtColor(
                np.asarray(image.convert("RGB")),
                cv2.COLOR_RGB2BGR,
            )
            path = self.diagnostics.save_frame(label, frame)
            self.diagnostics.event(
                "warehouse_sell_diagnostic_frame",
                label=label,
                screenshot=str(path),
            )

        workflow = WarehouseSellWorkflow(
            port,
            sell_ocr=sell_ocr,
            log=self._emit,
            on_checkpoint=on_checkpoint,
            on_diagnostic_frame=on_diagnostic_frame,
            on_first_batch_listed=lambda: self._claim_sale_proceeds_for_round(port),
            pause_at_safe_point=self._pause_at_safe_point,
            restart_game=self._restart_for_page_recovery,
        )
        prior_batches = tuple(getattr(self, "_resumed_prior_batches", ()))
        self._resumed_prior_batches = ()
        if forced_relist:
            prior_batches = ()
        completed_relist_batches: tuple[WarehouseListedBatch, ...] = ()
        checkpoint = getattr(self, "_loadout_checkpoint", None)
        sale_gate_passed = forced_relist or bool(
            checkpoint is not None and checkpoint.sale_gate_passed
        )
        recovery_attempts = 0
        while True:
            if not self._claim_sale_proceeds_for_round(port):
                self._emit(
                    "交易行货款领取后暂时无法返回出售页，进入安全恢复",
                    "warning",
                )
                port.wait(max(1.0, float(settings.sell_slot_poll_seconds)))
                if not port.navigate_to(
                    GamePageId.MARKET_SELL,
                    reason="货款领取后恢复交易行出售页",
                ):
                    continue
            result = workflow.run(
                request,
                prior_batches=prior_batches,
                sale_gate_passed=sale_gate_passed,
            )
            if result.outcome is WarehouseSellOutcome.RECOVERY_REQUIRED:
                checkpoint = getattr(self, "_loadout_checkpoint", None)
                if (
                    checkpoint is not None
                    and checkpoint.stage
                    is LoadoutCheckpointStage.LISTING_CONFIRM_PENDING
                    and recovery_attempts < 1
                ):
                    recovery_attempts += 1
                    self._emit(
                        "最终上架结果暂时不可见；先观察当前页和库存后自动续跑",
                        "warning",
                    )
                    self._resolve_uncertain_listing_checkpoint(port)
                    checkpoint = getattr(self, "_loadout_checkpoint", None)
                    prior_batches = (
                        self._prior_batches_from_checkpoint(checkpoint)
                        if checkpoint is not None
                        else result.batches
                    )
                    self._resumed_prior_batches = ()
                    sale_gate_passed = True
                    continue
                if recovery_attempts < 2 and port.navigate_to(
                    GamePageId.MARKET_SELL,
                    reason="交易行直售局部异常后返回出售页重试扫描",
                ):
                    recovery_attempts += 1
                    self._emit(
                        f"交易行直售局部异常，已返回出售页进行第 "
                        f"{recovery_attempts} 次重试：{result.message}",
                        "warning",
                    )
                    prior_batches = result.batches
                    checkpoint = getattr(self, "_loadout_checkpoint", None)
                    sale_gate_passed = forced_relist or bool(
                        checkpoint is not None and checkpoint.sale_gate_passed
                    )
                    continue
                self._emit(
                    f"交易行直售进入恢复等待：{result.message}；"
                    "等待后会重新绑定窗口和页面，不会重复购买或最终确认按钮",
                    "warning",
                )
                self._transition(
                    WorkflowState.LOADOUT_WAREHOUSE_SELL,
                    "交易行直售恢复等待",
                    "warning",
                )
                prior_batches = result.batches
                port.wait(max(1.0, float(settings.sell_slot_poll_seconds)))
                self._check_stop()
                port.navigate_to(
                    GamePageId.MARKET_SELL,
                    reason="交易行直售恢复等待后重新进入出售页",
                )
                recovery_attempts = 0
                checkpoint = getattr(self, "_loadout_checkpoint", None)
                sale_gate_passed = forced_relist or bool(
                    checkpoint is not None and checkpoint.sale_gate_passed
                )
                continue
            if result.outcome is WarehouseSellOutcome.LISTED:
                if forced_relist:
                    completed_relist_batches += result.batches
                    if remaining_relist_names:
                        request = self._forced_one_tick_lower_request(
                            remaining_relist_names.pop(0)
                        )
                        prior_batches = ()
                        sale_gate_passed = True
                        recovery_attempts = 0
                        self._transition(
                            WorkflowState.LOADOUT_WAREHOUSE_SELL,
                            f"继续按降一柱重卖 {request.ammunition_name}",
                        )
                        continue
                    result = WarehouseSellResult(
                        outcome=WarehouseSellOutcome.LISTED,
                        batches=completed_relist_batches,
                        message=(
                            "超时撤单涉及的子弹已全部按降一柱重新上架："
                            + "、".join(tracked_ammunition_names)
                        ),
                        first_lowest_price=(
                            completed_relist_batches[0].lowest_price
                            if completed_relist_batches
                            else result.first_lowest_price
                        ),
                    )
                break
            if result.outcome is not WarehouseSellOutcome.NO_SELL_SLOT:
                break
            prior_batches = result.batches
            sale_gate_passed = True
            if getattr(self, "_pending_warehouse_listing", None) is None:
                self._record_pending_warehouse_listing(
                    rule,
                    sum(batch.quantity for batch in completed_relist_batches)
                    + result.listed_quantity,
                    ammunition_names=tuple(tracked_ammunition_names),
                )
            pending = self._pending_warehouse_listing
            pending_ammunition_names = (
                pending.ammunition_names if pending is not None else ()
            )
            recovery = self._resolve_pending_listing_with_recovery(
                port=port,
                sell_ocr=sell_ocr,
            )
            if recovery.outcome is SellSlotRecoveryOutcome.SLOT_AVAILABLE:
                self._transition(
                    WorkflowState.LOADOUT_WAREHOUSE_SELL,
                    f"售位已释放，继续上架 {request.ammunition_name} 的剩余库存",
                )
                continue
            if recovery.outcome is SellSlotRecoveryOutcome.UNLISTED:
                tracked_ammunition_names = list(
                    self._unique_ammunition_names(
                        rule.ammunition_name,
                        *tracked_ammunition_names,
                        *pending_ammunition_names,
                        request.ammunition_name,
                        *remaining_relist_names,
                    )
                )
                forced_relist = True
                completed_relist_batches = ()
                prior_batches = ()
                request = self._forced_one_tick_lower_request(
                    tracked_ammunition_names[0]
                )
                remaining_relist_names = tracked_ammunition_names[1:]
                sale_gate_passed = True
                recovery_attempts = 0
                self._transition(
                    WorkflowState.LOADOUT_WAREHOUSE_SELL,
                    "售位再次超时，全部撤单后重新按降一柱卖出："
                    + "、".join(tracked_ammunition_names),
                    "warning",
                )
                continue
            elif recovery.outcome is SellSlotRecoveryOutcome.STOPPED:
                result = WarehouseSellResult(
                    outcome=WarehouseSellOutcome.STOPPED,
                    batches=result.batches,
                    message=recovery.message,
                    first_lowest_price=result.first_lowest_price,
                )
            else:
                result = WarehouseSellResult(
                    outcome=WarehouseSellOutcome.RECOVERY_REQUIRED,
                    batches=result.batches,
                    message=recovery.message,
                    first_lowest_price=result.first_lowest_price,
                )
            break

        if result.outcome is WarehouseSellOutcome.LISTED:
            self._record_pending_warehouse_listing(
                rule,
                result.listed_quantity,
                ammunition_names=tuple(tracked_ammunition_names),
            )
        self._emit_structured(
            "loadout_warehouse_sell_result",
            f"配装方案 {rule.scheme_index} 交易行直售结果：{result.message}",
            "info" if result.completed else "warning",
            scheme_index=rule.scheme_index,
            ammunition_name=rule.ammunition_name,
            ammunition_names=tracked_ammunition_names,
            forced_one_tick_lower=forced_relist,
            outcome=result.outcome.value,
            listed_quantity=result.listed_quantity,
            listed_batches=len(result.batches),
            first_lowest_price=result.first_lowest_price,
        )
        return result

    @staticmethod
    def _unique_ammunition_names(*names: str) -> tuple[str, ...]:
        unique: list[str] = []
        for raw_name in names:
            name = raw_name.strip()
            if name and name not in unique:
                unique.append(name)
        return tuple(unique)

    @staticmethod
    def _forced_one_tick_lower_request(ammunition_name: str) -> WarehouseSellRequest:
        return WarehouseSellRequest(
            ammunition_name=ammunition_name,
            sell_trigger_price=1,
            price_mode=ListingPriceMode.ONE_TICK_LOWER,
        )

    def _claim_sale_proceeds_for_round(self, port) -> bool:
        checkpoint = getattr(self, "_loadout_checkpoint", None)
        round_id = max(0, int(getattr(self, "_loadout_round_id", 0)))
        if (
            round_id < 2
            or checkpoint is None
            or checkpoint.listed_batch_count < 1
            or checkpoint.sale_proceeds_claim_attempted
        ):
            return True

        self._save_loadout_checkpoint(
            checkpoint.stage,
            sale_proceeds_claim_attempted=True,
            last_confirmed_page=GamePageId.MARKET_SELL.value,
            last_safe_action="本轮首批上架完成，准备领取交易行货款",
        )
        self._emit(f"第 {round_id} 轮首批上架完成，点击交易行货款领取按钮")
        claimed = port.claim_sale_proceeds()
        if claimed:
            self._save_loadout_checkpoint(
                checkpoint.stage,
                sale_proceeds_claimed=True,
                last_confirmed_page=GamePageId.MARKET_SELL.value,
                last_safe_action="本轮交易行货款已领取并返回出售页",
            )
            self._emit_structured(
                "loadout_sale_proceeds_claimed",
                f"第 {round_id} 轮已领取交易行售卖金额",
                round_id=round_id,
                listed_batch_count=checkpoint.listed_batch_count,
            )
            return True

        recovered = port.navigate_to(
            GamePageId.MARKET_SELL,
            reason="货款领取页面未按预期返回，恢复出售页",
        )
        self._emit(
            "本轮货款领取未确认完成；本轮不再重复点击金币按钮，"
            + ("已恢复出售页" if recovered else "尚未恢复出售页"),
            "warning",
        )
        return recovered

    def _get_warehouse_sell_ocr(self):
        sell_ocr = getattr(self, "_warehouse_sell_ocr", None)
        if sell_ocr is None:
            from bulletbot.ocr.sell_ocr import SellOcr

            sell_ocr = SellOcr()
            self._warehouse_sell_ocr = sell_ocr
        return sell_ocr

    def _resolve_uncertain_listing_checkpoint(self, port) -> None:
        checkpoint = getattr(self, "_loadout_checkpoint", None)
        if (
            checkpoint is None
            or checkpoint.stage is not LoadoutCheckpointStage.LISTING_CONFIRM_PENDING
        ):
            return

        observed = None
        for _attempt in range(3):
            self._check_stop()
            try:
                observed = port.observe_page()
            except RuntimeError:
                observed = None
            if observed in {GamePageId.MARKET_SELL, GamePageId.LISTING_EDITOR}:
                break
            port.wait(0.5)

        confirmed_quantity = checkpoint.listed_quantity_total
        confirmed_batches = checkpoint.listed_batch_count
        if observed is GamePageId.MARKET_SELL:
            confirmed_quantity += checkpoint.pending_batch_quantity
            confirmed_batches += 1
            action = "中断恢复观察到交易行出售页，确认上一次最终上架已生效"
        elif observed is GamePageId.LISTING_EDITOR:
            port.press_escape()
            port.wait(0.35)
            if not port.navigate_to(
                GamePageId.MARKET_SELL,
                reason="最终上架中断后退出未提交的编辑器",
            ):
                raise RuntimeError("最终上架中断后无法返回交易行出售页")
            action = "中断恢复确认仍在编辑器，已取消未提交上架"
        else:
            if not port.navigate_to(
                GamePageId.MARKET_SELL,
                reason="最终上架结果未知时返回出售页重新核对库存",
            ):
                raise RuntimeError("最终上架结果未知且无法返回交易行出售页")
            action = "最终上架结果未知，未重复点击并返回出售页重新扫描"

        checkpoint = self._save_loadout_checkpoint(
            LoadoutCheckpointStage.LISTING_CONFIRMED,
            sale_gate_passed=True,
            listed_batch_count=confirmed_batches,
            listed_quantity_total=confirmed_quantity,
            pending_batch_quantity=0,
            pending_batch_price=None,
            last_confirmed_page=GamePageId.MARKET_SELL.value,
            last_safe_action=action,
        )
        self._resumed_prior_batches = self._prior_batches_from_checkpoint(checkpoint)

    @staticmethod
    def _prior_batches_from_checkpoint(
        checkpoint: LoadoutPurchaseCheckpoint,
    ) -> tuple[WarehouseListedBatch, ...]:
        count = max(0, checkpoint.listed_batch_count)
        if count == 0:
            return ()
        quantities = [0] * count
        quantities[0] = max(0, checkpoint.listed_quantity_total)
        reference_price = max(1, int(checkpoint.first_lowest_price or 1))
        return tuple(
            WarehouseListedBatch(
                quantity=quantity,
                unit_price=reference_price,
                lowest_price=reference_price,
                inventory_before=0,
            )
            for quantity in quantities
        )

    def _record_pending_warehouse_listing(
        self,
        rule: LoadoutSchemeRule,
        listed_quantity: int,
        *,
        ammunition_names: tuple[str, ...] = (),
    ) -> PendingWarehouseListing:
        settings = self._loadout_purchase_settings
        if settings is None:
            raise RuntimeError("配装买入配置尚未初始化")
        started_at = time.time()
        tracked_names: list[str] = []
        for raw_name in (rule.ammunition_name, *ammunition_names):
            name = raw_name.strip()
            if name and name not in tracked_names:
                tracked_names.append(name)
        pending = PendingWarehouseListing(
            scheme_index=rule.scheme_index,
            ammunition_name=rule.ammunition_name,
            listed_quantity=max(0, int(listed_quantity)),
            started_at=started_at,
            deadline=(
                started_at + settings.sell_wait_timeout_minutes * 60.0
            ),
            additional_ammunition_names=tuple(tracked_names[1:]),
        )
        self._pending_warehouse_listing = pending
        checkpoint = getattr(self, "_loadout_checkpoint", None)
        self._save_loadout_checkpoint(
            LoadoutCheckpointStage.WAITING_SALE,
            scheme_index=rule.scheme_index,
            ammo_name=rule.ammunition_name,
            sale_gate_passed=True,
            listed_quantity_total=max(
                max(0, int(listed_quantity)),
                checkpoint.listed_quantity_total if checkpoint is not None else 0,
            ),
            pending_batch_quantity=0,
            pending_batch_price=None,
            last_confirmed_page=GamePageId.MARKET_SELL.value,
            last_safe_action="出售页库存处理完成，挂单已进入等待售位阶段",
        )
        self._emit(
            f"已记录 {'、'.join(pending.ammunition_names)} 挂单等待截止时间："
            f"{settings.sell_wait_timeout_minutes} 分钟后"
        )
        return pending

    def _run_pending_warehouse_listing_recovery(
        self,
        *,
        port=None,
        sell_ocr=None,
    ) -> SellSlotRecoveryResult:
        from bulletbot.loadout_purchase.warehouse_sell_recovery import (
            WarehouseSellRecoveryWorkflow,
        )
        from bulletbot.loadout_purchase.warehouse_sell_runtime import (
            WarehouseSellRuntimePort,
        )

        settings = self._loadout_purchase_settings
        pending = getattr(self, "_pending_warehouse_listing", None)
        if settings is None or pending is None:
            raise RuntimeError("当前没有可检查的交易行直售挂单")
        if sell_ocr is None:
            sell_ocr = self._get_warehouse_sell_ocr()
        if port is None:
            port = WarehouseSellRuntimePort(
                window_provider=self.game.refresh,
                generation_provider=lambda: getattr(self.game, "generation", 0),
                capture=self.capture,
                input_controller=self.input,
                navigation=self._unified_navigation,
                stop_event=self._stop,
                page_recognizer=self._trading_page_recognizer,
                sell_ocr=sell_ocr,
            )

        recovery_origin = getattr(self, "_loadout_checkpoint", None)

        def on_stage(stage: str, message: str) -> None:
            state = (
                WorkflowState.LOADOUT_SELL_UNLIST
                if stage == "unlisting"
                else WorkflowState.LOADOUT_SELL_SLOT_WAIT
            )
            self._transition(state, message)
            checkpoint_stage = (
                LoadoutCheckpointStage.SELL_UNLIST_ALL
                if stage == "unlisting"
                else LoadoutCheckpointStage.SELL_SLOT_WAIT
            )
            self._save_loadout_checkpoint(
                checkpoint_stage,
                last_confirmed_page=GamePageId.MARKET_SELL.value,
                last_safe_action=message,
            )

        def on_checkpoint(stage: str, data: dict[str, object]) -> None:
            checkpoint_stage = (
                LoadoutCheckpointStage.UNLIST_CONFIRM_PENDING
                if stage == "unlist_confirm_pending"
                else LoadoutCheckpointStage.SELL_UNLIST_ALL
            )
            self._save_loadout_checkpoint(
                checkpoint_stage,
                last_confirmed_page=GamePageId.MARKET_SELL.value,
                last_safe_action=(
                    "最终下架按钮点击前已确认标题和操作按钮"
                    if checkpoint_stage
                    is LoadoutCheckpointStage.UNLIST_CONFIRM_PENDING
                    else f"已确认下架 {int(data.get('unlisted_orders', 0))} 个挂单"
                ),
            )

        result = WarehouseSellRecoveryWorkflow(
            port,
            sell_ocr=sell_ocr,
            log=self._emit,
            on_stage=on_stage,
            on_checkpoint=on_checkpoint,
            pause_at_safe_point=self._pause_at_safe_point,
        ).resolve(
            pending,
            wait_when_slots_full=settings.wait_when_slots_full,
            poll_seconds=settings.sell_slot_poll_seconds,
            uncertain_confirm_pending=bool(
                recovery_origin is not None
                and recovery_origin.stage
                is LoadoutCheckpointStage.UNLIST_CONFIRM_PENDING
            ),
            force_unlist=bool(
                recovery_origin is not None
                and recovery_origin.stage
                in {
                    LoadoutCheckpointStage.SELL_UNLIST_ALL,
                    LoadoutCheckpointStage.UNLIST_CONFIRM_PENDING,
                }
            ),
        )
        if result.outcome in {
            SellSlotRecoveryOutcome.SLOT_AVAILABLE,
            SellSlotRecoveryOutcome.UNLISTED,
        }:
            self._pending_warehouse_listing = None
            if recovery_origin is not None:
                if (
                    result.outcome is SellSlotRecoveryOutcome.SLOT_AVAILABLE
                    and recovery_origin.sale_gate_passed
                    and recovery_origin.stage
                    in {
                        LoadoutCheckpointStage.LISTING_CONFIRMED,
                        LoadoutCheckpointStage.LISTING_CONFIRM_PENDING,
                        LoadoutCheckpointStage.SELL_SLOT_WAIT,
                        LoadoutCheckpointStage.SELL_UNLIST_ALL,
                        LoadoutCheckpointStage.UNLIST_CONFIRM_PENDING,
                    }
                ):
                    next_stage = LoadoutCheckpointStage.LISTING_CONFIRMED
                else:
                    next_stage = LoadoutCheckpointStage.BLANK_APPLIED
                self._save_loadout_checkpoint(
                    next_stage,
                    last_confirmed_page=GamePageId.MARKET_SELL.value,
                    last_safe_action=(
                        "售位释放，旧挂单不再阻塞"
                        if result.outcome is SellSlotRecoveryOutcome.SLOT_AVAILABLE
                        else "下架按钮连续消失，全部挂单已回库"
                    ),
                )
        if result.outcome is SellSlotRecoveryOutcome.SLOT_AVAILABLE:
            self._emit(
                f"检测到空闲售位：当前 {result.used_slots}/{result.total_slots}；"
                "旧挂单不再阻塞当前轮次上架"
            )
        elif result.outcome is SellSlotRecoveryOutcome.UNLISTED:
            self._emit(
                f"等待超时后已下架全部挂单，共确认 {result.unlisted_orders} 个"
            )
        self._emit_structured(
            "loadout_sell_slot_recovery",
            result.message,
            "info" if result.outcome in {
                SellSlotRecoveryOutcome.SLOT_AVAILABLE,
                SellSlotRecoveryOutcome.UNLISTED,
            } else "warning",
            outcome=result.outcome.value,
            pending_scheme_index=pending.scheme_index,
            pending_ammunition_name=pending.ammunition_name,
            pending_listed_quantity=pending.listed_quantity,
            used_slots=result.used_slots,
            total_slots=result.total_slots,
            unlisted_orders=result.unlisted_orders,
        )
        return result

    def _resolve_pending_listing_with_recovery(
        self,
        *,
        port=None,
        sell_ocr=None,
    ) -> SellSlotRecoveryResult:
        settings = self._loadout_purchase_settings
        if settings is None:
            raise RuntimeError("配装买入配置尚未初始化")
        while True:
            if port is None and sell_ocr is None:
                result = self._run_pending_warehouse_listing_recovery()
            else:
                if sell_ocr is None:
                    sell_ocr = self._get_warehouse_sell_ocr()
                result = self._run_pending_warehouse_listing_recovery(
                    port=port,
                    sell_ocr=sell_ocr,
                )
            if result.outcome is not SellSlotRecoveryOutcome.RECOVERY_REQUIRED:
                return result
            self._emit(
                f"售位等待或全部下架暂时无法确认：{result.message}；"
                "进入安全恢复等待，不会跳过剩余挂单",
                "warning",
            )
            wait_seconds = max(1.0, float(settings.sell_slot_poll_seconds))
            if port is None:
                self._sleep(wait_seconds)
            else:
                port.wait(wait_seconds)
                self._check_stop()

    def _wait_and_buy_loadout_scheme(
        self,
        settings: LoadoutPurchaseSettings,
    ) -> LoadoutPurchaseSelection:
        self._start_loadout_round_checkpoint()
        self._transition(
            WorkflowState.LOADOUT_PURCHASE_PREPARE,
            "进入零号大坝配装页并读取购买前余额",
        )
        self._prepare_game_for_trading_handoff()
        loadout = self._open_zero_dam_loadout()
        balance_before, balance_page = self._read_loadout_balance_with_page_recovery(
            "配装买入前",
            loadout,
        )
        self._emit_structured(
            "loadout_balance",
            f"配装买入本轮起始余额：{balance_before:,}",
            label="配装买入前",
            balance=balance_before,
        )

        if balance_page.page_type in LOADOUT_SCHEME_PAGES:
            opened = balance_page
        else:
            game = self._require_bound(self.game, "游戏")
            self._press_key(game, "loadout_scheme", "L：打开配装方案")
            opened = self._wait_page(
                self.game,
                LOADOUT_SCHEME_PAGES
                | {
                    PageType.LOADOUT,
                    PageType.LOADOUT_PURCHASE_CONFIRM,
                    PageType.LOADOUT_PRICE_CHANGE,
                },
                45,
                "配装买入方案列表或用户操作后的页面",
                include_ocr=True,
            )
        if opened.page_type in {
            PageType.LOADOUT_PURCHASE_CONFIRM,
            PageType.LOADOUT_PRICE_CHANGE,
        }:
            opened = self._cancel_loadout_purchase_page(opened.page_type)
        self._ensure_loadout_scheme_list(opened)

        warmup_started = time.monotonic()
        self._warm_fast_loadout_price_ocr()
        self._emit(
            "价格文字 OCR 已在点击方案前完成预热；耗时 "
            f"{(time.monotonic() - warmup_started) * 1000:.0f}ms"
        )

        self._transition(
            WorkflowState.LOADOUT_PURCHASE_SCAN,
            "按启用方案顺序切换，局部确认高亮并识别价格文字",
        )
        enabled_by_index = {
            rule.scheme_index: rule for rule in settings.enabled_schemes
        }
        scan_order = self._loadout_scan_order(settings)
        self._emit(
            "配装方案刷新顺序："
            + " -> ".join(str(index) for index in scan_order)
            + f"；切换后 {LOADOUT_SCHEME_INITIAL_SETTLE_SECONDS * 1000:.0f} ms "
            + "开始局部检查唯一强高亮，每 "
            + f"{LOADOUT_SCHEME_RETRY_INTERVAL_SECONDS * 1000:.0f} ms 检查，"
            + "局部命中后截取左右联合矩形，同帧复核高亮并执行价格文字 OCR，"
            + f"最长确认 {settings.scan_interval_seconds:.1f} 秒"
        )
        high_price_summaries: dict[int, _LoadoutHighPriceSummary] = {}
        quantity_inference = {
            rule.scheme_index: _LoadoutQuantityInference()
            for rule in settings.enabled_schemes
        }
        unconfirmed_streak = 0
        while True:
            self._check_stop()
            for scheme_index in scan_order:
                self._raise_if_scheduled_game_restart_due()
                refresh = self._switch_loadout_scheme_for_scan(
                    scheme_index,
                    settings.scan_interval_seconds,
                )
                if not refresh.confirmed:
                    unconfirmed_streak += 1
                    self._emit(
                        f"配装方案 {scheme_index} 唯一强高亮未确认："
                        f"{refresh.reason}；本次不购买，继续刷新",
                        "warning",
                    )
                    if (
                        refresh.status == "error"
                        or unconfirmed_streak
                        >= LOADOUT_HIGHLIGHT_UNCONFIRMED_RECOVERY_STREAK
                    ):
                        self._recover_loadout_scan_state(
                            scheme_index,
                            settings.scan_interval_seconds,
                            refresh.reason,
                        )
                        unconfirmed_streak = 0
                    continue
                unconfirmed_streak = 0
                rule = enabled_by_index.get(scheme_index)
                if rule is None:
                    continue
                assert refresh.selected_frame is not None
                assert refresh.selected_region is not None
                assert refresh.frame_size is not None

                offer = self._read_loadout_scheme_offer(
                    rule.scheme_index,
                    refresh.selected_frame,
                    None,
                    frame_region=refresh.selected_region,
                    frame_size=refresh.frame_size,
                )
                if not offer.selection_confirmed:
                    self._emit(
                        f"配装方案 {rule.scheme_index} 同帧高亮复核失败；"
                        "本次价格作废，不购买",
                        "warning",
                    )
                    continue
                if offer.price is None:
                    self._emit(
                        f"配装方案 {rule.scheme_index} 同帧局部 OCR 未识别到"
                        "整套金额；进入完整页面恢复，本次不购买",
                        "warning",
                    )
                    self._recover_loadout_scan_state(
                        scheme_index,
                        settings.scan_interval_seconds,
                        "同帧局部 OCR 不确定",
                    )
                    continue
                self._emit_loadout_price_recognition(
                    rule.scheme_index,
                    offer.price,
                    mode="normal",
                )
                validated_price = self._validate_loadout_scan_price(
                    rule,
                    offer.price,
                    quantity_inference[rule.scheme_index],
                )
                if validated_price is None:
                    continue
                offer_price = validated_price
                if offer_price > rule.max_total_price:
                    self._record_loadout_high_price_summary(
                        high_price_summaries,
                        rule,
                        offer_price,
                    )
                    continue
                high_price_summaries.pop(rule.scheme_index, None)

                self._emit_structured(
                    "loadout_low_price",
                    f"配装方案 {rule.scheme_index} 刷出低价：整套 "
                    f"{offer_price:,}，目标 {rule.total_quantity:,} 发",
                    scheme_index=rule.scheme_index,
                    quantity=rule.total_quantity,
                    total_price=offer_price,
                    displayed_unit_price=offer_price / rule.total_quantity,
                    max_total_price=rule.max_total_price,
                    max_unit_price=rule.max_unit_price,
                )

                self._transition(
                    WorkflowState.LOADOUT_PURCHASE_BUY,
                    f"配装方案 {rule.scheme_index} 达标，立即购买",
                )
                self._begin_loadout_purchase_action()
                self._save_loadout_checkpoint(
                    LoadoutCheckpointStage.PURCHASE_CONFIRM_PENDING,
                    scheme_index=rule.scheme_index,
                    ammo_name=rule.ammunition_name,
                    purchase_balance_before=balance_before,
                    purchase_listed_price=offer_price,
                    purchase_was_partial=False,
                    last_confirmed_page=PageType.LOADOUT_SCHEMES.value,
                    last_safe_action="最终购买点击前已确认方案、价格和余额",
                )
                # Once the purchase click can happen, keep pause requests
                # deferred until balance reconciliation and the resulting
                # safe checkpoint are complete.
                self._begin_pause_deferral()
                attempt = self._purchase_selected_loadout_scheme(rule, offer_price)
                if not attempt:
                    balance_label = (
                        "价格变动后余额"
                        if attempt.price_changed
                        else "购买未确认后余额"
                    )
                    balance_after_attempt, balance_page = (
                        self._read_loadout_balance_after_unconfirmed_purchase(
                            balance_label,
                            attempt.final_page,
                        )
                    )
                    spent = balance_before - balance_after_attempt
                    if spent > 0:
                        self._save_loadout_checkpoint(
                            LoadoutCheckpointStage.PURCHASE_CONFIRMED,
                            scheme_index=rule.scheme_index,
                            ammo_name=rule.ammunition_name,
                            purchase_balance_before=balance_before,
                            purchase_listed_price=offer_price,
                            purchase_was_partial=True,
                            last_confirmed_page=(
                                balance_page.page_type.value
                                if balance_page is not None
                                else None
                            ),
                            last_safe_action="余额减少，确认至少部分购买成交",
                        )
                        self._emit_loadout_purchase_result(
                            rule,
                            offer_price,
                            "partial",
                            balance_before,
                            balance_after_attempt,
                            spent=spent,
                        )
                        self._prepare_partial_loadout_for_mail(balance_page)
                        self._end_pause_deferral()
                        return LoadoutPurchaseSelection(
                            rule=rule,
                            listed_price=offer_price,
                            actual_average=None,
                            partial_spent=spent,
                        )
                    self._emit_loadout_purchase_result(
                        rule,
                        offer_price,
                        "none",
                        balance_before,
                        balance_after_attempt,
                        spent=spent,
                    )
                    if spent < 0:
                        self._emit(
                            f"购买尝试后余额由 {balance_before:,} 增加到 "
                            f"{balance_after_attempt:,}；未判定成交，"
                            "重新确认方案列表",
                            "warning",
                        )
                    else:
                        self._emit(
                            "购买尝试后余额未减少，确认本次完全没有成交；"
                            "重新确认方案列表并继续刷新"
                        )
                    balance_before = balance_after_attempt
                    self._save_loadout_checkpoint(
                        LoadoutCheckpointStage.PURCHASE_SCAN,
                        scheme_index=None,
                        ammo_name="",
                        purchase_balance_before=None,
                        purchase_listed_price=None,
                        purchase_was_partial=False,
                        last_confirmed_page=(
                            balance_page.page_type.value
                            if balance_page is not None
                            else None
                        ),
                        last_safe_action="余额未减少，确认本次没有购买成交",
                    )
                    self._ensure_loadout_scheme_list(balance_page)
                    self._transition(
                        WorkflowState.LOADOUT_PURCHASE_SCAN,
                        f"配装方案 {rule.scheme_index} 未完成购买，继续轮询",
                        "warning",
                    )
                    self._loadout_round_in_flight = False
                    self._end_pause_deferral()
                    continue
                balance_after, balance_page = (
                    self._read_loadout_balance_with_page_recovery(
                        "配装买入后",
                        attempt.final_page,
                    )
                )
                actual_average = self._calculate_loadout_average(
                    balance_before,
                    balance_after,
                    rule.total_quantity,
                )
                spent = balance_before - balance_after
                self._save_loadout_checkpoint(
                    LoadoutCheckpointStage.PURCHASE_CONFIRMED,
                    scheme_index=rule.scheme_index,
                    ammo_name=rule.ammunition_name,
                    purchase_balance_before=balance_before,
                    purchase_listed_price=offer_price,
                    purchase_was_partial=False,
                    last_confirmed_page=balance_page.page_type.value,
                    last_safe_action="购买后余额减少并确认完整成交",
                )
                self._emit_loadout_purchase_result(
                    rule,
                    offer_price,
                    "full",
                    balance_before,
                    balance_after,
                    spent=spent,
                    actual_average=actual_average,
                )
                self._apply_blank_loadout_scheme(
                    settings.blank_scheme_index,
                    schemes_open=balance_page.page_type in LOADOUT_SCHEME_PAGES,
                )
                self._save_loadout_checkpoint(
                    LoadoutCheckpointStage.BLANK_APPLIED,
                    last_confirmed_page=PageType.GAME_HOME_READY.value,
                    last_safe_action=f"已应用空白方案 {settings.blank_scheme_index}",
                )
                self._end_pause_deferral()
                return LoadoutPurchaseSelection(
                    rule=rule,
                    listed_price=offer_price,
                    actual_average=actual_average,
                )

    def _emit_loadout_price_recognition(
        self,
        scheme_index: int,
        price: int,
        *,
        mode: str,
    ) -> None:
        """Send a silent bookkeeping event; the UI persists it outside run.log."""
        callback = getattr(self, "on_event", None)
        if callback is None:
            return
        callback(
            WorkflowEvent(
                self._state,
                "",
                "debug",
                event_type="loadout_price_recognized",
                data={
                    "scheme_index": int(scheme_index),
                    "price": int(price),
                    "mode": mode,
                },
            )
        )

    def _validate_loadout_scan_price(
        self,
        rule: LoadoutSchemeRule,
        price: int,
        inference: _LoadoutQuantityInference,
    ) -> int | None:
        """Apply cheap price guards and the per-scheme quantity inference."""
        if price < rule.max_total_price * LOADOUT_PRICE_MIN_RATIO:
            if inference.last_guarded_price != price:
                self._emit(
                    f"配装方案 {rule.scheme_index} 识别价格 {price:,} 低于最高总价的"
                    f"{LOADOUT_PRICE_MIN_RATIO:.0%}；"
                    "按异常低价保护跳过，不加入发数推断样本",
                    "warning",
                )
                inference.last_guarded_price = price
            return None

        if inference.inferred_quantity is None:
            if price not in inference.samples:
                inference.samples.append(price)
                if len(inference.samples) > LOADOUT_QUANTITY_INFERENCE_MAX_SAMPLES:
                    inference.samples.pop(0)
            candidates = self._infer_loadout_quantities(rule, inference.samples)
            inference.candidates = candidates
            if len(inference.samples) >= LOADOUT_QUANTITY_INFERENCE_MIN_SAMPLES:
                if len(candidates) == 1:
                    inference.inferred_quantity = candidates[0]
                    slots = inference.inferred_quantity // rule.units_per_slot
                    self._emit(
                        f"配装方案 {rule.scheme_index} 已根据 "
                        f"{len(inference.samples)} 个不同价格推断总发数："
                        f"{inference.inferred_quantity:,} 发（约 {slots} 格）"
                    )
                elif (
                    len(inference.samples) >= LOADOUT_QUANTITY_INFERENCE_MAX_SAMPLES
                    and inference.unresolved_warning_count
                    < LOADOUT_QUANTITY_INFERENCE_MAX_WARNINGS
                ):
                    self._emit(
                        f"配装方案 {rule.scheme_index} 暂未唯一推断总发数；"
                        f"已收集 {len(inference.samples)} 个不同价格，继续按原逻辑扫描",
                        "warning",
                    )
                    inference.unresolved_warning_count += 1
            return price

        if price > rule.max_total_price:
            return price
        if price % inference.inferred_quantity == 0:
            return price

        corrected = [
            candidate
            for candidate in self._zero_eight_price_candidates(price)
            if candidate % inference.inferred_quantity == 0
            and candidate >= rule.max_total_price * LOADOUT_PRICE_MIN_RATIO
            and candidate <= rule.max_total_price
        ]
        if len(corrected) != 1:
            self._emit(
                f"配装方案 {rule.scheme_index} 价格 {price:,} 未通过已确认总发数 "
                f"{inference.inferred_quantity:,} 的整倍数校验；"
                + ("0/8 修正结果不唯一，" if len(corrected) > 1 else "")
                + "本次不购买",
                "warning",
            )
            return None
        self._emit(
            f"配装方案 {rule.scheme_index} 价格按 0/8 混淆规则修正："
            f"{price:,} -> {corrected[0]:,}",
            "warning",
        )
        return corrected[0]

    @staticmethod
    def _infer_loadout_quantities(
        rule: LoadoutSchemeRule,
        samples: list[int],
    ) -> tuple[int, ...]:
        if len(samples) < LOADOUT_QUANTITY_INFERENCE_MIN_SAMPLES:
            return ()
        configured = rule.total_quantity
        tolerance_percent = round(LOADOUT_QUANTITY_INFERENCE_TOLERANCE * 100)
        lower_numerator = configured * (100 - tolerance_percent)
        upper_numerator = configured * (100 + tolerance_percent)
        denominator = 100 * rule.units_per_slot
        min_slots = max(1, (lower_numerator + denominator - 1) // denominator)
        max_slots = upper_numerator // denominator
        return tuple(
            quantity
            for slots in range(min_slots, max_slots + 1)
            for quantity in (slots * rule.units_per_slot,)
            if all(price % quantity == 0 for price in samples)
        )

    @staticmethod
    def _zero_eight_price_candidates(price: int) -> tuple[int, ...]:
        """Return one-pass 0/8 substitutions, keeping a final digit at zero."""
        digits = list(str(price))
        if not digits:
            return ()
        if digits[-1] == "8":
            digits[-1] = "0"
        ambiguous = [
            index
            for index, digit in enumerate(digits[:-1])
            if digit in {"0", "8"}
        ]
        candidates: set[int] = set()
        for mask in range(1 << len(ambiguous)):
            variant = digits.copy()
            for bit, index in enumerate(ambiguous):
                variant[index] = "8" if mask & (1 << bit) else "0"
            if variant[0] == "0":
                continue
            value = int("".join(variant))
            if value >= 1_000:
                candidates.add(value)
        return tuple(sorted(candidates))

    def _record_loadout_high_price_summary(
        self,
        summaries: dict[int, _LoadoutHighPriceSummary],
        rule: LoadoutSchemeRule,
        price: int,
    ) -> None:
        now = self._monotonic()
        summary = summaries.get(rule.scheme_index)
        if summary is None:
            summaries[rule.scheme_index] = _LoadoutHighPriceSummary(
                started_at=now,
                samples=1,
                minimum=price,
                maximum=price,
                latest=price,
            )
            self._emit(
                f"配装方案 {rule.scheme_index} 当前整套 {price:,}，高于上限 "
                f"{rule.max_total_price:,}；后续每 "
                f"{LOADOUT_HIGH_PRICE_LOG_INTERVAL_SECONDS:.0f} 秒汇总高价扫描"
            )
            return

        if summary.samples == 0:
            summary.samples = 1
            summary.minimum = price
            summary.maximum = price
        else:
            summary.samples += 1
            summary.minimum = min(summary.minimum, price)
            summary.maximum = max(summary.maximum, price)
        summary.latest = price
        elapsed = now - summary.started_at
        if elapsed < LOADOUT_HIGH_PRICE_LOG_INTERVAL_SECONDS:
            return

        self._emit(
            f"配装方案 {rule.scheme_index} 最近 {elapsed:.1f} 秒高价扫描 "
            f"{summary.samples} 次：最低 {summary.minimum:,}，最高 "
            f"{summary.maximum:,}，最新 {summary.latest:,}；上限 "
            f"{rule.max_total_price:,}"
        )
        summary.started_at = now
        summary.samples = 0
        summary.minimum = 0
        summary.maximum = 0
        summary.latest = price

    def _wait_and_buy_fast_loadout_scheme(
        self,
        settings: LoadoutPurchaseSettings,
    ) -> LoadoutPurchaseSelection:
        self._start_loadout_round_checkpoint()
        self._fast_loadout_buy_locked = False
        self._transition(
            WorkflowState.LOADOUT_PURCHASE_PREPARE,
            "进入零号大坝配装页并读取极速配装购买前余额",
        )
        self._prepare_game_for_trading_handoff()
        loadout = self._open_zero_dam_loadout()
        balance_before, balance_page = self._read_loadout_balance_with_page_recovery(
            "极速配装买入前",
            loadout,
        )
        self._emit_structured(
            "loadout_balance",
            f"极速配装本轮起始余额：{balance_before:,}",
            label="极速配装买入前",
            balance=balance_before,
        )

        if balance_page.page_type in LOADOUT_SCHEME_PAGES:
            opened = balance_page
        else:
            game = self._require_bound(self.game, "游戏")
            self._press_key(game, "loadout_scheme", "L：打开配装方案")
            opened = self._wait_page(
                self.game,
                LOADOUT_SCHEME_PAGES
                | {
                    PageType.LOADOUT,
                    PageType.LOADOUT_PURCHASE_CONFIRM,
                    PageType.LOADOUT_PRICE_CHANGE,
                },
                45,
                "极速配装方案列表或用户操作后的页面",
                include_ocr=True,
            )
        if opened.page_type in {
            PageType.LOADOUT_PURCHASE_CONFIRM,
            PageType.LOADOUT_PRICE_CHANGE,
        }:
            opened = self._cancel_loadout_purchase_page(opened.page_type)
        self._ensure_loadout_scheme_list(opened)

        rules = settings.enabled_schemes
        if len(rules) != 2:
            raise RuntimeError("极速配装模式必须且只能启用两个方案")
        expected = (
            rules[0].slot_count,
            rules[0].units_per_slot,
            rules[0].max_unit_price,
        )
        if any(
            (rule.slot_count, rule.units_per_slot, rule.max_unit_price) != expected
            for rule in rules[1:]
        ):
            raise RuntimeError("极速配装的两个方案必须使用完全相同的购买参数")

        warmup_started = time.monotonic()
        self._warm_fast_loadout_price_ocr()
        self._emit(
            "极速价格 OCR 已在点击方案前完成预热；耗时 "
            f"{(time.monotonic() - warmup_started) * 1000:.0f}ms"
        )

        self._transition(
            WorkflowState.LOADOUT_PURCHASE_SCAN,
            "极速模式仅截取价格区域；高价切换，低价锁定购买",
        )
        self._emit(
            "极速配装顺序："
            + " -> ".join(str(rule.scheme_index) for rule in rules)
            + "；OCR 期间禁止输入，明确高价或连续三次识别无效并恢复后"
            "才切换方案"
        )
        invalid_streak = 0
        quantity_inference = {
            rule.scheme_index: _LoadoutQuantityInference() for rule in rules
        }
        rule_position = 0
        needs_scheme_click = True
        confirmed_switch_required = True
        while True:
            self._check_stop()
            self._raise_if_scheduled_game_restart_due()
            rule = rules[rule_position]
            if needs_scheme_click:
                if self._fast_loadout_buy_locked:
                    raise RuntimeError("极速配装购买锁未释放，拒绝切换方案")
                if confirmed_switch_required:
                    refresh = self._switch_loadout_scheme_for_scan(
                        rule.scheme_index,
                        settings.scan_interval_seconds,
                    )
                    if not refresh.confirmed:
                        self._recover_loadout_scan_state(
                            rule.scheme_index,
                            settings.scan_interval_seconds,
                            "未成交后切换方案未确认",
                        )
                        continue
                    confirmed_switch_required = False
                else:
                    self._sleep(settings.fast_loadout_switch_delay_ms / 1000.0)
                    self._click_fast_loadout_scheme(rule.scheme_index)
                    self._sleep(settings.fast_loadout_click_settle_ms / 1000.0)

            price: int | None = None
            while True:
                self._check_stop()
                price = self._read_fast_loadout_price(rule.scheme_index)
                if price is not None:
                    invalid_streak = 0
                    self._emit_loadout_price_recognition(
                        rule.scheme_index,
                        price,
                        mode="fast",
                    )
                    validated_price = self._validate_loadout_scan_price(
                        rule,
                        price,
                        quantity_inference[rule.scheme_index],
                    )
                    if validated_price is not None:
                        price = validated_price
                        break
                    continue
                invalid_streak += 1
                if invalid_streak < FAST_LOADOUT_INVALID_OCR_RECOVERY_STREAK:
                    continue
                diagnostic_path = (
                    self._save_fast_loadout_ocr_failure_diagnostic(
                        rule.scheme_index,
                    )
                )
                diagnostic_suffix = (
                    f"；失败价格裁切图已保存为 {diagnostic_path.name}"
                    if diagnostic_path is not None
                    else ""
                )
                self._emit(
                    f"极速配装方案 {rule.scheme_index} 连续 "
                    f"{invalid_streak} 次价格识别无效；恢复页面后切换到"
                    f"下一启用方案刷新报价{diagnostic_suffix}",
                    "warning",
                )
                self._recover_loadout_scan_state(
                    rule.scheme_index,
                    settings.scan_interval_seconds,
                    "极速价格 OCR 连续无效",
                )
                invalid_streak = 0
                rule_position = (rule_position + 1) % len(rules)
                needs_scheme_click = True
                break

            if price is None:
                continue

            if price > rule.max_total_price:
                rule_position = (rule_position + 1) % len(rules)
                needs_scheme_click = True
                continue

            # This lock is set before logging or purchase handling. No scheme
            # switch is permitted until a no-fill result is fully reconciled.
            self._fast_loadout_buy_locked = True
            selection, balance_before = self._buy_fast_loadout_offer(
                settings,
                rule,
                price,
                balance_before,
            )
            if selection is not None:
                return selection
            self._fast_loadout_buy_locked = False
            # A failed low-price attempt may leave the old low quote visible.
            # Confirm the next scheme once before trusting another price crop.
            rule_position = (rule_position + 1) % len(rules)
            needs_scheme_click = True
            confirmed_switch_required = True

    def _click_fast_loadout_scheme(self, scheme_index: int) -> None:
        if self._fast_loadout_buy_locked:
            raise RuntimeError("极速配装已锁定购买，拒绝切换方案")
        game = self._require_bound(self.game, "游戏")
        point = self.settings.action_points[f"loadout_scheme_{scheme_index}"]
        self.input.click_normalized(game, point)

    def _read_fast_loadout_price(self, scheme_index: int) -> int | None:
        expected_quantity = self._loadout_scheme_quantity(scheme_index)
        self._last_fast_loadout_ocr_expected_quantity = expected_quantity
        crop = self._capture_fast_loadout_price_crop()
        if crop is None:
            self._last_fast_loadout_ocr_attempts = []
            self._last_fast_loadout_ocr_rejection_reason = "unhealthy_crop"
            return None
        recognized = self._recognize_fast_loadout_price_crop(
            crop,
            expected_quantity=expected_quantity,
        )
        return recognized[0] if recognized is not None else None

    def _recognize_fast_loadout_price_crop(
        self,
        crop: np.ndarray,
        *,
        expected_quantity: int | None = None,
        enforce_quantity: bool = True,
    ) -> tuple[int, str, float] | None:
        """Recognize one raw price crop, with multi-scale voting on uncertainty."""
        attempts: list[dict[str, object]] = []
        self._last_fast_loadout_ocr_attempts = attempts
        self._last_fast_loadout_ocr_rejection_reason = ""
        self._last_fast_loadout_ocr_expected_quantity = expected_quantity
        primary = self._recognize_fast_loadout_price_at_scale(
            crop,
            FAST_LOADOUT_PRICE_SCALE,
            attempts=attempts,
        )
        if expected_quantity is not None:
            primary = self._correct_fast_loadout_price_trailing_eight(primary)
        if primary is not None:
            attempts[0]["normalized_price"] = primary[0]
        if primary is not None:
            if primary[2] >= FAST_LOADOUT_PRICE_MIN_CONFIDENCE:
                return primary
            if (
                enforce_quantity
                and
                expected_quantity is not None
                and self._price_matches_loadout_quantity(
                    primary[0],
                    expected_quantity,
                )
            ):
                return primary

        fallback_candidates = [
            self._recognize_fast_loadout_price_at_scale(
                crop,
                scale,
                attempts=attempts,
            )
            for scale in FAST_LOADOUT_PRICE_FALLBACK_SCALES
        ]
        if expected_quantity is not None:
            fallback_candidates = [
                self._correct_fast_loadout_price_trailing_eight(candidate)
                for candidate in fallback_candidates
            ]
        for attempt, candidate in zip(attempts[1:], fallback_candidates):
            if candidate is not None:
                attempt["normalized_price"] = candidate[0]

        if enforce_quantity and expected_quantity is not None:
            divisible_candidates = [
                candidate
                for candidate in fallback_candidates
                if candidate is not None
                and self._price_matches_loadout_quantity(
                    candidate[0],
                    expected_quantity,
                )
            ]
            divisible_prices = {candidate[0] for candidate in divisible_candidates}
            if len(divisible_prices) > 1:
                self._last_fast_loadout_ocr_rejection_reason = (
                    "multiple_divisible_prices"
                )
                return None
            if len(divisible_prices) == 1:
                price = next(iter(divisible_prices))
                return max(
                    (
                        candidate
                        for candidate in divisible_candidates
                        if candidate[0] == price
                    ),
                    key=lambda candidate: candidate[2],
                )

        candidates = [primary, *fallback_candidates]
        valid_candidates = [
            candidate
            for candidate in candidates
            if candidate is not None
        ]
        if not valid_candidates:
            self._last_fast_loadout_ocr_rejection_reason = "no_valid_candidates"
            return None

        counts = Counter(candidate[0] for candidate in valid_candidates)
        ranked = counts.most_common()
        price, votes = ranked[0]
        if votes < FAST_LOADOUT_PRICE_MIN_VOTES:
            self._last_fast_loadout_ocr_rejection_reason = "insufficient_votes"
            return None
        if len(ranked) > 1 and ranked[1][1] == votes:
            self._last_fast_loadout_ocr_rejection_reason = "tied_votes"
            return None
        matching = [candidate for candidate in valid_candidates if candidate[0] == price]
        return max(matching, key=lambda candidate: candidate[2])

    def _loadout_scheme_quantity(self, scheme_index: int) -> int | None:
        return next(
            (
                rule.total_quantity
                for rule in self._timing_settings().schemes
                if rule.scheme_index == scheme_index and rule.total_quantity > 0
            ),
            None,
        )

    @staticmethod
    def _price_matches_loadout_quantity(
        price: int,
        expected_quantity: int | None,
    ) -> bool:
        return expected_quantity is None or price % expected_quantity == 0

    @staticmethod
    def _correct_fast_loadout_price_trailing_eight(
        candidate: tuple[int, str, float] | None,
    ) -> tuple[int, str, float] | None:
        if candidate is None or candidate[0] % 10 != 8:
            return candidate
        price, _text, confidence = candidate
        corrected_price = price - 8
        return corrected_price, f"{corrected_price:,}", confidence

    def _recognize_fast_loadout_price_at_scale(
        self,
        crop: np.ndarray,
        scale: float,
        *,
        attempts: list[dict[str, object]] | None = None,
    ) -> tuple[int, str, float] | None:
        prepared = self._scale_fast_loadout_price_crop(crop, scale)
        recognized = self.vision.fast_price_ocr.recognize_line(prepared)
        if recognized is None:
            if attempts is not None:
                attempts.append(
                    {
                        "scale": scale,
                        "text": None,
                        "confidence": None,
                        "parsed_price": None,
                        "status": "no_text",
                    }
                )
            return None
        text, confidence = recognized
        price = self._parse_ocr_integer(text)
        if price is None:
            status = "parse_failed"
        elif price < 1_000:
            status = "price_too_small"
        elif not self._has_valid_ocr_thousands_grouping(text):
            status = "invalid_thousands_grouping"
        else:
            status = "candidate"
        if attempts is not None:
            attempts.append(
                {
                    "scale": scale,
                    "text": text,
                    "confidence": confidence,
                    "parsed_price": price,
                    "status": status,
                }
            )
        if status != "candidate":
            return None
        assert price is not None
        return price, text, confidence

    def _save_fast_loadout_ocr_failure_diagnostic(
        self,
        scheme_index: int,
    ) -> Path | None:
        crop = getattr(self, "_last_fast_loadout_price_crop", None)
        diagnostics = getattr(self, "diagnostics", None)
        if (
            diagnostics is None
            or not isinstance(crop, np.ndarray)
            or crop.size == 0
        ):
            return None
        try:
            path = diagnostics.save_frame(
                f"fast_price_ocr_invalid_scheme_{scheme_index}",
                crop,
            )
            diagnostics.event(
                "fast_loadout_price_ocr_invalid",
                scheme_index=scheme_index,
                rejection_reason=getattr(
                    self,
                    "_last_fast_loadout_ocr_rejection_reason",
                    "unknown",
                ),
                expected_quantity=getattr(
                    self,
                    "_last_fast_loadout_ocr_expected_quantity",
                    None,
                ),
                attempts=getattr(self, "_last_fast_loadout_ocr_attempts", []),
                screenshot=str(path),
                crop_shape=list(crop.shape),
            )
            return path
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return None

    def _warm_fast_loadout_price_ocr(self) -> None:
        for _attempt in range(3):
            crop = self._capture_fast_loadout_price_crop()
            if crop is not None:
                self.vision.fast_price_ocr.recognize_line(
                    self._scale_fast_loadout_price_crop(crop)
                )
                return
            self._sleep(0.05)
        raise RuntimeError("极速价格区域截图连续无效，无法安全预热 OCR")

    def _capture_fast_loadout_price_crop(self) -> np.ndarray | None:
        game = self._require_bound(self.game, "游戏")
        width = game.client_rect.width
        height = game.client_rect.height
        left_ratio, top_ratio, right_ratio, bottom_ratio = FAST_LOADOUT_PRICE_RECT
        rect = Rect(
            round(width * left_ratio),
            round(height * top_ratio),
            max(1, round(width * (right_ratio - left_ratio))),
            max(1, round(height * (bottom_ratio - top_ratio))),
        )
        frame = self.capture.capture_region(game, rect)
        self._last_fast_loadout_price_crop = frame.image
        if not frame.healthy:
            return None
        return frame.image

    def _save_loadout_price_trace_crop(
        self,
        scheme_index: int,
        stage: str,
        crop: np.ndarray | None,
    ) -> Path | None:
        diagnostics = getattr(self, "diagnostics", None)
        if (
            diagnostics is None
            or not isinstance(crop, np.ndarray)
            or crop.size == 0
        ):
            return None
        try:
            return diagnostics.save_price_trace_frame(
                f"loadout_scheme_{scheme_index}_{stage}",
                crop,
            )
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return None

    def _capture_loadout_price_trace_now(
        self,
        scheme_index: int,
        stage: str,
    ) -> Path | None:
        try:
            crop = self._capture_fast_loadout_price_crop()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return None
        return self._save_loadout_price_trace_crop(scheme_index, stage, crop)

    def _capture_loadout_price_trace_window(self, scheme_index: int) -> None:
        """Capture the price region from click time through 0.2 seconds after."""

        started = self._monotonic()
        deadline = started + LOADOUT_PRICE_TRACE_POST_CLICK_SECONDS
        index = 0
        max_samples = 1 + round(
            LOADOUT_PRICE_TRACE_POST_CLICK_SECONDS
            / LOADOUT_PRICE_TRACE_INTERVAL_SECONDS
        )
        while index < max_samples:
            self._capture_loadout_price_trace_now(
                scheme_index,
                f"post_click_{index:02d}",
            )
            index += 1
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                return
            self._sleep(min(LOADOUT_PRICE_TRACE_INTERVAL_SECONDS, remaining))

    def _buy_fast_loadout_offer(
        self,
        settings: LoadoutPurchaseSettings,
        rule: LoadoutSchemeRule,
        price: int,
        balance_before: int,
    ) -> tuple[LoadoutPurchaseSelection | None, int]:
        self._begin_loadout_purchase_action()
        self._state = WorkflowState.LOADOUT_PURCHASE_BUY
        game = self._require_bound(self.game, "游戏")
        self._save_loadout_checkpoint(
            LoadoutCheckpointStage.PURCHASE_CONFIRM_PENDING,
            scheme_index=rule.scheme_index,
            ammo_name=rule.ammunition_name,
            purchase_balance_before=balance_before,
            purchase_listed_price=price,
            purchase_was_partial=False,
            last_confirmed_page=PageType.LOADOUT_SCHEMES.value,
            last_safe_action="极速 OCR 已确认低价，最终购买点击前已锁定方案",
        )
        self._begin_pause_deferral()
        point = self.settings.action_points["use_scheme"]
        self._save_loadout_price_trace_crop(
            rule.scheme_index,
            "before_click",
            getattr(self, "_last_fast_loadout_price_crop", None),
        )
        self.input.click_normalized(game, point)
        self._capture_loadout_price_trace_window(rule.scheme_index)
        self._emit_structured(
            "loadout_low_price",
            f"极速配装方案 {rule.scheme_index} 刷出低价并已点击购买：整套 "
            f"{price:,}，目标 {rule.total_quantity:,} 发",
            scheme_index=rule.scheme_index,
            quantity=rule.total_quantity,
            total_price=price,
            displayed_unit_price=price / rule.total_quantity,
            max_total_price=rule.max_total_price,
            max_unit_price=rule.max_unit_price,
        )
        self.diagnostics.event(
            "input",
            action="click",
            label=f"极速购买配装方案 {rule.scheme_index}：{price:,}",
            point=point,
            hwnd=game.hwnd,
        )
        self._sleep(self.settings.action_settle_seconds)
        attempt = self._purchase_selected_loadout_scheme(
            rule,
            price,
            perform_click=False,
        )
        if not attempt:
            balance_label = (
                "价格变动后余额"
                if attempt.price_changed
                else "购买未确认后余额"
            )
            balance_after_attempt, balance_page = (
                self._read_loadout_balance_after_unconfirmed_purchase(
                    balance_label,
                    attempt.final_page,
                )
            )
            spent = balance_before - balance_after_attempt
            if spent > 0:
                self._save_loadout_checkpoint(
                    LoadoutCheckpointStage.PURCHASE_CONFIRMED,
                    scheme_index=rule.scheme_index,
                    ammo_name=rule.ammunition_name,
                    purchase_balance_before=balance_before,
                    purchase_listed_price=price,
                    purchase_was_partial=True,
                    last_confirmed_page=(
                        balance_page.page_type.value
                        if balance_page is not None
                        else None
                    ),
                    last_safe_action="余额减少，确认极速配装至少部分购买成交",
                )
                self._emit_loadout_purchase_result(
                    rule,
                    price,
                    "partial",
                    balance_before,
                    balance_after_attempt,
                    spent=spent,
                )
                self._prepare_partial_loadout_for_mail(balance_page)
                self._end_pause_deferral()
                return (
                    LoadoutPurchaseSelection(
                        rule=rule,
                        listed_price=price,
                        actual_average=None,
                        partial_spent=spent,
                    ),
                    balance_after_attempt,
                )
            self._emit_loadout_purchase_result(
                rule,
                price,
                "none",
                balance_before,
                balance_after_attempt,
                spent=spent,
            )
            self._save_loadout_checkpoint(
                LoadoutCheckpointStage.PURCHASE_SCAN,
                scheme_index=None,
                ammo_name="",
                purchase_balance_before=None,
                purchase_listed_price=None,
                purchase_was_partial=False,
                last_confirmed_page=(
                    balance_page.page_type.value
                    if balance_page is not None
                    else None
                ),
                last_safe_action="余额未减少，确认极速配装本次没有购买成交",
            )
            self._ensure_loadout_scheme_list(balance_page)
            self._transition(
                WorkflowState.LOADOUT_PURCHASE_SCAN,
                f"极速配装方案 {rule.scheme_index} 未成交，解除购买锁并继续",
                "warning",
            )
            self._loadout_round_in_flight = False
            self._end_pause_deferral()
            return None, balance_after_attempt

        balance_after, balance_page = self._read_loadout_balance_with_page_recovery(
            "极速配装买入后",
            attempt.final_page,
        )
        actual_average = self._calculate_loadout_average(
            balance_before,
            balance_after,
            rule.total_quantity,
        )
        spent = balance_before - balance_after
        self._save_loadout_checkpoint(
            LoadoutCheckpointStage.PURCHASE_CONFIRMED,
            scheme_index=rule.scheme_index,
            ammo_name=rule.ammunition_name,
            purchase_balance_before=balance_before,
            purchase_listed_price=price,
            purchase_was_partial=False,
            last_confirmed_page=balance_page.page_type.value,
            last_safe_action="极速配装购买后余额减少并确认完整成交",
        )
        self._emit_loadout_purchase_result(
            rule,
            price,
            "full",
            balance_before,
            balance_after,
            spent=spent,
            actual_average=actual_average,
        )
        self._apply_blank_loadout_scheme(
            settings.blank_scheme_index,
            schemes_open=balance_page.page_type in LOADOUT_SCHEME_PAGES,
        )
        self._save_loadout_checkpoint(
            LoadoutCheckpointStage.BLANK_APPLIED,
            last_confirmed_page=PageType.GAME_HOME_READY.value,
            last_safe_action=f"已应用空白方案 {settings.blank_scheme_index}",
        )
        self._end_pause_deferral()
        return (
            LoadoutPurchaseSelection(
                rule=rule,
                listed_price=price,
                actual_average=actual_average,
            ),
            balance_after,
        )

    @staticmethod
    def _loadout_scan_order(settings: LoadoutPurchaseSettings) -> tuple[int, ...]:
        enabled = tuple(rule.scheme_index for rule in settings.enabled_schemes)
        if len(enabled) == 1:
            return (settings.blank_scheme_index, enabled[0])
        return enabled

    def _switch_loadout_scheme_for_scan(
        self,
        scheme_index: int,
        settle_seconds: float,
    ) -> _LoadoutSchemeRefresh:
        game = self._require_bound(self.game, "游戏")
        point_name = f"loadout_scheme_{scheme_index}"
        point = self.settings.action_points[point_name]
        self.input.click_normalized(game, point)
        initial_settle_seconds = LOADOUT_SCHEME_INITIAL_SETTLE_SECONDS
        if getattr(self, "_loadout_purchase_fast_mode", False):
            initial_settle_seconds = (
                self._timing_settings().fast_loadout_click_settle_ms / 1000.0
            )
        confirmation_timeout = max(initial_settle_seconds, settle_seconds)
        started_at = self._monotonic()
        deadline = started_at + confirmation_timeout
        self._sleep(initial_settle_seconds)
        frame_size = (game.client_rect.width, game.client_rect.height)
        panel_rect = self._loadout_scheme_panel_rect(*frame_size)
        combined_rect = self._loadout_scheme_price_rect(*frame_size)
        attempts = 0
        last_selected_index: int | None = None
        while True:
            self._check_stop()
            try:
                # Poll only the scheme-card panel while the UI animation settles.
                # After a local hit, one combined left-panel/price frame provides
                # the same-frame selection check and recognition-only price OCR.
                panel_frame = self.capture.capture_region(game, panel_rect)
            except Exception as exc:
                reason = f"方案高亮局部截图失败：{exc}"
                self.diagnostics.event(
                    "loadout_scheme_highlight_confirm",
                    scheme_index=scheme_index,
                    attempts=attempts,
                    result="error",
                    reason=reason,
                )
                return _LoadoutSchemeRefresh("error", reason=reason)
            attempts += 1
            last_selected_index = self._detect_selected_loadout_scheme_region(
                panel_frame.image,
                panel_rect,
                (game.client_rect.width, game.client_rect.height),
            )
            if last_selected_index == scheme_index:
                try:
                    combined_frame = self.capture.capture_region(
                        game,
                        combined_rect,
                    )
                except Exception as exc:
                    reason = f"方案与价格联合区域截图失败：{exc}"
                    self.diagnostics.event(
                        "loadout_scheme_highlight_confirm",
                        scheme_index=scheme_index,
                        attempts=attempts,
                        result="error",
                        reason=reason,
                    )
                    return _LoadoutSchemeRefresh("error", reason=reason)
                last_selected_index = self._detect_selected_loadout_scheme_from_region(
                    combined_frame.image,
                    combined_rect,
                    frame_size,
                )
                if last_selected_index != scheme_index:
                    remaining = deadline - self._monotonic()
                    if remaining <= 0:
                        break
                    self._sleep(
                        min(LOADOUT_SCHEME_RETRY_INTERVAL_SECONDS, remaining)
                    )
                    continue
                return _LoadoutSchemeRefresh(
                    "confirmed",
                    selected_frame=combined_frame,
                    selected_region=combined_rect,
                    frame_size=frame_size,
                )
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                break
            self._sleep(
                min(LOADOUT_SCHEME_RETRY_INTERVAL_SECONDS, remaining)
            )
        actual = (
            str(last_selected_index)
            if last_selected_index is not None
            else "未识别或存在双高亮"
        )
        reason = (
            f"{confirmation_timeout:.1f} 秒内未确认目标方案唯一强高亮；"
            f"最后高亮 {actual}"
        )
        self.diagnostics.event(
            "loadout_scheme_highlight_confirm",
            scheme_index=scheme_index,
            attempts=attempts,
            result="unconfirmed",
            reason=reason,
        )
        return _LoadoutSchemeRefresh("unconfirmed", reason=reason)

    def _confirm_loadout_scheme_with_full_observation(
        self,
        game: WindowInfo,
        scheme_index: int,
        deadline: float,
    ) -> tuple[CapturedFrame, PageObservation, int | None]:
        while True:
            frame, observation = self._observe(
                game,
                "game",
                include_ocr=False,
                template_names=LOADOUT_SCAN_TEMPLATE_NAMES,
            )
            selected_index = (
                self._detect_selected_loadout_scheme(frame.image)
                if observation.page_type in LOADOUT_SCHEME_PAGES
                else None
            )
            if (
                selected_index == scheme_index
                and observation.page_type in LOADOUT_SCHEME_PAGES
            ) or observation.page_type in {
                PageType.LOADOUT_PURCHASE_CONFIRM,
                PageType.LOADOUT_PRICE_CHANGE,
            }:
                return frame, observation, selected_index
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                return frame, observation, selected_index
            self._sleep(
                min(
                    LOADOUT_SCHEME_FALLBACK_RETRY_INTERVAL_SECONDS,
                    remaining,
                )
            )

    def _recover_loadout_scan_state(
        self,
        scheme_index: int,
        settle_seconds: float,
        reason: str,
    ) -> None:
        self._emit(
            f"配装方案 {scheme_index} 局部刷新状态不确定（{reason}）；"
            "执行一次完整页面恢复后继续刷新",
            "warning",
        )
        game = self._require_bound(self.game, "游戏")
        deadline = self._monotonic() + max(
            LOADOUT_SCHEME_INITIAL_SETTLE_SECONDS,
            settle_seconds,
        )
        _frame, observation, selected_index = (
            self._confirm_loadout_scheme_with_full_observation(
                game,
                scheme_index,
                deadline,
            )
        )
        if observation.page_type in {
            PageType.LOADOUT_PURCHASE_CONFIRM,
            PageType.LOADOUT_PRICE_CHANGE,
        }:
            recovered = self._cancel_loadout_purchase_page(
                observation.page_type,
                scheme_index=scheme_index,
            )
            self._ensure_loadout_scheme_list(recovered)
            return
        if observation.page_type in LOADOUT_SCHEME_PAGES:
            actual = (
                str(selected_index)
                if selected_index is not None
                else "未识别"
            )
            self._emit(
                f"完整页面恢复已确认配装方案列表；当前高亮 {actual}，继续刷新"
            )
            return
        if observation.page_type == PageType.LOADOUT:
            self._ensure_loadout_scheme_list(observation)
            return
        self._ensure_loadout_scheme_list(None)

    def _open_zero_dam_loadout(self) -> PageObservation:
        game = self._require_bound(self.game, "游戏")
        self._click(game, "map_card", "打开端游对局地图")
        map_detail_pages = {
            PageType.MAP_OVERVIEW,
            PageType.MAP_OTHER,
            PageType.MAP_LONGBOW_START,
            PageType.MAP_LONGBOW_DOWNLOAD,
        }
        expected_pages = {
            PageType.MAP_SELECTION,
            PageType.MAP_ZERO_DAM_START,
            *map_detail_pages,
        }
        try:
            current = self._wait_page(
                self.game,
                expected_pages,
                self._timing_settings().map_fast_path_timeout_seconds,
                "配装买入地图入口",
                save_timeout=False,
            )
        except WorkflowTimeout:
            self._emit(
                "地图入口快速识别未通过；转入统一页面导航恢复",
                "warning",
            )
            result = self._unified_navigation.navigate(
                GamePageId.MAP_SELECTION,
                acceptable_pages=frozenset({GamePageId.MAP_ZERO_DAM_START}),
                reason="配装买入法恢复到零号大坝入口",
            )
            if not result.succeeded:
                raise RuntimeError(result.message)
            current = self._last_unified_mail_observation
            if current is None or current.page_type not in {
                PageType.MAP_SELECTION,
                PageType.MAP_ZERO_DAM_START,
            }:
                current = self._wait_page(
                    self.game,
                    {PageType.MAP_SELECTION, PageType.MAP_ZERO_DAM_START},
                    self._timing_settings().map_page_timeout_seconds,
                    "配装买入地图恢复结果",
                    save_timeout=False,
                )

        return self._enter_zero_dam_loadout(current)

    def _enter_zero_dam_loadout(
        self,
        current: PageObservation | None = None,
    ) -> PageObservation:
        map_detail_pages = {
            PageType.MAP_OVERVIEW,
            PageType.MAP_OTHER,
            PageType.MAP_LONGBOW_START,
            PageType.MAP_LONGBOW_DOWNLOAD,
        }
        recoverable_pages = {
            PageType.MAP_SELECTION,
            PageType.MAP_ZERO_DAM_START,
            PageType.LOADOUT,
            PageType.GENERIC_CONFIRM,
            PageType.GAME_HOME_READY,
            PageType.GAME_HOME_PREPARE,
            *map_detail_pages,
        }
        action_counts: dict[PageType, int] = {}
        unknown_recovery_actions = ("esc", "space")
        unknown_recovery_index = 0
        for _step in range(12):
            if current is None:
                try:
                    current = self._wait_page(
                        self.game,
                        recoverable_pages,
                        self._timing_settings().map_fast_path_timeout_seconds,
                        "零号大坝配装导航状态",
                        include_ocr=True,
                    )
                except WorkflowTimeout:
                    self._emit(
                        "零号大坝配装页面尚未识别；继续截图识别等待，暂不发送点击或按键",
                        "warning",
                    )
                    current = self._wait_page(
                        self.game,
                        recoverable_pages,
                        self._timing_settings().map_page_timeout_seconds,
                        "零号大坝配装导航状态恢复",
                        include_ocr=True,
                    )
            page_type = current.page_type
            if page_type == PageType.LOADOUT:
                return current

            action_counts[page_type] = action_counts.get(page_type, 0) + 1
            if action_counts[page_type] > 2:
                raise WorkflowTimeout(
                    f"零号大坝配装恢复连续停留在{page_type_label(page_type)}；"
                    "已停止重复操作"
                )

            game = self._require_bound(self.game, "游戏")
            if page_type in map_detail_pages:
                self._click(game, "map_expand", "返回地图全览")
            elif page_type == PageType.MAP_SELECTION:
                self._click_match(
                    game,
                    current,
                    ("real_zero_dam_node",),
                    "zero_dam_node",
                    "选择零号大坝",
                    require_match=True,
                )
            elif page_type == PageType.MAP_ZERO_DAM_START:
                self._click_match(
                    game,
                    current,
                    ("real_zero_dam_start",),
                    "start_action",
                    "零号大坝开始行动",
                    require_match=True,
                )
            elif page_type == PageType.GAME_HOME_PREPARE:
                self._click(game, "map_card", "打开地图")
            elif page_type == PageType.GAME_HOME_READY:
                self._click(game, "open_loadout", "打开配装")
            elif page_type == PageType.GENERIC_CONFIRM:
                self._click_match(
                    game,
                    current,
                    ("real_generic_confirm_button",),
                    "generic_confirm",
                    "系统提示确认",
                )
            else:
                raise RuntimeError(
                    f"无法从{page_type_label(page_type)}恢复到零号大坝配装"
                )
            current = None

        raise WorkflowTimeout("零号大坝配装恢复超过最大步骤数")

    def _read_loadout_scheme_offer(
        self,
        scheme_index: int,
        frame: CapturedFrame,
        observation: PageObservation | None,
        *,
        frame_region: Rect | None = None,
        frame_size: tuple[int, int] | None = None,
    ) -> LoadoutSchemeOffer:
        if frame_region is None:
            height, width = frame.image.shape[:2]
            frame_region = Rect(0, 0, width, height)
            frame_size = (width, height)
        if frame_size is None:
            raise ValueError("联合截图缺少完整游戏窗口尺寸")
        selection_confirmed = (
            self._detect_selected_loadout_scheme_from_region(
                frame.image,
                frame_region,
                frame_size,
            )
            == scheme_index
        )
        if not selection_confirmed:
            return LoadoutSchemeOffer(
                price=None,
                usable=False,
                partially_sold_out=False,
                selection_confirmed=False,
                observation=observation,
            )

        price = self._read_loadout_price_crop(
            scheme_index,
            self._loadout_price_ocr_crop_from_region(
                frame.image,
                frame_region,
                frame_size,
            ),
        )
        return LoadoutSchemeOffer(
            price=price,
            usable=False,
            partially_sold_out=False,
            selection_confirmed=True,
            observation=observation,
        )

    def _read_loadout_price_crop(
        self,
        scheme_index: int,
        price_crop: np.ndarray,
    ) -> int | None:
        self._last_fast_loadout_price_crop = price_crop
        if price_crop.size == 0:
            self._emit(
                f"配装方案 {scheme_index} 联合截图未完整覆盖价格区域；"
                "本次价格作废",
                "warning",
            )
            return None
        recognized = self._recognize_fast_loadout_price_crop(
            price_crop,
            expected_quantity=self._loadout_scheme_quantity(scheme_index),
        )
        if recognized is None:
            self._emit(
                f"配装方案 {scheme_index} 局部 OCR（文字模型）未通过金额校验；"
                "本次价格作废",
                "warning",
            )
            return None
        price, _text, _confidence = recognized
        return price

    def _detect_selected_loadout_scheme(self, image: np.ndarray) -> int | None:
        height, width = image.shape[:2]
        if height < 300 or width < 500:
            return None
        panel_rect = self._loadout_scheme_panel_rect(width, height)
        panel = image[
            panel_rect.top : panel_rect.bottom,
            panel_rect.left : panel_rect.right,
        ]
        return self._detect_selected_loadout_scheme_region(
            panel,
            panel_rect,
            (width, height),
        )

    def _detect_selected_loadout_scheme_from_region(
        self,
        image: np.ndarray,
        region: Rect,
        frame_size: tuple[int, int],
    ) -> int | None:
        panel = self._loadout_scheme_panel_rect(*frame_size)
        left = panel.left - region.left
        top = panel.top - region.top
        right = panel.right - region.left
        bottom = panel.bottom - region.top
        if (
            left < 0
            or top < 0
            or right > image.shape[1]
            or bottom > image.shape[0]
        ):
            return None
        return self._detect_selected_loadout_scheme_region(
            image[top:bottom, left:right],
            panel,
            frame_size,
        )

    def _loadout_scheme_panel_rect(self, width: int, height: int) -> Rect:
        search_radius = max(3, round(height * 0.004))
        centers = [
            self.settings.action_points[f"loadout_scheme_{index}"]
            for index in range(1, 6)
        ]
        left = max(
            0,
            min(round((center_x - 0.058) * width) for center_x, _ in centers)
            - search_radius,
        )
        right = min(
            width,
            max(round((center_x + 0.058) * width) for center_x, _ in centers)
            + search_radius
            + 1,
        )
        top = max(
            0,
            min(round((center_y - 0.028) * height) for _, center_y in centers)
            - search_radius,
        )
        bottom = min(
            height,
            max(round((center_y + 0.028) * height) for _, center_y in centers)
            + search_radius
            + 1,
        )
        return Rect(left, top, right - left, bottom - top)

    def _loadout_scheme_price_rect(self, width: int, height: int) -> Rect:
        panel = self._loadout_scheme_panel_rect(width, height)
        price = self._loadout_price_ocr_rect(width, height)
        left = min(panel.left, price.left)
        top = min(panel.top, price.top)
        right = max(panel.right, price.right)
        bottom = max(panel.bottom, price.bottom)
        return Rect(left, top, right - left, bottom - top)

    def _detect_selected_loadout_scheme_region(
        self,
        image: np.ndarray,
        region: Rect,
        frame_size: tuple[int, int],
    ) -> int | None:
        frame_width, frame_height = frame_size
        if image.size == 0 or frame_height < 300 or frame_width < 500:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scores: list[tuple[float, int]] = []
        for scheme_index in range(1, 6):
            center_x, center_y = self.settings.action_points[
                f"loadout_scheme_{scheme_index}"
            ]
            left = round((center_x - 0.058) * frame_width) - region.left
            right = round((center_x + 0.058) * frame_width) - region.left
            top = round((center_y - 0.028) * frame_height) - region.top
            bottom = round((center_y + 0.028) * frame_height) - region.top
            search_radius = max(3, round(frame_height * 0.004))
            vertical_top = max(0, top - search_radius)
            vertical_bottom = min(gray.shape[0], bottom + search_radius + 1)
            horizontal_left = max(0, left - search_radius)
            horizontal_right = min(gray.shape[1], right + search_radius + 1)

            left_continuity = self._best_vertical_border_continuity(
                gray,
                left,
                vertical_top,
                vertical_bottom,
                search_radius,
            )
            right_continuity = self._best_vertical_border_continuity(
                gray,
                right,
                vertical_top,
                vertical_bottom,
                search_radius,
            )
            bottom_continuity = self._best_horizontal_border_continuity(
                gray,
                bottom,
                horizontal_left,
                horizontal_right,
                search_radius,
            )
            scores.append(
                (
                    min(
                        left_continuity,
                        right_continuity,
                        bottom_continuity,
                    ),
                    scheme_index,
                )
            )
        scores.sort(reverse=True)
        best_score, best_index = scores[0]
        runner_up = scores[1][0]
        if (
            best_score < LOADOUT_SCHEME_BORDER_MIN_CONTINUITY
            or best_score - runner_up < 0.20
        ):
            return None
        return best_index

    @staticmethod
    def _longest_true_run(values: np.ndarray) -> int:
        padded = np.pad(values.astype(np.int8), (1, 1))
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        if starts.size == 0:
            return 0
        return int(np.max(ends - starts))

    @classmethod
    def _best_vertical_border_continuity(
        cls,
        gray: np.ndarray,
        expected_x: int,
        top: int,
        bottom: int,
        search_radius: int,
    ) -> float:
        if bottom <= top:
            return 0.0
        best = 0
        for x in range(
            max(0, expected_x - search_radius),
            min(gray.shape[1], expected_x + search_radius + 1),
        ):
            bright = gray[top:bottom, x] > LOADOUT_SCHEME_BORDER_BRIGHTNESS
            best = max(best, cls._longest_true_run(bright))
        return best / (bottom - top)

    @classmethod
    def _best_horizontal_border_continuity(
        cls,
        gray: np.ndarray,
        expected_y: int,
        left: int,
        right: int,
        search_radius: int,
    ) -> float:
        if right <= left:
            return 0.0
        best = 0
        for y in range(
            max(0, expected_y - search_radius),
            min(gray.shape[0], expected_y + search_radius + 1),
        ):
            bright = gray[y, left:right] > LOADOUT_SCHEME_BORDER_BRIGHTNESS
            best = max(best, cls._longest_true_run(bright))
        return best / (right - left)

    @staticmethod
    def _loadout_price_ocr_rect(width: int, height: int) -> Rect:
        left_ratio, top_ratio, right_ratio, bottom_ratio = LOADOUT_PRICE_OCR_RECT
        left = round(width * left_ratio)
        top = round(height * top_ratio)
        right = round(width * right_ratio)
        bottom = round(height * bottom_ratio)
        return Rect(left, top, right - left, bottom - top)

    @staticmethod
    def _fast_loadout_price_crop_from_image(image: np.ndarray) -> np.ndarray | None:
        height, width = image.shape[:2]
        if height <= 0 or width <= 0:
            return None
        left_ratio, top_ratio, right_ratio, bottom_ratio = FAST_LOADOUT_PRICE_RECT
        left = round(width * left_ratio)
        top = round(height * top_ratio)
        right = round(width * right_ratio)
        bottom = round(height * bottom_ratio)
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            return None
        return crop

    @staticmethod
    def _scale_fast_loadout_price_crop(
        crop: np.ndarray,
        scale: float = FAST_LOADOUT_PRICE_SCALE,
    ) -> np.ndarray:
        return cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

    @staticmethod
    def _loadout_price_ocr_crop(image: np.ndarray) -> np.ndarray:
        crop = WorkflowController._fast_loadout_price_crop_from_image(image)
        return crop if crop is not None else image[:0, :0]

    @staticmethod
    def _loadout_price_ocr_crop_from_region(
        image: np.ndarray,
        region: Rect,
        frame_size: tuple[int, int],
    ) -> np.ndarray:
        frame_width, frame_height = frame_size
        price = WorkflowController._loadout_price_ocr_rect(
            frame_width,
            frame_height,
        )
        left = price.left - region.left
        top = price.top - region.top
        right = price.right - region.left
        bottom = price.bottom - region.top
        if (
            left < 0
            or top < 0
            or right > image.shape[1]
            or bottom > image.shape[0]
        ):
            return image[:0, :0]
        return image[top:bottom, left:right]

    def _purchase_selected_loadout_scheme(
        self,
        rule: LoadoutSchemeRule,
        listed_price: int,
        *,
        perform_click: bool = True,
    ) -> LoadoutPurchaseAttempt:
        game = self._require_bound(self.game, "游戏")
        result_timeout = self._timing_settings().purchase_result_timeout_seconds

        if perform_click:
            self._save_loadout_price_trace_crop(
                rule.scheme_index,
                "before_click",
                getattr(self, "_last_fast_loadout_price_crop", None),
            )
            self._click(
                game,
                "use_scheme",
                f"购买配装方案 {rule.scheme_index}：{listed_price:,}",
                after_click=lambda: self._capture_loadout_price_trace_window(
                    rule.scheme_index,
                ),
            )
        try:
            page = self._wait_page(
                self.game,
                {
                    PageType.LOADOUT,
                    PageType.GENERIC_CONFIRM,
                    PageType.LOADOUT_PURCHASE_CONFIRM,
                    PageType.LOADOUT_PRICE_CHANGE,
                },
                result_timeout,
                "配装方案购买结果",
                include_ocr=True,
                save_timeout=False,
            )
        except WorkflowTimeout:
            self._emit(
                f"配装方案 {rule.scheme_index} 点击金额后未进入配装页，"
                "本次未确认购买",
                "warning",
            )
            return LoadoutPurchaseAttempt(False)
        if page.page_type in {
            PageType.LOADOUT_PURCHASE_CONFIRM,
            PageType.LOADOUT_PRICE_CHANGE,
        }:
            price_changed = page.page_type == PageType.LOADOUT_PRICE_CHANGE
            if price_changed:
                path = self._save_current_game_frame(
                    f"价格变动提醒_方案{rule.scheme_index}"
                )
                self._emit(f"已保存价格变动提醒截图：{path}", "warning")
            final_page = self._cancel_loadout_purchase_page(
                page.page_type,
                scheme_index=rule.scheme_index,
            )
            return LoadoutPurchaseAttempt(
                False,
                price_changed=price_changed,
                final_page=final_page,
            )
        if page.page_type == PageType.GENERIC_CONFIRM:
            game = self._require_bound(self.game, "游戏")
            self._click_match(
                game,
                page,
                ("real_generic_confirm_button",),
                "generic_confirm",
                "购买提示确认",
            )
            try:
                page = self._wait_page(
                    self.game,
                    {PageType.LOADOUT},
                    result_timeout,
                    "购买提示处理结果",
                    include_ocr=True,
                    save_timeout=False,
                )
            except WorkflowTimeout:
                self._emit(
                    f"配装方案 {rule.scheme_index} 的购买提示已确认，"
                    "但未进入配装页；继续轮询",
                    "warning",
                )
                return LoadoutPurchaseAttempt(False)
        if page.page_type == PageType.LOADOUT:
            return LoadoutPurchaseAttempt(True, final_page=page)
        self._emit(
            f"配装方案 {rule.scheme_index} 点击金额后仍停留在方案页，"
            "本次未确认购买",
            "warning",
        )
        return LoadoutPurchaseAttempt(False, final_page=page)

    def _cancel_loadout_purchase_page(
        self,
        page_type: PageType,
        *,
        scheme_index: int | None = None,
    ) -> PageObservation | None:
        page_name = page_type_label(page_type)
        scheme = f"配装方案 {scheme_index} " if scheme_index is not None else ""
        reason = (
            "价格可能已变化或商品已售罄"
            if page_type == PageType.LOADOUT_PRICE_CHANGE
            else "需要二次确认实现方案"
        )
        self._emit(
            f"{scheme}出现{page_name}（{reason}）；"
            "按 Esc 取消本次购买并继续刷新",
            "warning",
        )
        current_page = page_type
        purchase_modals = {
            PageType.LOADOUT_PURCHASE_CONFIRM,
            PageType.LOADOUT_PRICE_CHANGE,
        }
        expected_pages = LOADOUT_SCHEME_PAGES | purchase_modals | {PageType.LOADOUT}
        for attempt in range(3):
            current_name = page_type_label(current_page)
            game = self._require_bound(self.game, "游戏")
            self._press_key(game, "escape", f"Esc：关闭{current_name}")
            try:
                result = self._wait_page(
                    self.game,
                    set(expected_pages),
                    5,
                    f"关闭{current_name}后的页面",
                    include_ocr=True,
                    save_timeout=False,
                )
            except WorkflowTimeout:
                break
            if result.page_type in LOADOUT_SCHEME_PAGES | {PageType.LOADOUT}:
                if page_type == PageType.LOADOUT_PRICE_CHANGE:
                    path = self._save_current_game_frame(
                        f"价格变动退出后_{result.page_type.value}"
                    )
                    self._emit(
                        f"价格变动提醒退出后已定位到"
                        f"{page_type_label(result.page_type)}；截图：{path}"
                    )
                return result
            current_page = result.page_type
            if (
                page_type == PageType.LOADOUT_PRICE_CHANGE
                and attempt == 0
                and current_page == PageType.LOADOUT_PURCHASE_CONFIRM
            ):
                self._emit(
                    "价格变动提醒下方仍有确认实现方案页面；继续按 Esc 返回方案列表",
                    "warning",
                )
            elif attempt < 2:
                self._emit(
                    f"{page_type_label(current_page)}尚未关闭；重试一次 Esc",
                    "warning",
                )
        self._emit(
            f"{page_name}已尝试按 Esc 关闭，但尚未确认返回配装方案列表；"
            "禁止继续点击方案",
            "warning",
        )
        return None

    def _ensure_loadout_scheme_list(
        self,
        current: PageObservation | None,
    ) -> PageObservation:
        if current is None:
            current = self._wait_page(
                self.game,
                LOADOUT_SCHEME_PAGES | {PageType.LOADOUT},
                8,
                "配装买入异常恢复页面",
                include_ocr=True,
                save_timeout=False,
            )
        if current.page_type in LOADOUT_SCHEME_PAGES:
            return current
        if current.page_type != PageType.LOADOUT:
            raise RuntimeError(
                f"购买异常后停留在{page_type_label(current.page_type)}；"
                "尚未安全返回配装方案列表，禁止继续点击"
            )
        game = self._require_bound(self.game, "游戏")
        self._press_key(game, "loadout_scheme", "L：购买异常后重新打开配装方案")
        result = self._wait_page(
            self.game,
            set(LOADOUT_SCHEME_PAGES),
            12,
            "购买异常后配装方案列表",
            include_ocr=True,
        )
        self._emit(
            f"购买异常恢复完成：已确认{page_type_label(result.page_type)}"
        )
        return result

    def _prepare_partial_loadout_for_mail(
        self,
        current: PageObservation | None,
    ) -> None:
        if current is None:
            raise RuntimeError(
                "价格变动后无法确认当前页面；已检测到余额减少，"
                "不能安全处理部分成交"
            )
        if current.page_type in LOADOUT_SCHEME_PAGES:
            game = self._require_bound(self.game, "游戏")
            self._press_key(game, "escape", "Esc：部分成交后返回配装界面")
            current = self._wait_page(
                self.game,
                {PageType.LOADOUT},
                12,
                "部分成交配装界面",
                include_ocr=True,
            )
        if current.page_type != PageType.LOADOUT:
            raise RuntimeError(
                f"检测到部分成交，但当前为{page_type_label(current.page_type)}；"
                "不能安全确认当前部分配装"
            )
        path = self._save_current_game_frame("部分成交_确认当前配装前")
        self._emit(f"已保存部分成交配装截图：{path}", "warning")
        game = self._require_bound(self.game, "游戏")
        self._press_key(game, "loadout_scheme", "L：打开部分成交配装方案")
        initial = self._wait_step_outcome(
            self.game,
            self._partial_save_outcomes(),
            45,
            "部分成交配装保存起点",
            include_ocr=True,
        )
        self._reconcile_partial_loadout_save(initial)
        self._emit(
            f"当前部分配装已保存到中转方案 "
            f"{PARTIAL_TRANSFER_SCHEME_INDEX}；切换方案 5 空装后准备后续处理",
            "warning",
        )
        self._apply_blank_loadout_scheme(5, schemes_open=True)

    def _partial_save_outcomes(self) -> tuple[_StepOutcome, ...]:
        scheme_pages = frozenset(LOADOUT_SCHEME_PAGES)
        return (
            _StepOutcome(
                "方案 4 已保存",
                pages=scheme_pages,
                templates=frozenset({"real_use_scheme_button"}),
                predicate=lambda frame, _observation: (
                    self._detect_selected_loadout_scheme(frame.image)
                    == PARTIAL_TRANSFER_SCHEME_INDEX
                ),
            ),
            _StepOutcome(
                "等待覆盖确认",
                pages=frozenset({PageType.LOADOUT_SAVE_OVERWRITE}),
                templates=frozenset({"real_save_scheme_overwrite_confirm"}),
            ),
            _StepOutcome(
                "方案 4 保存位置已选中",
                pages=frozenset({PageType.LOADOUT_SAVE_DIALOG}),
                templates=frozenset({"real_save_scheme_submit"}),
                predicate=lambda frame, _observation: (
                    self._fixed_card_border_selected(
                        frame.image,
                        PARTIAL_TRANSFER_SAVE_SLOT_RECT,
                    )
                ),
            ),
            _StepOutcome(
                "选择保存位置",
                pages=frozenset({PageType.LOADOUT_SAVE_DIALOG}),
                templates=frozenset(
                    {
                        "real_save_scheme_dialog_header",
                        "real_save_scheme_submit",
                    }
                ),
            ),
            _StepOutcome(
                "当前配装已选中",
                pages=scheme_pages,
                templates=frozenset({"real_save_scheme_button"}),
                predicate=lambda frame, _observation: (
                    self._fixed_card_border_selected(
                        frame.image,
                        CURRENT_LOADOUT_CARD_RECT,
                    )
                ),
            ),
            _StepOutcome("配装方案列表", pages=scheme_pages),
        )

    def _reconcile_partial_loadout_save(self, initial: _StepResult) -> PageObservation:
        """Advance from the actual save UI and never repeat a submitted save."""

        result = initial
        deadline = self._monotonic() + 60
        attempts = {"select_current": 0, "open_save": 0, "select_slot": 0}
        while True:
            if result.outcome == "方案 4 已保存":
                self._emit("已确认中转方案 4 高亮且可使用，配装保存完成")
                return result.observation

            game = self._require_bound(self.game, "游戏")
            if result.outcome == "等待覆盖确认":
                self._emit("检测到覆盖确认弹窗，进入异常恢复分支")
                self._click_match(
                    game,
                    result.observation,
                    ("real_save_scheme_overwrite_confirm",),
                    "save_scheme_overwrite_confirm",
                    "确认覆盖中转方案 4",
                    require_match=True,
                )
                return self._wait_step_outcome(
                    self.game,
                    (self._partial_save_outcomes()[0],),
                    min(30, max(1.0, deadline - self._monotonic())),
                    "方案 4 覆盖后的保存结果",
                    include_ocr=True,
                ).observation

            if result.outcome == "方案 4 保存位置已选中":
                self._click_match(
                    game,
                    result.observation,
                    ("real_save_scheme_submit",),
                    "save_scheme_submit",
                    "保存到中转方案 4",
                    require_match=True,
                )
                submitted = self._wait_step_outcome(
                    self.game,
                    (
                        self._partial_save_outcomes()[0],
                        self._partial_save_outcomes()[1],
                    ),
                    min(20, max(1.0, deadline - self._monotonic())),
                    "方案 4 保存提交结果",
                    include_ocr=True,
                )
                if submitted.outcome == "方案 4 已保存":
                    self._emit("方案 4 原为空位，已直接保存，无需覆盖确认")
                    return submitted.observation
                result = submitted
                continue

            action_key: str
            if result.outcome == "选择保存位置":
                action_key = "select_slot"
                action = lambda: self._click(
                    game,
                    "save_scheme_slot_4",
                    "选择部分成交中转方案 4",
                )
            elif result.outcome == "当前配装已选中":
                action_key = "open_save"
                action = lambda: self._click_match(
                    game,
                    result.observation,
                    ("real_save_scheme_button",),
                    "save_scheme",
                    "保存当前部分配装",
                    require_match=True,
                )
            else:
                action_key = "select_current"
                action = lambda: self._click(
                    game,
                    "current_loadout",
                    "选择当前配装",
                )

            attempts[action_key] += 1
            if attempts[action_key] > 2:
                raise WorkflowTimeout(
                    f"部分成交配装保存停留在“{result.outcome}”；"
                    "可逆操作已重试一次，未继续重复点击"
                )
            if attempts[action_key] == 2:
                self._emit(
                    f"保存流程仍停留在“{result.outcome}”，重试一次可逆操作",
                    "warning",
                )
            action()
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                raise WorkflowTimeout("部分成交配装保存超过 60 秒恢复预算")
            result = self._wait_step_outcome(
                self.game,
                self._partial_save_outcomes(),
                remaining,
                "部分成交配装保存状态",
                include_ocr=True,
            )

    def _wait_fixed_card_selection(
        self,
        card_rect: tuple[int, int, int, int],
        required_templates: tuple[str, ...],
        timeout: float,
        label: str,
    ) -> PageObservation:
        started = self._monotonic()
        latest_frame: CapturedFrame | None = None
        latest_observation = PageObservation(PageType.UNKNOWN, 0.0)
        template_names = frozenset(required_templates)
        while True:
            self._check_stop()
            window = self.game.refresh()
            if window is not None:
                latest_frame, latest_observation = self._observe(
                    window,
                    "game",
                    include_ocr=False,
                    template_names=template_names,
                )
                templates_present = all(
                    name in latest_observation.matches for name in required_templates
                )
                selected = self._fixed_card_border_selected(
                    latest_frame.image,
                    card_rect,
                )
                if templates_present and selected:
                    return latest_observation
            if self._monotonic() - started >= timeout:
                if latest_frame is not None:
                    path = self.diagnostics.save_frame(
                        f"timeout_{label}",
                        latest_frame.image,
                    )
                    self._emit(f"已保存超时截图：{path}", "warning")
                raise WorkflowTimeout(f"等待{label}超时；未确认固定卡片白色选中边框")
            self._sleep(0.2)

    @staticmethod
    def _fixed_card_border_selected(
        image: np.ndarray,
        card_rect: tuple[int, int, int, int],
    ) -> bool:
        normalized = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
        left, top, right, bottom = card_rect
        thickness = 3
        border = np.concatenate(
            (
                gray[top : top + thickness, left:right].ravel(),
                gray[bottom - thickness : bottom, left:right].ravel(),
                gray[top:bottom, left : left + thickness].ravel(),
                gray[top:bottom, right - thickness : right].ravel(),
            )
        )
        return bool(
            np.mean(border > FIXED_CARD_BORDER_BRIGHTNESS)
            >= FIXED_CARD_BORDER_MIN_RATIO
        )

    def _save_current_game_frame(self, label: str) -> Path:
        game = self._require_bound(self.game, "游戏")
        frame = self.capture.capture(game)
        return self.diagnostics.save_frame(label, frame.image)

    def _read_loadout_balance(
        self,
        label: str,
        _page: PageObservation | None = None,
    ) -> int | None:
        game = self._require_bound(self.game, "游戏")
        point = self.settings.action_points["balance_hover"]
        self._emit(f"悬浮余额：{label} @ {point}")
        self.input.move_normalized(game, point)
        self._sleep(0.8)
        frame = self.capture.capture(game)
        regions = self.vision.ocr.detect(frame.image)
        height, width = frame.image.shape[:2]
        candidates: list[tuple[int, float, str]] = []
        for region in regions:
            left, top, right, bottom = region.bounds
            center_x = ((left + right) / 2) / width
            center_y = ((top + bottom) / 2) / height
            value = self._parse_balance_ocr_integer(region.text)
            if (
                value is not None
                and value >= 100_000
                and 0.72 <= center_x <= 0.98
                and 0.04 <= center_y <= 0.36
            ):
                candidates.append((value, region.confidence, region.text))
        path = self.diagnostics.save_frame(f"balance_{label}", frame.image)
        if not candidates:
            self._emit(f"{label}未识别到精确余额；截图：{path}", "warning")
            return None
        value, confidence, text = max(
            candidates,
            key=lambda item: (len(str(item[0])), item[1]),
        )
        self._emit(
            f"{label}余额 OCR：“{text}” -> {value:,} ({confidence:.0%})；截图：{path}"
        )
        return value

    def _read_loadout_balance_after_unconfirmed_purchase(
        self,
        label: str,
        current_page: PageObservation | None,
    ) -> tuple[int, PageObservation]:
        return self._read_loadout_balance_with_page_recovery(label, current_page)

    def _read_loadout_balance_with_page_recovery(
        self,
        label: str,
        current_page: PageObservation | None,
    ) -> tuple[int, PageObservation]:
        """Read an exact balance, alternating between safe loadout pages if needed."""

        safe_pages = LOADOUT_SCHEME_PAGES | {PageType.LOADOUT}
        page = (
            current_page
            if current_page is not None and current_page.page_type in safe_pages
            else None
        )
        attempt = 0
        failures_on_page = 0
        while True:
            self._check_stop()
            if page is None:
                try:
                    page = self._wait_page(
                        self.game,
                        set(safe_pages),
                        6,
                        f"{label}页面定位",
                        include_ocr=True,
                        save_timeout=False,
                    )
                    failures_on_page = 0
                    self._emit(
                        f"{label}已定位到{page_type_label(page.page_type)}；"
                        "继续悬浮读取余额"
                    )
                except WorkflowTimeout as exc:
                    last = exc.last_observation
                    if last is not None and last.page_type in safe_pages:
                        page = last
                        failures_on_page = 0
                        self._emit(
                            f"{label}已从最后一帧定位到"
                            f"{page_type_label(last.page_type)}；继续读取余额",
                            "warning",
                        )
                    elif last is not None and last.page_type in {
                        PageType.GENERIC_CONFIRM,
                        PageType.LOADOUT_PURCHASE_CONFIRM,
                        PageType.LOADOUT_PRICE_CHANGE,
                    }:
                        self._emit(
                            f"{label}检测到残留的"
                            f"{page_type_label(last.page_type)}；"
                            "先安全取消，再换到配装页面读取余额",
                            "warning",
                        )
                        page = self._cancel_loadout_purchase_page(last.page_type)
                        failures_on_page = 0
                        if page is None:
                            self._sleep(1.0)
                            continue
                    else:
                        self._emit(
                            f"{label}暂未定位到安全配装页面；继续等待，"
                            "不会点击购买",
                            "warning",
                        )
                        self._sleep(1.0)
                        continue

            attempt += 1
            attempt_label = label if attempt == 1 else f"{label}_重试{attempt}"
            try:
                balance = self._read_loadout_balance(attempt_label, page)
            except WorkflowStopped:
                raise
            except (OSError, RuntimeError) as exc:
                balance = None
                self._emit(
                    f"{attempt_label}读取发生临时异常：{exc}；"
                    "按余额识别失败继续恢复，不会点击购买",
                    "warning",
                )
            if balance is not None:
                if attempt > 1:
                    self._emit(
                        f"{label}经过 {attempt} 次尝试恢复识别：{balance:,}；"
                        "继续核对本次成交状态"
                    )
                return balance, page

            failures_on_page += 1
            if attempt == 1:
                self._emit(
                    f"{label}首次识别失败；保持当前页面并重新悬浮识别，"
                    "不会停止流程或再次点击购买",
                    "warning",
                )

            if failures_on_page >= 2:
                game = self._require_bound(self.game, "游戏")
                if page.page_type in LOADOUT_SCHEME_PAGES:
                    action = "escape"
                    action_label = "Esc：余额识别恢复返回配装界面"
                    target_pages = {PageType.LOADOUT}
                    target_label = "普通配装页"
                else:
                    action = "loadout_scheme"
                    action_label = "L：余额识别恢复打开配装方案"
                    target_pages = set(LOADOUT_SCHEME_PAGES)
                    target_label = "配装方案列表"
                self._emit(
                    f"{label}在{page_type_label(page.page_type)}连续识别失败；"
                    f"切换到{target_label}后继续读取余额",
                    "warning",
                )
                self._press_key(game, action, action_label)
                try:
                    page = self._wait_page(
                        self.game,
                        target_pages,
                        8,
                        f"{label}切换到{target_label}",
                        include_ocr=True,
                        save_timeout=False,
                    )
                except WorkflowTimeout:
                    page = None
                    self._emit(
                        f"{label}切换后尚未确认{target_label}；"
                        "继续定位安全页面并读取余额，不会点击购买",
                        "warning",
                    )
                failures_on_page = 0

            if attempt % 5 == 0:
                self._emit(
                    f"{label}已尝试 {attempt} 次仍未读到精确余额；"
                    "当前保持安全等待，不会再次购买；可按 Ctrl+Esc 停止",
                    "warning",
                )
            self._sleep(min(1.5, 0.3 + attempt * 0.1))

    @staticmethod
    def _calculate_loadout_average(
        balance_before: int | None,
        balance_after: int | None,
        quantity: int,
    ) -> int | None:
        if balance_before is None or balance_after is None or quantity <= 0:
            return None
        spent = balance_before - balance_after
        if spent <= 0:
            return None
        return round(spent / quantity)

    def _apply_blank_loadout_scheme(
        self,
        blank_scheme_index: int,
        *,
        schemes_open: bool = False,
    ) -> None:
        self._transition(
            WorkflowState.LOADOUT_PURCHASE_CLEAR,
            f"切换配装方案 {blank_scheme_index} 并确认空白配装",
        )
        self._apply_loadout_scheme_with_recovery(
            blank_scheme_index,
            f"空白配装方案 {blank_scheme_index}",
            schemes_open=schemes_open,
        )

    def _apply_loadout_scheme_with_recovery(
        self,
        scheme_index: int,
        label: str,
        *,
        schemes_open: bool,
    ) -> None:
        downstream = (
            _StepOutcome("确定配装页", pages=frozenset({PageType.LOADOUT})),
            _StepOutcome("入局提醒", pages=frozenset({PageType.ENTRY_WARNING})),
            _StepOutcome("配装已完成", pages=frozenset({PageType.GAME_HOME_READY})),
        )
        advanced_outcomes = {
            PageType.LOADOUT: "确定配装页",
            PageType.ENTRY_WARNING: "入局提醒",
            PageType.GAME_HOME_READY: "配装已完成",
        }
        selected: PageObservation
        if not schemes_open:
            game = self._require_bound(self.game, "游戏")
            self._press_key(game, "loadout_scheme", "L：购买完成后打开配装方案")
            opened = self._wait_step_outcome_with_delayed_recovery(
                self.game,
                (
                    _StepOutcome(
                        "配装方案列表",
                        pages=frozenset(LOADOUT_SCHEME_PAGES),
                    ),
                ),
                (),
                45,
                "配装方案列表",
                include_ocr=True,
            )
            if opened.observation.page_type in advanced_outcomes:
                selected = opened.observation
            else:
                selected = self._select_loadout_scheme_for_use(
                    scheme_index,
                    label,
                )
        else:
            selected = self._select_loadout_scheme_for_use(
                scheme_index,
                label,
            )
        if selected.page_type in advanced_outcomes:
            current = _StepResult(
                advanced_outcomes[selected.page_type],
                None,
                selected,
            )
            self._emit(
                f"{label}的“使用方案”步骤已被跳过或由用户完成，"
                f"从“{current.outcome}”继续",
                "warning",
            )
        else:
            game = self._require_bound(self.game, "游戏")
            self._click_match(
                game,
                selected,
                ("real_use_scheme_button",),
                "use_scheme",
                f"使用{label}",
                require_match=True,
            )
            current = self._wait_step_outcome_with_delayed_recovery(
                self.game,
                (downstream[0],),
                downstream[1:],
                45,
                f"{label}使用结果",
                include_ocr=True,
            )

        if current.outcome == "确定配装页":
            game = self._require_bound(self.game, "游戏")
            self._click(game, "confirm_equip", f"确认{label}")
            current = self._wait_step_outcome(
                self.game,
                downstream[1:],
                60,
                f"{label}确认结果",
                include_ocr=True,
            )

        if current.outcome == "入局提醒":
            game = self._require_bound(self.game, "游戏")
            self._click_match(
                game,
                current.observation,
                ("real_entry_warning_continue",),
                "entry_warning_continue",
                f"{label}仍要继续",
            )
            self._wait_step_outcome(
                self.game,
                (downstream[2],),
                60,
                f"{label}完成页",
                include_ocr=True,
            )

    @staticmethod
    def _parse_ocr_integer(text: str | None) -> int | None:
        return parse_ocr_integer(text)

    @staticmethod
    def _parse_balance_ocr_integer(text: str | None) -> int | None:
        return parse_ocr_integer(text)

    @staticmethod
    def _has_valid_ocr_thousands_grouping(text: str) -> bool:
        return has_valid_ocr_thousands_grouping(text)

    def _set_capture_backend(self, backend: str) -> None:
        if self.capture.preferred_backend == backend:
            return
        self.capture.set_backend(backend)
        self._emit(f"优先截图方式已设为 {backend.upper()}（{'卡邮件阶段' if backend == 'mss' else '配装刷新/交易阶段'}），失败时自动尝试备用方式，每 5 分钟探测恢复")

    def _preflight_mail_scheme(self) -> bool:
        """Inspect the target, then equip blank scheme 5 before starting a match.

        Loadout purchases call the single-cycle core directly and keep their
        existing post-purchase checks. A late purchase-required signal must
        still use the resource-cycle cleanup in _run_workflow.
        """
        self._transition(WorkflowState.PRECHECK, "卡邮件轮前检查：确认目标方案可直接使用")
        self._precheck()
        self._wait_window(self.game, "游戏")
        self._prepare_game_for_trading_handoff(allow_game_restart=False)
        loadout = self._open_zero_dam_loadout()
        if loadout.page_type != PageType.LOADOUT:
            raise RuntimeError("轮前检查未确认零号大坝配装页，禁止继续卡邮件")
        game = self._require_bound(self.game, "游戏")
        self._press_key(game, "loadout_scheme", "L：轮前检查目标配装方案")
        self._wait_step_outcome_with_delayed_recovery(
            self.game,
            (_StepOutcome("配装方案列表", pages=frozenset(LOADOUT_SCHEME_PAGES)),),
            (),
            45,
            "轮前检查配装方案列表",
            include_ocr=True,
        )
        usable = True
        try:
            self._select_loadout_scheme_for_use(
                self.settings.mail_scheme_index,
                f"轮前检查卡邮件方案 {self.settings.mail_scheme_index}",
            )
        except LoadoutSchemePurchaseRequired:
            usable = False
            self._emit("轮前检查发现目标方案需要购买，返回主界面并结束连续卡邮件")
        if usable:
            self._emit("目标卡邮件方案可直接使用；先应用空白配装方案 5，再开始本轮")
            try:
                self._apply_blank_loadout_scheme(5, schemes_open=True)
            except LoadoutSchemePurchaseRequired as exc:
                # No resource cycle has started. Do not let the outer late
                # purchase-required handler download/abandon a nonexistent match.
                raise RuntimeError(
                    "轮前空白配装方案 5 需要购买，禁止携带当前物资进入本轮；"
                    "请将方案 5 保存为空白配装"
                ) from exc
        # Never apply the target scheme here. Failed/uncertain blank application
        # also propagates instead of starting a resource cycle.
        self._prepare_game_for_trading_handoff(allow_game_restart=False)
        return usable

    def _run_workflow_cycle(
        self,
        round_number: int,
        *,
        mail_scheme_index: int | None = None,
    ) -> None:
        self._set_capture_backend("mss")
        selected_mail_scheme_index = (
            mail_scheme_index
            if mail_scheme_index is not None
            else self.settings.mail_scheme_index
        )
        self._emit(
            f"第 {round_number} 轮目标卡邮件配装方案："
            f"方案 {selected_mail_scheme_index}"
        )
        try:
            self._transition(WorkflowState.PRECHECK, "执行启动检查")
            self._precheck()

            self._transition(WorkflowState.WAIT_GAME, "等待并绑定游戏窗口")
            self._wait_window(self.game, "游戏")

            agent_page = self._run_mail_game_phase(
                "进入干员选择", self._prepare_mail_agent_selection,
            )
            if agent_page.page_type == PageType.AGENT_SELECT:
                close_message = "08：已进入干员选择界面，关闭游戏"
            else:
                close_message = (
                    "08：异常恢复确认已越过干员选择并进入游戏加载，关闭游戏"
                )
            self._transition(WorkflowState.CLOSE_GAME, close_message)
            game = self.game.refresh()
            termination: ProcessTreeTermination | None = None
            if game is None and not matching_process_running(self.game.fingerprint):
                self._emit("游戏已由用户关闭，跳过结束进程", "warning")
            else:
                termination = self._terminate_game_process_tree(
                    game,
                    reason="mail_agent_selection",
                    label="卡邮件关闭游戏",
                )

            self._transition(WorkflowState.WAIT_GAME_EXIT, "等待游戏窗口和进程退出")
            self._wait_window_gone(
                self.game,
                self._timing_settings().game_exit_timeout_seconds,
                expected_processes=(
                    termination.processes if termination is not None else ()
                ),
            )

            self._transition(WorkflowState.WAIT_LAUNCHER, "等待并重新绑定启动器")
            launcher = self._wait_window(
                self.launcher,
                "启动器",
                timeout=self._timing_settings().launcher_window_timeout_seconds,
            )

            # 游戏被强制结束后，启动器窗口通常仍然存在，但右下角按钮
            # 需要一段时间才会从“游戏运行中”刷新为“开始游戏”。
            # 在按钮就绪前不要进入资源管理，避免把过渡页面判为 unknown。
            self._wait_page(
                self.launcher,
                {PageType.LAUNCHER_HOME},
                self._timing_settings().launcher_window_timeout_seconds,
                "启动器“开始游戏”状态",
                include_ocr=True,
                require_launcher_start_ready=True,
            )

            self._transition(WorkflowState.STAGE_PAK, "09-10：游戏退出、启动器就绪后暂存 PAK")
            self.pak.stage()
            pak_status = self.pak.inspect()
            self.diagnostics.event("pak", action="stage", status=pak_status)
            self._emit(
                f"PAK 已暂存：{self.pak.source_display} → "
                f"{self.pak.staged_display}（状态：{pak_status}）"
            )

            self._transition(WorkflowState.REMOVE_RESOURCE, "10-11：定位并删除长弓溪谷资源")
            launcher_page = self._delete_longbow_launcher_resource()
            if launcher_page is not None:
                launcher_settings = self._require_bound(
                    self.launcher_settings,
                    "启动器设置",
                )
                self._click_match(
                    launcher_settings,
                    launcher_page,
                    (
                        "launcher_settings_title_compact",
                        "launcher_resource_header_popup",
                        "launcher_resource_header",
                    ),
                    "launcher_close_settings",
                    "关闭资源管理",
                )

            self._transition(WorkflowState.START_GAME, "13：从启动器开始游戏")
            if self.game.refresh() is None:
                launcher_home = self._wait_page(
                    self.launcher,
                    {PageType.LAUNCHER_HOME},
                    self._timing_settings().launcher_window_timeout_seconds,
                    "启动器首页",
                    include_ocr=True,
                    allow_game_started=True,
                )
                if self.game.refresh() is None:
                    launcher = self._require_bound(self.launcher, "启动器")
                    self._click_match(
                        launcher, launcher_home,
                        ("ocr_launcher_start_button", "launcher_start_button"),
                        "launcher_start_game", "开始游戏", require_match=True,
                    )
                else:
                    self._emit(
                        "等待启动器首页时游戏已由用户启动，跳过重复点击",
                        "warning",
                    )
            else:
                self._emit("游戏已由用户启动，跳过启动器“开始游戏”", "warning")

            self._transition(WorkflowState.REBIND_GAME, "14：等待新游戏窗口并自动重新绑定")
            self.game.clear()
            game = self._wait_window(
                self.game,
                "重新启动后的游戏",
                timeout=self._timing_settings().game_window_timeout_seconds,
            )
            self._emit(f"已重新绑定游戏窗口 HWND 0x{game.hwnd:X}")

            self._run_mail_game_phase(
                "资源提示与游戏启动", self._complete_mail_resource_intro,
                after_restart=lambda: None,
            )

            self._run_mail_game_phase(
                "下载地图并应用卡邮件方案",
                lambda: self._prepare_mail_scheme(selected_mail_scheme_index),
            )
            self._run_mail_game_phase(
                "等待地图下载", self._wait_for_longbow_download_completion,
                after_restart=self._resume_mail_download,
            )
            self._run_mail_game_phase(
                "切换全面战场并放弃对局", self._switch_to_warfare_and_abandon,
            )
            home = self._run_mail_game_phase(
                "结算并返回烽火地带", self._complete_warfare_round_and_return_firestorm,
                after_restart=self._mail_home_after_restart,
            )
            self._run_mail_game_phase(
                "领取邮件", lambda: self._collect_mail_rewards(home),
                after_restart=lambda: self._collect_mail_rewards(self._mail_home_after_restart()),
            )
            self._emit(f"第 {round_number} 轮已完成邮件领取，准备开始下一轮")
        except Exception:
            raise


    def _prepare_mail_agent_selection(self):
        self._transition(WorkflowState.OPEN_MAP, "01-03：选择长弓溪谷")
        self._navigate_to_longbow({PageType.MAP_LONGBOW_START})

        self._transition(WorkflowState.START_ACTION, "04：进入初次配装")
        departure_page = self._start_initial_loadout_with_recovery()

        self._transition(WorkflowState.DEPART, "06：配装出发")
        if departure_page.page_type == PageType.GAME_HOME_READY:
            game = self._require_bound(self.game, "游戏")
            self._click(game, "depart", "出发")
        else:
            self._emit(
                f"出发步骤已由用户完成，当前为"
                f"{page_type_label(departure_page.page_type)}",
                "warning",
            )

        self._transition(WorkflowState.WAIT_AGENT, "07-08：等待干员选择界面")
        agent_page = self._wait_for_agent_selection(departure_page)
        return agent_page

    def _complete_mail_resource_intro(self):
        self._transition(WorkflowState.DISMISS_DIALOGS, "15：等待第一次资源缺失提示")
        first_stage = self._wait_startup_resource(
            self.game,
            (
                _StepOutcome(
                    "第一次资源缺失提示",
                    pages=frozenset({PageType.MISSING_RESOURCE}),
                ),
            ),
            (
                _StepOutcome(
                    "初始模式选择",
                    pages=frozenset({PageType.INITIAL_MODE_SELECTION}),
                ),
                _StepOutcome(
                    "用户已进入后续页面",
                    pages=frozenset(INTRO_PROGRESS_PAGES),
                    predicate=self._confirmed_intro_progress,
                ),
            ),
            self._timing_settings().first_missing_resource_timeout_seconds,
            "第一次资源缺失提示",
            include_ocr=True,
        )
        if first_stage.observation.page_type == PageType.MISSING_RESOURCE:
            game = self._require_bound(self.game, "游戏")
            self._click_match(
                game,
                first_stage.observation,
                ("missing_resource_message",),
                "dialog_confirm",
                "第一次资源缺失确认",
            )
            current_intro: _StepResult | None = None
        else:
            current_intro = first_stage
            self._emit(
                f"当前已处于第一次资源提示的后续页面，从“{first_stage.outcome}”继续",
                "warning",
            )

        self._transition(WorkflowState.SELECT_FIRESTORM, "16：选择烽火地带")
        current_intro = self._confirm_restart_mode_selection(current_intro)

        self._transition(WorkflowState.RESTORE_PAK, "17：第二次资源缺失出现，先恢复 PAK")
        self._restore_pak_and_finish_intro(current_intro)

    def _prepare_mail_scheme(self, selected_mail_scheme_index):
        self._transition(WorkflowState.DOWNLOAD_MAP, "20-23：启动长弓溪谷下载")
        self._navigate_to_longbow({PageType.MAP_LONGBOW_DOWNLOAD, PageType.MAP_LONGBOW_START})
        download_progress = self._start_longbow_download_with_recovery()

        self._transition(WorkflowState.PREPARE_ZERO_DAM, "24-27：仅在零号大坝进入配装")
        loadout = self._enter_zero_dam_loadout(download_progress)

        self._transition(WorkflowState.APPLY_LOADOUT_SCHEME, "28-31：应用卡邮件方案")
        if loadout.page_type != PageType.LOADOUT:
            raise RuntimeError("未确认零号大坝配装页，禁止打开配装方案")
        self._apply_loadout_scheme_with_recovery(
            selected_mail_scheme_index,
            f"卡邮件配装方案 {selected_mail_scheme_index}",
            schemes_open=False,
        )

    def _run_mail_game_phase(self, label, action, *, after_restart=None):
        """Run existing bounded recovery first, then restart once at this checkpoint."""
        from bulletbot.navigation.action_recovery import PageActionStalled, PageRecoveryFailed

        previous = getattr(self, "_mail_game_phase_active", False)
        self._mail_game_phase_active = True
        try:
            try:
                return action()
            except (WorkflowTimeout, PageActionStalled) as exc:
                self._check_stop()
                if getattr(self, "_page_restart_in_progress", False):
                    raise PageRecoveryFailed(f"重启恢复中再次失败：{label}：{exc}") from exc
                self._emit(f"{label}的页面恢复已用尽，自动重启游戏后继续当前阶段：{exc}", "warning")
                self.diagnostics.event("mail_game_recovery_exhausted", phase=label, error=str(exc))
            enabled = getattr(self, "_page_recovery_enabled", False)
            self._mail_game_phase_active = False
            self._page_recovery_enabled = False
            self._page_restart_in_progress = True
            try:
                self._restart_game_without_mail_resource_cycle(LoadoutResumeTarget.MAIL_STORAGE)
            except WorkflowStopped:
                raise
            except Exception as exc:
                raise PageRecoveryFailed(f"{label}自动重启失败：{exc}") from exc
            finally:
                self._page_restart_in_progress = False
                self._page_recovery_enabled = enabled
                self._mail_game_phase_active = True
                self._set_capture_backend("mss")
            self._check_stop()
            try:
                return (after_restart or action)()
            except (WorkflowTimeout, PageActionStalled) as exc:
                raise PageRecoveryFailed(f"重启后{label}仍未恢复，停止重复重启：{exc}") from exc
        finally:
            self._mail_game_phase_active = previous

    def _resume_mail_download(self):
        self._navigate_to_longbow({PageType.MAP_LONGBOW_DOWNLOAD, PageType.MAP_LONGBOW_START})
        self._start_longbow_download_with_recovery()
        self._wait_for_longbow_download_completion()

    def _mail_home_after_restart(self):
        return self._wait_step_outcome(
            self.game,
            (_StepOutcome("重启后的烽火地带主页", pages=frozenset({
                PageType.GAME_HOME_PREPARE, PageType.GAME_HOME_READY,
            })),),
            120, "重启后的烽火地带主页", include_ocr=True,
        ).observation

    def _timing_settings(self) -> LoadoutPurchaseSettings:
        # Some recovery/test paths construct the controller without running
        # the full initializer; keep timing reads safe in those paths.
        settings = getattr(self, "_loadout_purchase_settings", None)
        return settings if settings is not None else LoadoutPurchaseSettings()

    def _reset_scheduled_game_restart_deadline(self) -> None:
        settings = self._timing_settings()
        self._scheduled_game_restart_deadline = (
            self._monotonic()
            + settings.scheduled_game_restart_interval_minutes * 60
            if settings.scheduled_game_restart_enabled
            else None
        )

    def _mark_loadout_price_refresh_resumed(
        self,
        rule: LoadoutSchemeRule,
    ) -> None:
        self._reset_scheduled_game_restart_deadline()
        self._emit_structured(
            "loadout_price_refresh_resumed",
            "邮件领取完成，重新开始刷新配装方案价格；定时重启已重新计时",
            scheme_index=rule.scheme_index,
        )

    def _raise_if_scheduled_game_restart_due(self) -> None:
        if self._loadout_finishing():
            self._check_finish_loadout_round()
            return
        deadline = getattr(self, "_scheduled_game_restart_deadline", None)
        if deadline is not None and self._monotonic() >= deadline:
            raise ScheduledGameRestartRequested()

    def _restart_for_page_recovery(self, reason: str) -> None:
        from bulletbot.navigation.action_recovery import PageRecoveryFailed

        if getattr(self, "_page_restart_in_progress", False):
            raise PageRecoveryFailed("页面恢复重启过程中再次卡住，停止重复重启")
        self._check_stop()
        state = self._state
        checkpoint = getattr(self, "_loadout_checkpoint", None)
        if checkpoint is not None:
            self._save_loadout_checkpoint(
                checkpoint.stage,
                recovery_reason=reason,
                last_safe_action="页面无响应，保留当前业务阶段和已确认结果后重启",
            )
        self._emit(f"{reason}；结束游戏进程树后重新启动，保留当前任务", "warning")
        enabled = getattr(self, "_page_recovery_enabled", False)
        self._page_restart_in_progress = True
        self._page_recovery_enabled = False
        try:
            # The caller resumes its own operation with fresh page/inventory data.
            self._restart_game_without_mail_resource_cycle(LoadoutResumeTarget.LOADOUT_SCAN)
            self._set_capture_backend("wgc")
            self._transition(state, "游戏重启完成，重新定位并继续原任务")
        except WorkflowStopped:
            raise
        except Exception as exc:
            raise PageRecoveryFailed(f"页面恢复重启失败：{exc}") from exc
        finally:
            self._page_recovery_enabled = enabled
            self._page_restart_in_progress = False

    def _restart_game_without_mail_resource_cycle(
        self,
        resume_target: LoadoutResumeTarget | None = None,
    ) -> None:
        settings = self._timing_settings()
        checkpoint = getattr(self, "_loadout_checkpoint", None)
        target = resume_target or (
            checkpoint.resume_target
            if checkpoint is not None and checkpoint.resume_target is not None
            else LoadoutResumeTarget.LOADOUT_SCAN
        )
        self._transition(
            WorkflowState.CLOSE_GAME,
            "定时重启：已到达安全节点，停止当前页面操作并关闭游戏",
        )
        game = self.game.refresh()
        termination: ProcessTreeTermination | None = None
        if game is None and not matching_process_running(self.game.fingerprint):
            self._emit("定时重启时游戏已关闭，跳过结束进程", "warning")
        else:
            termination = self._terminate_game_process_tree(
                game,
                reason="scheduled_game_restart",
                label="定时重启关闭游戏",
            )

        self._transition(WorkflowState.WAIT_GAME_EXIT, "定时重启：等待游戏完全退出")
        self._wait_window_gone(
            self.game,
            settings.game_exit_timeout_seconds,
            expected_processes=(
                termination.processes if termination is not None else ()
            ),
        )
        # Resource files can only be restored once the old game has exited.
        pak = getattr(self, "pak", None)
        if pak is not None and pak.inspect() in {"staged", "partial"}:
            if not self.settings.allow_file_operations:
                raise RuntimeError("自动重启前需要恢复 PAK，但文件操作未启用")
            pak.restore()
            if pak.inspect() != "ready":
                raise RuntimeError("自动重启前 PAK 恢复未完成")
            self._emit("自动重启前已恢复暂存 PAK，继续兼容资源校验或直接进入游戏")
            self.diagnostics.event("pak", action="restart_restore", status="ready")
        self._transition(
            WorkflowState.WAIT_LAUNCHER,
            "定时重启：等待启动器，不删除地图资源",
        )
        self._wait_window(
            self.launcher,
            "启动器",
            timeout=settings.launcher_window_timeout_seconds,
        )
        if self.game.refresh() is None:
            self._wait_page(
                self.launcher,
                {PageType.LAUNCHER_HOME},
                settings.launcher_window_timeout_seconds,
                "定时重启启动器首页",
                include_ocr=True,
                allow_game_started=True,
                require_launcher_start_ready=True,
            )
            if self.game.refresh() is None:
                launcher = self._require_bound(self.launcher, "启动器")
                self._click(launcher, "launcher_start_game", "定时重启：开始游戏")

        self._transition(WorkflowState.REBIND_GAME, "定时重启：等待新游戏窗口")
        self.game.clear()
        game = self._wait_window(
            self.game,
            "定时重启后的游戏",
            timeout=settings.game_window_timeout_seconds,
        )
        self._emit(f"定时重启已重新绑定游戏窗口 HWND 0x{game.hwnd:X}")
        self._finish_scheduled_restart_startup()
        self._prepare_game_for_trading_handoff(allow_game_restart=False)
        self._reset_scheduled_game_restart_deadline()
        self._resume_after_game_restart(target)
        self._emit_structured(
            "scheduled_game_restart_completed",
            f"定时重启完成，继续目的：{self._resume_target_label(target)}",
            interval_minutes=settings.scheduled_game_restart_interval_minutes,
            resume_target=target.value,
        )

    @staticmethod
    def _resume_target_label(target: LoadoutResumeTarget) -> str:
        return {
            LoadoutResumeTarget.LOADOUT_SCAN: "配装方案价格扫描",
            LoadoutResumeTarget.RETURN_HOME_THEN_SCAN: "返回主界面后开始配装价格扫描",
            LoadoutResumeTarget.WAREHOUSE_SELL: "交易行出售页处理",
            LoadoutResumeTarget.MAIL_STORAGE: "卡邮件/仓库处理",
        }[target]

    def _resume_after_game_restart(self, target: LoadoutResumeTarget) -> None:
        """Complete the purpose that was pending when the game was restarted."""
        if target in {
            LoadoutResumeTarget.LOADOUT_SCAN,
            LoadoutResumeTarget.RETURN_HOME_THEN_SCAN,
        }:
            self._emit(
                f"重启后已回到主界面，继续{self._resume_target_label(target)}"
            )
            return
        if target is LoadoutResumeTarget.WAREHOUSE_SELL:
            result = self._unified_navigation.navigate(
                GamePageId.MARKET_SELL,
                reason="重启后恢复交易行出售页处理",
            )
            if not result.succeeded:
                raise RuntimeError(result.message)
            self._emit("重启后已恢复到交易行出售页")
            return
        if target is LoadoutResumeTarget.MAIL_STORAGE:
            self._emit("重启后已回到主界面，继续卡邮件/仓库处理")

    @staticmethod
    def _startup_resource_waiting(page: PageObservation) -> bool:
        text = "".join("".join(page.ocr_texts).split())
        return not any(term in text for term in ("失败", "错误")) and any(
            term in text for term in (
                "正在校验", "校验中", "资源校验", "正在下载", "下载中", "检查资源", "正在修复",
            )
        )

    def _finish_scheduled_restart_startup(self) -> None:
        started = self._monotonic()
        deadline = started + self._timing_settings().game_window_timeout_seconds
        resource_deadline = started + max(
            self._timing_settings().game_window_timeout_seconds, LONGBOW_DOWNLOAD_TIMEOUT_SECONDS,
        )
        outcomes = (
            _StepOutcome(
                "资源缺失提示",
                pages=frozenset({PageType.MISSING_RESOURCE}),
            ),
            _StepOutcome(
                "初始模式选择",
                pages=frozenset({PageType.INITIAL_MODE_SELECTION}),
            ),
            _StepOutcome(
                "系统确认提示",
                pages=frozenset({PageType.GENERIC_CONFIRM}),
            ),
            _StepOutcome(
                "可继续的启动页面",
                pages=frozenset(INTRO_CONTINUE_PAGES),
            ),
            _StepOutcome(
                "游戏页面已就绪",
                pages=frozenset(INTRO_COMPLETE_PAGES),
            ),
            _StepOutcome("重启后模式或重连页面", pages=frozenset({
                PageType.WARFARE_HOME, PageType.SETTLEMENT, PageType.MODE_MENU,
                PageType.RECONNECT_PROMPT, PageType.ABANDON_PROMPT,
            })),
            _StepOutcome(
                "资源校验或下载中",
                predicate=lambda _frame, page: self._startup_resource_waiting(page),
            ),
        )
        resource_wait_reported = False
        while True:
            self._check_stop()
            if self._monotonic() >= deadline:
                raise WorkflowTimeout("等待游戏启动页面就绪超时")
            current = self._wait_startup_resource(
                self.game,
                outcomes,
                (),
                max(0.1, deadline - self._monotonic()),
                "定时重启后的游戏页面",
                include_ocr=True,
            ).observation
            if current.page_type in INTRO_COMPLETE_PAGES:
                return
            if current.page_type in {PageType.WARFARE_HOME, PageType.SETTLEMENT, PageType.MODE_MENU}:
                self._return_to_firestorm_home(current)
                return
            if current.page_type in {PageType.RECONNECT_PROMPT, PageType.ABANDON_PROMPT}:
                game = self._require_bound(self.game, "游戏")
                if current.page_type == PageType.RECONNECT_PROMPT:
                    self._click(game, "cancel_reconnect", "重启后取消重连")
                else:
                    self._click(game, "abandon_match", "重启后确认放弃对局")
                continue
            if self._startup_resource_waiting(current) and current.page_type not in {
                PageType.MISSING_RESOURCE, PageType.GENERIC_CONFIRM,
            }:
                deadline = resource_deadline
                if not resource_wait_reported:
                    self._emit("重启后正在资源校验或下载，等待完成后继续；无需每次都出现资源提示")
                    resource_wait_reported = True
                self._sleep(min(1.0, max(0.0, deadline - self._monotonic())))
                continue
            game = self._require_bound(self.game, "游戏")
            if current.page_type == PageType.MISSING_RESOURCE:
                self._click_match(
                    game,
                    current,
                    ("missing_resource_message",),
                    "dialog_confirm",
                    "定时重启：兼容资源缺失提示",
                )
            elif current.page_type == PageType.INITIAL_MODE_SELECTION:
                self._click_match(
                    game,
                    current,
                    ("season_2026_initial_mode_tabs", "real_initial_firestorm_card"),
                    "initial_firestorm_mode",
                    "定时重启：选择烽火地带",
                    require_match=False,
                )
            elif current.page_type == PageType.GAME_TRANSITION:
                self._press_key(game, "tab", "定时重启：Tab 开始游戏")
            elif current.page_type == PageType.ACTIVITY_REMINDER:
                self._press_key(game, "space", "定时重启：关闭活动提醒")
            else:
                self._click(game, "dialog_confirm", "定时重启：确认系统提示")

    def _wait_for_agent_selection(
        self,
        current: PageObservation,
    ) -> PageObservation:
        if current.page_type == PageType.AGENT_SELECT:
            return current
        if current.page_type == PageType.GAME_LOADING:
            self._emit(
                "进入干员选择等待前已处于游戏加载；"
                "按先前异常恢复已确认越过干员选择处理",
                "warning",
            )
            return current

        deadline = self._monotonic() + 240
        while True:
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                raise WorkflowTimeout("进入干员选择的恢复已用尽，页面仍未推进")
            result = self._wait_step_outcome_with_delayed_recovery(
                self.game,
                (
                    _StepOutcome(
                        "干员选择界面",
                        pages=frozenset({PageType.AGENT_SELECT}),
                        single_ocr_confirmation_terms=frozenset({"选择干员"}),
                    ),
                    _StepOutcome(
                        "入局提醒",
                        pages=frozenset({PageType.ENTRY_WARNING}),
                    ),
                ),
                (
                    _StepOutcome(
                        "已越过干员选择的游戏加载",
                        pages=frozenset({PageType.GAME_LOADING}),
                    ),
                ),
                remaining,
                "干员选择界面",
                include_ocr=True,
                ocr_interval=3,
                template_names=AGENT_SELECTION_TEMPLATE_NAMES,
                recovery_template_names=AGENT_SELECTION_RECOVERY_TEMPLATE_NAMES,
                ocr_crop=AGENT_SELECTION_OCR_CROP,
            )
            if result.observation.page_type in {
                PageType.AGENT_SELECT,
                PageType.GAME_LOADING,
            }:
                return result.observation
            game = self._require_bound(self.game, "游戏")
            self._click_match(
                game,
                result.observation,
                ("real_entry_warning_continue",),
                "entry_warning_continue",
                "仍要继续",
            )

    def _complete_warfare_round_and_return_firestorm(self) -> PageObservation:
        self._transition(
            WorkflowState.WAIT_SETTLEMENT,
            "40-42：等待结算或全面战场主页",
        )
        expected_pages = {
            PageType.GAME_LOADING,
            PageType.SETTLEMENT,
            PageType.ACTIVITY_REMINDER,
            PageType.WARFARE_HOME,
        }
        recovery_pages = {
            PageType.MODE_MENU,
            PageType.GAME_HOME_PREPARE,
            PageType.GAME_HOME_READY,
        }
        try:
            warfare_page = self._wait_step_outcome(
                self.game,
                (
                    _StepOutcome(
                        "全面战场结算或主界面",
                        pages=frozenset(expected_pages),
                    ),
                ),
                WARFARE_SETTLEMENT_TIMEOUT_SECONDS,
                "全面战场结算或主界面",
                include_ocr=True,
            ).observation
        except WorkflowTimeout as exc:
            last = exc.last_observation
            if last is None or not (
                last.page_type in expected_pages | recovery_pages
                or "real_space_continue_prompt" in last.matches
            ):
                raise
            warfare_page = last
            if (
                warfare_page.page_type not in expected_pages | recovery_pages
                and "real_space_continue_prompt" in warfare_page.matches
            ):
                warfare_page = PageObservation(
                    PageType.ACTIVITY_REMINDER,
                    warfare_page.confidence,
                    warfare_page.matches,
                    warfare_page.ocr_texts,
                    warfare_page.ocr_regions,
                )
            self._emit(
                f"结算等待超时，但最后仍识别到可恢复状态"
                f"{page_type_label(warfare_page.page_type)}；继续恢复",
                "warning",
            )
        if warfare_page.page_type in {
            PageType.MODE_MENU,
            PageType.GAME_HOME_PREPARE,
            PageType.GAME_HOME_READY,
        }:
            self._emit(
                f"结算/返回步骤已由用户推进到"
                f"{page_type_label(warfare_page.page_type)}",
                "warning",
            )
        elif warfare_page.page_type == PageType.WARFARE_HOME:
            self._emit("已进入全面战场主界面，无需继续等待结算")

        self._transition(WorkflowState.RETURN_FIRESTORM, "42-44：切回烽火地带")
        return self._return_to_firestorm_home(warfare_page)

    def _wait_mail_scheme_usable(
        self,
        timeout: float = 30,
        *,
        expected_scheme_index: int | None = None,
        save_timeout: bool = True,
        allow_advanced: bool = False,
    ) -> PageObservation:
        started = self._monotonic()
        latest_frame: CapturedFrame | None = None
        latest_page = PageType.UNKNOWN
        latest_selected_index: int | None = None
        mismatch_reported = False
        advanced_page: PageType | None = None
        advanced_count = 0
        while True:
            self._check_stop()
            window = self.game.refresh()
            if window is None:
                advanced_page = None
                advanced_count = 0
                self._sleep(0.5)
                if self._monotonic() - started >= timeout:
                    raise WorkflowTimeout("等待卡邮件方案时游戏窗口丢失")
                continue

            # Confirm the selected row and use button from the same fresh frame.
            # Full-screen OCR must not delay an already usable scheme.
            latest_frame, observation = self._observe(window, "game", include_ocr=False)
            latest_page = observation.page_type
            if allow_advanced and latest_page in {
                PageType.LOADOUT,
                PageType.ENTRY_WARNING,
                PageType.GAME_HOME_READY,
            }:
                if latest_page == advanced_page:
                    advanced_count += 1
                else:
                    advanced_page = latest_page
                    advanced_count = 1
            else:
                advanced_page = None
                advanced_count = 0
                latest_selected_index = self._detect_selected_loadout_scheme(
                    latest_frame.image
                )
                selection_confirmed = (
                    expected_scheme_index is None
                    or latest_selected_index == expected_scheme_index
                )
                if (
                    expected_scheme_index is not None
                    and not selection_confirmed
                    and not mismatch_reported
                ):
                    actual = (
                        str(latest_selected_index)
                        if latest_selected_index is not None
                        else "未识别"
                    )
                    self._emit(
                        f"等待配装方案 {expected_scheme_index} 高亮确认；"
                        f"当前高亮 {actual}，不会点击“使用方案”",
                        "warning",
                    )
                    mismatch_reported = True
                if (
                    selection_confirmed
                    and "real_use_scheme_button" in observation.matches
                ):
                    if expected_scheme_index is not None:
                        self._emit(
                            f"已确认配装方案 {expected_scheme_index} 高亮且显示“使用方案”"
                        )
                    return observation

                if (
                    selection_confirmed
                    and "real_loadout_schemes_header" in observation.matches
                    and "real_use_scheme_button" not in observation.matches
                ):
                    observation = self._analyze_mail_frame(
                        latest_frame, window, "game", include_ocr=True,
                        ocr_crop=(0.75, 0.72, 0.96, 0.88),
                    )

                price_text = self._find_scheme_price(
                    observation,
                    latest_frame.image.shape,
                )
                if (
                    price_text is not None
                    and selection_confirmed
                    and "real_loadout_schemes_header" in observation.matches
                    and self._monotonic() - started >= 2.0
                ):
                    path = self.diagnostics.save_frame(
                        "stop_卡邮件方案需要购买",
                        latest_frame.image,
                    )
                    self._emit(
                        f"卡邮件方案显示金额“{price_text}”，"
                        f"已保存停止截图：{path}",
                        "warning",
                    )
                    raise LoadoutSchemePurchaseRequired(
                        f"右下角显示金额“{price_text}”，不是“使用方案”"
                    )

            if self._monotonic() - started >= timeout:
                if allow_advanced and advanced_count >= 2:
                    self._emit(
                        f"选择方案正常等待超时后，确认页面已前进到"
                        f"{page_type_label(latest_page)}；"
                        "异常恢复按用户已完成“使用方案”处理",
                        "warning",
                    )
                    return observation
                if latest_frame is not None and save_timeout:
                    path = self.diagnostics.save_frame(
                        "timeout_卡邮件方案使用状态",
                        latest_frame.image,
                    )
                    self._emit(f"已保存超时截图：{path}", "warning")
                raise WorkflowTimeout(
                    "等待卡邮件方案右下角“使用方案”超时，"
                    f"最后识别为{page_type_label(latest_page)}，"
                    f"目标方案 {expected_scheme_index or '未指定'}，"
                    f"实际高亮 {latest_selected_index or '未识别'}"
                )
            self._sleep(self.settings.observation_interval_seconds)

    def _select_loadout_scheme_for_use(
        self,
        scheme_index: int,
        label: str,
    ) -> PageObservation:
        for attempt, timeout in ((1, 8), (2, 30)):
            game = self._require_bound(self.game, "游戏")
            self._click(
                game,
                f"loadout_scheme_{scheme_index}",
                label if attempt == 1 else f"重试：{label}",
            )
            try:
                return self._wait_mail_scheme_usable(
                    timeout,
                    expected_scheme_index=scheme_index,
                    save_timeout=attempt == 2,
                    allow_advanced=False,
                )
            except WorkflowTimeout:
                if attempt == 1:
                    self._emit(
                        f"{label}首次点击后等待方案 {scheme_index} 高亮及“使用方案”超时；"
                        "重新点击一次",
                        "warning",
                    )
                    continue
                raise
        raise AssertionError("unreachable")

    def _complete_purchase_required_mode_switch(self) -> None:
        self._wait_for_longbow_download_completion(
            context="配装方案需要购买",
        )
        self._switch_to_warfare_and_abandon(
            context="配装方案需要购买",
        )
        self._complete_warfare_round_and_return_firestorm()
        self._emit(
            "未应用配装方案；已等待长弓溪谷下载完成、放弃重连对局并返回烽火地带主界面，"
            "本轮卡邮件结束",
            "warning",
        )

    def _wait_for_longbow_download_completion(self, *, context: str = "") -> None:
        prefix = f"{context}：" if context else ""
        self._transition(
            WorkflowState.WAIT_DOWNLOAD,
            f"{prefix}等待长弓溪谷下载完成，不进入行动；每 {LONGBOW_DOWNLOAD_OBSERVATION_INTERVAL_SECONDS:g} 秒检查一次",
            "warning" if context else "info",
        )
        longbow = self._navigate_to_longbow(
            {PageType.MAP_LONGBOW_START, PageType.MAP_LONGBOW_DOWNLOAD},
        )
        deadline = self._monotonic() + LONGBOW_DOWNLOAD_TIMEOUT_SECONDS
        observable_pages = {
            PageType.MAP_LONGBOW_START,
            PageType.MAP_LONGBOW_DOWNLOAD,
            PageType.MAP_OVERVIEW,
            PageType.MAP_SELECTION,
            PageType.MAP_OTHER,
            PageType.MAP_ZERO_DAM_START,
            PageType.GAME_HOME_PREPARE,
            PageType.GAME_HOME_READY,
            PageType.LOADOUT,
            *LOADOUT_SCHEME_PAGES,
        }
        while longbow.page_type != PageType.MAP_LONGBOW_START:
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                raise WorkflowTimeout("等待长弓溪谷下载完成超时")
            try:
                result = self._wait_step_outcome(
                    self.game,
                    (
                        _StepOutcome(
                            "长弓溪谷下载进度",
                            pages=frozenset(observable_pages),
                        ),
                    ),
                    min(10.0, remaining),
                    "长弓溪谷下载进度",
                    include_ocr=True,
                    save_timeout=False,
                    observation_interval_seconds=LONGBOW_DOWNLOAD_OBSERVATION_INTERVAL_SECONDS,
                )
            except WorkflowTimeout:
                continue
            longbow = result.observation
            if longbow.page_type not in {
                PageType.MAP_LONGBOW_START,
                PageType.MAP_LONGBOW_DOWNLOAD,
            }:
                self._emit(
                    f"等待下载时页面被切换到"
                    f"{page_type_label(longbow.page_type)}；"
                    "重新定位长弓溪谷查看下载状态",
                    "warning",
                )
                longbow = self._navigate_to_longbow(
                    {PageType.MAP_LONGBOW_START, PageType.MAP_LONGBOW_DOWNLOAD},
                )
            elif longbow.page_type == PageType.MAP_LONGBOW_DOWNLOAD:
                self._sleep(min(LONGBOW_DOWNLOAD_OBSERVATION_INTERVAL_SECONDS, remaining))

    def _switch_to_warfare_and_abandon(self, *, context: str = "") -> None:
        prefix = f"{context}：" if context else ""
        switch_timeout = self._timing_settings().warfare_switch_timeout_seconds
        self._transition(
            WorkflowState.SWITCH_MODE,
            f"{prefix}退出地图并切换全面战场",
            "warning" if context else "info",
        )
        map_detail_pages = {
            PageType.MAP_OVERVIEW,
            PageType.MAP_OTHER,
            PageType.MAP_LONGBOW_START,
            PageType.MAP_LONGBOW_DOWNLOAD,
            PageType.MAP_ZERO_DAM_START,
        }
        home_pages = {PageType.GAME_HOME_PREPARE, PageType.GAME_HOME_READY}
        completed_pages = {
            PageType.GAME_LOADING,
            PageType.SETTLEMENT,
            PageType.ACTIVITY_REMINDER,
            PageType.WARFARE_HOME,
        }
        recoverable_pages = map_detail_pages | home_pages | completed_pages | {
            PageType.MAP_SELECTION, PageType.MODE_MENU,
            PageType.RECONNECT_PROMPT, PageType.ABANDON_PROMPT,
        }
        try:
            self._wait_step_or_recover(
                self.game,
                (
                    _StepOutcome(
                        "长弓溪谷开始行动页",
                        pages=frozenset({PageType.MAP_LONGBOW_START}),
                    ),
                ),
                switch_timeout,
                "切换全面战场前的长弓溪谷页面",
                include_ocr=True,
                recovery_pages=recoverable_pages,
            )
            game = self._require_bound(self.game, "游戏")
            self._press_key(game, "escape", "Esc：返回地图全览")
            self._wait_step_or_recover(
                self.game,
                (
                    _StepOutcome(
                        "地图全览",
                        pages=frozenset({PageType.MAP_SELECTION}),
                    ),
                ),
                switch_timeout,
                "地图全览",
                include_ocr=True,
                recovery_pages=recoverable_pages,
            )
            game = self._require_bound(self.game, "游戏")
            self._press_key(game, "escape", "Esc：返回游戏主界面")
            self._wait_step_or_recover(
                self.game,
                (
                    _StepOutcome(
                        "游戏主界面",
                        pages=frozenset(home_pages),
                    ),
                ),
                switch_timeout,
                "游戏主界面",
                include_ocr=True,
                recovery_pages=recoverable_pages,
            )
            game = self._require_bound(self.game, "游戏")
            self._press_key(game, "escape", "Esc：打开模式菜单")
            self._wait_step_or_recover(
                self.game,
                (
                    _StepOutcome(
                        "模式选择菜单",
                        pages=frozenset({PageType.MODE_MENU}),
                    ),
                ),
                switch_timeout,
                "模式选择菜单",
                include_ocr=True,
                recovery_pages=recoverable_pages,
            )
            game = self._require_bound(self.game, "游戏")
            self._click(game, "warfare_mode", "全面战场")

            self._transition(
                WorkflowState.CANCEL_RECONNECT,
                f"{prefix}取消重连并放弃对局",
                "warning" if context else "info",
            )
            self._wait_step_or_recover(
                self.game,
                (
                    _StepOutcome(
                        "取消重连提示",
                        pages=frozenset({PageType.RECONNECT_PROMPT}),
                    ),
                ),
                switch_timeout,
                "取消重连提示",
                include_ocr=True,
                recovery_pages=recoverable_pages,
            )
            game = self._require_bound(self.game, "游戏")
            self._click(game, "cancel_reconnect", "取消重连")
            self._wait_step_or_recover(
                self.game,
                (
                    _StepOutcome(
                        "放弃对局确认",
                        pages=frozenset({PageType.ABANDON_PROMPT}),
                    ),
                ),
                switch_timeout,
                "放弃对局确认",
                include_ocr=True,
                recovery_pages=recoverable_pages,
            )
            game = self._require_bound(self.game, "游戏")
            self._click(game, "abandon_match", "放弃对局")
            completed = self._wait_step_or_recover(
                self.game,
                (
                    _StepOutcome(
                        "放弃对局后的进度",
                        pages=frozenset(completed_pages),
                    ),
                ),
                switch_timeout,
                "放弃对局后的进度",
                include_ocr=True,
                recovery_pages=recoverable_pages,
            ).observation
            self._emit(
                f"切换全面战场已前进到"
                f"{page_type_label(completed.page_type)}"
            )
            return
        except WorkflowTimeout:
            self._emit(
                "切换全面战场步骤偏离预期或等待超时；"
                "进入异常恢复并识别用户可能推进到的页面",
                "warning",
            )

        self._transition(
            WorkflowState.CANCEL_RECONNECT,
            f"{prefix}异常恢复：取消重连并放弃对局",
            "warning",
        )
        observable_pages = (
            map_detail_pages
            | home_pages
            | completed_pages
            | {
                PageType.MAP_SELECTION,
                PageType.MODE_MENU,
                PageType.RECONNECT_PROMPT,
                PageType.ABANDON_PROMPT,
            }
        )
        action_counts: dict[PageType, int] = {}
        for _step in range(14):
            current = self._wait_step_outcome(
                self.game,
                (
                    _StepOutcome(
                        "切换全面战场当前状态",
                        pages=frozenset(observable_pages),
                    ),
                ),
                switch_timeout,
                "切换全面战场或放弃对局进度",
                include_ocr=True,
            ).observation
            page_type = current.page_type
            if page_type in completed_pages:
                self._emit(
                    f"切换全面战场已前进到{page_type_label(page_type)}"
                )
                return

            action_counts[page_type] = action_counts.get(page_type, 0) + 1
            max_actions = (
                1
                if page_type
                in {PageType.RECONNECT_PROMPT, PageType.ABANDON_PROMPT}
                else 2
            )
            if action_counts[page_type] > max_actions:
                raise WorkflowTimeout(
                    f"切换全面战场连续停留在{page_type_label(page_type)}；"
                    "已停止重复操作"
                )
            game = self._require_bound(self.game, "游戏")
            if page_type in map_detail_pages:
                self._press_key(game, "escape", "Esc：返回地图全览")
            elif page_type == PageType.MAP_SELECTION:
                self._press_key(game, "escape", "Esc：返回游戏主界面")
            elif page_type in home_pages:
                self._press_key(game, "escape", "Esc：打开模式菜单")
            elif page_type == PageType.MODE_MENU:
                self._click(game, "warfare_mode", "全面战场")
            elif page_type == PageType.RECONNECT_PROMPT:
                self._click(game, "cancel_reconnect", "取消重连")
            elif page_type == PageType.ABANDON_PROMPT:
                self._click(game, "abandon_match", "放弃对局")
        raise WorkflowTimeout("切换全面战场并放弃对局超过最大恢复步骤数")

    def _return_to_firestorm_home(
        self,
        current: PageObservation | None = None,
    ) -> PageObservation:
        home_pages = {PageType.GAME_HOME_PREPARE, PageType.GAME_HOME_READY}
        settlement_pages = {
            PageType.SETTLEMENT,
            PageType.ACTIVITY_REMINDER,
            PageType.WARFARE_HOME,
        }
        progress_pages = {
            PageType.GAME_LOADING,
            *settlement_pages,
            PageType.MODE_MENU,
            *home_pages,
        }
        try:
            if current is None or current.page_type == PageType.GAME_LOADING:
                current = self._wait_step_or_recover(
                    self.game,
                    (
                        _StepOutcome(
                            "全面战场结算或主界面",
                            pages=frozenset(settlement_pages),
                        ),
                    ),
                    180,
                    "全面战场结算或主界面",
                    include_ocr=True,
                    recovery_pages=progress_pages - {PageType.GAME_LOADING},
                ).observation

            settlement_actions = 0
            while current.page_type in {
                PageType.SETTLEMENT,
                PageType.ACTIVITY_REMINDER,
            }:
                settlement_actions += 1
                if settlement_actions > 3:
                    raise WorkflowTimeout(
                        "正常结算步骤连续三次仍未进入全面战场主界面",
                        last_observation=current,
                    )
                game = self._require_bound(self.game, "游戏")
                self._press_key(game, "space", "空格：继续结算")
                current = self._wait_step_or_recover(
                    self.game,
                    (
                        _StepOutcome(
                            "继续结算后的页面",
                            pages=frozenset(settlement_pages),
                        ),
                    ),
                    180,
                    "继续结算后的全面战场主界面",
                    include_ocr=True,
                    recovery_pages=progress_pages - {PageType.GAME_LOADING},
                ).observation

            if current.page_type != PageType.WARFARE_HOME:
                raise WorkflowTimeout(
                    "正常返回流程未处于全面战场主界面",
                    last_observation=current,
                )
            game = self._require_bound(self.game, "游戏")
            self._press_key(game, "escape", "Esc：打开模式菜单")
            current = self._wait_step_or_recover(
                self.game,
                (
                    _StepOutcome(
                        "模式选择菜单",
                        pages=frozenset({PageType.MODE_MENU}),
                    ),
                ),
                45,
                "返回烽火地带的模式选择菜单",
                include_ocr=True,
                recovery_pages=progress_pages - {PageType.GAME_LOADING},
            ).observation
            game = self._require_bound(self.game, "游戏")
            self._click(game, "firestorm_mode", "烽火地带")
            return self._wait_step_or_recover(
                self.game,
                (
                    _StepOutcome(
                        "烽火地带主界面",
                        pages=frozenset(home_pages),
                    ),
                ),
                120,
                "烽火地带主界面",
                include_ocr=True,
                recovery_pages=progress_pages - {PageType.GAME_LOADING},
            ).observation
        except WorkflowTimeout as exc:
            self._emit(
                "返回烽火地带步骤偏离预期或等待超时；"
                "进入异常恢复并识别用户可能推进到的页面",
                "warning",
            )
            current = (
                exc.last_observation
                if exc.last_observation is not None
                and exc.last_observation.page_type in progress_pages
                else None
            )

        action_counts: dict[PageType, int] = {}
        for _step in range(12):
            if current is None:
                current = self._wait_step_outcome(
                    self.game,
                    (
                        _StepOutcome(
                            "返回烽火地带当前状态",
                            pages=frozenset(progress_pages),
                        ),
                    ),
                    180,
                    "全面战场结算或返回烽火地带进度",
                    include_ocr=True,
                ).observation
            page_type = current.page_type
            if page_type in home_pages:
                return current
            if page_type == PageType.GAME_LOADING:
                current = self._wait_step_outcome(
                    self.game,
                    (_StepOutcome("加载结束后的页面", pages=frozenset(
                        progress_pages - {PageType.GAME_LOADING}
                    )),),
                    180, "等待加载结束", include_ocr=True,
                ).observation
                continue

            action_counts[page_type] = action_counts.get(page_type, 0) + 1
            if action_counts[page_type] > 2:
                raise WorkflowTimeout(
                    f"返回烽火地带连续停留在{page_type_label(page_type)}；"
                    "已停止重复操作"
                )
            game = self._require_bound(self.game, "游戏")
            if page_type in {PageType.SETTLEMENT, PageType.ACTIVITY_REMINDER}:
                self._press_key(game, "space", "空格：继续结算")
            elif page_type == PageType.WARFARE_HOME:
                self._press_key(game, "escape", "Esc：打开模式菜单")
            elif page_type == PageType.MODE_MENU:
                self._click(game, "firestorm_mode", "烽火地带")
            current = None
        raise WorkflowTimeout("返回烽火地带超过最大恢复步骤数")

    @staticmethod

    def _find_scheme_price(
        observation: PageObservation,
        image_shape: tuple[int, ...],
    ) -> str | None:
        height, width = image_shape[:2]
        for region in observation.ocr_regions:
            left, top, right, bottom = region.bounds
            center_x = ((left + right) / 2) / width
            center_y = ((top + bottom) / 2) / height
            if (
                0.75 <= center_x <= 0.96
                and 0.72 <= center_y <= 0.88
                and parse_ocr_integer(region.text) is not None
            ):
                return region.text.strip()
        return None

    def _collect_mail_rewards(self, home: PageObservation) -> None:
        self._transition(WorkflowState.COLLECT_MAIL, "45-53：领取邮件中的胸挂和背包")
        notification = _StepOutcome(
            "新邮件入口",
            templates=frozenset({"real_mail_notification_icon"}),
        )
        inbox_outcome = _StepOutcome(
            "邮件附件界面",
            pages=frozenset({PageType.MAIL_INBOX}),
            read_text=True,
        )
        claim_complete = _StepOutcome(
            "领取完成",
            pages=frozenset({PageType.MAIL_CLAIM_COMPLETE}),
        )
        home_complete = _StepOutcome(
            "已返回行前备战",
            pages=frozenset(
                {PageType.GAME_HOME_PREPARE, PageType.GAME_HOME_READY}
            ),
        )

        if home.page_type == PageType.MAIL_INBOX:
            current = _StepResult(
                "邮件附件界面",
                None,
                home,
            )
        elif home.page_type == PageType.MAIL_CLAIM_COMPLETE:
            current = _StepResult(
                "领取完成",
                None,
                home,
            )
        elif "real_mail_notification_icon" in home.matches:
            current = _StepResult(
                "新邮件入口",
                None,
                home,
            )
        else:
            self._emit("尚未看到邮件感叹号，等待新邮件到达")
            current = self._wait_step_outcome_with_delayed_recovery(
                self.game,
                (notification,),
                (inbox_outcome, claim_complete),
                120,
                "新邮件入口",
                include_ocr=True,
            )

        if current.outcome == "新邮件入口":
            game = self._require_bound(self.game, "游戏")
            self._click_match(
                game,
                current.observation,
                ("real_mail_notification_icon",),
                "mail_icon",
                "打开邮件",
                require_match=True,
            )
            current = self._wait_step_outcome_with_delayed_recovery(
                self.game,
                (inbox_outcome,),
                (claim_complete,),
                60,
                "物资异常丢失补偿邮件",
                include_ocr=True,
            )
        else:
            self._emit(
                f"邮件入口步骤已被跳过或由用户完成，从“{current.outcome}”继续",
                "warning",
            )

        if current.outcome == "领取完成":
            self._finish_mail_claim(current, inbox_outcome, home_complete)
            return

        partial_selected = _StepOutcome(
            "部分领取已开启",
            pages=frozenset({PageType.MAIL_INBOX}),
            templates=frozenset({"real_mail_partial_selected"}),
            read_text=True,
        )
        partial_available = _StepOutcome(
            "可以开启部分领取",
            pages=frozenset({PageType.MAIL_INBOX}),
            templates=frozenset({"real_mail_partial_claim"}),
            read_text=True,
        )

        if "real_mail_partial_selected" in current.observation.matches:
            selection_start = _StepResult("部分领取已开启", None, current.observation)
        elif "real_mail_partial_claim" in current.observation.matches:
            selection_start = _StepResult("可以开启部分领取", None, current.observation)
        else:
            self._emit("邮件页未立即显示部分领取控件，进入恢复识别", "warning")
            selection_start = self._wait_step_outcome(
                self.game,
                (partial_selected, partial_available, claim_complete),
                12,
                "邮件部分领取起点",
                include_ocr=True,
            )
        if selection_start.outcome == "领取完成":
            self._finish_mail_claim(selection_start, inbox_outcome, home_complete)
            return

        game = self._require_bound(self.game, "游戏")
        # The first slot is the only reliable cheap discriminator.  A gear
        # item keeps the historical first/second-slot path; ammunition means
        # the two gear items are at the bottom of the attachment strip.
        if selection_start.outcome == "部分领取已开启":
            selection_mode = selection_start.observation
            first_is_gear = self._mail_first_attachment_is_gear(selection_mode)
            self._emit("检测到用户已开启部分领取，保留现有附件选择状态")
        else:
            self._click(game, "mail_attachment_first", "查看第一格邮件附件")
            first_attachment = self._observe(
                game,
                "game",
                include_ocr=True,
                template_names=MAIL_ATTACHMENT_TEMPLATE_NAMES,
            )[1]
            first_is_gear = self._mail_first_attachment_is_gear(first_attachment)
            if first_is_gear:
                self._emit("首格附件识别为胸挂或背包，使用前两格领取流程")
            else:
                self._emit(
                    "首格名称未确认是胸挂/背包；转入滚动及附件外观差异检查",
                    "info",
                )
            if "real_mail_partial_selected" in first_attachment.matches:
                selection_start = _StepResult(
                    "部分领取已开启",
                    None,
                    first_attachment,
                )
            elif "real_mail_partial_claim" in first_attachment.matches:
                selection_start = _StepResult(
                    "可以开启部分领取",
                    None,
                    first_attachment,
                )
            else:
                self._emit("部分领取控件状态不确定，进入恢复识别", "warning")
                selection_start = self._wait_step_outcome(
                    self.game,
                    (partial_selected, partial_available, claim_complete),
                    8,
                    "邮件部分领取状态复核",
                    include_ocr=True,
                )
            if selection_start.outcome == "领取完成":
                self._finish_mail_claim(selection_start, inbox_outcome, home_complete)
                return
            if selection_start.outcome == "部分领取已开启":
                selection_mode = selection_start.observation
                self._emit("用户已提前开启部分领取，从当前选择状态继续", "warning")
            else:
                game = self._require_bound(self.game, "游戏")
                self._click_match(
                    game,
                    selection_start.observation,
                    ("real_mail_partial_claim",),
                    "mail_partial_claim",
                    "部分领取",
                    require_match=True,
                )
                selected_result = self._wait_step_outcome_with_delayed_recovery(
                    self.game,
                    (partial_selected,),
                    (claim_complete,),
                    30,
                    "邮件部分领取选择状态",
                    include_ocr=True,
                )
                if selected_result.outcome == "领取完成":
                    self._finish_mail_claim(
                        selected_result,
                        inbox_outcome,
                        home_complete,
                    )
                    return
                selection_mode = selected_result.observation

        game = self._require_bound(self.game, "游戏")
        # The partial-claim state is a second chance to classify the first
        # card when the tooltip frame did not expose a usable OCR result.
        if not first_is_gear and self._mail_first_attachment_is_gear(selection_mode):
            first_is_gear = True
            self._emit("部分领取状态确认首格为胸挂或背包，取消弹药滚动分支")
        if first_is_gear:
            selected_points = self._select_mail_front_gear_attachments(
                game,
                selection_mode,
            )
            selected = self._wait_attachment_selections(selected_points, 30)
        else:
            selection_mode = self._scroll_mail_attachments_to_bottom(game)
            selected_points, selected = self._select_mail_bottom_gear_attachments(
                game,
                selection_mode,
            )

        game = self._require_bound(self.game, "游戏")
        self._click_match(
            game,
            selected,
            ("real_mail_claim_button",),
            "mail_claim",
            "领取胸挂和背包",
            require_match=True,
        )
        claimed = self._wait_step_outcome_with_delayed_recovery(
            self.game,
            (claim_complete,),
            (),
            60,
            "领取完成提示（未确认前不开始下一轮）",
            include_ocr=True,
        )
        self._finish_mail_claim(claimed, inbox_outcome, home_complete)

    def _finish_mail_claim(
        self,
        current: _StepResult,
        inbox_outcome: _StepOutcome,
        home_complete: _StepOutcome,
    ) -> None:
        if current.outcome == "领取完成":
            game = self._require_bound(self.game, "游戏")
            self._press_key(game, "space", "空格：领取完成后继续")
            current = self._wait_step_outcome_with_delayed_recovery(
                self.game,
                (inbox_outcome,),
                (home_complete,),
                60,
                "领取后的邮件界面",
                include_ocr=True,
            )

        if current.outcome == "已返回行前备战":
            self._emit("用户已完成邮件返回步骤，从行前备战继续", "warning")
            return

        game = self._require_bound(self.game, "游戏")
        self._press_key(game, "escape", "Esc：返回行前备战")
        self._wait_step_outcome(
            self.game,
            (home_complete,),
            60,
            "下一轮行前备战",
            include_ocr=True,
        )

    @staticmethod
    def _mail_first_attachment_is_gear(observation: PageObservation) -> bool:
        """Identify the first attachment from its tooltip text, not its skin."""
        for region in observation.ocr_regions:
            text = region.text.replace(" ", "")
            if not any(keyword in text for keyword in MAIL_ATTACHMENT_GEAR_KEYWORDS):
                continue
            left, top, right, bottom = region.bounds
            center_x = (left + right) / 2 / 1920
            center_y = (top + bottom) / 2 / 1080
            if 0.25 <= center_x <= 0.72 and 0.18 <= center_y <= 0.78:
                return True
        return False

    @staticmethod
    def _mail_attachment_strip_signature(image: np.ndarray) -> np.ndarray:
        left, top, right, bottom = MAIL_ATTACHMENT_STRIP_RECT
        normalized = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_AREA)
        strip = normalized[top:bottom, left:right]
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (480, 80), interpolation=cv2.INTER_AREA)

    def _scroll_mail_attachments_to_bottom(self, game: WindowInfo) -> PageObservation:
        previous: np.ndarray | None = None
        stable_frames = 0
        latest_observation = PageObservation(PageType.UNKNOWN, 0.0)
        for step in range(1, MAIL_ATTACHMENT_SCROLL_MAX_STEPS + 1):
            self._check_stop()
            point = self.settings.action_points["mail_attachment_scroll"]
            self.input.scroll_normalized(game, point, MAIL_ATTACHMENT_SCROLL_DELTA)
            self.diagnostics.event(
                "input",
                action="scroll",
                delta=MAIL_ATTACHMENT_SCROLL_DELTA,
                point=point,
                step=step,
                hwnd=game.hwnd,
            )
            self._sleep(MAIL_ATTACHMENT_SCROLL_SETTLE_SECONDS)
            frame, latest_observation = self._observe(
                game,
                "game",
                include_ocr=False,
                template_names=MAIL_ATTACHMENT_TEMPLATE_NAMES,
            )
            signature = self._mail_attachment_strip_signature(frame.image)
            if previous is not None:
                difference = float(np.mean(cv2.absdiff(signature, previous)))
                if difference < 2.5:
                    stable_frames += 1
                else:
                    stable_frames = 0
            previous = signature
            if stable_frames >= MAIL_ATTACHMENT_SCROLL_STABLE_FRAMES:
                self._emit(f"邮件附件栏已滚动到底部（第 {step} 步画面稳定）")
                return latest_observation
        self._emit(
            f"邮件附件栏达到最大滚动步数 {MAIL_ATTACHMENT_SCROLL_MAX_STEPS}，按当前底部位置继续",
            "warning",
        )
        return latest_observation

    def _select_mail_front_gear_attachments(
        self,
        game: WindowInfo,
        observation: PageObservation,
    ) -> list[tuple[float, float]]:
        first_selected = "real_mail_first_selected" in observation.matches
        second_selected = "real_mail_second_selected" in observation.matches
        self._emit("首格名称识别为胸挂/背包，按前两格选择")
        selected_points = [
            self.settings.action_points["mail_attachment_first"],
            self.settings.action_points["mail_attachment_second"],
        ]
        if not first_selected:
            self._click(game, "mail_attachment_first", "勾选第一个附件（胸挂回退位置）")
        if not second_selected:
            self._click(game, "mail_attachment_second", "勾选第二个附件（背包回退位置）")
        return selected_points

    def _select_mail_bottom_gear_attachments(
        self,
        game: WindowInfo,
        observation: PageObservation,
    ) -> tuple[list[tuple[float, float]], PageObservation]:
        if "real_mail_first_selected" in observation.matches:
            self._click(game, "mail_attachment_first", "取消首格弹药选择")

        attempt = 0
        last_location_key: tuple[tuple[int, int, int, int], ...] | None = None
        candidate_confirmation_key: (
            tuple[tuple[tuple[int, int, int, int], ...], tuple[int, int]] | None
        ) = None
        candidate_confirmations = 0
        latest = observation
        while attempt < MAIL_ATTACHMENT_IDENTIFICATION_MAX_ATTEMPTS:
            self._check_stop()
            attempt += 1
            game = self._require_bound(self.game, "游戏")
            self.input.move_normalized(game, MAIL_ATTACHMENT_SAFE_CURSOR_POINT)
            self.diagnostics.event(
                "input",
                action="move",
                label="移出邮件附件栏以稳定识别",
                point=MAIL_ATTACHMENT_SAFE_CURSOR_POINT,
                hwnd=game.hwnd,
            )
            self._sleep(MAIL_ATTACHMENT_SCROLL_SETTLE_SECONDS)
            frame, latest = self._observe(
                game,
                "game",
                include_ocr=False,
                template_names=MAIL_ATTACHMENT_TEMPLATE_NAMES,
            )
            cards, candidate_indices, reason = self._analyze_mail_attachment_cards(
                frame.image,
                latest,
            )
            location_key = tuple(card.bounds for card in cards)
            self.diagnostics.event(
                "mail_attachment_analysis",
                attempt=attempt,
                card_bounds=[card.bounds for card in cards],
                difference_scores=[round(card.difference_score, 2) for card in cards],
                candidate_indices=candidate_indices,
                reason=reason,
            )

            if len(candidate_indices) == 2:
                confirmation_key = (
                    location_key,
                    (candidate_indices[0], candidate_indices[1]),
                )
                if confirmation_key == candidate_confirmation_key:
                    candidate_confirmations += 1
                else:
                    candidate_confirmation_key = confirmation_key
                    candidate_confirmations = 1
                if candidate_confirmations < MAIL_ATTACHMENT_OUTLIER_CONFIRMATIONS:
                    if candidate_confirmations == 1:
                        self._emit(
                            "邮件附件发现两个装备候选；将保持当前位置连续复核，"
                            "不会按单帧结果领取",
                            "warning",
                        )
                    continue

                selected_cards = [cards[index] for index in candidate_indices]
                selected_points = [
                    card.center_normalized for card in selected_cards
                ]
                if location_key != last_location_key:
                    descriptions = "、".join(
                        f"第 {index + 1} 格 {cards[index].difference_score:.1f}"
                        for index in candidate_indices
                    )
                    self._emit(
                        f"邮件附件动态分格完成，共 {len(cards)} 格；"
                        f"非弹药候选为{descriptions}，按格子中心选择（{reason}）"
                    )
                last_location_key = location_key

                selection_scores = self._mail_attachment_selection_scores(
                    frame.image,
                    selected_points,
                )
                for index, (point, score) in enumerate(
                    zip(selected_points, selection_scores),
                    start=1,
                ):
                    if score >= 0.76:
                        continue
                    self._click_normalized(
                        game,
                        point,
                        f"勾选动态定位的非弹药附件 {index}",
                        source="mail_attachment_card_center",
                    )
                try:
                    selected = self._wait_attachment_selections(
                        selected_points,
                        MAIL_ATTACHMENT_SELECTION_RETRY_SECONDS,
                        save_timeout=False,
                    )
                except WorkflowTimeout:
                    if attempt == 1 or attempt % 5 == 0:
                        self._emit(
                            f"邮件附件第 {attempt} 次选择后尚未确认两个勾选；"
                            "将重新分格并只补点缺失项，不会点击领取或停止任务",
                            "warning",
                        )
                    continue
                return selected_points, selected

            candidate_confirmation_key = None
            candidate_confirmations = 0
            if attempt == 1 or attempt % 5 == 0:
                self._emit(
                    f"邮件附件第 {attempt} 次动态识别未得到两个可靠非弹药格："
                    f"{reason}；继续向右滚动并重新识别，不会按固定位置误点",
                    "warning",
                )
            point = self.settings.action_points["mail_attachment_scroll"]
            self.input.scroll_normalized(game, point, MAIL_ATTACHMENT_SCROLL_DELTA)
            self.diagnostics.event(
                "input",
                action="scroll",
                delta=MAIL_ATTACHMENT_SCROLL_DELTA,
                point=point,
                recovery_attempt=attempt,
                hwnd=game.hwnd,
            )
            self._sleep(MAIL_ATTACHMENT_SCROLL_SETTLE_SECONDS)
        raise WorkflowTimeout(
            "邮件附件在有限重试内未能连续确认相邻的胸挂和背包；"
            "已停止领取，避免把子弹误领为装备",
            last_observation=latest,
        )

    @staticmethod
    def _mail_attachment_card_boundaries(image: np.ndarray) -> list[int]:
        normalized = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_AREA)
        left, _top, right, _bottom = MAIL_ATTACHMENT_STRIP_RECT
        gray = cv2.cvtColor(
            normalized[MAIL_ATTACHMENT_CARD_TOP:MAIL_ATTACHMENT_CARD_BOTTOM, left:right],
            cv2.COLOR_BGR2GRAY,
        )
        edge_strength = np.mean(
            np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)),
            axis=0,
        )
        threshold = max(100.0, float(np.percentile(edge_strength, 97)))
        peaks = [
            x
            for x in range(2, len(edge_strength) - 2)
            if edge_strength[x] >= threshold
            and edge_strength[x] >= np.max(edge_strength[x - 2 : x + 3])
        ]
        groups: list[list[int]] = []
        for peak in peaks:
            if groups and peak - groups[-1][-1] <= MAIL_ATTACHMENT_EDGE_GROUP_DISTANCE:
                groups[-1].append(peak)
            else:
                groups.append([peak])
        candidates = [
            max(group, key=lambda x: edge_strength[x]) + left for group in groups
        ]

        # Ignore strong edges inside an item image by selecting the longest
        # regularly spaced chain of card boundaries.
        best_paths: list[list[int]] = []
        for index, boundary in enumerate(candidates):
            path = [boundary]
            for previous_index in range(index):
                distance = boundary - candidates[previous_index]
                if not (
                    MAIL_ATTACHMENT_CARD_MIN_WIDTH
                    <= distance
                    <= MAIL_ATTACHMENT_CARD_MAX_WIDTH
                ):
                    continue
                candidate_path = best_paths[previous_index] + [boundary]
                if len(candidate_path) > len(path):
                    path = candidate_path
            best_paths.append(path)
        return max(best_paths, key=len, default=[])

    @classmethod
    def _analyze_mail_attachment_cards(
        cls,
        image: np.ndarray,
        observation: PageObservation | None = None,
    ) -> tuple[list[_MailAttachmentCard], list[int], str]:
        normalized = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_AREA)
        boundaries = cls._mail_attachment_card_boundaries(normalized)
        if len(boundaries) < 5:
            return [], [], f"只检测到 {max(0, len(boundaries) - 1)} 个完整附件格"

        card_images: list[np.ndarray] = []
        card_bounds: list[tuple[int, int, int, int]] = []
        for left, right in zip(boundaries, boundaries[1:]):
            bounds = (
                left,
                MAIL_ATTACHMENT_CARD_TOP,
                right,
                MAIL_ATTACHMENT_CARD_BOTTOM,
            )
            inner = normalized[
                MAIL_ATTACHMENT_CARD_TOP + 9 : MAIL_ATTACHMENT_CARD_BOTTOM - 8,
                left + 6 : right - 6,
            ]
            if inner.size == 0:
                continue
            card_bounds.append(bounds)
            gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
            card_images.append(
                cv2.resize(gray, (72, 64), interpolation=cv2.INTER_AREA)
            )
        if len(card_images) < 4:
            return [], [], f"只检测到 {len(card_images)} 个可比较附件格"

        pairwise = np.asarray(
            [
                [
                    float(
                        np.mean(
                            np.abs(
                                first.astype(np.float32)
                                - second.astype(np.float32)
                            )
                        )
                    )
                    for second in card_images
                ]
                for first in card_images
            ],
            dtype=np.float32,
        )
        medoid_index = int(np.argmin(np.median(pairwise, axis=1)))
        scores = pairwise[medoid_index]
        cards = [
            _MailAttachmentCard(bounds=bounds, difference_score=float(score))
            for bounds, score in zip(card_bounds, scores)
        ]
        order = np.argsort(scores)
        candidate_indices = sorted(int(index) for index in order[-2:])
        if candidate_indices[1] != candidate_indices[0] + 1:
            return cards, [], "两个外观异常格不相邻，不能确认为胸挂和背包"
        baseline = scores[order[:-2]]
        baseline_max = float(np.max(baseline))
        baseline_median = float(np.median(baseline))
        baseline_mad = float(np.median(np.abs(baseline - baseline_median)))
        lower_outlier_score = min(scores[index] for index in candidate_indices)
        required_score = max(
            MAIL_ATTACHMENT_OUTLIER_MIN_SCORE,
            baseline_median + max(6.0, 4.0 * baseline_mad),
        )
        gap = float(lower_outlier_score - baseline_max)
        ratio = float(lower_outlier_score / max(1.0, baseline_max))
        if (
            lower_outlier_score < required_score
            or gap < MAIL_ATTACHMENT_OUTLIER_MIN_GAP
            or ratio < 1.5
        ):
            return (
                cards,
                [],
                f"异常格断层不足（候选 {lower_outlier_score:.1f}，"
                f"弹药上限 {baseline_max:.1f}，差值 {gap:.1f}）",
            )

        reason = (
            f"外观差值 {scores[candidate_indices[0]]:.1f}/"
            f"{scores[candidate_indices[1]]:.1f}，弹药上限 {baseline_max:.1f}"
        )
        return cards, candidate_indices, reason

    def _prepare_game_for_trading_handoff(
        self,
        *,
        allow_game_restart: bool | None = None,
    ) -> None:
        if allow_game_restart is None:
            allow_game_restart = not getattr(self, "_page_restart_in_progress", False)
        if self._try_fast_return_to_firestorm_home():
            self._emit("已通过轻量模板确认烽火地带主界面")
            # A restart may have been requested while returning from the
            # market.  The fast path is still a successful arrival, so clear
            # the pending purpose before the next scheduled-restart check can
            # mistake it for an unfinished handoff.
            self._mark_resume_target_reached()
            return

        self._emit(
            "轻量页面路径无法返回主界面；转入统一页面导航异常恢复",
            "warning",
        )
        resume_target = self._resume_target_for_current_node()
        self._navigate_game_home_with_recovery(
            resume_target=resume_target,
            reason="返回可继续自动流程的游戏主界面",
            allow_game_restart=allow_game_restart,
        )

    def _resume_target_for_current_node(self) -> LoadoutResumeTarget:
        """Map the durable checkpoint node to the work to resume after restart."""
        checkpoint = getattr(self, "_loadout_checkpoint", None)
        if checkpoint is not None and checkpoint.resume_target is not None:
            return checkpoint.resume_target
        if checkpoint is not None:
            if checkpoint.stage in {
                LoadoutCheckpointStage.WAREHOUSE_SCAN,
                LoadoutCheckpointStage.LISTING_CONFIRM_PENDING,
                LoadoutCheckpointStage.LISTING_CONFIRMED,
                LoadoutCheckpointStage.WAITING_SALE,
                LoadoutCheckpointStage.SELL_SLOT_WAIT,
                LoadoutCheckpointStage.SELL_UNLIST_ALL,
                LoadoutCheckpointStage.UNLIST_CONFIRM_PENDING,
            }:
                return LoadoutResumeTarget.WAREHOUSE_SELL
            if checkpoint.stage is LoadoutCheckpointStage.MAIL_STORAGE:
                return LoadoutResumeTarget.MAIL_STORAGE
        return LoadoutResumeTarget.LOADOUT_SCAN

    def _navigate_game_home_with_recovery(
        self,
        *,
        resume_target: LoadoutResumeTarget,
        reason: str,
        allow_game_restart: bool = True,
    ) -> None:
        """Navigate to game home, then restart the game once if it remains stuck.

        The unified navigator performs the normal action and its single retry.
        For ``open_game_home`` the retry is delayed by two minutes.  If that
        still does not reach home, persist the intended continuation before
        reusing the existing game-process restart flow.
        """
        result = self._unified_navigation.navigate(
            GamePageId.GAME_HOME_PREPARE,
            acceptable_pages=frozenset({GamePageId.GAME_HOME_READY}),
            reason=reason,
        )
        if result.succeeded:
            self._emit("统一导航已恢复到烽火地带主界面")
            self._mark_resume_target_reached()
            return

        if result.status is NavigationStatus.STOPPED:
            raise WorkflowStopped()

        # A missing route or ambiguous window requires operator input; a game
        # restart cannot safely resolve either condition.  Only a bounded
        # action/recovery failure is eligible for the automatic game restart.
        if result.status not in {
            NavigationStatus.FAILED,
            NavigationStatus.RETRYABLE,
        } or not allow_game_restart:
            raise RuntimeError(result.message)

        current_page = (
            result.current_page.value if result.current_page is not None else None
        )
        self._save_loadout_checkpoint(
            LoadoutCheckpointStage.RETURN_HOME_PENDING,
            resume_target=resume_target,
            pending_action="open_game_home",
            recovery_reason=result.message,
            last_confirmed_page=current_page,
            last_safe_action="返回游戏主界面动作两次未完成，准备重启游戏恢复",
        )
        self._emit(
            f"返回游戏主界面失败：{result.message}；"
            "已保存重启目的，开始重启游戏进程",
            "warning",
        )
        self._restart_game_without_mail_resource_cycle(resume_target)

    def _mark_resume_target_reached(self) -> None:
        checkpoint = getattr(self, "_loadout_checkpoint", None)
        if checkpoint is None or checkpoint.resume_target is None:
            return
        stage = checkpoint.stage
        if stage is LoadoutCheckpointStage.RETURN_HOME_PENDING:
            stage = LoadoutCheckpointStage.PURCHASE_SCAN
        self._save_loadout_checkpoint(
            stage,
            resume_target=None,
            pending_action="",
            recovery_reason="",
            last_confirmed_page=GamePageId.GAME_HOME_READY.value,
            last_safe_action="已确认重启后返回游戏主界面，继续原任务",
        )

    def _return_home_after_warehouse_sell(self) -> None:
        self._emit(
            "交易行上架完成，等待 2 秒让出售页稳定后点击顶部“开始游戏”"
        )
        self._sleep(MARKET_SELL_HOME_CLICK_DELAY_SECONDS)
        self._check_stop()
        if getattr(self, "_unified_navigation", None) is not None:
            self._save_loadout_checkpoint(
                LoadoutCheckpointStage.RETURN_HOME_PENDING,
                resume_target=LoadoutResumeTarget.RETURN_HOME_THEN_SCAN,
                pending_action="open_game_home",
                recovery_reason="",
                last_confirmed_page=GamePageId.MARKET_SELL.value,
                last_safe_action="交易行上架完成，准备返回游戏主界面",
            )
            self._navigate_game_home_with_recovery(
                resume_target=LoadoutResumeTarget.RETURN_HOME_THEN_SCAN,
                reason="交易行上架完成后返回可继续自动流程的游戏主界面",
            )
            return

        # Keep this narrow fallback for lightweight unit-test controllers that
        # are intentionally constructed without a navigation object.
        game = self._require_bound(self.game, "游戏")
        self._click(
            game,
            "game_home_tab",
            "交易行顶部“开始游戏”",
        )

    def _try_fast_return_to_firestorm_home(self) -> bool:
        home_pages = {PageType.GAME_HOME_PREPARE, PageType.GAME_HOME_READY}
        escape_pages = {
            PageType.MAP_OVERVIEW,
            PageType.MAP_SELECTION,
            PageType.MAP_OTHER,
            PageType.MAP_LONGBOW_START,
            PageType.MAP_LONGBOW_DOWNLOAD,
            PageType.MAP_ZERO_DAM_START,
            PageType.LOADOUT,
            PageType.LOADOUT_PURCHASE_CONFIRM,
            PageType.LOADOUT_PRICE_CHANGE,
            PageType.LOADOUT_SCHEMES,
            PageType.MAIL_SCHEME_SELECTED,
            PageType.MODE_MENU,
            PageType.MAIL_INBOX,
        }
        observable_pages = home_pages | escape_pages | {PageType.ACTIVITY_REMINDER}

        # These pages have a known reversible route. Keep the normal path on the
        # lightweight Mail templates and reserve cross-workflow OCR for failures.
        for step in range(4):
            try:
                page = self._wait_page(
                    self.game,
                    observable_pages,
                    4,
                    f"快速返回主界面第 {step + 1} 步",
                    save_timeout=False,
                )
            except WorkflowTimeout:
                return False
            if page.page_type in home_pages:
                return True

            game = self._require_bound(self.game, "游戏")
            if page.page_type == PageType.ACTIVITY_REMINDER:
                self._emit("快速返回主界面：检测到活动提醒，按空格继续")
                self._press_key(game, "space", "空格：关闭活动提醒")
            else:
                self._emit(
                    f"快速返回主界面：当前为{page_type_label(page.page_type)}，按 Esc 返回"
                )
                self._press_key(game, "escape", "Esc：沿已知页面路径返回主界面")
        return False

    def _prepare_mss_window(self, window):
        deadline = getattr(self, '_startup_capture_deadline', None)
        if deadline is None:
            self.input.activate(window)
        else:
            self.input.activate(window, timeout_seconds=max(0, deadline - self._monotonic()),
                                check_stop=self._check_stop)

    def _wait_startup_resource(self, binding, expected, recovery, timeout, label, **kwargs):
        from .startup_wait import StartupWait

        wait = StartupWait(self, binding, timeout)
        previous = getattr(self, '_startup_capture_deadline', None)
        self._startup_capture_deadline = wait.deadline
        outcome = 'failed'
        try:
            result = self._wait_step_outcome(binding, (*expected, *recovery), timeout, label,
                                             startup_wait=wait, **kwargs)
            outcome = result.outcome
            return result
        except WorkflowStopped:
            outcome = 'stopped'
            raise
        except WorkflowTimeout:
            outcome = 'timeout'
            raise
        except Exception as exc:
            wait.last_error = f"{type(exc).__name__}: {exc}"
            wait.event('startup_unrecoverable_error', error=wait.last_error)
            raise
        finally:
            self._startup_capture_deadline = previous
            wait.finish(outcome)

    def _wait_step_outcome(
        self,
        binding: WindowBindingManager,
        outcomes: tuple[_StepOutcome, ...],
        timeout: float,
        label: str,
        *,
        include_ocr: bool = False,
        ocr_interval: int = 4,
        template_names: frozenset[str] | None = None,
        ocr_crop: tuple[float, float, float, float] | None = None,
        save_timeout: bool = True,
        startup_wait=None,
        observation_interval_seconds: float | None = None,
    ) -> _StepResult:
        """Wait for any valid next state without assuming one fixed UI sequence."""

        if ocr_interval < 1:
            raise ValueError("OCR 观察间隔必须大于零")

        def accept_template(page: PageObservation) -> bool:
            matching = [
                outcome for outcome in outcomes
                if (not outcome.pages or page.page_type in outcome.pages)
                and outcome.templates.issubset(page.matches)
            ]
            if any(outcome.read_text for outcome in matching):
                return False
            return any(
                outcome.predicate is None and not outcome.read_text
                for outcome in matching
            )

        started = self._monotonic()
        scan_count = 0
        latest_frame: CapturedFrame | None = None
        latest_observation = PageObservation(PageType.UNKNOWN, 0.0)
        while True:
            self._check_stop()
            if startup_wait is not None and self._monotonic() >= startup_wait.deadline:
                raise WorkflowTimeout(f"等待{label}超时；窗口恢复次数 {startup_wait.failures}；"
                                      f"最后异常：{startup_wait.last_error or '无'}")
            window = startup_wait.refresh() if startup_wait is not None else binding.refresh()
            if window is None:
                self._sleep(min(0.5, max(0, startup_wait.deadline - self._monotonic()))
                            if startup_wait is not None else 0.5)
            else:
                scan_count += 1
                observe_options: dict[str, object] = {}
                if include_ocr:
                    observe_options["accept_template"] = accept_template
                if template_names is not None:
                    observe_options["template_names"] = template_names
                if ocr_crop is not None:
                    observe_options["ocr_crop"] = ocr_crop
                observe = startup_wait.observe if startup_wait is not None else self._observe
                use_ocr = include_ocr and scan_count % ocr_interval == 0
                latest_frame, latest_observation = observe(
                    window,
                    "game" if binding is self.game else "launcher",
                    use_ocr,
                    **observe_options,
                )
                if startup_wait is not None and self._monotonic() >= startup_wait.deadline:
                    raise WorkflowTimeout(f"等待{label}超时；最后页面：{latest_observation.page_type.value}；"
                                          f"最后异常：{startup_wait.last_error or '无'}")
                matched_outcome: _StepOutcome | None = None
                for outcome in outcomes:
                    if include_ocr and outcome.templates and latest_observation.page_type == PageType.UNKNOWN:
                        continue
                    if (
                        outcome.pages
                        and latest_observation.page_type not in outcome.pages
                    ):
                        continue
                    if not outcome.templates.issubset(latest_observation.matches):
                        continue
                    if outcome.read_text and include_ocr and not use_ocr:
                        # A later generic outcome must not bypass the content
                        # read required by this matching, higher-priority one.
                        break
                    if (
                        outcome.predicate is not None
                        and not outcome.predicate(latest_frame, latest_observation)
                    ):
                        continue
                    matched_outcome = outcome
                    break

                if matched_outcome is not None:
                    return _StepResult(
                        matched_outcome.name,
                        latest_frame,
                        latest_observation,
                    )

            if self._monotonic() - started >= timeout:
                if latest_frame is not None and save_timeout:
                    path = self.diagnostics.save_frame(
                        f"timeout_{label}",
                        latest_frame.image,
                    )
                    self._emit(f"已保存超时截图：{path}", "warning")
                expected = "、".join(outcome.name for outcome in outcomes)
                raise WorkflowTimeout(
                    f"等待{label}超时；允许结果：{expected}；"
                    f"最后识别为{page_type_label(latest_observation.page_type)}。"
                    "界面可能被跳过、被用户操作，或仍在切换中",
                    last_observation=latest_observation,
                )
            delay = (observation_interval_seconds if observation_interval_seconds is not None else self.settings.observation_interval_seconds)
            if startup_wait is not None:
                delay = min(delay, max(0, startup_wait.deadline - self._monotonic()))
            self._sleep(delay)

    @staticmethod
    def _ocr_confirms_outcome(
        outcome: _StepOutcome,
        observation: PageObservation,
    ) -> bool:
        if (
            not outcome.single_ocr_confirmation_terms
            or not observation.ocr_texts
        ):
            return False
        normalized = "".join("".join(observation.ocr_texts).split())
        return any(
            term in normalized
            for term in outcome.single_ocr_confirmation_terms
        )

    def _wait_step_or_recover(
        self,
        binding: WindowBindingManager,
        outcomes: tuple[_StepOutcome, ...],
        timeout: float,
        label: str,
        *,
        recovery_pages: set[PageType],
        **kwargs,
    ) -> _StepResult:
        result = self._wait_step_outcome(
            binding, (*outcomes, _StepOutcome("转入页面恢复", pages=frozenset(recovery_pages))),
            timeout, label, **kwargs,
        )
        if result.outcome == "转入页面恢复":
            raise WorkflowTimeout(
                f"{label}期间页面已变化，立即按当前页面恢复",
                last_observation=result.observation,
            )
        return result

    def _wait_step_outcome_with_delayed_recovery(
        self,
        binding: WindowBindingManager,
        expected_outcomes: tuple[_StepOutcome, ...],
        recovery_outcomes: tuple[_StepOutcome, ...],
        timeout: float,
        label: str,
        *,
        recovery_timeout: float = 15,
        include_ocr: bool = False,
        ocr_interval: int = 4,
        template_names: frozenset[str] | None = None,
        recovery_template_names: frozenset[str] | None = None,
        ocr_crop: tuple[float, float, float, float] | None = None,
        save_timeout: bool = True,
    ) -> _StepResult:
        """Observe safe recovery pages immediately, with one final bounded retry.

        Callers must not list unchanged pre-action pages as proof of completion.
        """
        try:
            return self._wait_step_outcome(
                binding,
                (*expected_outcomes, *recovery_outcomes),
                timeout,
                label,
                include_ocr=include_ocr,
                ocr_interval=ocr_interval,
                template_names=recovery_template_names or template_names,
                ocr_crop=ocr_crop,
                save_timeout=save_timeout,
            )
        except WorkflowTimeout:
            self._emit(
                f"等待{label}超时；进入异常恢复，"
                "检查页面是否已被用户操作或跳过",
                "warning",
            )
        return self._wait_step_outcome(
            binding,
            (*expected_outcomes, *recovery_outcomes),
            recovery_timeout,
            f"{label}异常恢复",
            include_ocr=include_ocr,
            ocr_interval=ocr_interval,
            template_names=recovery_template_names or template_names,
            ocr_crop=ocr_crop,
            save_timeout=save_timeout,
        )

    def _wait_matches(
        self,
        binding: WindowBindingManager,
        required: tuple[str, ...],
        timeout: float,
        label: str,
    ) -> PageObservation:
        started = self._monotonic()
        latest_frame: CapturedFrame | None = None
        latest_observation = PageObservation(PageType.UNKNOWN, 0.0)
        while True:
            self._check_stop()
            window = binding.refresh()
            if window is None:
                self._sleep(0.5)
            else:
                latest_frame, latest_observation = self._observe(
                    window,
                    "game" if binding is self.game else "launcher",
                    include_ocr=False,
                )
                if all(name in latest_observation.matches for name in required):
                    return latest_observation
            if self._monotonic() - started >= timeout:
                if latest_frame is not None:
                    path = self.diagnostics.save_frame(f"timeout_{label}", latest_frame.image)
                    self._emit(f"已保存超时截图：{path}", "warning")
                missing = "、".join(name for name in required if name not in latest_observation.matches)
                raise WorkflowTimeout(f"等待{label}超时，缺少模板：{missing}")
            self._sleep(self.settings.observation_interval_seconds)

    def _wait_attachment_selections(
        self,
        selected_points: list[tuple[float, float]],
        timeout: float,
        *,
        save_timeout: bool = True,
    ) -> PageObservation:
        started = self._monotonic()
        latest_frame: CapturedFrame | None = None
        latest_observation = PageObservation(PageType.UNKNOWN, 0.0)
        while True:
            self._check_stop()
            window = self.game.refresh()
            if window is not None:
                latest_frame, latest_observation = self._observe(
                    window,
                    "game",
                    include_ocr=False,
                )
                scores = self._mail_attachment_selection_scores(
                    latest_frame.image,
                    selected_points,
                )
                if len(scores) == 2 and all(score >= 0.76 for score in scores):
                    return latest_observation
            if self._monotonic() - started >= timeout:
                if save_timeout and latest_frame is not None:
                    path = self.diagnostics.save_frame(
                        "timeout_胸挂和背包勾选状态",
                        latest_frame.image,
                    )
                    self._emit(f"已保存超时截图：{path}", "warning")
                raise WorkflowTimeout("未确认胸挂和背包均已勾选")
            self._sleep(self.settings.observation_interval_seconds)

    def _mail_attachment_selection_scores(
        self,
        image: np.ndarray,
        selected_points: list[tuple[float, float]],
    ) -> list[float]:
        template = self.vision.catalog.images["real_mail_first_selected"]
        normalized = cv2.resize(
            image,
            (1920, 1080),
            interpolation=cv2.INTER_AREA,
        )
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        scores: list[float] = []
        for point_x, point_y in selected_points:
            center_x = int(round(point_x * 1920 + 30))
            center_y = int(round(point_y * 1080 - 32))
            left = max(0, center_x - 25)
            top = max(0, center_y - 25)
            right = min(1920, center_x + 26)
            bottom = min(1080, center_y + 26)
            search = normalized[top:bottom, left:right]
            if (
                search.shape[0] < template.shape[0]
                or search.shape[1] < template.shape[1]
            ):
                scores.append(0.0)
                continue
            result = cv2.matchTemplate(
                cv2.cvtColor(search, cv2.COLOR_BGR2GRAY),
                template_gray,
                cv2.TM_CCOEFF_NORMED,
            )
            scores.append(float(cv2.minMaxLoc(result)[1]))
        return scores

    def _start_initial_loadout_with_recovery(self) -> PageObservation:
        downstream_pages = {
            PageType.LOADOUT,
            PageType.ENTRY_WARNING,
            PageType.GAME_HOME_READY,
            PageType.AGENT_SELECT,
            PageType.MATCHING,
            PageType.GAME_LOADING,
        }
        current = self._wait_step_outcome_with_delayed_recovery(
            self.game,
            (
                _StepOutcome(
                    "长弓溪谷开始行动",
                    pages=frozenset({PageType.MAP_LONGBOW_START}),
                ),
            ),
            (
                _StepOutcome(
                    "用户已进入初次配装后续页面",
                    pages=frozenset(downstream_pages),
                ),
            ),
            45,
            "长弓溪谷开始行动",
            include_ocr=True,
        )
        if current.observation.page_type == PageType.MAP_LONGBOW_START:
            game = self._require_bound(self.game, "游戏")
            self._click_match(
                game,
                current.observation,
                ("map_start_button",),
                "start_action",
                "长弓溪谷开始行动",
                require_match=True,
            )
            return self._complete_initial_loadout()

        self._emit(
            f"长弓溪谷开始行动已由用户完成，当前为"
            f"{page_type_label(current.observation.page_type)}",
            "warning",
        )
        return self._complete_initial_loadout(current.observation)

    def _start_longbow_download_with_recovery(
        self,
    ) -> PageObservation | None:
        downstream_pages = {
            PageType.MAP_LONGBOW_START,
            PageType.MAP_OVERVIEW,
            PageType.MAP_SELECTION,
            PageType.MAP_OTHER,
            PageType.MAP_ZERO_DAM_START,
            PageType.GAME_HOME_READY,
            PageType.LOADOUT,
        }
        current = self._wait_step_outcome_with_delayed_recovery(
            self.game,
            (
                _StepOutcome(
                    "长弓溪谷下载按钮",
                    pages=frozenset({PageType.MAP_LONGBOW_DOWNLOAD}),
                    predicate=lambda _frame, page: bool(
                        {"map_download_icon", "map_download_button"} & page.matches.keys()
                    ),
                ),
            ),
            (
                _StepOutcome(
                    "长弓溪谷正在下载",
                    pages=frozenset({PageType.MAP_LONGBOW_DOWNLOAD}),
                    templates=frozenset({"map_download_progress_icon"}),
                ),
                _StepOutcome(
                    "已处于下载步骤后续页面",
                    pages=frozenset(downstream_pages),
                ),
            ),
            45,
            "长弓溪谷下载按钮",
            include_ocr=True,
        )
        if current.outcome == "长弓溪谷下载按钮":
            game = self._require_bound(self.game, "游戏")
            self._click_match(
                game,
                current.observation,
                ("map_download_icon", "map_download_button"),
                "start_action",
                "下载长弓溪谷",
                require_match=True,
            )
            return None

        self._emit(
            f"已处于长弓下载步骤的后续页面，稍后仍需核验下载状态，当前为"
            f"{page_type_label(current.observation.page_type)}",
            "warning",
        )
        return current.observation

    def _complete_initial_loadout(
        self,
        current: PageObservation | None = None,
    ) -> PageObservation:
        progress_pages = {
            PageType.LOADOUT,
            PageType.ENTRY_WARNING,
            PageType.GAME_HOME_READY,
            PageType.AGENT_SELECT,
            PageType.MATCHING,
            PageType.GAME_LOADING,
        }
        recovery_pages = {
            PageType.AGENT_SELECT,
            PageType.MATCHING,
            PageType.GAME_LOADING,
        }
        page = current
        if page is None:
            page = self._wait_step_outcome_with_delayed_recovery(
                self.game,
                (
                    _StepOutcome(
                        "初次配装进度",
                        pages=frozenset(progress_pages - recovery_pages),
                    ),
                ),
                (
                    _StepOutcome(
                        "用户已进入初次配装后续页面",
                        pages=frozenset(recovery_pages),
                    ),
                ),
                60,
                "初次配装或配装出发页",
                include_ocr=True,
            ).observation
        if page.page_type == PageType.LOADOUT:
            game = self._require_bound(self.game, "游戏")
            self._click(game, "confirm_loadout", "确认装配")
            page = self._wait_step_outcome_with_delayed_recovery(
                self.game,
                (
                    _StepOutcome(
                        "确认装配后的进度",
                        pages=frozenset(
                            {
                                PageType.ENTRY_WARNING,
                                PageType.GAME_HOME_READY,
                            }
                        ),
                    ),
                ),
                (
                    _StepOutcome(
                        "用户已进入确认装配后续页面",
                        pages=frozenset(recovery_pages),
                    ),
                ),
                60,
                "入局提醒或配装出发页",
                include_ocr=True,
            ).observation
        if page.page_type == PageType.ENTRY_WARNING:
            game = self._require_bound(self.game, "游戏")
            self._click_match(
                game,
                page,
                ("real_entry_warning_continue",),
                "entry_warning_continue",
                "仍要继续",
            )
            page = self._wait_step_outcome_with_delayed_recovery(
                self.game,
                (
                    _StepOutcome(
                        "入局提醒后的进度",
                        pages=frozenset({PageType.GAME_HOME_READY}),
                    ),
                ),
                (
                    _StepOutcome(
                        "用户已进入配装出发后续页面",
                        pages=frozenset(recovery_pages),
                    ),
                ),
                60,
                "配装出发页",
                include_ocr=True,
            ).observation
        return page

    @staticmethod
    def _confirmed_intro_progress(_frame, observation: PageObservation) -> bool:
        if observation.page_type not in INTRO_PROGRESS_PAGES:
            return False
        # A Space hint alone does not prove that the resource/mode sequence
        # has completed. Keep waiting for an independent page anchor.
        if observation.page_type == PageType.ACTIVITY_REMINDER:
            return "real_activity_header" in observation.matches
        return True

    def _confirm_restart_mode_selection(
        self, current: _StepResult | None,
    ) -> _StepResult | None:
        """A repeated identical dialog is not evidence that mode selection ran."""
        retries = 0
        if current is not None and current.observation.page_type in INTRO_PROGRESS_PAGES:
            if not self._confirmed_intro_progress(None, current.observation):
                current = None
        while True:
            if current is None:
                current = self._wait_step_outcome(
                    self.game,
                    (
                        _StepOutcome("模式选择", pages=frozenset({PageType.INITIAL_MODE_SELECTION})),
                        _StepOutcome("资源提示仍存在", pages=frozenset({PageType.MISSING_RESOURCE})),
                        _StepOutcome("已进入游戏页面", pages=frozenset(INTRO_PROGRESS_PAGES),
                                     predicate=self._confirmed_intro_progress),
                    ),
                    60, "资源确认后的模式选择或游戏页面", include_ocr=True,
                )
            page = current.observation
            if page.page_type == PageType.INITIAL_MODE_SELECTION:
                self._click_match(
                    self._require_bound(self.game, "游戏"), page,
                    ("season_2026_initial_mode_tabs", "real_initial_firestorm_card"),
                    "initial_firestorm_mode", "烽火地带",
                )
                return None
            if page.page_type in INTRO_PROGRESS_PAGES:
                return current
            if page.page_type != PageType.MISSING_RESOURCE:
                raise WorkflowTimeout("重启后的模式选择阶段无法确认", last_observation=page)
            if retries >= 2:
                raise WorkflowTimeout(
                    "资源提示重复出现，未确认经过模式选择；停止将其当作第二次提示",
                    last_observation=page,
                )
            retries += 1
            self._emit(f"模式选择尚未确认，重新确认当前资源提示 {retries}/2", "warning")
            self._click_match(
                self._require_bound(self.game, "游戏"), page,
                ("missing_resource_message",), "dialog_confirm", "重试当前资源提示确认",
            )
            current = None

    def _restore_pak_and_finish_intro(
        self,
        current: _StepResult | None,
    ) -> None:
        if current is not None and current.observation.page_type in INTRO_PROGRESS_PAGES:
            if not self._confirmed_intro_progress(None, current.observation):
                current = None
        for attempt in range(3):
            if current is None:
                current = self._wait_step_outcome(
                    self.game,
                    (
                        _StepOutcome("第二次资源缺失提示", pages=frozenset({PageType.MISSING_RESOURCE})),
                        _StepOutcome("模式选择仍存在", pages=frozenset({PageType.INITIAL_MODE_SELECTION})),
                        _StepOutcome("已进入游戏页面", pages=frozenset(INTRO_PROGRESS_PAGES),
                                     predicate=self._confirmed_intro_progress),
                    ),
                    90, "模式选择后的资源提示或游戏页面", include_ocr=True,
                )
            if current.observation.page_type != PageType.INITIAL_MODE_SELECTION:
                break
            if attempt == 2:
                raise WorkflowTimeout("选择烽火地带后仍未离开模式选择", last_observation=current.observation)
            self._click_match(
                self._require_bound(self.game, "游戏"), current.observation,
                ("season_2026_initial_mode_tabs", "real_initial_firestorm_card"),
                "initial_firestorm_mode", "重试选择烽火地带",
            )
            current = None

        self.pak.restore()
        pak_status = self.pak.inspect()
        self.diagnostics.event("pak", action="restore", status=pak_status)
        self._emit(
            f"PAK 已恢复：{self.pak.staged_display} → "
            f"{self.pak.source_display}（状态：{pak_status}）"
        )

        if current.observation.page_type == PageType.MISSING_RESOURCE:
            game = self._require_bound(self.game, "游戏")
            self._click_match(
                game,
                current.observation,
                ("missing_resource_message",),
                "dialog_confirm",
                "PAK 恢复后确认资源提示",
            )
            self._finish_optional_intro_pages()
            return

        self._emit(
            f"当前已处于资源提示的后续页面，当前为"
            f"{page_type_label(current.observation.page_type)}；"
            "PAK 恢复后从当前页面继续",
            "warning",
        )
        self._finish_optional_intro_pages(current.observation)

    def _finish_optional_intro_pages(
        self,
        current: PageObservation | None = None,
    ) -> None:
        unknown_recovery_actions = ("space", "tab", "escape")
        unknown_recovery_index = 0
        unknown_frames = 0

        def persistent_unknown(_frame, observation):
            nonlocal unknown_frames
            if observation.page_type == PageType.UNKNOWN:
                unknown_frames += 1
            else:
                unknown_frames = 0
            # Include a periodic OCR observation before trying a fallback key.
            return unknown_frames >= 4

        action_counts: dict[PageType, int] = {}
        for _step in range(12):
            if current is None:
                current = self._wait_step_outcome(
                    self.game,
                    (
                        _StepOutcome("连续未知页面", predicate=persistent_unknown),
                        _StepOutcome("可继续的介绍页面", pages=frozenset(INTRO_CONTINUE_PAGES)),
                        _StepOutcome("恢复后的模式或资源提示", pages=frozenset({
                            PageType.INITIAL_MODE_SELECTION, PageType.MISSING_RESOURCE,
                        })),
                        _StepOutcome("介绍流程已完成", pages=frozenset(INTRO_COMPLETE_PAGES)),
                    ),
                    180, "过渡页面或已完成页面", include_ocr=True,
                    ocr_interval=1,
                ).observation
            if current.page_type in INTRO_COMPLETE_PAGES:
                return
            if current.page_type == PageType.UNKNOWN:
                if unknown_recovery_index >= len(unknown_recovery_actions):
                    raise WorkflowTimeout(
                        "资源恢复后依次尝试空格、Tab、Esc，仍无法识别页面",
                        last_observation=current,
                    )
                key = unknown_recovery_actions[unknown_recovery_index]
                unknown_recovery_index += 1
                self._press_key(
                    self._require_bound(self.game, "游戏"),
                    key,
                    f"未知页面兜底：按{key}尝试关闭备用推广页（{unknown_recovery_index}/3）",
                )
                unknown_frames = 0
                current = None
                continue
            action_counts[current.page_type] = action_counts.get(current.page_type, 0) + 1
            if action_counts[current.page_type] > 3:
                raise WorkflowTimeout("重启页面重复操作后仍未推进", last_observation=current)
            game = self._require_bound(self.game, "游戏")
            if current.page_type == PageType.GAME_TRANSITION:
                self._press_key(game, "tab", "Tab：开始游戏")
            elif current.page_type == PageType.ACTIVITY_REMINDER:
                self._press_key(game, "space", "空格：跳过活动提醒")
            elif current.page_type == PageType.INITIAL_MODE_SELECTION:
                self._click_match(game, current,
                    ("season_2026_initial_mode_tabs", "real_initial_firestorm_card"),
                    "initial_firestorm_mode", "恢复后选择烽火地带")
            elif current.page_type == PageType.MISSING_RESOURCE:
                self._click_match(game, current, ("missing_resource_message",),
                    "dialog_confirm", "恢复后确认资源提示")
            else:
                raise WorkflowTimeout("重启恢复遇到未支持页面", last_observation=current)
            current = None
        raise WorkflowTimeout("重启页面恢复超过最大步骤数")

    def _navigate_to_longbow(
        self,
        expected: set[PageType],
    ) -> PageObservation:
        fast_result = self._try_fast_navigate_to_longbow(expected)
        if fast_result is not None:
            return fast_result

        self._emit(
            "长弓溪谷轻量导航未按预期到达；转入统一页面导航异常恢复",
            "warning",
        )
        expected_ids = frozenset(GamePageId(item.value) for item in expected)
        primary = (
            GamePageId.MAP_LONGBOW_START
            if GamePageId.MAP_LONGBOW_START in expected_ids
            else sorted(expected_ids, key=lambda item: item.value)[0]
        )
        result = self._unified_navigation.navigate(
            primary,
            acceptable_pages=expected_ids - {primary},
            reason="卡邮件流程进入长弓溪谷",
        )
        if not result.succeeded or result.current_page is None:
            raise RuntimeError(result.message)
        page_type = PageType(result.current_page.value)
        latest = self._last_unified_mail_observation
        if latest is not None and latest.page_type == page_type:
            return latest
        return PageObservation(page_type, 1.0)

    def _try_fast_navigate_to_longbow(
        self,
        expected: set[PageType],
    ) -> PageObservation | None:
        home_pages = {PageType.GAME_HOME_PREPARE, PageType.GAME_HOME_READY}
        scheme_pages = {
            PageType.LOADOUT_SCHEMES,
            PageType.MAIL_SCHEME_SELECTED,
        }
        map_detail_pages = {
            PageType.MAP_OVERVIEW,
            PageType.MAP_OTHER,
            PageType.MAP_ZERO_DAM_START,
            PageType.MAP_LONGBOW_START,
            PageType.MAP_LONGBOW_DOWNLOAD,
        }
        map_pages = {PageType.MAP_SELECTION, *map_detail_pages}
        try:
            current = self._wait_page(
                self.game,
                home_pages | scheme_pages | {PageType.LOADOUT} | map_pages,
                self._timing_settings().map_fast_path_timeout_seconds,
                "长弓溪谷轻量导航起点",
                save_timeout=False,
            )
        except WorkflowTimeout:
            return None

        if current.page_type in expected:
            self._emit("长弓溪谷轻量导航：当前已经位于目标页面")
            return current

        if current.page_type in scheme_pages:
            game = self._require_bound(self.game, "游戏")
            self._press_key(game, "escape", "Esc：从配装方案列表返回配装界面")
            try:
                current = self._wait_page(
                    self.game,
                    {PageType.LOADOUT},
                    self._timing_settings().map_fast_path_timeout_seconds,
                    "配装方案列表返回配装界面",
                    save_timeout=False,
                )
            except WorkflowTimeout:
                return None

        if current.page_type == PageType.LOADOUT:
            game = self._require_bound(self.game, "游戏")
            self._press_key(game, "escape", "Esc：从配装界面返回游戏主界面")
            try:
                current = self._wait_page(
                    self.game,
                    home_pages,
                    self._timing_settings().map_fast_path_timeout_seconds,
                    "配装界面返回游戏主界面",
                    save_timeout=False,
                )
            except WorkflowTimeout:
                return None

        if current.page_type in home_pages:
            game = self._require_bound(self.game, "游戏")
            self._click(game, "map_card", "打开端游对局地图")
            try:
                current = self._wait_page(
                    self.game,
                    map_pages,
                    self._timing_settings().map_fast_path_timeout_seconds,
                    "长弓溪谷地图入口",
                    save_timeout=False,
                )
            except WorkflowTimeout:
                return None
            if current.page_type in expected:
                return current

        if current.page_type in map_detail_pages:
            game = self._require_bound(self.game, "游戏")
            self._click(game, "map_expand", "返回地图全览")
            try:
                current = self._wait_page(
                    self.game,
                    {PageType.MAP_SELECTION},
                    self._timing_settings().map_fast_path_timeout_seconds,
                    "长弓溪谷地图全览",
                    save_timeout=False,
                )
            except WorkflowTimeout:
                return None

        if current.page_type != PageType.MAP_SELECTION:
            return None
        game = self._require_bound(self.game, "游戏")
        self._click(game, "longbow_node", "选择长弓溪谷")
        try:
            return self._wait_page(
                self.game,
                expected,
                self._timing_settings().map_fast_path_timeout_seconds,
                "长弓溪谷目标页面",
                save_timeout=False,
            )
        except WorkflowTimeout:
            return None

    def _press_key(self, window: WindowInfo, key: str, label: str) -> None:
        actions = {
            "escape": self.input.press_escape,
            "tab": self.input.press_tab,
            "space": self.input.press_space,
            "loadout_scheme": self.input.press_loadout_scheme,
        }
        action = actions.get(key)
        if action is None:
            raise ValueError(f"未知按键动作：{key}")
        self._emit(f"按键：{label}")
        action(window)
        self.diagnostics.event("input", action="key", key=key, label=label, hwnd=window.hwnd)
        self._sleep(self.settings.action_settle_seconds)

    def _precheck(self) -> None:
        if not self.settings.allow_input:
            raise RuntimeError("未启用真实鼠标键盘输入，请先使用观察模式或在界面中明确启用")
        if not self.settings.allow_file_operations:
            raise RuntimeError("未启用真实 PAK 文件操作")
        if self.pak.record_path.exists() or self.pak.inspect() in {
            "staged",
            "partial",
        }:
            self._emit("发现上次未完成的 PAK 事务，先执行恢复检查", "warning")
            self.pak.recover_if_needed()
        status = self.pak.inspect()
        if status != "ready":
            raise PakTransactionError(f"PAK 启动状态异常：{status}")

    def _wait_window(
        self,
        binding: WindowBindingManager,
        label: str,
        timeout: float | None = None,
    ) -> WindowInfo:
        from .windows import is_delta_force_splash_window

        started = self._monotonic()
        last_splash_report = float('-inf')
        while True:
            self._check_stop()
            if binding is self.game:
                candidates = enumerate_windows()
                window = binding.refresh(candidates)
                splash = [w for w in candidates if is_delta_force_splash_window(w)]
                if splash and window is None and self._monotonic() - last_splash_report >= 10:
                    elapsed = self._monotonic() - started
                    self._emit(f"检测到游戏启动画面窗口，暂不绑定；等待正式游戏窗口（已等待 {elapsed:.1f} 秒）")
                    self.diagnostics.event('game_splash_wait', label=label,
                        elapsed_seconds=elapsed,
                        remaining_seconds=max(0, timeout - elapsed) if timeout is not None else None,
                        excluded=[dict(hwnd=w.hwnd, title=w.title, class_name=w.class_name,
                                       pid=w.pid, process=w.process_name,
                                       client_size=[w.client_rect.width, w.client_rect.height],
                                       reason='splash_screen') for w in splash])
                    last_splash_report = self._monotonic()
            else:
                window = binding.refresh()
            if binding is getattr(self, "launcher", None) and self._tick_launcher_recovery(window):
                self._sleep(.5)
                continue
            if window is not None:
                self.diagnostics.event(
                    "window_bound",
                    label=label,
                    hwnd=window.hwnd,
                    title=window.title,
                    process=window.process_name,
                )
                return window
            if binding.ambiguous:
                raise RuntimeError(f"{label}存在多个相似窗口，请回到主界面重新选择")
            if timeout is not None and self._monotonic() - started >= timeout:
                raise WorkflowTimeout(f"等待{label}窗口超时")
            self._sleep(0.5)

    def _terminate_game_process_tree(
        self,
        game: WindowInfo | None,
        *,
        reason: str,
        label: str,
    ) -> ProcessTreeTermination:
        termination = terminate_game_process_tree(
            game.hwnd if game is not None else None
        )
        if termination.kill_errors:
            self._emit(
                f"{label}时部分进程首次结束失败，将复查残留并检查启动器恢复条件："
                f"{termination.kill_errors}",
                "warning",
            )
        self._emit(
            f"{label}：已向根 PID {termination.root_pid} 及其进程树发送强制结束，"
            f"共 {len(termination.processes)} 个进程"
        )
        self.diagnostics.event(
            "process",
            action="terminate_game_tree",
            reason=reason,
            root_pid=termination.root_pid,
            process_count=len(termination.processes),
            attempted_pids=list(termination.kill_attempted_pids),
            errors=termination.kill_errors,
        )
        return termination

    def _wait_window_gone(
        self,
        binding: WindowBindingManager,
        timeout: float,
        *,
        expected_processes: list[ProcessRecord] | tuple[ProcessRecord, ...] = (),
    ) -> None:
        started = self._monotonic()
        launcher = getattr(self, "launcher", None)
        fingerprint = getattr(launcher, "fingerprint", None)
        path = getattr(fingerprint, "process_path", "")
        if isinstance(path, str) and launcher is not None:
            self._launcher_exit_recovery = LauncherRecovery(path)
            if launcher.refresh() is None:
                self._launcher_exit_recovery.missing_since = started
        next_launcher_probe = started + 5
        cleanup = ProcessExitCleanup(expected_processes, started)
        while True:
            self._check_stop()
            remaining, retry = cleanup.poll(self._monotonic())
            if getattr(self, "_launcher_exit_recovery", None) is not None and self._monotonic() >= next_launcher_probe:
                launcher_window = self.launcher.refresh()
                running = False
                if launcher_window is not None and self._launcher_exit_recovery.closing is None:
                    _, page = self._observe(launcher_window, "launcher", include_ocr=True,
                                            ocr_crop=(.65, .72, .98, .98))
                    running = "ocr_launcher_game_running" in page.matches
                self._tick_launcher_recovery(launcher_window, running=running)
                next_launcher_probe = self._monotonic() + 5
            if retry is not None:
                self.diagnostics.event("process", action="retry_game_exit_cleanup", **retry)
                self._emit(
                    f"游戏退出残留清理 {retry['attempt']}/3："
                    f"再次请求结束 PID {retry['attempted_pids']}；"
                    f"复查仍存在 {retry['remaining_pids']}；错误 {retry['errors']}",
                    "warning",
                )
            if (
                binding.refresh() is None
                and not matching_process_running(binding.fingerprint)
                and not remaining
            ):
                return
            if self._monotonic() - started >= timeout:
                raise WorkflowTimeout(
                    f"等待游戏退出超时；残留 PID {[record.pid for record in remaining]}；"
                    f"已重试清理 {cleanup.attempts} 次，未确认完全退出"
                )
            self._sleep(0.5)

    def _tick_launcher_recovery(self, window, *, running: bool = False) -> bool:
        recovery = getattr(self, "_launcher_exit_recovery", None)
        if recovery is None:
            return False
        message = recovery.tick(self._monotonic(), visible=window is not None, running=running)
        if message:
            self._emit(f"启动器退出恢复：{message}", "warning")
            self.diagnostics.event("launcher_recovery", message=message, executable=recovery.executable)
            self.launcher.clear()
        return bool(message or recovery.closing is not None or running)

    def _wait_page(
        self,
        binding: WindowBindingManager,
        expected: set[PageType],
        timeout: float,
        label: str,
        include_ocr: bool = False,
        save_timeout: bool = True,
        stable_count: int = 1,
        allow_game_started: bool = False,
        require_launcher_start_ready: bool = False,
    ) -> PageObservation:
        if stable_count < 1:
            raise ValueError("页面稳定确认次数必须大于零")
        started = self._monotonic()
        consecutive_page: PageType | None = None
        consecutive_count = 0
        scan_count = 0
        latest_frame: CapturedFrame | None = None
        while True:
            self._check_stop()
            if (allow_game_started and binding is getattr(self, "launcher", None)
                    and expected == {PageType.LAUNCHER_HOME}
                    and self.game.refresh() is not None):
                return PageObservation(PageType.GAME_STARTUP, 1.0)
            window = binding.refresh()
            if binding is getattr(self, "launcher", None) and self._tick_launcher_recovery(window):
                self._sleep(.5)
                continue
            if window is None:
                consecutive_page = None
                consecutive_count = 0
                self._sleep(0.5)
                if self._monotonic() - started >= timeout:
                    raise WorkflowTimeout(f"等待{label}时窗口丢失")
                continue
            scan_count += 1
            use_ocr = include_ocr and scan_count % 4 == 0
            observe_options = {}
            if include_ocr:
                observe_options["accept_template"] = lambda page: (
                    page.page_type in expected
                    and (not require_launcher_start_ready
                         or "ocr_launcher_start_button" in page.matches)
                )
            latest_frame, observation = self._observe(
                window, "game" if binding is self.game else "launcher", use_ocr,
                **observe_options,
            )
            if binding is getattr(self, "launcher", None) and self._tick_launcher_recovery(
                window, running="ocr_launcher_game_running" in observation.matches,
            ):
                self._sleep(.5)
                continue
            launcher_start_ready = (
                not require_launcher_start_ready
                or "ocr_launcher_start_button" in observation.matches
            )
            if observation.page_type in expected and launcher_start_ready:
                if (
                    (
                        observation.page_type == PageType.LAUNCHER_RESOURCES
                        and "ocr_launcher_longbow_resource" in observation.matches
                    )
                    or (
                        observation.page_type == PageType.LAUNCHER_DELETE_CONFIRM
                        and "ocr_launcher_delete_confirm" in observation.matches
                    )
                ):
                    return observation
                if observation.page_type == consecutive_page:
                    consecutive_count += 1
                else:
                    consecutive_page = observation.page_type
                    consecutive_count = 1
                if consecutive_count >= stable_count:
                    return observation
            else:
                consecutive_page = None
                consecutive_count = 0
            if self._monotonic() - started >= timeout:
                if latest_frame is not None and save_timeout:
                    path = self.diagnostics.save_frame(f"timeout_{label}", latest_frame.image)
                    self._emit(f"已保存超时截图：{path}", "warning")
                raise WorkflowTimeout(
                    f"等待{label}超时，最后识别为{page_type_label(observation.page_type)}",
                    last_observation=observation,
                )
            self._sleep(self.settings.observation_interval_seconds)

    def _observe(
        self,
        window: WindowInfo,
        target: str,
        include_ocr: bool,
        template_names: frozenset[str] | None = None,
        ocr_crop: tuple[float, float, float, float] | None = None,
        accept_template: Callable[[PageObservation], bool] | None = None,
    ) -> tuple[CapturedFrame, PageObservation]:
        if window.minimized and target == "launcher":
            self.input.activate(window, check_stop=self._check_stop)
            frame = self.capture.capture(window)
        elif window.minimized:
            frame = CapturedFrame(np.zeros((1, 1, 3), dtype=np.uint8), time.time(), 0.0, 0.0)
        else:
            frame = self.capture.capture(window)
        observation = self._analyze_mail_frame(
            frame,
            window,
            target,
            include_ocr=include_ocr,
            template_names=template_names,
            ocr_crop=ocr_crop,
            accept_template=accept_template,
        )
        return frame, observation

    def _analyze_mail_frame(
        self,
        frame: CapturedFrame,
        window: WindowInfo,
        target: str,
        *,
        include_ocr: bool,
        template_names: frozenset[str] | None = None,
        ocr_crop: tuple[float, float, float, float] | None = None,
        accept_template: Callable[[PageObservation], bool] | None = None,
    ) -> PageObservation:
        observe_options: dict[str, object] = {}
        if template_names is not None:
            observe_options["template_names"] = template_names
        if ocr_crop is not None:
            observe_options["ocr_crop"] = ocr_crop
        if accept_template is not None:
            observe_options["accept_template"] = accept_template
        observation = self.vision.observe(
            frame,
            target=target,
            include_ocr=include_ocr,
            **observe_options,
        )
        unified = adapt_mail_observation(observation, target=target)
        previous = getattr(self, "_diagnostic_page_by_target", {})
        if (getattr(getattr(self, "capture", None), "backend", None) == "mss"
                and previous.get(target) != observation.page_type
                and observation.page_type not in {
            PageType.UNKNOWN, PageType.INVALID_FRAME,
        }):
            self.diagnostics.save_frame(f"page_{target}_{observation.page_type.value}", frame.image)
            previous[target] = observation.page_type
            self._diagnostic_page_by_target = previous
        self._last_page = unified.page_id.value
        self.on_frame(frame.image, observation, target)
        self.diagnostics.event(
            "observation",
            target=target,
            hwnd=window.hwnd,
            page=unified.page_id.value,
            page_label=unified.chinese_name,
            page_surface=unified.surface.value,
            page_evidence=unified.evidence,
            page_stable=unified.stable,
            confidence=round(unified.confidence, 4),
            frame_mean=round(frame.mean, 2),
            frame_std=round(frame.std, 2),
            ocr=observation.ocr_texts,
        )
        return observation

    def _observe_unified_game(self) -> UnifiedPageObservation:
        recognition_started = time.perf_counter()
        window = self.game.refresh()
        if window is None:
            if self.game.ambiguous:
                raise RuntimeError("游戏窗口存在多个相似候选，请手动选择")
            raise RuntimeError("游戏窗口已丢失")
        frame, mail_observation = self._observe(
            window,
            "game",
            include_ocr=False,
        )
        generation = getattr(self.game, "generation", 0)
        mail = adapt_mail_observation(
            mail_observation,
            target="game",
            window_generation=generation,
        )
        recognition_mode = "home_template_fast_path"
        if self._is_unified_home_fast_path(mail_observation):
            resolved = mail
        else:
            rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
            trading_observation = self._trading_page_recognizer.recognize(
                Image.fromarray(rgb)
            )
            trading = adapt_trading_observation(
                trading_observation,
                captured_at=frame.timestamp,
                window_generation=generation,
            )
            resolved = resolve_page_observations(trading, mail)
            recognition_mode = "trading_page_recognition"
            if resolved.page_id is GamePageId.UNKNOWN:
                mail_observation = self._analyze_mail_frame(
                    frame,
                    window,
                    "game",
                    include_ocr=True,
                )
                mail = adapt_mail_observation(
                    mail_observation,
                    target="game",
                    window_generation=generation,
                )
                resolved = resolve_page_observations(trading, mail)
                recognition_mode = "full_ocr_fallback"
        unified = self._unified_page_stability.update(resolved)
        self._last_page = unified.page_id.value
        self._last_unified_frame = frame
        self._last_unified_mail_observation = mail_observation
        recognition_ms = round(
            (time.perf_counter() - recognition_started) * 1000,
        )
        self.diagnostics.event(
            "unified_navigation_observation",
            page=unified.page_id.value,
            page_label=unified.chinese_name,
            confidence=round(unified.confidence, 4),
            stable=unified.stable,
            window_generation=unified.window_generation,
            evidence=unified.evidence,
            recognition_mode=recognition_mode,
            recognition_ms=recognition_ms,
        )
        return unified

    @staticmethod
    def _is_unified_home_fast_path(observation: PageObservation) -> bool:
        anchor_names = UNIFIED_HOME_FAST_PATH_ANCHORS.get(observation.page_type)
        if anchor_names is None:
            return False
        anchors = [
            observation.matches[name]
            for name in anchor_names
            if name in observation.matches
        ]
        return bool(
            observation.confidence >= UNIFIED_HOME_FAST_PATH_MIN_CONFIDENCE
            and any(
                anchor.score >= UNIFIED_HOME_FAST_PATH_MIN_CONFIDENCE
                for anchor in anchors
            )
        )

    def _capture_unified_navigation_failure(self, label: str) -> str | None:
        frame = self._last_unified_frame
        if frame is None:
            self._observe_unified_game()
            frame = self._last_unified_frame
        if frame is None:
            return None
        return str(self.diagnostics.save_frame(label, frame.image))

    def _click(
        self,
        window: WindowInfo,
        point_name: str,
        label: str,
        *,
        after_click: Callable[[], None] | None = None,
    ) -> None:
        point = self.settings.action_points[point_name]
        self._emit(f"点击：{label} @ {point}")
        self.input.click_normalized(window, point)
        self.diagnostics.event("input", action="click", label=label, point=point, hwnd=window.hwnd)
        if after_click is not None:
            after_click()
        self._sleep(self.settings.action_settle_seconds)

    def _click_normalized(
        self,
        window: WindowInfo,
        point: tuple[float, float],
        label: str,
        *,
        source: str,
    ) -> None:
        self._emit(f"点击：{label} @ {point}（来源：{source}）")
        self.input.click_normalized(window, point)
        self.diagnostics.event(
            "input",
            action="click",
            label=label,
            point=point,
            source=source,
            hwnd=window.hwnd,
        )
        self._sleep(self.settings.action_settle_seconds)

    def _click_match(
        self,
        window: WindowInfo,
        observation: PageObservation,
        template_names: tuple[str, ...],
        fallback_point_name: str,
        label: str,
        require_match: bool = False,
    ) -> None:
        point = None
        source = fallback_point_name
        for template_name in template_names:
            match = observation.matches.get(template_name)
            if match is not None and match.click_normalized is not None:
                point = match.click_normalized
                source = template_name
                break
        if point is None:
            if require_match:
                raise RuntimeError(f"未定位到“{label}”模板，已停止以避免误点")
            point = self.settings.action_points[fallback_point_name]
        self._emit(f"点击：{label} @ {point}（来源：{source}）")
        self.input.click_normalized(window, point)
        self.diagnostics.event(
            "input",
            action="click",
            label=label,
            point=point,
            source=source,
            hwnd=window.hwnd,
        )
        self._sleep(self.settings.action_settle_seconds)

    def _delete_longbow_launcher_resource(self) -> PageObservation | None:
        launcher_page = self._open_launcher_resources()
        for attempt in range(2):
            try:
                launcher_page = self._find_launcher_resource(launcher_page)
            except LauncherResourceAbsent:
                self._emit(
                    "已完整扫描资源列表，长弓溪谷资源已不存在；"
                    "按已由用户删除或此前已完成继续",
                    "warning",
                )
                return launcher_page

            launcher_settings = self._require_bound(
                self.launcher_settings,
                "启动器设置",
            )
            self._click_match(
                launcher_settings,
                launcher_page,
                ("ocr_launcher_longbow_resource",),
                "launcher_delete_longbow",
                "烽火地带-长弓溪谷删除按钮",
                require_match=True,
            )

            self._transition(WorkflowState.CONFIRM_DELETE, "12：确认删除长弓溪谷资源")
            try:
                result = self._wait_step_outcome(
                    self.launcher_settings,
                    (
                        _StepOutcome(
                            "删除资源确认框",
                            pages=frozenset({PageType.LAUNCHER_DELETE_CONFIRM}),
                            single_ocr_confirmation_terms=frozenset(
                                {"是否确定删除资源包"}
                            ),
                        ),
                        # OCR can positively identify the modal and its
                        # confirm button even while the classifier still sees
                        # the resource-list background.  Accept that signal as
                        # a safe fallback so the workflow can click 确定.
                        _StepOutcome(
                            "删除资源确认框（OCR兜底）",
                            templates=frozenset({"ocr_launcher_delete_confirm"}),
                        ),
                    ),
                    12,
                    "删除资源确认框",
                    include_ocr=True,
                    ocr_interval=1,
                    save_timeout=False,
                ).observation
            except WorkflowTimeout:
                if self.game.refresh() is not None:
                    self._emit(
                        "删除动作后游戏已由用户启动；从游戏重启阶段继续",
                        "warning",
                    )
                    return None
                self._emit(
                    "删除确认框未稳定出现；进入异常恢复，"
                    "检查确认框是否已被用户处理",
                    "warning",
                )
                try:
                    result = self._wait_step_outcome(
                        self.launcher_settings,
                        (
                            _StepOutcome(
                                "删除资源确认框",
                                pages=frozenset(
                                    {PageType.LAUNCHER_DELETE_CONFIRM}
                                ),
                            ),
                            _StepOutcome(
                                "删除资源确认框（OCR兜底）",
                                templates=frozenset({"ocr_launcher_delete_confirm"}),
                            ),
                            _StepOutcome(
                                "删除后资源管理",
                                pages=frozenset(
                                    {PageType.LAUNCHER_RESOURCES}
                                ),
                                predicate=(
                                    self._launcher_resource_scan_is_unobstructed
                                ),
                            ),
                        ),
                        15,
                        "删除确认异常恢复",
                        include_ocr=True,
                        ocr_interval=1,
                        save_timeout=False,
                    ).observation
                except WorkflowTimeout:
                    if attempt >= 1:
                        raise WorkflowTimeout(
                            "两次点击长弓溪谷删除按钮后仍无法确认删除结果"
                        )
                    self._emit(
                        "异常恢复仍未确认删除结果；"
                        "重新打开资源管理并核对",
                        "warning",
                    )
                    launcher_page = self._open_launcher_resources()
                    continue

            if result.page_type == PageType.LAUNCHER_RESOURCES:
                self._emit(
                    "删除确认框可能已由用户处理，重新扫描资源列表确认结果",
                    "warning",
                )
                try:
                    launcher_page = self._find_launcher_resource(result)
                except LauncherResourceAbsent:
                    self._emit(
                        "完整扫描后长弓溪谷资源已不存在；"
                        "确认用户已完成删除",
                        "warning",
                    )
                    return result
                if attempt >= 1:
                    raise WorkflowTimeout(
                        "资源列表中仍存在长弓溪谷，删除结果未确认"
                    )
                continue

            launcher_settings = self._require_bound(
                self.launcher_settings,
                "启动器设置",
            )
            self._click_match(
                launcher_settings,
                result,
                ("ocr_launcher_delete_confirm", "real_launcher_delete_confirm"),
                "launcher_confirm_delete",
                "确认删除资源",
                require_match=True,
            )
            launcher_page = self._wait_step_outcome(
                self.launcher_settings,
                (
                    _StepOutcome(
                        "删除后的资源管理窗口",
                        pages=frozenset({PageType.LAUNCHER_RESOURCES}),
                        predicate=(
                            self._launcher_resource_scan_is_unobstructed
                        ),
                    ),
                ),
                45,
                "删除后的资源管理窗口",
                include_ocr=True,
                ocr_interval=1,
            ).observation

            # Deleted resources may remain listed as downloadable entries.
            # The confirmed modal closing is the normal-path completion signal.
            return launcher_page

        raise WorkflowTimeout("资源列表中仍存在长弓溪谷，删除结果未确认")

    def _find_launcher_resource(self, _initial: PageObservation) -> PageObservation:
        resource_name = "ocr_launcher_longbow_resource"
        launcher_settings = self._require_bound(self.launcher_settings, "启动器设置")
        frame, latest = self._observe(launcher_settings, "launcher_settings", include_ocr=True)
        self._require_unobstructed_launcher_resource_scan(latest)
        saw_longbow_name = any(
            VisionEngine._contains_launcher_longbow_text(text)
            for text in latest.ocr_texts
        )
        if resource_name in latest.matches:
            self._record_launcher_delete_location(frame, latest)
            return latest

        # Reset toward the top first, then scan downward in small steps. OCR
        # must identify the longbow row and a trash icon on that same row.
        scrolls = [120] * 6 + [-120] * 14
        for delta in scrolls:
            self._check_stop()
            launcher_settings = self._require_bound(self.launcher_settings, "启动器设置")
            self.input.scroll_normalized(launcher_settings, (0.62, 0.58), delta)
            self.diagnostics.event(
                "input", action="scroll", delta=delta, hwnd=launcher_settings.hwnd
            )
            self._sleep(0.6)
            frame, latest = self._observe(
                launcher_settings,
                "launcher_settings",
                include_ocr=True,
            )
            self._require_unobstructed_launcher_resource_scan(latest)
            saw_longbow_name = saw_longbow_name or any(
                VisionEngine._contains_launcher_longbow_text(text)
                for text in latest.ocr_texts
            )
            if resource_name in latest.matches:
                self._record_launcher_delete_location(frame, latest)
                return latest
        if saw_longbow_name:
            raise RuntimeError(
                "OCR 识别到长弓溪谷资源名称，但未确认同一行删除按钮；"
                "已停止以避免误点"
            )
        raise LauncherResourceAbsent("完整扫描后未找到长弓溪谷资源")

    @staticmethod
    def _launcher_delete_confirmation_visible(
        observation: PageObservation,
    ) -> bool:
        if observation.page_type == PageType.LAUNCHER_DELETE_CONFIRM:
            return True
        if "ocr_launcher_delete_confirm" in observation.matches:
            return True
        combined = "".join(text.replace(" ", "") for text in observation.ocr_texts)
        return "确定删除资源包" in combined

    def _launcher_resource_scan_is_unobstructed(
        self,
        _frame: CapturedFrame,
        observation: PageObservation,
    ) -> bool:
        return self._launcher_resource_observation_is_unobstructed(observation)

    def _launcher_resource_observation_is_unobstructed(
        self,
        observation: PageObservation,
    ) -> bool:
        return (
            observation.page_type == PageType.LAUNCHER_RESOURCES
            and (bool(observation.ocr_texts) or bool({
                "launcher_resource_header", "launcher_resource_header_popup",
                "launcher_resource_menu_compact",
            } & observation.matches.keys()))
            and not self._launcher_delete_confirmation_visible(observation)
        )

    def _require_unobstructed_launcher_resource_scan(
        self,
        observation: PageObservation,
    ) -> None:
        if self._launcher_delete_confirmation_visible(observation):
            raise WorkflowTimeout(
                "资源列表扫描期间仍显示删除确认框；已停止，"
                "拒绝将被弹窗遮挡的资源行判定为已删除",
                last_observation=observation,
            )
        if not self._launcher_resource_observation_is_unobstructed(observation):
            raise WorkflowTimeout(
                "资源列表扫描画面未通过完整 OCR 校验；已停止，"
                "无法确认长弓溪谷资源是否存在",
                last_observation=observation,
            )

    def _record_launcher_delete_location(
        self,
        frame: CapturedFrame,
        observation: PageObservation,
    ) -> None:
        match = observation.matches["ocr_launcher_longbow_resource"]
        path = self.diagnostics.save_frame("before_delete_longbow", frame.image)
        self._emit(
            f"OCR 已确认长弓资源行及垃圾桶，删除坐标：{match.click_normalized}；定位截图：{path}"
        )

    def _open_launcher_resources(self) -> PageObservation:
        # An existing "Settings" window may be on a different page. Always
        # use the fixed account menu route; the bottom shortcut icons vary by
        # launcher installation and currently available actions.
        launcher_timeout = self._timing_settings().launcher_window_timeout_seconds
        self._wait_page(
            self.launcher,
            {PageType.LAUNCHER_HOME},
            launcher_timeout,
            "启动器首页",
        )
        launcher = self._require_bound(self.launcher, "启动器")
        self._click(launcher, "launcher_account_button", "打开右侧账号菜单")
        self._click(
            launcher,
            "launcher_account_resource_entry",
            "从账号菜单打开资源管理",
        )
        settings_window = self._wait_launcher_settings_window(launcher_timeout)
        self._bind_launcher_settings_window(settings_window)
        self._log_launcher_settings_bound(settings_window)
        return self._wait_page(
            self.launcher_settings,
            {PageType.LAUNCHER_RESOURCES},
            launcher_timeout,
            "启动器设置中的资源管理",
            include_ocr=True,
        )

    def _find_launcher_settings_window(self) -> WindowInfo | None:
        launcher = self._require_bound(self.launcher, "启动器")
        minimum_size = self.launcher_settings.fingerprint.minimum_client_size
        candidates = [
            window
            for window in enumerate_windows()
            if window.hwnd != launcher.hwnd
            and window.client_rect.valid(minimum_size)
            and (
                window.pid == launcher.pid
                or self.launcher_settings.fingerprint.score(window) >= 95
            )
        ]
        if not candidates:
            return None
        return self.launcher_settings.refresh(candidates)

    def _wait_launcher_settings_window(self, timeout: float) -> WindowInfo:
        started = self._monotonic()
        while True:
            self._check_stop()
            window = self._find_launcher_settings_window()
            if window is not None:
                self._bind_launcher_settings_window(window)
                self._log_launcher_settings_bound(window)
                return window
            if self._monotonic() - started >= timeout:
                raise WorkflowTimeout("等待独立的启动器设置窗口超时")
            self._sleep(0.25)

    def _bind_launcher_settings_window(self, window: WindowInfo) -> None:
        fingerprint = self.launcher_settings.bind(window)
        fingerprint.minimum_client_size = (
            min(window.client_rect.width, 600),
            min(window.client_rect.height, 400),
        )

    def _log_launcher_settings_bound(self, window: WindowInfo) -> None:
        self._emit(
            f"已绑定启动器设置窗口 HWND 0x{window.hwnd:X}，客户区 "
            f"{window.client_rect.width}×{window.client_rect.height}"
        )
        self.diagnostics.event(
            "window_bound",
            label="启动器设置",
            hwnd=window.hwnd,
            title=window.title,
            process=window.process_name,
            client_size=[window.client_rect.width, window.client_rect.height],
        )

    def _require_bound(self, binding: WindowBindingManager, label: str) -> WindowInfo:
        window = binding.refresh()
        if window is None:
            raise RuntimeError(f"{label}窗口已失效")
        return window

    def _transition(self, state: WorkflowState, message: str, level: str = "info") -> None:
        self._state = state
        self.on_event(WorkflowEvent(state, message, level))
        self.diagnostics.event("state", state=state.value, message=message, level=level)

    def _emit(self, message: str, level: str = "info") -> None:
        self.on_event(WorkflowEvent(self._state, message, level))
        self.diagnostics.event("message", state=self._state.value, message=message, level=level)

    def _emit_structured(
        self,
        event_type: str,
        message: str,
        level: str = "info",
        **data: object,
    ) -> None:
        self.on_event(
            WorkflowEvent(
                self._state,
                message,
                level,
                event_type=event_type,
                data=data,
            )
        )
        self.diagnostics.event(
            event_type,
            state=self._state.value,
            message=message,
            level=level,
            **data,
        )

    def _emit_loadout_purchase_result(
        self,
        rule: LoadoutSchemeRule,
        listed_price: int,
        outcome: str,
        balance_before: int | None,
        balance_after: int | None,
        *,
        spent: int | None = None,
        actual_average: int | None = None,
    ) -> None:
        labels = {
            "full": "全部成交",
            "partial": "部分成交",
            "none": "完全没有成交",
            "unknown": "成交状态未知",
        }
        label = labels[outcome]
        balance_text = (
            f"余额 {balance_before:,} -> {balance_after:,}"
            if balance_before is not None and balance_after is not None
            else "余额变化未完整识别"
        )
        self._emit_structured(
            "loadout_purchase_result",
            f"配装方案 {rule.scheme_index} 购买结果：{label}；{balance_text}",
            "warning" if outcome != "full" else "info",
            scheme_index=rule.scheme_index,
            outcome=outcome,
            quantity=rule.total_quantity if outcome == "full" else 0,
            listed_price=listed_price,
            displayed_unit_price=listed_price / rule.total_quantity,
            balance_before=balance_before,
            balance_after=balance_after,
            spent=spent,
            actual_average=actual_average,
            included_in_average=outcome == "full" and actual_average is not None,
        )

    def _sleep(self, seconds: float) -> None:
        self._pause_at_safe_point()
        deadline = self._monotonic() + max(seconds, 0.0)
        while True:
            if self._stop.is_set():
                raise WorkflowStopped()
            self._pause_at_safe_point()
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                return
            if self._stop.wait(min(0.10, remaining)):
                raise WorkflowStopped()

    def _check_stop(self) -> None:
        if self._stop.is_set():
            raise WorkflowStopped()
        self._check_finish_loadout_round()
        self._pause_at_safe_point()
        self._check_finish_loadout_round()

    def _pause_at_safe_point(self) -> None:
        pause_requested = getattr(self, "_pause_requested", None)
        if (
            pause_requested is None
            or not pause_requested.is_set()
            or getattr(self, "_pause_deferral_depth", 0) > 0
        ):
            return
        paused_at = time.monotonic()
        self._paused.set()
        self._emit_structured(
            "workflow_pause_state",
            "自动流程已在安全点暂停；请尽量恢复暂停前的页面，再按 Ctrl+空格继续；Ctrl+Esc 终止本轮",
            status="paused",
            page=self._last_page,
        )
        while pause_requested.is_set():
            if self._scheduled_finishing() or self._loadout_finishing():
                pause_requested.clear()
                break
            self._check_finish_loadout_round()
            if self._stop.wait(0.10):
                self._paused.clear()
                raise WorkflowStopped()
        paused_seconds = max(0.0, time.monotonic() - paused_at)
        self._paused_duration_total += paused_seconds
        self._extend_pending_listing_deadline(paused_seconds)
        self._revalidate_game_after_pause()
        self._paused.clear()
        self._emit_structured(
            "workflow_pause_state",
            f"暂停结束，已重新绑定并识别当前页面；暂停 {paused_seconds:.1f} 秒",
            status="resumed",
            page=self._last_page,
            paused_seconds=round(paused_seconds, 3),
        )

    def _revalidate_game_after_pause(self) -> None:
        while True:
            if self._stop.is_set():
                raise WorkflowStopped()
            window = self.game.refresh()
            if window is None:
                # Some mail-storage stages intentionally run while the game
                # is closed and the launcher is active.
                return
            try:
                observation = self._observe_unified_game()
            except (OSError, RuntimeError):
                if self._stop.wait(0.50):
                    raise WorkflowStopped()
                continue
            if observation.page_id is not GamePageId.UNKNOWN:
                return
            self._emit(
                "继续前暂未识别到游戏页面；保持无输入等待，不会重放暂停前动作",
                "warning",
            )
            if self._stop.wait(0.50):
                raise WorkflowStopped()

    def _extend_pending_listing_deadline(self, paused_seconds: float) -> None:
        pending = getattr(self, "_pending_warehouse_listing", None)
        if pending is None or paused_seconds <= 0:
            return
        self._pending_warehouse_listing = replace(
            pending,
            started_at=pending.started_at + paused_seconds,
            deadline=pending.deadline + paused_seconds,
        )
        checkpoint = getattr(self, "_loadout_checkpoint", None)
        if checkpoint is not None:
            self._save_loadout_checkpoint(
                checkpoint.stage,
                last_safe_action=checkpoint.last_safe_action,
            )

    def _monotonic(self) -> float:
        return time.monotonic() - getattr(self, "_paused_duration_total", 0.0)

    def _begin_pause_deferral(self) -> None:
        self._pause_deferral_depth = getattr(self, "_pause_deferral_depth", 0) + 1

    def _end_pause_deferral(self) -> None:
        self._pause_deferral_depth = max(
            0,
            getattr(self, "_pause_deferral_depth", 0) - 1,
        )
        self._pause_at_safe_point()

    def _restore_on_exit(self) -> tuple[bool, str | None]:
        try:
            status = self.pak.inspect()
            if status in {"staged", "partial"} and self.settings.allow_file_operations:
                self.pak.restore()
                status = self.pak.inspect()
                message = (
                    f"退出前已恢复 PAK：{self.pak.staged_display} → "
                    f"{self.pak.source_display}（状态：{status}）"
                )
                self.on_event(WorkflowEvent(self._state, message, "warning"))
                self.diagnostics.event("pak", action="emergency_restore", status=status)
            if status != "ready":
                return False, f"PAK 最终状态为 {status}"
            return True, None
        except Exception as exc:
            self.on_event(WorkflowEvent(self._state, f"退出前恢复 PAK 失败：{exc}", "error"))
            return False, str(exc)

    def _close_runtime_resources(self) -> None:
        input_controller = getattr(self, "input", None)
        if input_controller is not None:
            input_controller.restore_launcher_topmost()
        capture = getattr(self, "capture", None)
        if capture is not None:
            capture.close()
        self.diagnostics.close()
        from bulletbot.ocr.isolated_runtime import close_ocr_worker_client

        close_ocr_worker_client()

    def _save_failure_frame(self) -> None:
        for binding, label in (
            (self.launcher_settings, "launcher_settings_failure"),
            (self.game, "game_failure"),
            (self.launcher, "launcher_failure"),
        ):
            try:
                window = binding.refresh()
                if window is None:
                    continue
                frame = self.capture.capture(window)
                self.diagnostics.save_frame(label, frame.image)
                return
            except Exception:
                continue
